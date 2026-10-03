# =============================================================================
# 02_clustering_pca.py
#
# Cluster UK local authorities on 15 local indicators (productivity, skills,
# health, connectivity, housing) using k-means, then project the clusters
# into two dimensions with PCA. Reports which cluster Glasgow City sits in,
# and identifies LADs in that cluster that outperform Glasgow on total IS8
# employment share (aspiration LADs).
#
# Depends on: data/raw/Employee_counts_IS8_LADs.parquet,
#             data/raw/merged_cleaned.csv,
#             data/raw/uk_lad_boundaries_2023.geojson
# =============================================================================

from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from adjustText import adjust_text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---- Setup ------------------------------------------------------------------

DATA_DIR = Path("data/raw")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CHOSEN_K = 5

GCR_LADS = [
    "Glasgow City", "East Dunbartonshire", "West Dunbartonshire",
    "North Lanarkshire", "South Lanarkshire", "East Renfrewshire",
    "Renfrewshire", "Inverclyde",
]

GCR_CODES = [
    "S12000049", "S12000045", "S12000039", "S12000050",
    "S12000029", "S12000011", "S12000038", "S12000018",
]

# Elastic net picks (see 03_elastic_net.py) plus related economic indicators.
CLUSTER_VARS = [
    "GVA per hour", "Level 3+ qualifications", "High growth enterprises",
    "Broadband availability", "4G area coverage", "research_income_000s",
    "total_HE_students", "Under 75 mortality rate", "median_house_price",
    "Weekly pay", "Apprenticeship achievements", "Smokers",
    "Employment rate", "GDHI per head", "Active enterprises",
]


# ---- Load data --------------------------------------------------------------

print("Loading data...")

employee_counts = pd.read_parquet(
    DATA_DIR / "Employee_counts_IS8_LADs.parquet", engine="fastparquet"
)
merged = pd.read_csv(DATA_DIR / "merged_cleaned.csv")

print(f"  Employee counts: {len(employee_counts):,} rows")
print(f"  Merged indicators: {len(merged)} LADs")


# ---- Build IS8 employment shares from latest BRES year ---------------------

latest_year = employee_counts["YEAR"].max()
latest = employee_counts[employee_counts["YEAR"] == latest_year].copy()

emp_total = (
    latest[latest["IS8_SECTOR"] == "Total"]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("total_employment")
    .reset_index()
)

emp_is8 = (
    latest[latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum()
    .reset_index()
)
emp_is8 = emp_is8.merge(emp_total, on="GEOGRAPHY_CODE", how="left")
emp_is8["emp_share"] = emp_is8["OBS_VALUE"] / emp_is8["total_employment"] * 100

emp_share_wide = emp_is8.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="emp_share", aggfunc="first",
).reset_index()
emp_share_wide.columns = ["GEOGRAPHY_CODE"] + [
    f"emp_share_{col}" for col in emp_share_wide.columns[1:]
]
emp_share_wide["emp_share_total_IS8"] = emp_share_wide[
    [c for c in emp_share_wide.columns
     if c.startswith("emp_share_") and c != "emp_share_total_IS8"]
].sum(axis=1)

merged = merged.merge(
    emp_share_wide, left_on="LAD23CD", right_on="GEOGRAPHY_CODE", how="left"
).drop(columns=["GEOGRAPHY_CODE"], errors="ignore")

print(f"  Latest BRES year: {latest_year}")


# ---- Assemble clustering dataset --------------------------------------------

available_vars = [c for c in CLUSTER_VARS if c in merged.columns]
missing_vars = [c for c in CLUSTER_VARS if c not in merged.columns]

if missing_vars:
    print(f"\n  WARNING: {len(missing_vars)} variables not in merged, dropped:")
    for v in missing_vars:
        print(f"    {v}")

cluster_df = merged[["LAD23CD", "LAD23NM"] + available_vars].dropna().copy()
n_dropped = len(merged) - len(cluster_df)

print(f"\nLADs with complete data: {len(cluster_df)} of {len(merged)} "
      f"({n_dropped} dropped for missing values)")
print(f"Variables used: {len(available_vars)}")

# Glasgow needs to survive the NaN drop, or the exercise is pointless.
gla_in = (cluster_df["LAD23NM"] == "Glasgow City").any()
print(f"Glasgow City included: {'Yes' if gla_in else 'NO, check missingness'}")


# ---- Standardise ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[available_vars].values)


# ---- Choose k with elbow and silhouette ------------------------------------

print("\nEvaluating k = 2 to 11...")

k_range = range(2, 12)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, "bo-")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_title("Elbow method")

axes[1].plot(k_range, silhouettes, "ro-")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score (higher is better)")
axes[1].axvline(CHOSEN_K, color="grey", linestyle="--", alpha=0.5,
                label=f"Chosen k={CHOSEN_K}")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()

print("Silhouette scores:")
for k, sil in zip(k_range, silhouettes):
    marker = "  (chosen)" if k == CHOSEN_K else ""
    print(f"  k={k:2d}: {sil:.3f}{marker}")


# ---- Fit final model --------------------------------------------------------

km = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_STATE, n_init=10)
cluster_df["cluster"] = km.fit_predict(X_scaled)

print("\nCluster sizes:")
for c, n in cluster_df["cluster"].value_counts().sort_index().items():
    print(f"  Cluster {c}: {n} LADs")


# ---- Interpret clusters ----------------------------------------------------

# Normalise so cells read as a multiple of the UK average.
cluster_profiles = cluster_df.groupby("cluster")[available_vars].mean()
uk_means = cluster_df[available_vars].mean()
cluster_normalised = cluster_profiles / uk_means

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
    cluster_normalised, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
    ax=ax, linewidths=0.5, vmin=0.3, vmax=2.0,
    cbar_kws={"label": "Ratio to sample mean"},
)
ax.set_title("Cluster profiles vs sample mean (green above, red below)")
ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nCluster interpretation (features > 1.2x or < 0.8x sample mean):")
for c in range(CHOSEN_K):
    profile = cluster_normalised.loc[c]
    n = (cluster_df["cluster"] == c).sum()
    highs = profile[profile > 1.2].index.tolist()
    lows = profile[profile < 0.8].index.tolist()
    print(f"\n  Cluster {c} (n={n}):")
    if highs:
        print(f"    Above: {', '.join(highs)}")
    if lows:
        print(f"    Below: {', '.join(lows)}")


# ---- Glasgow's cluster placement -------------------------------------------

print("\n---- Glasgow City Region cluster assignments ----")

gcr = cluster_df[cluster_df["LAD23NM"].isin(GCR_LADS)].sort_values("cluster")
print(gcr[["LAD23NM", "cluster"]].to_string(index=False))

glasgow_cluster = cluster_df.loc[
    cluster_df["LAD23NM"] == "Glasgow City", "cluster"
].iloc[0]
peers = cluster_df[cluster_df["cluster"] == glasgow_cluster]["LAD23NM"].sort_values()

print(f"\nGlasgow City is in Cluster {glasgow_cluster}.")
print(f"Peers ({len(peers)} LADs):")
for p in peers:
    print(f"  {p}")


# ---- Aspiration LADs -------------------------------------------------------

# Peers in Glasgow's cluster that beat it on total IS8 share.
print("\n---- Aspiration LADs: same cluster as Glasgow, higher IS8 share ----")

peer_data = cluster_df[cluster_df["cluster"] == glasgow_cluster].merge(
    merged[["LAD23CD", "emp_share_total_IS8"]], on="LAD23CD", how="left"
)
glasgow_is8 = peer_data.loc[
    peer_data["LAD23NM"] == "Glasgow City", "emp_share_total_IS8"
].iloc[0]

aspirations = peer_data[
    peer_data["emp_share_total_IS8"] > glasgow_is8
].sort_values("emp_share_total_IS8", ascending=False)

print(f"\nGlasgow's total IS8 share: {glasgow_is8:.2f}%")
print(f"LADs in same cluster with higher share: {len(aspirations)}")

if len(aspirations) > 0:
    print("\nTop 15 by IS8 share:")
    print(aspirations[["LAD23NM", "emp_share_total_IS8"]]
          .head(15).to_string(index=False))

    asp_means = aspirations[available_vars].mean()
    gla_vals = peer_data.loc[
        peer_data["LAD23NM"] == "Glasgow City", available_vars
    ].iloc[0]
    diff = (asp_means - gla_vals).sort_values(ascending=False)

    print("\nFeature means, aspiration LADs vs Glasgow:")
    print(f"  {'Feature':<32s} {'Glasgow':>12s} {'Aspirations':>12s} {'Diff':>12s}")
    print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*12}")
    for v in diff.index:
        print(f"  {v:<32s} {gla_vals[v]:>12,.2f} "
              f"{asp_means[v]:>12,.2f} {diff[v]:>+12,.2f}")


# ---- Visualise clusters in PCA space ---------------------------------------

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_result = pca.fit_transform(X_scaled)
cluster_df["PC1"] = pca_result[:, 0]
cluster_df["PC2"] = pca_result[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set2(np.linspace(0, 1, CHOSEN_K))
for c in range(CHOSEN_K):
    mask = cluster_df["cluster"] == c
    ax.scatter(
        cluster_df.loc[mask, "PC1"], cluster_df.loc[mask, "PC2"],
        c=[colors[c]], s=20, alpha=0.5, label=f"Cluster {c}",
    )

gcr_data = cluster_df[cluster_df["LAD23NM"].isin(GCR_LADS)]
ax.scatter(
    gcr_data["PC1"], gcr_data["PC2"], c="red", s=100,
    edgecolors="black", linewidth=1.5, zorder=5, label="Glasgow City Region",
)

texts = [
    ax.text(row["PC1"], row["PC2"], row["LAD23NM"], fontsize=8)
    for _, row in gcr_data.iterrows()
]
adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
    expand=(2.0, 2.0),
    force_text=(0.5, 0.8),
    force_static=(0.3, 0.5),
    min_arrow_len=5,
)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)")
ax.set_title(f"K-means clusters (k={CHOSEN_K}) in PCA space, GCR highlighted")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_pca.png", dpi=150, bbox_inches="tight")
plt.close()


# ---- Interactive cluster map -----------------------------------------------

print("\nBuilding interactive cluster map...")

boundaries = gpd.read_file(DATA_DIR / "uk_lad_boundaries_2023.geojson")

gdf = boundaries.merge(
    cluster_df[["LAD23CD", "LAD23NM", "cluster"]],
    on="LAD23CD", how="inner",
    suffixes=("", "_drop"),
).to_crs("EPSG:4326")
gdf = gdf.drop(columns=[c for c in gdf.columns if c.endswith("_drop")])

# Simplify polygons to keep the HTML file small.
gdf["geometry"] = gdf["geometry"].simplify(
    tolerance=0.01, preserve_topology=True
)

m = folium.Map(location=[54.5, -2], zoom_start=6, tiles=None)

folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    data=gdf[["LAD23CD", "cluster"]],
    columns=["LAD23CD", "cluster"],
    key_on="feature.properties.LAD23CD",
    fill_color="Set2",
    fill_opacity=0.6,
    line_opacity=0.3,
    legend_name=f"K-means cluster (k={CHOSEN_K})",
    nan_fill_color="lightgrey",
).add_to(m)

# GCR outline with thick black border.
gcr_gdf = gdf[gdf["LAD23CD"].isin(GCR_CODES)]
folium.GeoJson(
    gcr_gdf.__geo_interface__,
    style_function=lambda x: {"fillOpacity": 0, "color": "black", "weight": 1.5},
    name="Glasgow City Region",
).add_to(m)

# Invisible layer for hover tooltips.
folium.GeoJson(
    gdf.__geo_interface__,
    style_function=lambda x: {"fillOpacity": 0, "weight": 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["LAD23NM", "cluster"],
        aliases=["Local Authority:", "Cluster:"],
        localize=True, sticky=True,
    ),
).add_to(m)

folium.LayerControl().add_to(m)

map_path = OUT_DIR / "cluster_map.html"
m.save(str(map_path))
print(f"  Saved to {map_path}")

print("\nDone.")