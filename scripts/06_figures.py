# =============================================================================
# 06_readme_figures.py
#
# Produces polished, Poppins-styled versions of the six headline figures
# embedded in the README. Rebuilds each figure's data from source rather
# than loading the diagnostic PNGs from earlier scripts, so visual styling
# can be tuned without rerunning the full pipeline. Outputs go to
# figures/readme/ as both PNG (for the README) and PDF (for writing
# samples).
#
# Depends on: data/raw/Employee_counts_IS8_LADs.parquet,
#             data/raw/Business_counts_IS8_LADs.parquet,
#             data/raw/merged_cleaned.csv,
#             data/raw/uk_lad_boundaries_2023.geojson,
#             fonts/Poppins-Regular.ttf, Poppins-SemiBold.ttf, Poppins-Bold.ttf
# =============================================================================

import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ---- Setup ------------------------------------------------------------------

DATA_DIR = Path("data/raw")
FONT_DIR = Path("fonts")
OUT_DIR = Path("figures/readme")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for font_file in ["Poppins-Regular.ttf", "Poppins-SemiBold.ttf", "Poppins-Bold.ttf"]:
    font_path = FONT_DIR / font_file
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams.update({
    "font.family":        "Poppins",
    "font.size":          12,
    "axes.titlesize":     16,
    "axes.titleweight":   "semibold",
    "axes.labelsize":     13,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#333333",
    "axes.labelcolor":    "#333333",
    "axes.linewidth":     0.8,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    12,
    "legend.frameon":     False,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "savefig.bbox":       "tight",
    "savefig.dpi":        180,
})

NAVY     = "#1a1f3a"
ACCENT   = "#3b5bdb"
RED      = "#d73027"
BLUE     = "#4575b4"
ORANGE   = "#fc8d59"
LIGHTBLU = "#91bfdb"
GREY     = "#999999"
TEXT     = "#333333"

SECTOR_COLORS = {
    "Advanced manufacturing":             "#8c510a",
    "Creative Industries":                RED,
    "Defence sector":                     "#4d4d4d",
    "Digital and Technology":             BLUE,
    "Financial Services":                 ORANGE,
    "Life Sciences":                      "#1a9850",
    "Professional and Business Services": "#762a83",
}

RANDOM_STATE = 42
CV_FOLDS     = 10
L1_RATIOS    = [0.1, 0.3, 0.5, 0.7, 0.9]

GCR_CODES = [
    "S12000049", "S12000045", "S12000039", "S12000050",
    "S12000029", "S12000011", "S12000038", "S12000018",
]
GLASGOW_CODE = "S12000049"
LAD_TYPE_FILTER = "district|unitary"

EXCLUDE_FROM_X = [
    "LAD23CD", "LAD23NM", "LONG", "LAT", "geometry",
    "GEOGRAPHY_CODE", "GEOGRAPHY_NAME", "GEOGRAPHY_TYPE",
    "Local Authority District", "County or Unitary Authority",
    "Country", "Nation", "Region", "Combined Authority or City Region",
    "BNG_E", "BNG_N", "FID", "GlobalID",
]


def save_fig(fig, name):
    """Save a figure as both PNG and PDF to figures/readme/."""
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / name}.png (+ pdf)")


def wrap_title(text, fig_width_inches, chars_per_inch=10):
    """Wrap title text to roughly match the figure's rendered width."""
    max_chars = int(fig_width_inches * chars_per_inch)
    return textwrap.fill(text, width=max_chars, break_long_words=False)


# ---- Load data --------------------------------------------------------------

print("Loading data...")
business_counts = pd.read_parquet(
    DATA_DIR / "Business_counts_IS8_LADs.parquet", engine="fastparquet"
)
employee_counts = pd.read_parquet(
    DATA_DIR / "Employee_counts_IS8_LADs.parquet", engine="fastparquet"
)
merged = pd.read_csv(DATA_DIR / "merged_cleaned.csv")

latest_emp_year = employee_counts["YEAR"].max()
earliest_emp_year = employee_counts["YEAR"].min()
latest_biz_year = business_counts["YEAR"].max()


# ---- Build outcomes and LQs ------------------------------------------------

emp_lads_latest = employee_counts[
    (employee_counts["YEAR"] == latest_emp_year)
    & employee_counts["GEOGRAPHY_TYPE"].str.contains(
        LAD_TYPE_FILTER, case=False, na=False
    )
].copy()
biz_lads_latest = business_counts[
    (business_counts["YEAR"] == latest_biz_year)
    & business_counts["GEOGRAPHY_TYPE"].str.contains(
        LAD_TYPE_FILTER, case=False, na=False
    )
].copy()

emp_total_lad = (
    emp_lads_latest[emp_lads_latest["IS8_SECTOR"] == "Total"]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("total_employment").reset_index()
)
emp_is8 = (
    emp_lads_latest[emp_lads_latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
    .merge(emp_total_lad, on="GEOGRAPHY_CODE", how="left")
)
emp_is8["emp_share"] = emp_is8["OBS_VALUE"] / emp_is8["total_employment"] * 100

emp_share_wide = emp_is8.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="emp_share", aggfunc="first",
).reset_index()
emp_share_wide.columns = ["GEOGRAPHY_CODE"] + [
    f"emp_share_{c}" for c in emp_share_wide.columns[1:]
]
emp_share_wide["emp_share_total_IS8"] = emp_share_wide[
    [c for c in emp_share_wide.columns
     if c.startswith("emp_share_") and c != "emp_share_total_IS8"]
].sum(axis=1)

biz_total_lad = (
    biz_lads_latest[biz_lads_latest["IS8_SECTOR"] == "Total"]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("total_businesses").reset_index()
)
biz_is8 = (
    biz_lads_latest[biz_lads_latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
    .merge(biz_total_lad, on="GEOGRAPHY_CODE", how="left")
)
biz_is8["biz_share"] = biz_is8["OBS_VALUE"] / biz_is8["total_businesses"] * 100

biz_share_wide = biz_is8.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="biz_share", aggfunc="first",
).reset_index()
biz_share_wide.columns = ["GEOGRAPHY_CODE"] + [
    f"biz_share_{c}" for c in biz_share_wide.columns[1:]
]
biz_share_wide["biz_share_total_IS8"] = biz_share_wide[
    [c for c in biz_share_wide.columns
     if c.startswith("biz_share_") and c != "biz_share_total_IS8"]
].sum(axis=1)

for df_to_merge in [emp_share_wide, biz_share_wide]:
    merged = merged.merge(
        df_to_merge, left_on="LAD23CD", right_on="GEOGRAPHY_CODE", how="left"
    ).drop(columns=["GEOGRAPHY_CODE"], errors="ignore")

uk_emp_by_sector = emp_lads_latest.groupby("IS8_SECTOR")["OBS_VALUE"].sum()
uk_total_emp = float(uk_emp_by_sector["Total"])
uk_sector_share = uk_emp_by_sector.drop("Total") / uk_total_emp

lad_sector = (
    emp_lads_latest[emp_lads_latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
    .merge(emp_total_lad, on="GEOGRAPHY_CODE", how="left")
)
lad_sector["local_share"] = lad_sector["OBS_VALUE"] / lad_sector["total_employment"]
lad_sector["uk_share"] = lad_sector["IS8_SECTOR"].map(uk_sector_share)
lad_sector["LQ"] = lad_sector["local_share"] / lad_sector["uk_share"]
lq_wide = lad_sector.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="LQ", aggfunc="first",
).reset_index()
lq_wide.columns = ["GEOGRAPHY_CODE"] + [f"LQ_{c}" for c in lq_wide.columns[1:]]

merged = merged.merge(
    lq_wide, left_on="LAD23CD", right_on="GEOGRAPHY_CODE", how="left"
).drop(columns=["GEOGRAPHY_CODE"], errors="ignore")

lq_cols = [c for c in merged.columns if c.startswith("LQ_")]
outcome_cols = [c for c in merged.columns
                if c.startswith(("emp_share_", "biz_share_"))]
all_numeric = [
    c for c in merged.columns
    if c not in EXCLUDE_FROM_X and c not in lq_cols and c not in outcome_cols
    and merged[c].dtype in ["float64", "int64", "int16"]
]
scottish_rows = merged[merged["LAD23CD"].str.startswith("S12")]
X_cols_uk = [c for c in all_numeric if not scottish_rows[c].isna().all()]

print(f"  Merged: {merged.shape}, Model A features: {len(X_cols_uk)}")


# ---- Figure 1: UK IS8 growth trends ----------------------------------------

print("\n---- Figure 1: UK IS8 growth trends ----")

emp_timeseries = employee_counts[
    employee_counts["GEOGRAPHY_TYPE"].str.contains(
        LAD_TYPE_FILTER, case=False, na=False
    )
].copy()
sector_by_year = (
    emp_timeseries[emp_timeseries["IS8_SECTOR"] != "Total"]
    .groupby(["YEAR", "IS8_SECTOR"])["OBS_VALUE"].sum().unstack()
)
indexed = sector_by_year.div(sector_by_year.iloc[0]) * 100

order = indexed.iloc[-1].sort_values(ascending=False).index.tolist()


def spread_labels(endpoints, min_gap=1.8):
    sorted_items = sorted(endpoints.items(), key=lambda x: x[1], reverse=True)
    adjusted = {}
    last_y = None
    for sector, y in sorted_items:
        if last_y is not None and (last_y - y) < min_gap:
            y = last_y - min_gap
        adjusted[sector] = y
        last_y = y
    return adjusted


endpoint_ys = {s: float(indexed[s].iloc[-1]) for s in order}
label_ys = spread_labels(endpoint_ys, min_gap=1.8)

fig, ax = plt.subplots(figsize=(11, 6.5))
for sector in order:
    ax.plot(
        indexed.index, indexed[sector],
        label=sector, color=SECTOR_COLORS.get(sector, GREY),
        linewidth=2.4, marker="o", markersize=4,
    )

ax.axhline(y=100, color=GREY, linewidth=0.6, linestyle="--", zorder=0)
ax.set_xlabel("Year", color=TEXT)
ax.set_ylabel(f"Employment index ({earliest_emp_year} = 100)", color=TEXT)
ax.set_xticks(sector_by_year.index)
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

for sector in order:
    y_label = label_ys[sector]
    y_actual = endpoint_ys[sector]
    if abs(y_actual - y_label) > 0.3:
        ax.plot(
            [indexed.index[-1], indexed.index[-1] + 0.4],
            [y_actual, y_label],
            color=SECTOR_COLORS.get(sector, GREY),
            linewidth=0.9, alpha=0.5, zorder=1, clip_on=False,
        )
    ax.annotate(
        sector,
        xy=(indexed.index[-1] + 0.4, y_label),
        xytext=(5, 0), textcoords="offset points",
        color=SECTOR_COLORS.get(sector, GREY),
        fontsize=10, fontweight="semibold",
        va="center",
    )

ax.set_xlim(indexed.index[0] - 0.3, indexed.index[-1] + 3.8)

fig.suptitle(
    wrap_title(
        "Life Sciences led UK IS8 growth with +32%, while Advanced Manufacturing was the only sector to decline",
        fig.get_size_inches()[0], chars_per_inch=9,
    ),
    x=0.02, y=0.98, ha="left", color=NAVY,
    fontsize=16, fontweight="semibold",
)
fig.subplots_adjust(top=0.88)
save_fig(fig, "01_uk_growth_trends")


# ---- Figure 2: UK LAD clusters map -----------------------------------------

print("\n---- Figure 2: UK LAD clusters map ----")

lq_matrix = merged[["LAD23CD"] + lq_cols].dropna()
X_lq = StandardScaler().fit_transform(lq_matrix[lq_cols].values)
km = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
lq_matrix = lq_matrix.copy()
lq_matrix["cluster"] = km.fit_predict(X_lq)

# Dominant cluster (by size) gets a muted slate background that keeps some
# character without being as heavy as navy; smaller clusters get bright accents.
cluster_sizes = lq_matrix.groupby("cluster").size().sort_values(ascending=False)
accent_palette = ["#1a9850", "#fd8d3c", "#3b5bdb", "#762a83"]
CLUSTER_COLORS = {}
CLUSTER_LABELS = {}
for i, (cluster_id, size) in enumerate(cluster_sizes.items()):
    if i == 0:
        CLUSTER_COLORS[cluster_id] = "#4a5c73"
    else:
        CLUSTER_COLORS[cluster_id] = accent_palette[i - 1]
    CLUSTER_LABELS[cluster_id] = f"Cluster {cluster_id} (n={size:.0f})"

lads = gpd.read_file(DATA_DIR / "uk_lad_boundaries_2023.geojson")
lads = lads.merge(
    lq_matrix[["LAD23CD", "cluster"]],
    on="LAD23CD", how="left",
)

fig, ax = plt.subplots(figsize=(9, 11))
lads.plot(
    ax=ax,
    color=lads["cluster"].map(CLUSTER_COLORS).fillna("#eeeeee"),
    edgecolor="white", linewidth=0.2,
)
ax.set_axis_off()

# Pin title, map and legend to the same left anchor; tighten the title-to-map
# gap by lifting the top of the axes closer to the title.
fig.suptitle(
    wrap_title(
        "73% of UK local authorities share one baseline industrial profile; just 27% specialise in distinct clusters",
        fig.get_size_inches()[0], chars_per_inch=9,
    ),
    x=0.08, y=0.98, ha="left", color=NAVY,
    fontsize=16, fontweight="semibold",
)

legend_patches = [
    mpatches.Patch(color=CLUSTER_COLORS[cid], label=CLUSTER_LABELS[cid])
    for cid in sorted(CLUSTER_LABELS)
]
fig.legend(
    handles=legend_patches,
    loc="lower left",
    ncol=5,
    bbox_to_anchor=(0.08, 0.03),
    fontsize=11,
    frameon=False,
    handlelength=1.4,
    handleheight=1.1,
    columnspacing=1.8,
)
fig.subplots_adjust(top=0.96, bottom=0.08, left=0.0, right=1.0)
save_fig(fig, "02_uk_clusters_map")


# ---- Figure 3: Elastic net top predictors by sector ------------------------

print("\n---- Figure 3: Elastic net top predictors by sector ----")

SECTOR_OUTCOMES = {
    "Advanced Manufacturing":  "emp_share_Advanced manufacturing",
    "Creative Industries":     "emp_share_Creative Industries",
    "Digital & Technology":    "emp_share_Digital and Technology",
    "Financial Services":      "emp_share_Financial Services",
    "Life Sciences":           "emp_share_Life Sciences",
    "Prof. & Business Svcs":   "emp_share_Professional and Business Services",
}

LABEL_OVERRIDES_HEATMAP = {
    "research_income_000s":        "Research income\n(£ thousands)",
    "median_house_price":          "Median house price",
    "total_HE_students":           "Total HE students",
    "Apprenticeship achievements": "Apprenticeship\nachievements",
}

top_predictors = {}
for sector_label, outcome_col in SECTOR_OUTCOMES.items():
    if outcome_col not in merged.columns:
        continue
    model_df = merged[merged[outcome_col].notna()].dropna(subset=X_cols_uk).copy()
    if len(model_df) < 50:
        continue
    X = model_df[X_cols_uk].values
    y = model_df[outcome_col].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    enet = ElasticNetCV(l1_ratio=L1_RATIOS, cv=cv,
                        random_state=RANDOM_STATE, max_iter=10000)
    enet.fit(Xs, y)
    coefs = pd.Series(enet.coef_, index=X_cols_uk)
    nonzero = coefs[coefs != 0]
    top5 = nonzero.reindex(nonzero.abs().sort_values(ascending=False).index[:5])
    top_predictors[sector_label] = top5

rank_cols = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]
rows = list(top_predictors.keys())
grid = pd.DataFrame(index=rows, columns=rank_cols, dtype=float)
labels = pd.DataFrame(index=rows, columns=rank_cols, dtype=object)
for sector_label, series in top_predictors.items():
    for i, (var, coef) in enumerate(series.items()):
        if i >= 5:
            break
        grid.loc[sector_label, rank_cols[i]] = coef
        sign = "+" if coef > 0 else "–"
        display_var = LABEL_OVERRIDES_HEATMAP.get(var, var)
        labels.loc[sector_label, rank_cols[i]] = f"{display_var}\n({sign})"

fig, ax = plt.subplots(figsize=(14, 6.2))
vmax = float(np.nanmax(np.abs(grid.values)))
im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

for i, row in enumerate(rows):
    for j, col in enumerate(rank_cols):
        text = labels.loc[row, col]
        if pd.isna(text):
            continue
        coef = grid.loc[row, col]
        text_color = "white" if abs(coef) > 0.6 * vmax else TEXT
        ax.text(j, i, text, ha="center", va="center",
                color=text_color, fontsize=9, linespacing=1.25)

ax.set_xticks(range(len(rank_cols)))
ax.set_xticklabels(rank_cols)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(rank_cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Standardised coefficient", fontsize=10, color=TEXT)
cbar.ax.tick_params(labelsize=9)

fig.suptitle(
    wrap_title(
        "Productivity (GVA per hour) and high-growth enterprises predict IS8 presence across most sectors",
        fig.get_size_inches()[0], chars_per_inch=9,
    ),
    x=0.02, y=0.98, ha="left", color=NAVY,
    fontsize=16, fontweight="semibold",
)
fig.subplots_adjust(top=0.88)
save_fig(fig, "03_elastic_net_top_predictors")


# ---- Figure 4: Glasgow predicted vs actual (dumbbell) ----------------------

print("\n---- Figure 4: Glasgow predicted vs actual (dumbbell) ----")

OUTCOMES = {
    "Total IS8 Employment Share":    "emp_share_total_IS8",
    "Total IS8 Business Share":      "biz_share_total_IS8",
    "Financial Services":            "emp_share_Financial Services",
    "Prof & Business Services":      "emp_share_Professional and Business Services",
    "Adv Manufacturing":             "emp_share_Advanced manufacturing",
    "Digital & Tech":                "emp_share_Digital and Technology",
    "Defence":                       "emp_share_Defence sector",
    "Life Sciences":                 "emp_share_Life Sciences",
    "Creative Industries":           "emp_share_Creative Industries",
}

pva_rows = []
for name, col in OUTCOMES.items():
    if col not in merged.columns:
        continue
    model_df = merged[merged[col].notna()].dropna(subset=X_cols_uk).copy()
    if len(model_df) < 50:
        continue
    X = model_df[X_cols_uk].values
    y = model_df[col].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    enet = ElasticNetCV(l1_ratio=L1_RATIOS, cv=cv,
                        random_state=RANDOM_STATE, max_iter=10000)
    enet.fit(Xs, y)
    gcr_row = model_df[model_df["LAD23CD"] == GLASGOW_CODE]
    if len(gcr_row) == 0:
        continue
    actual = float(gcr_row[col].iloc[0])
    predicted = float(enet.predict(scaler.transform(gcr_row[X_cols_uk].values))[0])
    pva_rows.append({
        "Outcome": name, "Actual": actual,
        "Predicted": predicted, "Residual": actual - predicted,
    })

pva_df = pd.DataFrame(pva_rows).sort_values("Residual", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 7))

y_positions = list(range(len(pva_df)))
for i, row in pva_df.iterrows():
    pred = row["Predicted"]
    actual = row["Actual"]
    residual = row["Residual"]
    color = ACCENT if residual >= 0 else RED

    ax.plot([pred, actual], [i, i],
            color="#cccccc", linewidth=2.2, zorder=1)
    ax.scatter(pred, i, s=130, color="white",
               edgecolor="#777777", linewidth=1.6, zorder=2)
    ax.scatter(actual, i, s=160, color=color, zorder=3)

    # Larger offset so the pp label clears the actual dot.
    if residual >= 0:
        label_x = actual + 2.0
        ha = "left"
    else:
        label_x = actual - 2.0
        ha = "right"
    ax.text(label_x, i, f"{residual:+.2f}pp",
            va="center", ha=ha, fontsize=13,
            color=color, fontweight="semibold")

ax.set_yticks(y_positions)
ax.set_yticklabels(pva_df["Outcome"], fontsize=12)
ax.invert_yaxis()

ax.set_xlabel("Employment or business share (%)", fontsize=13)
ax.grid(axis="x", alpha=0.25, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

xmax = max(pva_df["Actual"].max(), pva_df["Predicted"].max())
# Extra room on both sides: left to clear the red '-xx pp' labels past 0,
# right to clear the blue labels after the actual-share dot.
ax.set_xlim(-7, xmax + 12)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#777777", markeredgewidth=1.6,
           markersize=11, label="Predicted", linewidth=0),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT,
           markersize=12, label="Actual (over-performing)", linewidth=0),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
           markersize=12, label="Actual (under-performing)", linewidth=0),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

fig.suptitle(
    wrap_title(
        "Glasgow's IS8 employment exceeds model predictions by +4.21pp, led by Financial Services",
        fig.get_size_inches()[0], chars_per_inch=9,
    ),
    x=0.02, y=0.97, ha="left", color=NAVY,
    fontsize=16, fontweight="semibold",
)
fig.subplots_adjust(top=0.90)
save_fig(fig, "04_glasgow_predicted_vs_actual")


# ---- Figure 5: Shift-share decomposition -----------------------------------

print("\n---- Figure 5: Shift-share decomposition ----")

earliest_year = employee_counts["YEAR"].min()
lad_mask = employee_counts["GEOGRAPHY_TYPE"].str.contains(
    LAD_TYPE_FILTER, case=False, na=False
)
uk_early = employee_counts[(employee_counts["YEAR"] == earliest_year) & lad_mask] \
    .groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()
uk_late = employee_counts[(employee_counts["YEAR"] == latest_emp_year) & lad_mask] \
    .groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()
gcr_early = employee_counts[
    (employee_counts["YEAR"] == earliest_year)
    & (employee_counts["GEOGRAPHY_CODE"].isin(GCR_CODES))
].groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()
gcr_late = employee_counts[
    (employee_counts["YEAR"] == latest_emp_year)
    & (employee_counts["GEOGRAPHY_CODE"].isin(GCR_CODES))
].groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()

uk_total_early = uk_early["Total"]
uk_total_late = uk_late["Total"]
nat_growth = (uk_total_late - uk_total_early) / uk_total_early

shift_rows = []
for sector in [s for s in uk_early if s != "Total"]:
    e_ir = gcr_early.get(sector, 0)
    if e_ir == 0:
        continue
    nat_eff = e_ir * nat_growth
    uk_s_early = uk_early.get(sector, 1)
    uk_s_late = uk_late.get(sector, 0)
    sec_growth = ((uk_s_late - uk_s_early) / uk_s_early) if uk_s_early > 0 else 0
    ind_mix = e_ir * (sec_growth - nat_growth)
    gcr_sec_growth = ((gcr_late.get(sector, 0) - e_ir) / e_ir) if e_ir > 0 else 0
    loc_comp = e_ir * (gcr_sec_growth - sec_growth)
    shift_rows.append({
        "Sector": sector,
        "National": nat_eff,
        "Industry mix": ind_mix,
        "Local competitiveness": loc_comp,
        "Total": gcr_late.get(sector, 0) - e_ir,
    })
ss_df = pd.DataFrame(shift_rows).sort_values("Total")
total_change = ss_df["Total"].sum()
totals = ss_df[["National", "Industry mix", "Local competitiveness"]].sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         gridspec_kw={"width_ratios": [2, 1]})

ss_df.set_index("Sector")[["National", "Industry mix", "Local competitiveness"]].plot(
    kind="barh", stacked=True, ax=axes[0],
    color=[BLUE, ORANGE, LIGHTBLU], width=0.75, zorder=2,
)
axes[0].axvline(x=0, color=TEXT, linewidth=0.8, zorder=1)
axes[0].set_xlabel("Employment change, 2015 to 2024", fontsize=14)
axes[0].set_ylabel("")
axes[0].legend(loc="lower right", fontsize=12)
axes[0].grid(axis="x", alpha=0.25, linewidth=0.5, zorder=0)
axes[0].set_axisbelow(True)
axes[0].tick_params(axis="y", labelsize=12)
axes[0].tick_params(axis="x", labelsize=11)
axes[0].xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{int(x):+,}" if x else "0")
)

bar_colors = [BLUE, ORANGE, LIGHTBLU]
bars = axes[1].bar(range(3), totals.values, color=bar_colors, width=0.6, zorder=2)
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["National", "Industry\nmix", "Local\ncomp."], fontsize=12)
axes[1].axhline(y=0, color=TEXT, linewidth=0.8, zorder=1)
axes[1].set_ylabel("Employment change", fontsize=14)
axes[1].grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
axes[1].set_axisbelow(True)
axes[1].tick_params(axis="y", labelsize=11)
axes[1].yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{int(x):+,}" if x else "0")
)
axes[1].set_ylim(0, totals.max() * 1.25)
for bar, val in zip(bars, totals.values):
    pct = val / total_change * 100
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + totals.max() * 0.02,
                 f"{val:+,.0f}\n({pct:.0f}%)",
                 ha="center", va="bottom", fontsize=12,
                 color=TEXT, fontweight="semibold")

fig.suptitle(
    wrap_title(
        f"Only 15% of Glasgow's {total_change:+,.0f} new IS8 jobs reflect genuine local advantage",
        fig.get_size_inches()[0], chars_per_inch=10,
    ),
    x=0.05, y=0.96, ha="left", color=NAVY,
    fontsize=18, fontweight="semibold",
)
plt.tight_layout()
fig.subplots_adjust(top=0.90)
save_fig(fig, "05_glasgow_shift_share")


# ---- Figure 6: Barriers quantified in jobs ---------------------------------
# Strengths get BLUE (positive framing — what's working), barriers get RED
# (the gaps costing Glasgow jobs). Clearer subtitles below.

print("\n---- Figure 6: Glasgow barriers quantified in jobs ----")

VARIABLE_DIRECTIONS = {
    "Under 75 mortality rate":     "lower is better",
    "Broadband availability":      "higher is better",
    "Female HLE":                  "higher is better",
    "Male HLE":                    "higher is better",
    "High growth enterprises":     "higher is better",
    "Level 3+ qualifications":     "higher is better",
    "GVA per hour":                "higher is better",
    "median_house_price":          "higher is better",
    "research_income_000s":        "higher is better",
    "Smokers":                     "lower is better",
    "total_HE_students":           "higher is better",
    "Worthwhile":                  "higher is better",
    "4G area coverage":            "higher is better",
    "Anxiety":                     "lower is better",
    "Weekly pay":                  "higher is better",
    "Apprenticeship achievements": "higher is better",
    "Unemployment rate":           "lower is better",
}

LABEL_OVERRIDES = {
    "median_house_price":   "Median house price",
    "research_income_000s": "Research income",
    "total_HE_students":    "HE students",
    "Male HLE":             "Healthy life expectancy (M)",
    "Female HLE":           "Healthy life expectancy (F)",
    "Worthwhile":           "Life worthwhile score",
}

glasgow = merged[merged["LAD23NM"] == "Glasgow City"].iloc[0]
outcome_col = "emp_share_total_IS8"
model_df = merged[merged[outcome_col].notna()].dropna(subset=X_cols_uk).copy()
X = model_df[X_cols_uk].values
y = model_df[outcome_col].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
enet = ElasticNetCV(l1_ratio=L1_RATIOS, cv=cv,
                    random_state=RANDOM_STATE, max_iter=10000)
enet.fit(Xs, y)

gcr_row = model_df[model_df["LAD23CD"] == GLASGOW_CODE].copy()
baseline_pred = float(enet.predict(scaler.transform(gcr_row[X_cols_uk].values))[0])

gcr_total_emp = float(
    employee_counts[
        (employee_counts["YEAR"] == latest_emp_year)
        & (employee_counts["GEOGRAPHY_CODE"].isin(GCR_CODES))
        & (employee_counts["IS8_SECTOR"] == "Total")
    ]["OBS_VALUE"].sum()
)

barrier_rows = []
strength_rows = []
for var, direction in VARIABLE_DIRECTIONS.items():
    if var not in X_cols_uk or var not in merged.columns:
        continue
    gla_val = float(glasgow[var])
    uk_mean = float(merged[var].mean())
    gap = gla_val - uk_mean
    is_barrier = (
        (direction == "higher is better" and gap < 0)
        or (direction == "lower is better" and gap > 0)
    )
    cf = gcr_row.copy()
    cf[var] = uk_mean
    cf_pred = float(enet.predict(scaler.transform(cf[X_cols_uk].values))[0])
    change = cf_pred - baseline_pred
    jobs = change / 100 * gcr_total_emp
    label = LABEL_OVERRIDES.get(var, var)
    row = {"Variable": label, "Change (pp)": change, "Jobs": jobs}
    if is_barrier and change > 0:
        barrier_rows.append(row)
    elif not is_barrier and change < 0:
        strength_rows.append(row)

barriers_df = pd.DataFrame(barrier_rows).sort_values("Jobs", ascending=True)
strengths_df = pd.DataFrame(strength_rows).sort_values("Jobs", ascending=False)
strengths_df["Jobs at risk"] = -strengths_df["Jobs"]

health_vars = ["Under 75 mortality rate", "Male HLE", "Female HLE", "Smokers"]
health_labels = [LABEL_OVERRIDES.get(v, v) for v in health_vars]
health_jobs = barriers_df[barriers_df["Variable"].isin(health_labels)]["Jobs"].sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False)

# Barriers: RED (the gaps costing Glasgow jobs).
axes[0].barh(barriers_df["Variable"], barriers_df["Jobs"],
             color=RED, zorder=2)
for _, row in barriers_df.iterrows():
    axes[0].text(row["Jobs"] + 400, row["Variable"],
                 f"+{int(round(row['Jobs'])):,}",
                 va="center", ha="left", fontsize=13,
                 color=TEXT, fontweight="semibold")
axes[0].set_xlabel("Jobs", fontsize=14)
axes[0].text(-0.3, 1.08, "Barriers: jobs Glasgow could gain by closing each gap",
             transform=axes[0].transAxes, color=NAVY,
             fontsize=15, fontweight="semibold", ha="left", va="bottom")
axes[0].grid(axis="x", alpha=0.25, linewidth=0.5, zorder=0)
axes[0].set_axisbelow(True)
axes[0].tick_params(axis="y", labelsize=13)
axes[0].tick_params(axis="x", labelsize=11)
axes[0].xaxis.set_major_locator(MaxNLocator(5))
axes[0].xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{int(x):+,}" if x else "0")
)
axes[0].set_xlim(0, barriers_df["Jobs"].max() * 1.3)

# Strengths: BLUE (what's working for Glasgow today).
axes[1].barh(strengths_df["Variable"], strengths_df["Jobs at risk"],
             color=ACCENT, zorder=2)
for _, row in strengths_df.iterrows():
    axes[1].text(row["Jobs at risk"] + 1200, row["Variable"],
                 f"−{int(round(row['Jobs at risk'])):,}",
                 va="center", ha="left", fontsize=13,
                 color=TEXT, fontweight="semibold")
axes[1].set_xlabel("Jobs", fontsize=14)
axes[1].text(-0.33, 1.08, "Strengths: jobs Glasgow would lose without each strength",
             transform=axes[1].transAxes, color=NAVY,
             fontsize=15, fontweight="semibold", ha="left", va="bottom")
axes[1].grid(axis="x", alpha=0.25, linewidth=0.5, zorder=0)
axes[1].set_axisbelow(True)
axes[1].tick_params(axis="y", labelsize=13)
axes[1].tick_params(axis="x", labelsize=11)
axes[1].xaxis.set_major_locator(MaxNLocator(5))
axes[1].xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"−{int(x):,}" if x else "0")
)
axes[1].set_xlim(0, strengths_df["Jobs at risk"].max() * 1.3)

fig.suptitle(
    wrap_title(
        f"Health-related barriers alone account for almost {int(round(health_jobs / 1000)) * 1000:,} potential Glasgow IS8 jobs",
        fig.get_size_inches()[0], chars_per_inch=10,
    ),
    x=0.02, y=0.97, ha="left", color=NAVY,
    fontsize=18, fontweight="semibold",
)
plt.tight_layout()
fig.subplots_adjust(top=0.81)
save_fig(fig, "06_glasgow_barriers_strengths_jobs")

print("\nDone.")