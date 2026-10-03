# =============================================================================
# 05_glasgow_analysis.py
#
# Regional deep-dive on Glasgow City Region (GCR). Computes GCR's IS8 sector
# profile against the UK baseline (employment shares and location quotients).
# Fits elastic net on multiple IS8 outcomes to produce predicted-vs-actual
# residuals for Glasgow, identifying which sectors over- and underperform
# their modelled baseline. Runs a shift-share decomposition of GCR
# employment growth into national, industry mix, and local competitiveness
# components. Builds a sector co-location correlation matrix across UK LADs
# to identify natural adjacencies. Quantifies Glasgow's structural barriers
# and strengths through individual and combined counterfactuals on the
# elastic net, then converts the percentage-point impacts into estimated
# job counts.
#
# Depends on: data/raw/Employee_counts_IS8_LADs.parquet,
#             data/raw/Business_counts_IS8_LADs.parquet,
#             data/raw/merged_cleaned.csv
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ---- Setup ------------------------------------------------------------------

DATA_DIR = Path("data/raw")
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 10
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

GCR_LADS = {
    "Glasgow City":        "S12000049",
    "East Dunbartonshire": "S12000045",
    "West Dunbartonshire": "S12000039",
    "North Lanarkshire":   "S12000050",
    "South Lanarkshire":   "S12000029",
    "East Renfrewshire":   "S12000011",
    "Renfrewshire":        "S12000038",
    "Inverclyde":          "S12000018",
}
GCR_CODES = list(GCR_LADS.values())
GLASGOW_CODE = "S12000049"

EXCLUDE_FROM_X = [
    "LAD23CD", "LAD23NM", "LONG", "LAT", "geometry",
    "GEOGRAPHY_CODE", "GEOGRAPHY_NAME", "GEOGRAPHY_TYPE",
    "Local Authority District", "County or Unitary Authority",
    "Country", "Nation", "Region", "Combined Authority or City Region",
    "BNG_E", "BNG_N", "FID", "GlobalID",
]

# Direction for each variable in the barriers analysis: whether a higher
# value is better or worse for IS8 presence. Taken from the elastic net
# coefficient signs in 03_elastic_net.py.
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


# ---- Load data --------------------------------------------------------------

print("Loading data...")
business_counts = pd.read_parquet(
    DATA_DIR / "Business_counts_IS8_LADs.parquet", engine="fastparquet"
)
employee_counts = pd.read_parquet(
    DATA_DIR / "Employee_counts_IS8_LADs.parquet", engine="fastparquet"
)
merged = pd.read_csv(DATA_DIR / "merged_cleaned.csv")


# ---- Build IS8 outcome variables (same construction as 03_elastic_net.py) --

latest_emp_year = employee_counts["YEAR"].max()
emp_latest = employee_counts[employee_counts["YEAR"] == latest_emp_year].copy()

emp_total = (
    emp_latest[emp_latest["IS8_SECTOR"] == "Total"]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("total_employment").reset_index()
)
emp_is8 = (
    emp_latest[emp_latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
    .merge(emp_total, on="GEOGRAPHY_CODE", how="left")
)
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

latest_biz_year = business_counts["YEAR"].max()
biz_latest = business_counts[business_counts["YEAR"] == latest_biz_year].copy()
biz_total = (
    biz_latest[biz_latest["IS8_SECTOR"] == "Total"]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("total_businesses").reset_index()
)
biz_is8 = (
    biz_latest[biz_latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
    .merge(biz_total, on="GEOGRAPHY_CODE", how="left")
)
biz_is8["biz_share"] = biz_is8["OBS_VALUE"] / biz_is8["total_businesses"] * 100
biz_share_wide = biz_is8.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="biz_share", aggfunc="first",
).reset_index()
biz_share_wide.columns = ["GEOGRAPHY_CODE"] + [
    f"biz_share_{col}" for col in biz_share_wide.columns[1:]
]
biz_share_wide["biz_share_total_IS8"] = biz_share_wide[
    [c for c in biz_share_wide.columns
     if c.startswith("biz_share_") and c != "biz_share_total_IS8"]
].sum(axis=1)

for df_to_merge in [emp_share_wide, biz_share_wide]:
    merged = merged.merge(
        df_to_merge, left_on="LAD23CD", right_on="GEOGRAPHY_CODE", how="left"
    ).drop(columns=["GEOGRAPHY_CODE"], errors="ignore")


# ---- Define Model A feature set --------------------------------------------

lq_cols = [c for c in merged.columns if c.startswith("LQ_")]
outcome_cols = [c for c in merged.columns
                if c.startswith(("emp_share_", "biz_share_"))]
all_numeric = [
    c for c in merged.columns
    if c not in EXCLUDE_FROM_X and c not in lq_cols
    and c not in outcome_cols
    and merged[c].dtype in ["float64", "int64", "int16"]
]
scottish_rows = merged[merged["LAD23CD"].str.startswith("S12")]
X_cols_uk = [c for c in all_numeric if not scottish_rows[c].isna().all()]

print(f"  Merged dataset: {merged.shape}")
print(f"  Model A features: {len(X_cols_uk)} variables")


# ---- Section 1: GCR IS8 sector profile -------------------------------------

print("\n---- Section 1: GCR IS8 sector profile ----")

emp_gcr = emp_latest[emp_latest["GEOGRAPHY_CODE"].isin(GCR_CODES)].copy()
gcr_by_sector = emp_gcr.groupby("IS8_SECTOR")["OBS_VALUE"].sum().reset_index()
gcr_total_emp = float(
    gcr_by_sector.loc[gcr_by_sector["IS8_SECTOR"] == "Total", "OBS_VALUE"].iloc[0]
)
gcr_sectors = gcr_by_sector[gcr_by_sector["IS8_SECTOR"] != "Total"].copy()
gcr_sectors["emp_share"] = gcr_sectors["OBS_VALUE"] / gcr_total_emp * 100

# UK baseline: aggregated across all LADs, not using country-level aggregates.
emp_uk = emp_latest[
    emp_latest["GEOGRAPHY_TYPE"].str.contains("district|unitary", case=False, na=False)
].copy()
uk_by_sector = emp_uk.groupby("IS8_SECTOR")["OBS_VALUE"].sum().reset_index()
uk_total_emp = float(
    uk_by_sector.loc[uk_by_sector["IS8_SECTOR"] == "Total", "OBS_VALUE"].iloc[0]
)
uk_sectors = uk_by_sector[uk_by_sector["IS8_SECTOR"] != "Total"].copy()
uk_sectors["emp_share"] = uk_sectors["OBS_VALUE"] / uk_total_emp * 100

profile = gcr_sectors[["IS8_SECTOR", "OBS_VALUE", "emp_share"]].merge(
    uk_sectors[["IS8_SECTOR", "emp_share"]],
    on="IS8_SECTOR", suffixes=("_gcr", "_uk"),
)
profile["LQ"] = profile["emp_share_gcr"] / profile["emp_share_uk"]
profile = profile.sort_values("LQ", ascending=False)

print(f"\n  GCR total employment: {gcr_total_emp:,.0f}")
print(f"  UK total employment:  {uk_total_emp:,.0f}")
print(f"\n  {'Sector':<40s} {'GCR emp':>10s} {'GCR %':>8s} {'UK %':>8s} {'LQ':>6s}")
print("  " + "-" * 76)
for _, row in profile.iterrows():
    print(f"  {row['IS8_SECTOR']:<40s} {row['OBS_VALUE']:>10,.0f} "
          f"{row['emp_share_gcr']:>7.2f}% {row['emp_share_uk']:>7.2f}% "
          f"{row['LQ']:>5.2f}")

profile.to_csv(RESULTS_DIR / "gcr_sector_profile.csv", index=False, float_format="%.3f")

# Figure: GCR vs UK shares and LQ bar chart.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = range(len(profile))
width = 0.35
axes[0].bar([i - width/2 for i in x], profile["emp_share_gcr"], width,
            label="GCR", color="#d73027")
axes[0].bar([i + width/2 for i in x], profile["emp_share_uk"], width,
            label="UK", color="#4575b4")
axes[0].set_xticks(x)
axes[0].set_xticklabels(profile["IS8_SECTOR"], rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Employment share (%)")
axes[0].set_title("IS8 employment shares: GCR vs UK")
axes[0].legend()

colors = ["#d73027" if lq >= 1 else "#4575b4" for lq in profile["LQ"]]
axes[1].barh(profile["IS8_SECTOR"], profile["LQ"], color=colors)
axes[1].axvline(x=1, color="black", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("Location quotient (1.0 = UK average)")
axes[1].set_title("GCR IS8 location quotients")
plt.tight_layout()
plt.savefig(FIG_DIR / "gcr_sector_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'gcr_sector_profile.png'}")


# ---- Section 2: Predicted vs actual IS8 presence ---------------------------

print("\n---- Section 2: Predicted vs actual IS8 presence ----")

OUTCOMES = {
    "Total IS8 Employment Share":    "emp_share_total_IS8",
    "Total IS8 Business Share":      "biz_share_total_IS8",
    "Financial Services":            "emp_share_Financial Services",
    "Digital & Tech":                "emp_share_Digital and Technology",
    "Adv Manufacturing":             "emp_share_Advanced manufacturing",
    "Life Sciences":                 "emp_share_Life Sciences",
    "Creative Industries":           "emp_share_Creative Industries",
    "Prof & Business Services":      "emp_share_Professional and Business Services",
    "Defence":                       "emp_share_Defence sector",
}

pred_vs_actual = []
for outcome_name, outcome_col in OUTCOMES.items():
    if outcome_col not in merged.columns:
        continue

    model_df = merged[merged[outcome_col].notna()].dropna(subset=X_cols_uk).copy()
    if len(model_df) < 50:
        continue

    X = model_df[X_cols_uk].values
    y = model_df[outcome_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    enet = ElasticNetCV(
        l1_ratio=L1_RATIOS, cv=cv,
        random_state=RANDOM_STATE, max_iter=10000,
    )
    enet.fit(X_scaled, y)

    gcr_row = model_df[model_df["LAD23CD"] == GLASGOW_CODE]
    if len(gcr_row) == 0:
        continue
    actual = float(gcr_row[outcome_col].iloc[0])
    predicted = float(enet.predict(scaler.transform(gcr_row[X_cols_uk].values))[0])
    residual = actual - predicted

    pred_vs_actual.append({
        "Outcome": outcome_name,
        "Actual": actual,
        "Predicted": predicted,
        "Residual (pp)": residual,
        "Status": "over-performing" if residual > 0 else "under-performing",
    })

pva_df = pd.DataFrame(pred_vs_actual)
print(f"\n  {'Outcome':<28s} {'Actual':>10s} {'Predicted':>10s} {'Residual':>10s} {'Status':>18s}")
print("  " + "-" * 80)
for _, row in pva_df.iterrows():
    print(f"  {row['Outcome']:<28s} {row['Actual']:>9.2f}% "
          f"{row['Predicted']:>9.2f}% {row['Residual (pp)']:>+9.2f}pp "
          f"{row['Status']:>18s}")

pva_df.to_csv(
    RESULTS_DIR / "glasgow_predicted_vs_actual.csv",
    index=False, float_format="%.3f",
)

# Figure: residuals as a diverging bar chart.
fig, ax = plt.subplots(figsize=(10, 6))
sorted_pva = pva_df.sort_values("Residual (pp)")
colors = ["#d73027" if r < 0 else "#4575b4" for r in sorted_pva["Residual (pp)"]]
ax.barh(sorted_pva["Outcome"], sorted_pva["Residual (pp)"], color=colors)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_xlabel("Residual (percentage points, actual minus predicted)")
ax.set_title("Glasgow City: predicted vs actual IS8 presence (elastic net)")
plt.tight_layout()
plt.savefig(FIG_DIR / "glasgow_predicted_vs_actual.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'glasgow_predicted_vs_actual.png'}")


# ---- Section 3: Shift-share decomposition ----------------------------------

print("\n---- Section 3: Shift-share decomposition ----")

earliest_year = employee_counts["YEAR"].min()
lad_filter = employee_counts["GEOGRAPHY_TYPE"].str.contains(
    "district|unitary", case=False, na=False
)

uk_early = employee_counts[
    (employee_counts["YEAR"] == earliest_year) & lad_filter
].groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()
uk_late = employee_counts[
    (employee_counts["YEAR"] == latest_emp_year) & lad_filter
].groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()

gcr_early = employee_counts[
    (employee_counts["YEAR"] == earliest_year)
    & (employee_counts["GEOGRAPHY_CODE"].isin(GCR_CODES))
].groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()
gcr_late = employee_counts[
    (employee_counts["YEAR"] == latest_emp_year)
    & (employee_counts["GEOGRAPHY_CODE"].isin(GCR_CODES))
].groupby("IS8_SECTOR")["OBS_VALUE"].sum().to_dict()

uk_total_early = uk_early.get("Total", 1)
uk_total_late = uk_late.get("Total", 1)
national_growth_rate = (uk_total_late - uk_total_early) / uk_total_early

print(f"\n  Period: {earliest_year} to {latest_emp_year}")
print(f"  National employment growth rate: {national_growth_rate*100:.1f}%")

shift_share = []
sectors = [s for s in uk_early.keys() if s != "Total"]
for sector in sectors:
    e_ir = gcr_early.get(sector, 0)
    if e_ir == 0:
        continue
    actual_change = gcr_late.get(sector, 0) - e_ir
    national_effect = e_ir * national_growth_rate
    uk_sector_early = uk_early.get(sector, 1)
    uk_sector_late = uk_late.get(sector, 0)
    sector_national_growth = (
        (uk_sector_late - uk_sector_early) / uk_sector_early
        if uk_sector_early > 0 else 0
    )
    industry_mix = e_ir * (sector_national_growth - national_growth_rate)
    gcr_sector_growth = (
        (gcr_late.get(sector, 0) - e_ir) / e_ir if e_ir > 0 else 0
    )
    local_comp = e_ir * (gcr_sector_growth - sector_national_growth)
    shift_share.append({
        "Sector": sector,
        "National Effect": national_effect,
        "Industry Mix": industry_mix,
        "Local Competitiveness": local_comp,
        "Total Change": actual_change,
        "GCR Early": e_ir,
        "GCR Late": gcr_late.get(sector, 0),
    })

ss_df = pd.DataFrame(shift_share)
print(f"\n  {'Sector':<40s} {'National':>10s} {'Industry':>10s} {'Local':>10s} {'Total':>10s}")
print("  " + "-" * 85)
for _, row in ss_df.iterrows():
    print(f"  {row['Sector']:<40s} {row['National Effect']:>+10,.0f} "
          f"{row['Industry Mix']:>+10,.0f} {row['Local Competitiveness']:>+10,.0f} "
          f"{row['Total Change']:>+10,.0f}")
print(f"  {'TOTAL':<40s} {ss_df['National Effect'].sum():>+10,.0f} "
      f"{ss_df['Industry Mix'].sum():>+10,.0f} "
      f"{ss_df['Local Competitiveness'].sum():>+10,.0f} "
      f"{ss_df['Total Change'].sum():>+10,.0f}")

total_change = ss_df["Total Change"].sum()
nat_eff = ss_df["National Effect"].sum()
ind_mix = ss_df["Industry Mix"].sum()
loc_comp = ss_df["Local Competitiveness"].sum()
print(f"\n  Breakdown of total change ({total_change:+,.0f}):")
print(f"    National effect:       {nat_eff:+,.0f} ({nat_eff/total_change*100:.1f}%)")
print(f"    Industry mix:          {ind_mix:+,.0f} ({ind_mix/total_change*100:.1f}%)")
print(f"    Local competitiveness: {loc_comp:+,.0f} ({loc_comp/total_change*100:.1f}%)")

ss_df.to_csv(RESULTS_DIR / "shift_share_decomposition.csv",
             index=False, float_format="%.0f")

# Figure: stacked bars by sector + totals summary.
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ss_plot = ss_df.set_index("Sector")[
    ["National Effect", "Industry Mix", "Local Competitiveness"]
]
ss_plot.plot(kind="barh", stacked=True, ax=axes[0],
             color=["#4575b4", "#fc8d59", "#91bfdb"])
axes[0].axvline(x=0, color="black", linewidth=0.5)
axes[0].set_xlabel("Employment change")
axes[0].set_title("Shift-share decomposition by IS8 sector (GCR)")
axes[0].legend(fontsize=9)

totals = ss_df[["National Effect", "Industry Mix", "Local Competitiveness"]].sum()
axes[1].bar(range(3), totals.values,
            color=["#4575b4", "#fc8d59", "#91bfdb"])
axes[1].set_xticks(range(3))
axes[1].set_xticklabels([
    f"National\n({totals.iloc[0]:+,.0f})",
    f"Industry mix\n({totals.iloc[1]:+,.0f})",
    f"Local comp\n({totals.iloc[2]:+,.0f})",
], fontsize=10)
axes[1].axhline(y=0, color="black", linewidth=0.5)
axes[1].set_ylabel("Employment change")
axes[1].set_title(f"GCR total IS8 change: {total_change:+,.0f}")
plt.tight_layout()
plt.savefig(FIG_DIR / "shift_share_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'shift_share_decomposition.png'}")


# ---- Section 4: Sector co-location correlation -----------------------------

print("\n---- Section 4: Sector co-location correlation ----")

emp_all = employee_counts[
    (employee_counts["YEAR"] == latest_emp_year) & lad_filter
].copy()
all_lad_sectors = emp_all.groupby(
    ["GEOGRAPHY_CODE", "IS8_SECTOR"]
)["OBS_VALUE"].sum().reset_index()
all_totals = (
    all_lad_sectors[all_lad_sectors["IS8_SECTOR"] == "Total"]
    [["GEOGRAPHY_CODE", "OBS_VALUE"]]
    .rename(columns={"OBS_VALUE": "total_emp"})
)
all_lad_sectors = all_lad_sectors[
    all_lad_sectors["IS8_SECTOR"] != "Total"
].merge(all_totals, on="GEOGRAPHY_CODE")
all_lad_sectors["emp_share"] = (
    all_lad_sectors["OBS_VALUE"] / all_lad_sectors["total_emp"] * 100
)
all_wide = all_lad_sectors.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="emp_share", aggfunc="first",
).dropna()

sector_corr = all_wide.corr()
sector_corr.to_csv(RESULTS_DIR / "sector_colocation_correlation.csv",
                   float_format="%.3f")

print(f"\n  Strong positive correlations (r > 0.5):")
for i, sec_i in enumerate(sector_corr.columns):
    for j, sec_j in enumerate(sector_corr.columns):
        if i < j and sector_corr.iloc[i, j] > 0.5:
            print(f"    {sec_i} + {sec_j}: r={sector_corr.iloc[i, j]:.3f}")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sector_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("IS8 sector co-location: employment share correlation across UK LADs")
plt.tight_layout()
plt.savefig(FIG_DIR / "sector_colocation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'sector_colocation_heatmap.png'}")


# ---- Section 5: Barriers quantification ------------------------------------

print("\n---- Section 5: Barriers quantification ----")

glasgow = merged[merged["LAD23NM"] == "Glasgow City"].iloc[0]

print(f"\n  {'Variable':<35s} {'Glasgow':>12s} {'UK mean':>12s} {'Gap':>12s} {'Assessment':>12s}")
print("  " + "-" * 86)

barrier_list = []
strength_list = []
assessments = []
for var, direction in VARIABLE_DIRECTIONS.items():
    if var not in merged.columns:
        print(f"  {var:<35s} NOT IN DATASET, skipping")
        continue
    gla_val = float(glasgow[var])
    uk_mean = float(merged[var].mean())
    gap = gla_val - uk_mean
    if direction == "higher is better":
        assessment = "STRENGTH" if gap > 0 else "BARRIER"
    else:
        assessment = "BARRIER" if gap > 0 else "STRENGTH"
    (strength_list if assessment == "STRENGTH" else barrier_list).append(var)
    assessments.append({
        "Variable": var, "Glasgow": gla_val, "UK mean": uk_mean,
        "Gap": gap, "Direction": direction, "Assessment": assessment,
    })
    print(f"  {var:<35s} {gla_val:>12,.2f} {uk_mean:>12,.2f} "
          f"{gap:>+12,.2f} {assessment:>12s}")

pd.DataFrame(assessments).to_csv(
    RESULTS_DIR / "glasgow_barriers_strengths.csv",
    index=False, float_format="%.3f",
)

# Fit elastic net on Total IS8 Employment Share for counterfactuals.
outcome_col = "emp_share_total_IS8"
model_df = merged[merged[outcome_col].notna()].dropna(subset=X_cols_uk).copy()
X = model_df[X_cols_uk].values
y = model_df[outcome_col].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
enet = ElasticNetCV(cv=cv, l1_ratio=L1_RATIOS,
                    random_state=RANDOM_STATE, max_iter=10000)
enet.fit(X_scaled, y)

gcr_row = model_df[model_df["LAD23CD"] == GLASGOW_CODE].copy()
actual = float(gcr_row[outcome_col].iloc[0])
baseline_pred = float(enet.predict(scaler.transform(gcr_row[X_cols_uk].values))[0])

print(f"\n  Elastic net fit: alpha={enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}")
print(f"  Baseline prediction for Glasgow: {baseline_pred:.2f}%")
print(f"  Actual: {actual:.2f}%")
print(f"  Unexplained residual: {actual - baseline_pred:+.2f}pp "
      "(anchor institutions, agglomeration, etc.)")

# Individual counterfactuals.
individual = []
for var in VARIABLE_DIRECTIONS:
    if var not in X_cols_uk:
        continue
    gcr_cf = gcr_row.copy()
    gcr_cf[var] = merged[var].mean()
    cf_pred = float(enet.predict(scaler.transform(gcr_cf[X_cols_uk].values))[0])
    change = cf_pred - baseline_pred
    var_type = "BARRIER" if var in barrier_list else "STRENGTH"
    individual.append({
        "Variable": var, "Type": var_type,
        "Counterfactual pred": cf_pred, "Change (pp)": change,
    })

ind_df = pd.DataFrame(individual).sort_values(
    "Change (pp)", key=abs, ascending=False
)
print(f"\n  Individual counterfactuals (each variable set to UK mean, ranked by impact):")
print(f"  {'Variable':<35s} {'Type':>10s} {'New pred':>10s} {'Change':>10s}")
print("  " + "-" * 71)
for _, row in ind_df.iterrows():
    print(f"  {row['Variable']:<35s} {row['Type']:>10s} "
          f"{row['Counterfactual pred']:>9.2f}% {row['Change (pp)']:>+9.2f}pp")

ind_df.to_csv(RESULTS_DIR / "barriers_individual_counterfactuals.csv",
              index=False, float_format="%.3f")

# Combined counterfactuals.
def _combined_pred(vars_to_change):
    cf = gcr_row.copy()
    for v in vars_to_change:
        if v in X_cols_uk:
            cf[v] = merged[v].mean()
    return float(enet.predict(scaler.transform(cf[X_cols_uk].values))[0])

barriers_pred = _combined_pred(barrier_list)
strengths_pred = _combined_pred(strength_list)
all_pred = _combined_pred(list(VARIABLE_DIRECTIONS))

combined = pd.DataFrame([
    {"Scenario": "All barriers set to UK mean",
     "Prediction": barriers_pred, "Change (pp)": barriers_pred - baseline_pred},
    {"Scenario": "All strengths set to UK mean",
     "Prediction": strengths_pred, "Change (pp)": strengths_pred - baseline_pred},
    {"Scenario": "All variables set to UK mean",
     "Prediction": all_pred, "Change (pp)": all_pred - baseline_pred},
])
print(f"\n  Combined counterfactuals:")
print(f"  {'Scenario':<35s} {'Prediction':>12s} {'Change':>10s}")
print("  " + "-" * 60)
for _, row in combined.iterrows():
    print(f"  {row['Scenario']:<35s} {row['Prediction']:>11.2f}% "
          f"{row['Change (pp)']:>+9.2f}pp")

combined.to_csv(RESULTS_DIR / "barriers_combined_counterfactuals.csv",
                index=False, float_format="%.3f")


# ---- Section 6: Convert barrier impacts to jobs ----------------------------

print("\n---- Section 6: Jobs equivalent of barrier and strength impacts ----")

# gcr_total_emp was computed in Section 1.
print(f"\n  GCR total employment: {gcr_total_emp:,.0f}")

ind_df["Jobs equivalent"] = ind_df["Change (pp)"] / 100 * gcr_total_emp
combined["Jobs equivalent"] = combined["Change (pp)"] / 100 * gcr_total_emp

# Health-specific counterfactual: Glasgow's largest category of barriers.
health_vars = ["Under 75 mortality rate", "Male HLE", "Female HLE", "Smokers"]
health_pred = _combined_pred([v for v in health_vars if v in VARIABLE_DIRECTIONS])
health_change = health_pred - baseline_pred
health_jobs = health_change / 100 * gcr_total_emp

print(f"\n  Individual barrier impacts (jobs, barriers only, change > 0):")
barriers_only = ind_df[(ind_df["Type"] == "BARRIER") & (ind_df["Change (pp)"] > 0)]
for _, row in barriers_only.iterrows():
    print(f"    {row['Variable']:<35s} {row['Change (pp)']:>+6.2f}pp  "
          f"{row['Jobs equivalent']:>+10,.0f} jobs")

print(f"\n  Combined scenarios:")
for _, row in combined.iterrows():
    print(f"    {row['Scenario']:<35s} {row['Change (pp)']:>+6.2f}pp  "
          f"{row['Jobs equivalent']:>+10,.0f} jobs")

print(f"\n  Health barriers only ({', '.join(health_vars)}):")
print(f"    Combined impact: {health_change:+.2f}pp  {health_jobs:+,.0f} jobs")

ind_df.to_csv(RESULTS_DIR / "barriers_individual_counterfactuals.csv",
              index=False, float_format="%.3f")
combined.to_csv(RESULTS_DIR / "barriers_combined_counterfactuals.csv",
                index=False, float_format="%.3f")

# Figure: barrier and strength impacts, sorted by magnitude.
fig, ax = plt.subplots(figsize=(10, 8))
plot_df = ind_df.sort_values("Change (pp)")
colors = ["#4575b4" if r["Type"] == "BARRIER" else "#d73027"
          for _, r in plot_df.iterrows()]
ax.barh(plot_df["Variable"], plot_df["Change (pp)"], color=colors)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_xlabel("Change in predicted IS8 employment share if variable set to UK mean (pp)")
ax.set_title("Glasgow: barrier and strength impacts via elastic net counterfactuals\n"
             "(blue = barrier removed increases share; red = strength lost decreases share)")
plt.tight_layout()
plt.savefig(FIG_DIR / "glasgow_barriers_quantification.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'glasgow_barriers_quantification.png'}")

print("\nDone.")