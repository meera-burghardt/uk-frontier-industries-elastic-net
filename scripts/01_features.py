# =============================================================================
# 01_uk_landscape.py
#
# UK-wide descriptive analysis of the eight Industrial Strategy sectors.
# Builds the location quotients and IS8 outcome variables (emp_share_*,
# biz_share_*) that the modelling scripts (02 to 05) also construct, so
# this script doubles as the canonical reference for how those variables
# are defined. Produces three overviews: national sector totals and
# shares, per-sector employment timeseries from 2015 to the latest year,
# and the dominant IS8 sector in each of the seven non-London candidate
# combined authority regions from the project brief. Entry point into
# the repo's analysis chain.
#
# Depends on: data/raw/Employee_counts_IS8_LADs.parquet,
#             data/raw/Business_counts_IS8_LADs.parquet,
#             data/raw/merged_cleaned.csv
#
# Note on merged_cleaned.csv: produced by the group's cleaning pipeline
# (not reproduced in this repo). Contains ONS local indicators, HESA
# research income and student counts, median house prices, and Scottish
# backfill for apprenticeship and mortality variables. See data/raw/README.md
# for details.
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ---- Setup ------------------------------------------------------------------

DATA_DIR = Path("data/raw")
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# The seven non-London urban areas from the project brief.
CANDIDATE_REGIONS = {
    "Glasgow City Region": [
        "S12000049", "S12000045", "S12000039", "S12000050",
        "S12000029", "S12000011", "S12000038", "S12000018",
    ],
    "South Yorkshire": ["E08000016", "E08000017", "E08000018", "E08000019"],
    "West Midlands": [
        "E08000025", "E08000026", "E08000027", "E08000028",
        "E08000029", "E08000030", "E08000031",
    ],
    "West Yorkshire": [
        "E08000032", "E08000033", "E08000034", "E08000035", "E08000036",
    ],
    "Greater Manchester": [
        "E08000001", "E08000002", "E08000003", "E08000004", "E08000005",
        "E08000006", "E08000007", "E08000008", "E08000009", "E08000010",
    ],
    "West of England": ["E06000023", "E06000022", "E06000025"],
    "Cardiff-Newport":  ["W06000015", "W06000022"],
}

# Keeps LAD-level rows and drops country/region aggregates.
LAD_TYPE_FILTER = "district|unitary"

SECTOR_COLORS = {
    "Advanced manufacturing":             "#8c510a",
    "Creative Industries":                "#d73027",
    "Defence sector":                     "#4d4d4d",
    "Digital and Technology":             "#4575b4",
    "Financial Services":                 "#f46d43",
    "Life Sciences":                      "#1a9850",
    "Professional and Business Services": "#762a83",
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

earliest_emp_year = employee_counts["YEAR"].min()
latest_emp_year = employee_counts["YEAR"].max()
latest_biz_year = business_counts["YEAR"].max()

print(f"  Employee counts: {employee_counts.shape}, "
      f"years {earliest_emp_year}-{latest_emp_year}")
print(f"  Business counts: {business_counts.shape}, "
      f"years {business_counts['YEAR'].min()}-{latest_biz_year}")
print(f"  Merged indicators: {merged.shape}")


# ---- Latest-year LAD subsets and UK aggregates -----------------------------

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

uk_emp_by_sector = emp_lads_latest.groupby("IS8_SECTOR")["OBS_VALUE"].sum()
uk_total_emp = float(uk_emp_by_sector["Total"])
uk_emp_sector_only = uk_emp_by_sector.drop("Total")
uk_sector_share = uk_emp_sector_only / uk_total_emp

uk_biz_by_sector = biz_lads_latest.groupby("IS8_SECTOR")["OBS_VALUE"].sum()
uk_total_biz = float(uk_biz_by_sector["Total"])
uk_biz_sector_only = uk_biz_by_sector.drop("Total")


# ---- Section 1: IS8 outcome variables --------------------------------------

print("\n---- Section 1: Building IS8 outcome variables ----")

# emp_share_<sector> = local sector employment / local total employment
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

# biz_share_<sector> = local sector businesses / local total businesses
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

outcomes = emp_share_wide.merge(biz_share_wide, on="GEOGRAPHY_CODE", how="outer")
outcomes.to_csv(
    RESULTS_DIR / "lad_outcome_variables.csv",
    index=False, float_format="%.3f",
)
print(f"  Saved {len(outcomes)} LAD rows x {len(outcomes.columns)-1} outcomes "
      f"to results/lad_outcome_variables.csv")


# ---- Section 2: Location quotients -----------------------------------------

print("\n---- Section 2: Computing location quotients ----")

# LQ = (local sector share) / (national sector share)
# A value above 1.0 means the LAD is more specialised in the sector than
# the UK average.
lad_sector = (
    emp_lads_latest[emp_lads_latest["IS8_SECTOR"] != "Total"]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
    .merge(emp_total_lad, on="GEOGRAPHY_CODE", how="left")
)
lad_sector["local_share"] = (
    lad_sector["OBS_VALUE"] / lad_sector["total_employment"]
)
lad_sector["uk_share"] = lad_sector["IS8_SECTOR"].map(uk_sector_share)
lad_sector["LQ"] = lad_sector["local_share"] / lad_sector["uk_share"]

lq_wide = lad_sector.pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="LQ", aggfunc="first",
).reset_index()
lq_wide.columns = ["GEOGRAPHY_CODE"] + [f"LQ_{c}" for c in lq_wide.columns[1:]]
lq_wide.to_csv(
    RESULTS_DIR / "lad_location_quotients.csv",
    index=False, float_format="%.3f",
)
print(f"  Saved LQs for {len(lq_wide)} LADs x "
      f"{len(lq_wide.columns)-1} sectors")


# ---- Section 3: UK sector totals and shares --------------------------------

print("\n---- Section 3: UK IS8 sector totals ----")

sector_totals = pd.DataFrame({
    "UK employment": uk_emp_sector_only,
    "UK employment share (%)": uk_emp_sector_only / uk_total_emp * 100,
    "UK businesses": uk_biz_sector_only,
    "UK business share (%)": uk_biz_sector_only / uk_total_biz * 100,
}).sort_values("UK employment", ascending=False)
sector_totals.index.name = "IS8 sector"

print(sector_totals.round(2).to_string())
sector_totals.to_csv(RESULTS_DIR / "uk_sector_totals.csv", float_format="%.3f")

# Figure: side-by-side employment and business bar charts.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

st_emp = sector_totals.sort_values("UK employment", ascending=True)
bar_colors = [SECTOR_COLORS.get(s, "grey") for s in st_emp.index]
axes[0].barh(st_emp.index, st_emp["UK employment"] / 1e6, color=bar_colors)
axes[0].set_xlabel("Employment (millions)")
axes[0].set_title(f"UK IS8 employment by sector ({latest_emp_year})")

st_biz = sector_totals.sort_values("UK businesses", ascending=True)
bar_colors = [SECTOR_COLORS.get(s, "grey") for s in st_biz.index]
axes[1].barh(st_biz.index, st_biz["UK businesses"] / 1e3, color=bar_colors)
axes[1].set_xlabel("Businesses (thousands)")
axes[1].set_title(f"UK IS8 businesses by sector ({latest_biz_year})")

plt.tight_layout()
plt.savefig(FIG_DIR / "uk_is8_sector_overview.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'uk_is8_sector_overview.png'}")


# ---- Section 4: Sector growth trends ---------------------------------------

print(f"\n---- Section 4: Sector growth trends "
      f"{earliest_emp_year}-{latest_emp_year} ----")

emp_timeseries = employee_counts[
    employee_counts["GEOGRAPHY_TYPE"].str.contains(
        LAD_TYPE_FILTER, case=False, na=False
    )
].copy()
sector_by_year = (
    emp_timeseries[emp_timeseries["IS8_SECTOR"] != "Total"]
    .groupby(["YEAR", "IS8_SECTOR"])["OBS_VALUE"].sum().unstack()
)

years_elapsed = latest_emp_year - earliest_emp_year
growth = pd.DataFrame({
    f"{earliest_emp_year}": sector_by_year.iloc[0],
    f"{latest_emp_year}": sector_by_year.iloc[-1],
})
growth["Total growth (%)"] = (
    (growth[f"{latest_emp_year}"] - growth[f"{earliest_emp_year}"])
    / growth[f"{earliest_emp_year}"] * 100
)
growth["CAGR (%)"] = (
    (growth[f"{latest_emp_year}"] / growth[f"{earliest_emp_year}"])
    ** (1 / years_elapsed) - 1
) * 100
growth = growth.sort_values("Total growth (%)", ascending=False)
growth.index.name = "IS8 sector"

print(growth.round(2).to_string())
sector_by_year.to_csv(
    RESULTS_DIR / "uk_sector_growth_timeseries.csv", float_format="%.0f"
)
growth.to_csv(RESULTS_DIR / "uk_sector_growth_rates.csv", float_format="%.3f")

# Figure: indexed growth lines (earliest year = 100).
fig, ax = plt.subplots(figsize=(11, 6))
indexed = sector_by_year.div(sector_by_year.iloc[0]) * 100
for sector in indexed.columns:
    ax.plot(indexed.index, indexed[sector], label=sector,
            color=SECTOR_COLORS.get(sector, "grey"), linewidth=2)
ax.axhline(y=100, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel(f"Employment index ({earliest_emp_year} = 100)")
ax.set_title(f"UK IS8 sector employment, {earliest_emp_year}-{latest_emp_year}")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG_DIR / "uk_is8_growth_trends.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'uk_is8_growth_trends.png'}")


# ---- Section 5: Dominant sector per candidate region -----------------------

print("\n---- Section 5: Dominant IS8 sector per candidate region ----")

region_rows = []
for region_name, lad_codes in CANDIDATE_REGIONS.items():
    region_emp = emp_lads_latest[
        emp_lads_latest["GEOGRAPHY_CODE"].isin(lad_codes)
    ]
    region_total = float(
        region_emp.loc[region_emp["IS8_SECTOR"] == "Total", "OBS_VALUE"].sum()
    )
    region_sector = (
        region_emp[region_emp["IS8_SECTOR"] != "Total"]
        .groupby("IS8_SECTOR")["OBS_VALUE"].sum()
    )
    for sector, emp in region_sector.items():
        local_share = emp / region_total if region_total > 0 else np.nan
        uk_share = float(uk_sector_share.get(sector, np.nan))
        region_rows.append({
            "Region": region_name,
            "IS8 sector": sector,
            "Employment": emp,
            "Local share (%)": local_share * 100,
            "LQ": local_share / uk_share if uk_share > 0 else np.nan,
        })

region_df = pd.DataFrame(region_rows)
dominant = (
    region_df.sort_values("LQ", ascending=False)
    .groupby("Region").head(1).reset_index(drop=True)
)

print(f"\n  {'Region':<24s} {'Dominant sector':<40s} {'LQ':>6s} {'Share':>10s}")
print("  " + "-" * 84)
for _, row in dominant.iterrows():
    print(f"  {row['Region']:<24s} {row['IS8 sector']:<40s} "
          f"{row['LQ']:>6.2f} {row['Local share (%)']:>9.2f}%")

region_df.to_csv(
    RESULTS_DIR / "regional_sector_breakdown.csv",
    index=False, float_format="%.3f",
)
dominant.to_csv(
    RESULTS_DIR / "regional_dominant_sectors.csv",
    index=False, float_format="%.3f",
)

# Figure: horizontal bar per region, length = LQ of its dominant sector,
# coloured by sector.
fig, ax = plt.subplots(figsize=(11, 5))
dominant_sorted = dominant.sort_values("LQ")
bar_colors = [SECTOR_COLORS.get(s, "grey") for s in dominant_sorted["IS8 sector"]]
ax.barh(dominant_sorted["Region"], dominant_sorted["LQ"], color=bar_colors)
ax.axvline(x=1, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Location quotient of dominant IS8 sector (1.0 = UK average)")
ax.set_title("Dominant IS8 sector by candidate region")

unique_sectors = sorted(dominant_sorted["IS8 sector"].unique())
legend_elements = [
    Patch(facecolor=SECTOR_COLORS.get(s, "grey"), label=s)
    for s in unique_sectors
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "uk_regional_dominant_sectors.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved {FIG_DIR / 'uk_regional_dominant_sectors.png'}")

print("\nDone.")