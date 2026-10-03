# =============================================================================
# 04_ols_rf_robustness.py
#
# Robustness checks for the elastic net results in 03_elastic_net.py. Fits
# ordinary least squares on the small set of variables the elastic net
# picked out, so coefficients can be read in raw units with p-values, then
# fits a random forest on the full UK feature set to see whether a non-linear
# estimator surfaces the same predictors. Compares random forest's top
# features against the lasso selections to check agreement across methods.
#
# Depends on: data/raw/Employee_counts_IS8_LADs.parquet,
#             data/raw/Business_counts_IS8_LADs.parquet,
#             data/raw/merged_cleaned.csv
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ---- Setup ------------------------------------------------------------------

DATA_DIR = Path("data/raw")
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 5

EXCLUDE_FROM_X = [
    "LAD23CD", "LAD23NM", "LONG", "LAT", "geometry",
    "GEOGRAPHY_CODE", "GEOGRAPHY_NAME", "GEOGRAPHY_TYPE",
    "Local Authority District", "County or Unitary Authority",
    "Country", "Nation", "Region", "Combined Authority or City Region",
    "BNG_E", "BNG_N", "FID", "GlobalID",
]


def _slug(name):
    return (name.lower().replace(" ", "_")
            .replace("&", "and").replace(",", ""))


# ---- Load data --------------------------------------------------------------

print("Loading data...")
business_counts = pd.read_parquet(
    DATA_DIR / "Business_counts_IS8_LADs.parquet", engine="fastparquet"
)
employee_counts = pd.read_parquet(
    DATA_DIR / "Employee_counts_IS8_LADs.parquet", engine="fastparquet"
)
merged = pd.read_csv(DATA_DIR / "merged_cleaned.csv")


# ---- Build outcomes (mirrors construction in 03_elastic_net.py) ------------

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


# ---- Outcomes and the lasso selections to test -----------------------------

OUTCOMES = {
    "Total IS8 Employment Share":   "emp_share_total_IS8",
    "Total IS8 Business Share":     "biz_share_total_IS8",
    "Financial Services Emp Share": "emp_share_Financial Services",
    "Digital & Tech Emp Share":     "emp_share_Digital and Technology",
    "Adv Manufacturing Emp Share":  "emp_share_Advanced manufacturing",
}

# Top variables the lasso picked for each outcome in 03_elastic_net.py.
# Hardcoded rather than loaded from results/ so this script stands alone.
LASSO_SELECTED_VARS = {
    "Total IS8 Employment Share": [
        "High growth enterprises", "GVA per hour", "Under 75 mortality rate",
        "total_HE_students", "Level 3+ qualifications",
    ],
    "Total IS8 Business Share": [
        "High growth enterprises", "4G area coverage", "Level 3+ qualifications",
        "Apprenticeship achievements", "Under 75 mortality rate",
    ],
    "Financial Services Emp Share": [
        "High growth enterprises", "GVA per hour",
    ],
    "Digital & Tech Emp Share": [
        "GVA per hour", "High growth enterprises", "total_HE_students",
        "Level 3+ qualifications", "Employment rate",
    ],
    "Adv Manufacturing Emp Share": [
        "Apprenticeship achievements", "Broadband availability",
        "Deaths of enterprises", "Apprenticeship starts", "Smokers",
    ],
}


# ---- Part 1: OLS on lasso-selected variables -------------------------------

print("\n---- Part 1: OLS on lasso-selected variables ----")

glasgow_rows = []
for outcome_name, outcome_col in OUTCOMES.items():
    selected = [v for v in LASSO_SELECTED_VARS[outcome_name] if v in merged.columns]
    if not selected:
        print(f"\n  SKIP {outcome_name}: no selected vars in data")
        continue

    model_df = merged[merged[outcome_col].notna()].dropna(subset=selected).copy()
    X = sm.add_constant(model_df[selected])
    y = model_df[outcome_col]
    ols = sm.OLS(y, X).fit()

    print(f"\n  {outcome_name}:")
    print(f"    R²={ols.rsquared:.3f}, adj R²={ols.rsquared_adj:.3f}, n={int(ols.nobs)}")
    print(f"    Significant predictors (p<0.05):")
    for var in selected:
        coef = ols.params[var]
        pval = ols.pvalues[var]
        if pval < 0.05:
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
            print(f"      {var:35s} {coef:+10.4f}  (p={pval:.3f}) {sig}")

    # Save the full regression table to a text file so the detailed
    # standard errors, t-stats, and CIs are inspectable without a rerun.
    with open(RESULTS_DIR / f"ols_{_slug(outcome_name)}.txt", "w") as f:
        f.write(str(ols.summary2()))

    # Quick Glasgow predicted vs actual on the OLS; the elastic net
    # version (which is the one the memo cites) is in 05_glasgow_analysis.py.
    glasgow = model_df[model_df["LAD23CD"] == "S12000049"]
    if len(glasgow):
        gla_X = sm.add_constant(glasgow[selected], has_constant="add")
        pred = float(ols.predict(gla_X).iloc[0])
        actual = float(glasgow[outcome_col].iloc[0])
        print(f"    Glasgow: actual={actual:.2f}%, predicted={pred:.2f}%, "
              f"residual={actual - pred:+.2f}pp")
        glasgow_rows.append({
            "Outcome": outcome_name, "Actual": actual,
            "Predicted (OLS)": pred, "Residual (pp)": actual - pred,
        })

pd.DataFrame(glasgow_rows).to_csv(
    RESULTS_DIR / "ols_glasgow_cross_check.csv",
    index=False, float_format="%.2f",
)


# ---- Part 2: Random forest feature importance ------------------------------

print("\n---- Part 2: Random forest feature importance ----")

rf_results = {}
for outcome_name, outcome_col in OUTCOMES.items():
    model_df = merged[merged[outcome_col].notna()].dropna(subset=X_cols_uk).copy()
    if len(model_df) < 50:
        continue

    X = model_df[X_cols_uk].values
    y = model_df[outcome_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "Variable": X_cols_uk,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    r2_train = rf.score(X_train, y_train)
    r2_test = rf.score(X_test, y_test)

    print(f"\n  {outcome_name}:")
    print(f"    R² train={r2_train:.3f}, R² test={r2_test:.3f}")
    print(f"    Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"      {row['Variable']:35s} {row['Importance']:.4f}")

    rf_results[outcome_name] = {
        "importance": importance, "r2_train": r2_train, "r2_test": r2_test,
    }
    importance.to_csv(
        RESULTS_DIR / f"rf_importance_{_slug(outcome_name)}.csv",
        index=False, float_format="%.4f",
    )


# ---- Part 3: Lasso vs Random Forest agreement ------------------------------

print("\n---- Part 3: Lasso vs Random Forest agreement ----")

agreement_rows = []
for outcome_name, r in rf_results.items():
    rf_top10 = set(r["importance"].head(10)["Variable"].tolist())
    lasso_top = set(LASSO_SELECTED_VARS.get(outcome_name, []))
    if not lasso_top:
        continue
    overlap = rf_top10 & lasso_top
    print(f"\n  {outcome_name}:")
    print(f"    Lasso selected ({len(lasso_top)}): {sorted(lasso_top)}")
    print(f"    Shared with RF top 10 ({len(overlap)}): {sorted(overlap)}")
    agreement_rows.append({
        "Outcome": outcome_name,
        "Lasso selected": len(lasso_top),
        "RF top 10 overlap": len(overlap),
        "Shared variables": ", ".join(sorted(overlap)),
    })

pd.DataFrame(agreement_rows).to_csv(
    RESULTS_DIR / "lasso_rf_agreement.csv", index=False,
)


# ---- Feature importance figure ---------------------------------------------

print("\n---- Saving feature importance figure ----")

plot_outcomes = list(rf_results.keys())
n_plots = len(plot_outcomes)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))
axes = axes.flatten() if n_plots > 1 else [axes]

for i, outcome_name in enumerate(plot_outcomes):
    imp = rf_results[outcome_name]["importance"].head(12)
    r2 = rf_results[outcome_name]["r2_test"]
    axes[i].barh(range(len(imp)), imp["Importance"].values, color="#4575b4")
    axes[i].set_yticks(range(len(imp)))
    axes[i].set_yticklabels(imp["Variable"].values, fontsize=9)
    axes[i].set_xlabel("Feature importance")
    axes[i].set_title(f"{outcome_name}\n(RF test R²={r2:.3f})", fontsize=10)
    axes[i].invert_yaxis()

for j in range(n_plots, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Random Forest: feature importance (Model A, UK)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {FIG_DIR / 'rf_feature_importance.png'}")

print("\nDone.")