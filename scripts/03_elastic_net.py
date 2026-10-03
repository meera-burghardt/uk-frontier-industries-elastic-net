# =============================================================================
# 03_elastic_net.py
#
# Fit Lasso and Elastic Net regularised regression models to predict IS8
# sector presence and growth across UK local authorities. Compares Model A
# (UK-wide, variables available across all nations) with Model B (England
# only, which adds education and transport indicators missing for Scotland).
# Fits both estimators for seven outcomes: total IS8 employment share and
# growth, business share and growth, plus five sector-specific employment
# shares. Reports Lasso vs Elastic Net agreement and variable selection
# frequency across outcomes.
#
# Depends on: data/raw/Employee_counts_IS8_LADs.parquet,
#             data/raw/Business_counts_IS8_LADs.parquet,
#             data/raw/merged_cleaned.csv
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


# ---- Setup ------------------------------------------------------------------

DATA_DIR = Path("data/raw")
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 10
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Identifiers, geography codes, and location columns to keep out of X.
EXCLUDE_FROM_X = [
    "LAD23CD", "LAD23NM", "LONG", "LAT", "geometry",
    "GEOGRAPHY_CODE", "GEOGRAPHY_NAME", "GEOGRAPHY_TYPE",
    "Local Authority District", "County or Unitary Authority",
    "Country", "Nation", "Region", "Combined Authority or City Region",
    "BNG_E", "BNG_N", "FID", "GlobalID",
]


# ---- Load data --------------------------------------------------------------

print("Loading data...")

business_counts = pd.read_parquet(
    DATA_DIR / "Business_counts_IS8_LADs.parquet", engine="fastparquet"
)
employee_counts = pd.read_parquet(
    DATA_DIR / "Employee_counts_IS8_LADs.parquet", engine="fastparquet"
)
merged = pd.read_csv(DATA_DIR / "merged_cleaned.csv")

print(f"  Business counts: {len(business_counts):,} rows")
print(f"  Employee counts: {len(employee_counts):,} rows")
print(f"  Merged indicators: {len(merged)} LADs")


# ---- Build IS8 outcome variables -------------------------------------------

# Employment shares at the latest BRES year.
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

# Employment growth, earliest to latest BRES year.
earliest_year = employee_counts["YEAR"].min()
emp_early_g = (
    employee_counts[employee_counts["YEAR"] == earliest_year]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
)
emp_late_g = (
    employee_counts[employee_counts["YEAR"] == latest_emp_year]
    .groupby(["GEOGRAPHY_CODE", "IS8_SECTOR"])["OBS_VALUE"].sum().reset_index()
)
growth = emp_early_g.merge(
    emp_late_g, on=["GEOGRAPHY_CODE", "IS8_SECTOR"],
    suffixes=("_early", "_late"),
)
growth["emp_growth"] = np.where(
    growth["OBS_VALUE_early"] > 0,
    (growth["OBS_VALUE_late"] - growth["OBS_VALUE_early"])
    / growth["OBS_VALUE_early"] * 100,
    np.nan,
)
growth_wide = growth[growth["IS8_SECTOR"] != "Total"].pivot_table(
    index="GEOGRAPHY_CODE", columns="IS8_SECTOR",
    values="emp_growth", aggfunc="first",
).reset_index()
growth_wide.columns = ["GEOGRAPHY_CODE"] + [
    f"emp_growth_{col}" for col in growth_wide.columns[1:]
]

# Total IS8 growth: aggregate across sectors, not the mean of per-sector
# rates (which would overweight small sectors like Defence).
total_growth = growth[growth["IS8_SECTOR"] != "Total"].groupby(
    "GEOGRAPHY_CODE"
).apply(lambda x: (
    (x["OBS_VALUE_late"].sum() - x["OBS_VALUE_early"].sum())
    / x["OBS_VALUE_early"].sum() * 100
) if x["OBS_VALUE_early"].sum() > 0 else np.nan).reset_index()
total_growth.columns = ["GEOGRAPHY_CODE", "emp_growth_total_IS8"]
growth_wide = growth_wide.merge(total_growth, on="GEOGRAPHY_CODE", how="left")

# Business shares at the latest UK Business Counts year.
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

# Overall business growth rate (all sectors, not just IS8).
earliest_biz_year = business_counts["YEAR"].min()
biz_early = (
    business_counts[
        (business_counts["YEAR"] == earliest_biz_year)
        & (business_counts["IS8_SECTOR"] == "Total")
    ]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("biz_early").reset_index()
)
biz_late = (
    business_counts[
        (business_counts["YEAR"] == latest_biz_year)
        & (business_counts["IS8_SECTOR"] == "Total")
    ]
    .groupby("GEOGRAPHY_CODE")["OBS_VALUE"].sum()
    .rename("biz_late").reset_index()
)
biz_growth = biz_early.merge(biz_late, on="GEOGRAPHY_CODE")
biz_growth["biz_growth_rate"] = np.where(
    biz_growth["biz_early"] > 0,
    (biz_growth["biz_late"] - biz_growth["biz_early"])
    / biz_growth["biz_early"] * 100,
    np.nan,
)

print(f"  Employment years: {earliest_year} to {latest_emp_year}")
print(f"  Business years: {earliest_biz_year} to {latest_biz_year}")


# ---- Merge outcomes into feature frame -------------------------------------

for df_to_merge in [emp_share_wide, growth_wide, biz_share_wide]:
    merged = merged.merge(
        df_to_merge, left_on="LAD23CD", right_on="GEOGRAPHY_CODE", how="left"
    ).drop(columns=["GEOGRAPHY_CODE"], errors="ignore")

merged = merged.merge(
    biz_growth[["GEOGRAPHY_CODE", "biz_growth_rate"]],
    left_on="LAD23CD", right_on="GEOGRAPHY_CODE", how="left",
).drop(columns=["GEOGRAPHY_CODE"], errors="ignore")

print(f"\nMerged dataset: {merged.shape}")


# ---- Define feature sets ---------------------------------------------------

lq_cols = [c for c in merged.columns if c.startswith("LQ_")]
outcome_cols = [c for c in merged.columns
                if c.startswith(("emp_share_", "emp_growth_",
                                 "biz_share_", "biz_growth_"))]

all_numeric = [
    c for c in merged.columns
    if c not in EXCLUDE_FROM_X
    and c not in lq_cols
    and c not in outcome_cols
    and merged[c].dtype in ["float64", "int64", "int16"]
]

# Model A: variables with at least one non-null for Scottish LADs.
scottish_rows = merged[merged["LAD23CD"].str.startswith("S12")]
scotland_missing = [c for c in all_numeric if scottish_rows[c].isna().all()]
X_cols_uk = [c for c in all_numeric if c not in scotland_missing]

# Model B: variables with at least 60% coverage among English LADs.
english_rows = merged[merged["LAD23CD"].str.startswith("E")]
X_cols_england = [
    c for c in all_numeric if english_rows[c].notna().mean() >= 0.6
]

print(f"\nModel A (UK): {len(X_cols_uk)} variables")
print(f"Model B (England): {len(X_cols_england)} variables")
print(f"  Scotland missing: {len(scotland_missing)} variables")


# ---- Outcomes to model -----------------------------------------------------

OUTCOMES = {
    "Total IS8 Employment Share":    "emp_share_total_IS8",
    "Total IS8 Employment Growth":   "emp_growth_total_IS8",
    "Total IS8 Business Share":      "biz_share_total_IS8",
    "Total IS8 Business Growth":     "biz_growth_rate",
    "Financial Services Emp Share":  "emp_share_Financial Services",
    "Digital & Tech Emp Share":      "emp_share_Digital and Technology",
    "Adv Manufacturing Emp Share":   "emp_share_Advanced manufacturing",
    "Creative Industries Emp Share": "emp_share_Creative Industries",
    "Life Sciences Emp Share":       "emp_share_Life Sciences",
    "Defence Sector Emp Share":      "emp_share_Defence sector",
    "Prof & Business Svcs Emp Share": "emp_share_Professional and Business Services",
}
OUTCOMES = {k: v for k, v in OUTCOMES.items() if v in merged.columns}


# ---- Fitting function ------------------------------------------------------

def fit_lasso_and_enet(df, X_cols, y_col, outcome_name):
    """Fit LassoCV and ElasticNetCV with a 70/30 train-test split, 10-fold CV.

    Scaler fit on train only, applied to test. Returns dict with coefficient
    tables and R² scores for both estimators, or None if fewer than 50
    complete rows are available.
    """
    model_df = df[df[y_col].notna()].dropna(subset=X_cols).copy()
    if len(model_df) < 50:
        print(f"  SKIP: only {len(model_df)} complete rows")
        return None

    X = model_df[X_cols].values
    y = model_df[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    lasso = LassoCV(cv=cv, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train_s, y_train)

    enet = ElasticNetCV(
        l1_ratio=L1_RATIOS, cv=cv,
        random_state=RANDOM_STATE, max_iter=10000,
    )
    enet.fit(X_train_s, y_train)

    def _coef_df(model):
        return pd.DataFrame({
            "Variable": X_cols,
            "Coefficient": model.coef_,
        }).sort_values("Coefficient", key=abs, ascending=False)

    results = {
        "outcome": outcome_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "lasso": {
            "coefs": _coef_df(lasso),
            "r2_train": lasso.score(X_train_s, y_train),
            "r2_test": lasso.score(X_test_s, y_test),
            "alpha": lasso.alpha_,
        },
        "enet": {
            "coefs": _coef_df(enet),
            "r2_train": enet.score(X_train_s, y_train),
            "r2_test": enet.score(X_test_s, y_test),
            "alpha": enet.alpha_,
            "l1_ratio": enet.l1_ratio_,
        },
    }

    for method, label in [("lasso", "LASSO"), ("enet", "ELASTIC NET")]:
        r = results[method]
        n_sel = (r["coefs"]["Coefficient"] != 0).sum()
        print(f"  {label:12s} train R²={r['r2_train']:.3f}  "
              f"test R²={r['r2_test']:.3f}  "
              f"{n_sel}/{len(X_cols)} vars selected")

    return results


# ---- Fit Model A and Model B -----------------------------------------------

print("\n---- Model A (UK-wide) ----")
results_A = {}
for name, col in OUTCOMES.items():
    print(f"\n  {name}")
    r = fit_lasso_and_enet(merged, X_cols_uk, col, name)
    if r:
        results_A[name] = r

print("\n---- Model B (England only) ----")
england_df = merged[merged["LAD23CD"].str.startswith("E")].copy()
results_B = {}
for name, col in OUTCOMES.items():
    print(f"\n  {name}")
    r = fit_lasso_and_enet(england_df, X_cols_england, col, name)
    if r:
        results_B[name] = r


# ---- Save artefacts --------------------------------------------------------

def _slug(name):
    return (name.lower()
            .replace(" ", "_").replace("&", "and")
            .replace(":", "").replace(",", ""))

# R² comparison across models.
r2_rows = []
for name in OUTCOMES:
    row = {"Outcome": name}
    for scope, r in [("A", results_A.get(name)), ("B", results_B.get(name))]:
        if r:
            row[f"Model {scope} Lasso Train"] = r["lasso"]["r2_train"]
            row[f"Model {scope} Lasso Test"]  = r["lasso"]["r2_test"]
            row[f"Model {scope} ENet Train"]  = r["enet"]["r2_train"]
            row[f"Model {scope} ENet Test"]   = r["enet"]["r2_test"]
    r2_rows.append(row)
pd.DataFrame(r2_rows).to_csv(
    RESULTS_DIR / "model_r2_comparison.csv", index=False, float_format="%.3f"
)

# Coefficient tables for every UK outcome.
for name, r in results_A.items():
    r["lasso"]["coefs"].to_csv(
        RESULTS_DIR / f"coefs_lasso_A_{_slug(name)}.csv",
        index=False, float_format="%.4f",
    )
    r["enet"]["coefs"].to_csv(
        RESULTS_DIR / f"coefs_enet_A_{_slug(name)}.csv",
        index=False, float_format="%.4f",
    )

# How often each variable gets selected across the UK models.
lasso_freq, enet_freq = {}, {}
for r in results_A.values():
    for _, row in r["lasso"]["coefs"].iterrows():
        lasso_freq[row["Variable"]] = lasso_freq.get(row["Variable"], 0) + int(row["Coefficient"] != 0)
    for _, row in r["enet"]["coefs"].iterrows():
        enet_freq[row["Variable"]] = enet_freq.get(row["Variable"], 0) + int(row["Coefficient"] != 0)
freq_table = pd.DataFrame({
    "Lasso selections": pd.Series(lasso_freq),
    "ENet selections":  pd.Series(enet_freq),
}).sort_values("Lasso selections", ascending=False)
freq_table.to_csv(RESULTS_DIR / "variable_selection_frequency.csv")

print("\n---- Variables selected in 2+ UK outcomes ----")
show = freq_table[(freq_table["Lasso selections"] >= 2)
                  | (freq_table["ENet selections"] >= 2)]
print(show.to_string())


# ---- Lasso vs Elastic Net agreement ----------------------------------------

print("\n---- Top 5 agreement, Lasso vs Elastic Net (UK models) ----")
for name, r in results_A.items():
    lasso_top = set(r["lasso"]["coefs"]
                    [r["lasso"]["coefs"]["Coefficient"] != 0]
                    .head(5)["Variable"].tolist())
    enet_top  = set(r["enet"]["coefs"]
                    [r["enet"]["coefs"]["Coefficient"] != 0]
                    .head(5)["Variable"].tolist())
    overlap = lasso_top & enet_top
    denom = min(5, len(lasso_top), len(enet_top))
    print(f"\n  {name}:")
    print(f"    Agreement: {len(overlap)}/{denom}")
    if overlap:
        print(f"    Shared: {', '.join(sorted(overlap))}")


# ---- Diagnostic figures ----------------------------------------------------

print("\n---- Saving diagnostic figures ----")

# Headline: Elastic Net top predictors for employment share and business share.
headline = [("Total IS8 Employment Share", "Employment Share"),
            ("Total IS8 Business Share",   "Business Share")]
if all(name in results_A for name, _ in headline):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for i, (name, short) in enumerate(headline):
        r = results_A[name]["enet"]
        sel = r["coefs"][r["coefs"]["Coefficient"] != 0].sort_values("Coefficient")
        colors = ["#d73027" if v < 0 else "#4575b4" for v in sel["Coefficient"]]
        axes[i].barh(range(len(sel)), sel["Coefficient"].values, color=colors)
        axes[i].set_yticks(range(len(sel)))
        axes[i].set_yticklabels(sel["Variable"].values, fontsize=9)
        axes[i].set_xlabel("Standardised coefficient")
        axes[i].set_title(
            f"{short}\n"
            f"Train R²={r['r2_train']:.3f}, Test R²={r['r2_test']:.3f}  |  "
            f"α={r['alpha']:.4f}, L1={r['l1_ratio']:.1f}  |  "
            f"{len(sel)}/{len(r['coefs'])} vars",
            fontsize=10,
        )
        axes[i].axvline(x=0, color="black", linewidth=0.5)
    fig.suptitle("Model A (UK): Elastic Net predictors of IS8 presence",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "enet_predictors_emp_vs_biz_share.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIG_DIR / 'enet_predictors_emp_vs_biz_share.png'}")

# Method comparison: Lasso vs Elastic Net on the main model.
if "Total IS8 Employment Share" in results_A:
    r = results_A["Total IS8 Employment Share"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for i, (method, label) in enumerate([("lasso", "Lasso"), ("enet", "Elastic Net")]):
        m = r[method]
        sel = m["coefs"][m["coefs"]["Coefficient"] != 0].sort_values("Coefficient")
        colors = ["#d73027" if v < 0 else "#4575b4" for v in sel["Coefficient"]]
        axes[i].barh(range(len(sel)), sel["Coefficient"].values, color=colors)
        axes[i].set_yticks(range(len(sel)))
        axes[i].set_yticklabels(sel["Variable"].values, fontsize=9)
        axes[i].set_xlabel("Standardised coefficient")
        axes[i].set_title(f"{label} (Train R²={m['r2_train']:.3f}, "
                          f"Test R²={m['r2_test']:.3f})")
        axes[i].axvline(x=0, color="black", linewidth=0.5)
    fig.suptitle("Model A (UK): Lasso vs Elastic Net on Total IS8 Employment Share",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lasso_vs_enet_total_is8_emp_share.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIG_DIR / 'lasso_vs_enet_total_is8_emp_share.png'}")

print("\nDone.")