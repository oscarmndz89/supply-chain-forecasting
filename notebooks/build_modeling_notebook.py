"""
Builds notebooks/03_modeling.ipynb programmatically.
Run: python notebooks/build_modeling_notebook.py
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# Predictive Modeling — Supply Chain Forecasting
**Portfolio Project | Data Science**

Three models trained on the engineered feature set from notebook 02:

| Model | Type | Target | Algorithm |
|---|---|---|---|
| **DelayClassifier** | Binary classification | `is_delayed` | XGBoost + GridSearchCV |
| **CostVarianceRegressor** | Regression | `cost_variance_usd` | XGBoost + GridSearchCV |
| **LeadTimeForecaster** | Time-series forecast | daily avg lead time | Prophet (per carrier) |

All supervised models use only **pre-delivery features** — no leakage from
actual arrival data.
"""))

# ── 0. Setup ───────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 0. Setup"))

cells.append(new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

from src.models import DelayClassifier, CostVarianceRegressor, LeadTimeForecaster

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

CARRIER_COLORS = {
    "FastFreight": "#2ecc71", "RelayEx": "#3498db", "PrimeHaul": "#5dade2",
    "SwiftLog": "#f39c12",   "CargoLink": "#e67e22", "NorthStar": "#e74c3c",
    "DirectMove": "#c0392b",
}

DATA = Path("../data")
print("Setup complete.")
"""))

# ── 1. Load Data ───────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## 1. Load Engineered Feature Matrix
"""))

cells.append(new_code_cell("""\
df = pd.read_csv(DATA / "shipments_featured.csv", parse_dates=["date"])
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"is_delayed base rate: {df['is_delayed'].mean():.1%}")
print(f"cost_variance_usd mean: ${df['cost_variance_usd'].mean():.3f}")
"""))

# ── 2. DelayClassifier ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## 2. Delay Classifier (XGBoost + GridSearchCV)

**Task:** predict whether a shipment will arrive late, using only information
available at dispatch time (carrier history, lane risk, shipment characteristics,
temporal features).

We use a **time-based train/test split** — train on the first 80% of dates,
test on the remaining 20% — to simulate realistic deployment conditions.
"""))

cells.append(new_code_cell("""\
# Time-based split
df_sorted = df.sort_values("date").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.80)
split_date = df_sorted.loc[split_idx, "date"].date()

train_df = df_sorted.iloc[:split_idx]
test_df  = df_sorted.iloc[split_idx:]

print(f"Train: {len(train_df):,} rows  ({train_df['date'].min().date()} — {train_df['date'].max().date()})")
print(f"Test : {len(test_df):,}  rows  ({test_df['date'].min().date()} — {test_df['date'].max().date()})")
print(f"Split date: {split_date}")
print(f"Train delay rate: {train_df['is_delayed'].mean():.1%}  |  Test delay rate: {test_df['is_delayed'].mean():.1%}")
"""))

cells.append(new_code_cell("""\
# Prepare features
X_train_clf, y_train_clf = DelayClassifier.prepare_features(train_df)
X_test_clf,  y_test_clf  = DelayClassifier.prepare_features(test_df)

print(f"Feature matrix: {X_train_clf.shape[1]} features")
print("Features:", list(X_train_clf.columns))
"""))

cells.append(new_code_cell("""\
# Fit with GridSearchCV
clf = DelayClassifier()
clf.fit(X_train_clf, y_train_clf, cv=3, scoring="roc_auc")

print(f"Best params : {clf.best_params_}")
print(f"Best CV AUC : {max(clf.cv_results_['mean_test_score']):.4f}")
"""))

cells.append(new_code_cell("""\
# Evaluate on held-out test set
metrics_clf = clf.evaluate(X_test_clf, y_test_clf)
print("Test-set performance:")
print(f"  AUC       : {metrics_clf['auc']:.4f}")
print(f"  Precision : {metrics_clf['precision']:.4f}")
print(f"  Recall    : {metrics_clf['recall']:.4f}")
print(f"  F1        : {metrics_clf['f1']:.4f}")
"""))

cells.append(new_markdown_cell("""\
### 2a. Confusion Matrix & ROC Curve
"""))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
ax = axes[0]
cm = metrics_clf["confusion_matrix"]
disp = ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Delayed"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set", fontweight="bold")

# ROC curve
ax = axes[1]
y_proba = clf.predict_proba(X_test_clf)
fpr, tpr, _ = roc_curve(y_test_clf, y_proba)
roc_auc_val = auc(fpr, tpr)
ax.plot(fpr, tpr, color="#2c3e50", lw=2, label=f"AUC = {roc_auc_val:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="#2c3e50")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Delay Classifier", fontweight="bold")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("../data/fig_model_clf_roc_cm.png", bbox_inches="tight")
plt.show()
"""))

cells.append(new_markdown_cell("""\
### 2b. SHAP Beeswarm — Which features drive delay predictions?

Each dot is one test-set shipment.
- **Red dots** = high feature value; **blue dots** = low value
- Dots pushed right = increased predicted delay probability
"""))

cells.append(new_code_cell("""\
# Compute SHAP on test set (capped at 500 rows for speed)
shap_sample = X_test_clf.sample(min(500, len(X_test_clf)), random_state=42)
shap_exp_clf = clf.shap_explanation(shap_sample)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_exp_clf.values,
    shap_sample,
    plot_type="dot",
    max_display=15,
    show=False,
    plot_size=None,
)
plt.title("SHAP Beeswarm — Delay Classifier", fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("../data/fig_model_clf_shap_beeswarm.png", bbox_inches="tight")
plt.show()
"""))

cells.append(new_markdown_cell("""\
### 2c. Top 10 Feature Importances (mean |SHAP|)
"""))

cells.append(new_code_cell("""\
mean_abs_shap_clf = (
    pd.Series(
        np.abs(shap_exp_clf.values).mean(axis=0),
        index=shap_sample.columns,
    )
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 4))
colors = sns.color_palette("RdYlGn_r", len(mean_abs_shap_clf))
ax.barh(mean_abs_shap_clf.index[::-1], mean_abs_shap_clf.values[::-1],
        color=colors[::-1], edgecolor="white", height=0.65)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 10 Features — Delay Classifier", fontweight="bold")
for i, (feat, val) in enumerate(zip(mean_abs_shap_clf.index[::-1],
                                     mean_abs_shap_clf.values[::-1])):
    ax.text(val + mean_abs_shap_clf.max() * 0.01, i,
            f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("../data/fig_model_clf_importance.png", bbox_inches="tight")
plt.show()

print("Top 10 features by mean |SHAP|:")
for feat, val in mean_abs_shap_clf.items():
    print(f"  {feat:<30s}  {val:.5f}")

top_clf_feature = mean_abs_shap_clf.index[0]
print()
print(f"Top predictive feature: {top_clf_feature}")
"""))

# ── 3. CostVarianceRegressor ───────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## 3. Cost Variance Regressor (XGBoost + GridSearchCV)

**Task:** predict the dollar gap between estimated freight cost and the invoiced
amount, using only pre-delivery features.  This helps flag shipments likely to
incur billing overruns before the invoice arrives.
"""))

cells.append(new_code_cell("""\
X_train_reg, y_train_reg = CostVarianceRegressor.prepare_features(train_df)
X_test_reg,  y_test_reg  = CostVarianceRegressor.prepare_features(test_df)

reg = CostVarianceRegressor()
reg.fit(X_train_reg, y_train_reg, cv=3)

print(f"Best params : {reg.best_params_}")
print(f"Best CV RMSE: {-max(reg.cv_results_['mean_test_score']):.4f}")
"""))

cells.append(new_code_cell("""\
metrics_reg = reg.evaluate(X_test_reg, y_test_reg)
print("Test-set performance:")
print(f"  RMSE : ${metrics_reg['rmse']:.4f}")
print(f"  MAE  : ${metrics_reg['mae']:.4f}")
print(f"  R2   : {metrics_reg['r2']:.4f}")
"""))

cells.append(new_markdown_cell("""\
### 3a. Actual vs Predicted & Residuals
"""))

cells.append(new_code_cell("""\
y_pred_reg = reg.predict(X_test_reg)
residuals  = y_test_reg.values - y_pred_reg

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs predicted scatter
ax = axes[0]
lim_lo = min(y_test_reg.min(), y_pred_reg.min()) - 0.1
lim_hi = max(y_test_reg.max(), y_pred_reg.max()) + 0.1
ax.scatter(y_test_reg, y_pred_reg, alpha=0.35, s=14,
           c="#3498db", linewidths=0, rasterized=True)
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual cost_variance_usd")
ax.set_ylabel("Predicted cost_variance_usd")
ax.set_title("Actual vs Predicted — Cost Variance", fontweight="bold")
ax.legend()
stats_str = (f"RMSE=${metrics_reg['rmse']:.3f}  "
             f"MAE=${metrics_reg['mae']:.3f}  "
             f"R2={metrics_reg['r2']:.3f}")
ax.text(0.05, 0.92, stats_str,
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Residual plot
ax = axes[1]
ax.scatter(y_pred_reg, residuals, alpha=0.35, s=14,
           c="#e74c3c", linewidths=0, rasterized=True)
ax.axhline(0, color="black", lw=1.5, ls="--")
ax.axhline( residuals.std(), color="grey", lw=1, ls=":", label="+1 std")
ax.axhline(-residuals.std(), color="grey", lw=1, ls=":", label="-1 std")
ax.set_xlabel("Predicted cost_variance_usd")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual Plot", fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("../data/fig_model_reg_scatter.png", bbox_inches="tight")
plt.show()
"""))

cells.append(new_markdown_cell("""\
### 3b. SHAP Bar Chart — Cost Variance Drivers
"""))

cells.append(new_code_cell("""\
shap_sample_reg = X_test_reg.sample(min(500, len(X_test_reg)), random_state=42)
shap_exp_reg    = reg.shap_explanation(shap_sample_reg)

mean_abs_shap_reg = (
    pd.Series(
        np.abs(shap_exp_reg.values).mean(axis=0),
        index=shap_sample_reg.columns,
    )
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 4))
colors = sns.color_palette("Blues_d", len(mean_abs_shap_reg))
ax.barh(mean_abs_shap_reg.index[::-1], mean_abs_shap_reg.values[::-1],
        color=colors[::-1], edgecolor="white", height=0.65)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 10 Features — Cost Variance Regressor", fontweight="bold")
for i, (feat, val) in enumerate(zip(mean_abs_shap_reg.index[::-1],
                                     mean_abs_shap_reg.values[::-1])):
    ax.text(val + mean_abs_shap_reg.max() * 0.01, i,
            f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("../data/fig_model_reg_shap.png", bbox_inches="tight")
plt.show()

top_reg_feature = mean_abs_shap_reg.index[0]
print(f"Top predictive feature for cost variance: {top_reg_feature}")
"""))

# ── 4. LeadTimeForecaster ──────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## 4. Lead Time Forecaster (Prophet, per carrier)

**Task:** forecast the average daily lead time for each carrier over the next
30 days, using the full historical shipment record.

Prophet decomposes the time series into trend + weekly + yearly seasonality,
producing 80% confidence intervals around each day's forecast.
"""))

cells.append(new_code_cell("""\
# Fit one Prophet model per carrier on the FULL dataset (no train/test split)
df_raw = pd.read_csv(DATA / "shipments.csv", parse_dates=["date"])

forecaster = LeadTimeForecaster(
    forecast_horizon=30,
    yearly_seasonality=True,
    weekly_seasonality=True,
)
forecaster.fit(df_raw)
print(f"Fitted {len(forecaster.models_)} carrier models: {list(forecaster.models_.keys())}")
"""))

cells.append(new_code_cell("""\
# Generate forecasts
forecasts = forecaster.forecast_all()

# Show tail of one carrier as a sanity check
fc_sample = forecasts["FastFreight"]
last_hist_date = forecaster.history_["FastFreight"]["ds"].max()
future_rows    = fc_sample[fc_sample["ds"] > last_hist_date]
print(f"FastFreight — {len(future_rows)} forecast days")
print(future_rows[["ds","yhat","yhat_lower","yhat_upper"]].tail(5).to_string(index=False))
"""))

cells.append(new_markdown_cell("""\
### 4a. 30-Day Forecast — All Carriers

The dashed vertical line marks today (the first forecast date).
Shaded bands show the 80% prediction interval.
"""))

cells.append(new_code_cell("""\
fig, ax = plt.subplots(figsize=(15, 6))

for carrier, fc in forecasts.items():
    color = CARRIER_COLORS[carrier]
    hist  = forecaster.history_[carrier]

    last_hist = hist["ds"].max()
    future    = fc[fc["ds"] > last_hist].copy()

    # Historical line (last 60 days for readability)
    recent_hist = hist[hist["ds"] >= hist["ds"].max() - pd.Timedelta(days=60)]
    ax.plot(recent_hist["ds"], recent_hist["y"],
            color=color, lw=1.0, alpha=0.45)

    # Forecast line + CI band
    ax.plot(future["ds"], future["yhat"],
            color=color, lw=2.2, label=carrier)
    ax.fill_between(future["ds"], future["yhat_lower"], future["yhat_upper"],
                    color=color, alpha=0.12)

ax.axvline(last_hist, color="black", lw=1.2, ls="--", label="Forecast start")
ax.set_xlabel("Date")
ax.set_ylabel("Avg Lead Time (days)")
ax.set_title("30-Day Lead Time Forecast per Carrier  (80% CI)", fontweight="bold")
ax.legend(fontsize=9, ncol=2, loc="upper left")
ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("../data/fig_model_prophet_forecast.png", bbox_inches="tight")
plt.show()
"""))

cells.append(new_markdown_cell("""\
### 4b. Forecast Summary Table
"""))

cells.append(new_code_cell("""\
summary_rows = []
for carrier, fc in forecasts.items():
    last_hist = forecaster.history_[carrier]["ds"].max()
    future = fc[fc["ds"] > last_hist]
    summary_rows.append({
        "carrier":          carrier,
        "forecast_avg_lead": round(future["yhat"].mean(), 2),
        "forecast_hi_ci":   round(future["yhat_upper"].mean(), 2),
        "forecast_lo_ci":   round(future["yhat_lower"].mean(), 2),
        "trend":            round(future["trend"].iloc[-1] - future["trend"].iloc[0], 3),
    })

fc_summary = pd.DataFrame(summary_rows).sort_values("forecast_avg_lead")
print("30-day forecast summary:")
print(fc_summary.to_string(index=False))
"""))

# ── 5. Score Full Dataset ──────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## 5. Score Full Dataset

Apply both supervised models to all 3,000 shipments and save to
`data/scored_shipments_full.csv`.  The `in_test_set` flag indicates
which rows were held out during training.
"""))

cells.append(new_code_cell("""\
# Score every shipment
X_full_clf, _ = DelayClassifier.prepare_features(df)
X_full_reg, _ = CostVarianceRegressor.prepare_features(df)

df_scored = df.copy()
df_scored["pred_is_delayed"]   = clf.predict(X_full_clf)
df_scored["pred_delay_proba"]  = clf.predict_proba(X_full_clf).round(4)
df_scored["pred_cost_variance"] = reg.predict(X_full_reg).round(4)

# Flag rows that were in the test set
test_ids = set(test_df["shipment_id"])
df_scored["in_test_set"] = df_scored["shipment_id"].isin(test_ids).astype(int)

out_path = DATA / "scored_shipments_full.csv"
df_scored.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df_scored.shape}")
print(f"Test-set rows: {df_scored['in_test_set'].sum()}")
print()
print("Score distribution:")
print(df_scored[["pred_is_delayed","pred_delay_proba","pred_cost_variance"]].describe().round(4))
"""))

# ── 6. Summary Table ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## 6. Model Summary

Key results across all three models in one place.
"""))

cells.append(new_code_cell("""\
# Build summary
summary = pd.DataFrame([
    {
        "Model":              "DelayClassifier",
        "Type":               "Binary classification",
        "Target":             "is_delayed",
        "Algorithm":          "XGBoost",
        "Key metric":         f"AUC = {metrics_clf['auc']:.4f}",
        "Also":               f"F1 = {metrics_clf['f1']:.4f}",
        "Top feature":        top_clf_feature,
        "Best params":        str(clf.best_params_),
    },
    {
        "Model":              "CostVarianceRegressor",
        "Type":               "Regression",
        "Target":             "cost_variance_usd",
        "Algorithm":          "XGBoost",
        "Key metric":         f"RMSE = ${metrics_reg['rmse']:.4f}",
        "Also":               f"R2 = {metrics_reg['r2']:.4f}",
        "Top feature":        top_reg_feature,
        "Best params":        str(reg.best_params_),
    },
    {
        "Model":              "LeadTimeForecaster",
        "Type":               "Time-series forecast",
        "Target":             "actual_lead_days (daily avg)",
        "Algorithm":          "Prophet",
        "Key metric":         "30-day horizon per carrier",
        "Also":               "80% prediction interval",
        "Top feature":        "Trend + yearly/weekly seasonality",
        "Best params":        "yearly + weekly seasonality",
    },
])

pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 160)
print("=" * 100)
print("MODEL SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))
"""))

# ── Build notebook ─────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

out_path = Path(__file__).parent / "03_modeling.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

code_cells     = sum(1 for c in cells if c["cell_type"] == "code")
markdown_cells = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Written: {out_path}")
print(f"Cells: {len(cells)} total ({code_cells} code, {markdown_cells} markdown)")
