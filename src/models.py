"""
Model classes for supply chain forecasting.

Classes
-------
DelayClassifier       XGBoost binary classifier → is_delayed
CostVarianceRegressor XGBoost regressor         → cost_variance_usd
LeadTimeForecaster    Prophet per-carrier daily lead-time forecast

All models operate on the output of FeatureEngineer.fit_transform().
Both supervised models use only pre-delivery features to avoid leakage.
"""
from __future__ import annotations

import contextlib
import io
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

# Prophet import — suppress noisy Stan/cmdstan output on import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from prophet import Prophet


# ── Pre-delivery feature set (shared by both supervised models) ───────────────

# Columns available before a shipment arrives (no actual_lead_days / delay info)
_BASE_FEATURES: list[str] = [
    # temporal
    "month", "quarter", "day_of_week", "is_weekend", "is_nov_dec",
    # shipment characteristics
    "weight_bucket", "weight_kg", "quantity", "promised_lead_days",
    "freight_cost_usd", "cost_per_kg",
    # carrier rolling history (30-day lookback)
    "rolling_avg_delay", "rolling_on_time_rate",
    "rolling_damage_rate", "rolling_cost_variance",
    # lane risk
    "lane_avg_delay", "lane_delay_rate", "lane_risk_numeric",
    # encoded categoricals (added by prepare_features)
    "direction_enc", "carrier_enc", "sku_enc",
]

_CARRIER_RANK: dict[str, int] = {
    "FastFreight": 0, "PrimeHaul": 1, "RelayEx": 2, "SwiftLog": 3,
    "CargoLink": 4, "NorthStar": 5, "DirectMove": 6,
}

_DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5],
    "learning_rate": [0.05, 0.10],
}


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction_enc, carrier_enc, sku_enc columns (in-place copy)."""
    out = df.copy()
    out["direction_enc"] = (out["direction"] == "inbound").astype(int)
    out["carrier_enc"]   = out["carrier"].map(_CARRIER_RANK).fillna(3).astype(int)
    sku_cats = sorted(out["sku_category"].unique())
    sku_map  = {s: i for i, s in enumerate(sku_cats)}
    out["sku_enc"] = out["sku_category"].map(sku_map).fillna(0).astype(int)
    return out


# ── DelayClassifier ────────────────────────────────────────────────────────────

class DelayClassifier:
    """
    XGBoost binary classifier predicting ``is_delayed``.

    Parameters
    ----------
    param_grid : optional grid for GridSearchCV; uses _DEFAULT_PARAM_GRID if None.

    Attributes set after fit()
    --------------------------
    best_estimator_ : fitted XGBClassifier (best from grid search)
    best_params_    : dict of winning hyper-parameters
    feature_names_  : list of feature column names
    cv_results_     : full GridSearchCV cv_results_ dict
    """

    TARGET = "is_delayed"
    FEATURE_COLS = _BASE_FEATURES

    def __init__(self, param_grid: dict | None = None) -> None:
        self.param_grid = param_grid or _DEFAULT_PARAM_GRID
        self.best_estimator_: XGBClassifier | None = None
        self.best_params_: dict = {}
        self.feature_names_: list[str] = []
        self.cv_results_: dict = {}
        self._explainer: shap.TreeExplainer | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def prepare_features(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Encode categoricals and return (X, y) ready for fit/predict.

        Parameters
        ----------
        df : output of FeatureEngineer.fit_transform() — must contain
             feature columns and the ``is_delayed`` target.

        Returns
        -------
        X : DataFrame of shape (n, len(FEATURE_COLS))
        y : Series of 0/1 labels
        """
        enc = _encode_categoricals(df)
        X = enc[_BASE_FEATURES].copy()
        y = enc[DelayClassifier.TARGET]
        return X, y

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 3,
        scoring: str = "roc_auc",
    ) -> "DelayClassifier":
        """Grid-search over XGBClassifier hyper-parameters."""
        base = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=1,
        )
        gs = GridSearchCV(
            base,
            self.param_grid,
            cv=cv,
            scoring=scoring,
            refit=True,
            n_jobs=1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        self.best_estimator_ = gs.best_estimator_
        self.best_params_    = gs.best_params_
        self.feature_names_  = list(X_train.columns)
        self.cv_results_     = gs.cv_results_
        self._explainer      = None  # reset cache
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions."""
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of delay (class 1)."""
        return self.best_estimator_.predict_proba(X)[:, 1]

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """
        Returns
        -------
        dict with keys: precision, recall, f1, auc, confusion_matrix
        """
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        return {
            "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1":               round(f1_score(y_test, y_pred, zero_division=0),        4),
            "auc":              round(roc_auc_score(y_test, y_proba),                   4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def shap_explanation(self, X: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP values; caches the TreeExplainer."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.best_estimator_)
        return self._explainer(X)

    def feature_importance_df(self) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, sorted descending."""
        raise NotImplementedError(
            "Call shap_explanation(X) first, then compute mean(|SHAP|) externally."
        )


# ── CostVarianceRegressor ──────────────────────────────────────────────────────

class CostVarianceRegressor:
    """
    XGBoost regressor predicting ``cost_variance_usd``.

    Uses the same pre-delivery feature set as DelayClassifier so both
    models can be applied at dispatch time.
    """

    TARGET = "cost_variance_usd"
    FEATURE_COLS = _BASE_FEATURES

    def __init__(self, param_grid: dict | None = None) -> None:
        self.param_grid = param_grid or _DEFAULT_PARAM_GRID
        self.best_estimator_: XGBRegressor | None = None
        self.best_params_: dict = {}
        self.feature_names_: list[str] = []
        self.cv_results_: dict = {}
        self._explainer: shap.TreeExplainer | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def prepare_features(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        enc = _encode_categoricals(df)
        X = enc[_BASE_FEATURES].copy()
        y = enc[CostVarianceRegressor.TARGET]
        return X, y

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 3,
        scoring: str = "neg_root_mean_squared_error",
    ) -> "CostVarianceRegressor":
        base = XGBRegressor(
            random_state=42,
            n_jobs=1,
        )
        gs = GridSearchCV(
            base,
            self.param_grid,
            cv=cv,
            scoring=scoring,
            refit=True,
            n_jobs=1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        self.best_estimator_ = gs.best_estimator_
        self.best_params_    = gs.best_params_
        self.feature_names_  = list(X_train.columns)
        self.cv_results_     = gs.cv_results_
        self._explainer      = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.best_estimator_.predict(X)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """Returns dict with keys: rmse, mae."""
        y_pred = self.predict(X_test)
        residuals = y_test.values - y_pred
        return {
            "rmse":      round(float(np.sqrt(np.mean(residuals ** 2))), 4),
            "mae":       round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2":        round(float(1 - np.sum(residuals**2) /
                               np.sum((y_test.values - y_test.mean())**2)), 4),
        }

    def shap_explanation(self, X: pd.DataFrame) -> shap.Explanation:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.best_estimator_)
        return self._explainer(X)


# ── LeadTimeForecaster ─────────────────────────────────────────────────────────

class LeadTimeForecaster:
    """
    Per-carrier daily lead-time forecaster using Facebook Prophet.

    Fits one Prophet model per carrier on historical daily average
    ``actual_lead_days``.  Missing dates (no shipments) are forward-filled
    with the carrier's rolling 7-day mean before fitting.

    Attributes set after fit()
    --------------------------
    models_    : dict[carrier_name → fitted Prophet]
    history_   : dict[carrier_name → training DataFrame (ds, y)]
    """

    def __init__(
        self,
        forecast_horizon: int = 30,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
    ) -> None:
        self.forecast_horizon   = forecast_horizon
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality  = daily_seasonality

        self.models_:  dict[str, Prophet] = {}
        self.history_: dict[str, pd.DataFrame] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, carriers: list[str] | None = None) -> "LeadTimeForecaster":
        """
        Fit one Prophet model per carrier.

        Parameters
        ----------
        df       : shipments DataFrame with ``date``, ``carrier``,
                   ``actual_lead_days`` columns.
        carriers : subset of carriers to fit; defaults to all unique carriers.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if carriers is None:
            carriers = sorted(df["carrier"].unique())

        for carrier in carriers:
            series = self._build_daily_series(df, carrier)
            self.history_[carrier] = series

            m = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                interval_width=0.80,
            )
            # Suppress Prophet/Stan stdout/stderr
            with open(os.devnull, "w") as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(series)

            self.models_[carrier] = m

        return self

    def forecast(self, carrier: str) -> pd.DataFrame:
        """
        Return a forecast DataFrame for ``carrier``.

        Columns: ds, yhat, yhat_lower, yhat_upper, trend.
        """
        if carrier not in self.models_:
            raise ValueError(f"No fitted model for carrier '{carrier}'. Call fit() first.")

        m      = self.models_[carrier]
        future = m.make_future_dataframe(periods=self.forecast_horizon, freq="D")

        with open(os.devnull, "w") as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = m.predict(future)

        return fc[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]

    def forecast_all(self) -> dict[str, pd.DataFrame]:
        """Return forecast DataFrames for every fitted carrier."""
        return {c: self.forecast(c) for c in self.models_}

    # ── Internal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_daily_series(df: pd.DataFrame, carrier: str) -> pd.DataFrame:
        """
        Aggregate to daily average lead time, fill missing dates with
        a 7-day rolling mean, and return a Prophet-ready (ds, y) DataFrame.
        """
        sub = df[df["carrier"] == carrier].copy()
        daily = (
            sub.groupby("date")["actual_lead_days"]
            .mean()
            .rename("y")
            .reset_index()
            .rename(columns={"date": "ds"})
        )

        # Fill calendar gaps so Prophet sees a contiguous series
        full_range = pd.DataFrame(
            {"ds": pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")}
        )
        daily = full_range.merge(daily, on="ds", how="left")

        # Fill NaN with 7-day rolling mean, then back-fill any leading NaNs
        daily["y"] = (
            daily["y"]
            .fillna(daily["y"].rolling(7, min_periods=1, center=True).mean())
            .bfill()
            .ffill()
        )
        return daily
