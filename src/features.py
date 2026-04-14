"""
Feature engineering for supply chain shipments data.

Usage
-----
    from src.features import FeatureEngineer

    fe = FeatureEngineer(data_dir="data/")
    df_raw = pd.read_csv("data/shipments.csv", parse_dates=["date"])
    df_feat = fe.fit_transform(df_raw)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Builds a feature matrix from shipments.csv enriched with lane_risk.csv.

    Feature groups
    --------------
    1. Carrier history   — rolling 30-day window per carrier (no-leakage)
    2. Lane risk         — merged from lane_risk.csv, risk tier encoded as int
    3. Temporal          — month, day_of_week, is_weekend, is_nov_dec, quarter
    4. Shipment          — weight_bucket, cost_per_kg, delay_ratio
    5. Targets (kept)    — is_delayed, delay_days, cost_variance_usd
    """

    RISK_MAP: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self._lane_risk_cache: pd.DataFrame | None = None

    # ── Public API ──────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature groups and return the enriched DataFrame.

        Parameters
        ----------
        df : raw shipments DataFrame (output of pd.read_csv("shipments.csv")).

        Returns
        -------
        DataFrame with original columns plus all engineered features.
        """
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])

        out = self._add_temporal_features(out)
        out = self._add_shipment_features(out)
        out = self._add_carrier_rolling_features(out)
        out = self._add_lane_risk_features(out)
        return out

    @property
    def feature_cols(self) -> list[str]:
        """Ordered list of engineered feature column names (excludes targets & IDs)."""
        return [
            # temporal
            "month", "quarter", "day_of_week", "is_weekend", "is_nov_dec",
            # shipment
            "weight_bucket", "cost_per_kg", "delay_ratio",
            # carrier rolling
            "rolling_avg_delay", "rolling_on_time_rate",
            "rolling_damage_rate", "rolling_cost_variance",
            # lane risk
            "lane_avg_delay", "lane_delay_rate", "lane_risk_numeric",
        ]

    @property
    def target_cols(self) -> list[str]:
        return ["is_delayed", "delay_days", "cost_variance_usd"]

    # ── Feature group implementations ───────────────────────────────────────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["month"]       = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek          # Mon=0 … Sun=6
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_nov_dec"]  = df["month"].isin([11, 12]).astype(int)
        df["quarter"]     = df["date"].dt.quarter
        return df

    def _add_shipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Weight bucket by tertile (0=light, 1=medium, 2=heavy)
        q33, q66 = df["weight_kg"].quantile([1 / 3, 2 / 3])
        df["weight_bucket"] = pd.cut(
            df["weight_kg"],
            bins=[-np.inf, q33, q66, np.inf],
            labels=[0, 1, 2],
        ).astype(int)

        df["cost_per_kg"] = (
            df["freight_cost_usd"] / df["weight_kg"].replace(0.0, np.nan)
        ).round(4)

        df["delay_ratio"] = (
            df["actual_lead_days"]
            / df["promised_lead_days"].replace(0, np.nan)
        ).round(4)

        return df

    def _add_carrier_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-carrier 30-day rolling window features.

        ``closed="left"`` ensures the current row is excluded from its own
        window — no target leakage.  The very first row(s) per carrier that
        have no prior history within 30 days receive NaN, which are then
        back-filled with each carrier's global mean.
        """
        df = df.sort_values("date").reset_index(drop=True)

        SOURCE = {
            "rolling_avg_delay":     "delay_days",
            "rolling_on_time_rate":  "is_delayed",   # inverted to on-time
            "rolling_damage_rate":   "damage_flag",
            "rolling_cost_variance": "cost_variance_usd",
        }

        pieces: list[pd.DataFrame] = []

        for carrier, group in df.groupby("carrier", sort=False):
            g = group.set_index("date").sort_index()
            r = g.rolling("30D", min_periods=1, closed="left")

            g["rolling_avg_delay"]     = r["delay_days"].mean()
            g["rolling_on_time_rate"]  = 1.0 - r["is_delayed"].mean()
            g["rolling_damage_rate"]   = r["damage_flag"].mean()
            g["rolling_cost_variance"] = r["cost_variance_usd"].mean()
            pieces.append(g)

        rolled = (
            pd.concat(pieces)
            .reset_index()[["shipment_id"] + list(SOURCE.keys())]
        )

        df = df.merge(rolled, on="shipment_id", how="left")

        # Fill cold-start NaNs with each carrier's global mean
        for col, src in SOURCE.items():
            nan_mask = df[col].isna()
            if nan_mask.any():
                if col == "rolling_on_time_rate":
                    fill = df.groupby("carrier")["is_delayed"].transform(
                        lambda x: 1.0 - x.mean()
                    )
                else:
                    fill = df.groupby("carrier")[src].transform("mean")
                df.loc[nan_mask, col] = fill[nan_mask]

        return df

    def _add_lane_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # lane_id encodes state pairs only, so multiple city routes share the same
        # lane_id.  Aggregate to one row per (lane_id, direction) before merging
        # to avoid row explosion.
        lr_raw = self._load_lane_risk()
        lr = (
            lr_raw.groupby(["lane_id", "direction"], as_index=False)
            .agg(
                lane_avg_delay    = ("avg_delay_days", "mean"),
                lane_delay_rate   = ("delay_rate",     "mean"),
                lane_risk_numeric = ("risk_tier",
                                     lambda x: round(x.map(self.RISK_MAP).mean())),
            )
        )
        out = df.merge(lr, on=["lane_id", "direction"], how="left")
        # Fill NaN for lanes too sparse to appear in lane_risk (< 5 shipments)
        out["lane_avg_delay"]  = out["lane_avg_delay"].fillna(out["lane_avg_delay"].mean())
        out["lane_delay_rate"] = out["lane_delay_rate"].fillna(out["lane_delay_rate"].mean())
        # Keep risk numeric as integer 0/1/2 — round after mean-fill
        out["lane_risk_numeric"] = (
            out["lane_risk_numeric"]
            .fillna(out["lane_risk_numeric"].mean())
            .round()
            .astype(int)
        )
        return out

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _load_lane_risk(self) -> pd.DataFrame:
        if self._lane_risk_cache is None:
            self._lane_risk_cache = pd.read_csv(self.data_dir / "lane_risk.csv")
        return self._lane_risk_cache
