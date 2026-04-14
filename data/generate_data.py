"""
Supply Chain Dataset Generator
Generates 3 tables: shipments, carrier_scorecard, lane_risk
"""

import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path

fake = Faker()
np.random.seed(42)

# --Constants ────────────────────────────────────────────────────────────────

CARRIERS = ["FastFreight", "RelayEx", "PrimeHaul", "SwiftLog",
            "CargoLink", "NorthStar", "DirectMove"]

ORIGINS = [
    "Chicago, IL", "Los Angeles, CA", "Houston, TX", "Atlanta, GA",
    "Dallas, TX", "New York, NY", "Seattle, WA", "Phoenix, AZ",
    "Denver, CO", "Miami, FL", "Detroit, MI", "Memphis, TN",
]

DESTINATIONS = [
    "Columbus, OH", "Louisville, KY", "Indianapolis, IN", "Kansas City, MO",
    "Nashville, TN", "Charlotte, NC", "Portland, OR", "Salt Lake City, UT",
    "Minneapolis, MN", "San Antonio, TX", "Baltimore, MD", "Tampa, FL",
]

SKU_CATEGORIES = [
    "Electronics", "Apparel", "Food & Beverage", "Industrial Parts",
    "Pharmaceuticals", "Consumer Goods", "Automotive", "Office Supplies",
]

# Carrier base delay profiles (mean extra days, std) — FastFreight best, DirectMove worst
CARRIER_PROFILES = {
    "FastFreight": {"delay_mu": -0.3, "delay_sigma": 0.8, "cost_bias": -0.02, "damage_p": 0.008},
    "RelayEx":     {"delay_mu":  0.4, "delay_sigma": 1.2, "cost_bias":  0.03, "damage_p": 0.018},
    "PrimeHaul":   {"delay_mu":  0.2, "delay_sigma": 1.0, "cost_bias":  0.01, "damage_p": 0.014},
    "SwiftLog":    {"delay_mu":  0.6, "delay_sigma": 1.4, "cost_bias":  0.04, "damage_p": 0.022},
    "CargoLink":   {"delay_mu":  0.8, "delay_sigma": 1.6, "cost_bias":  0.05, "damage_p": 0.025},
    "NorthStar":   {"delay_mu":  1.0, "delay_sigma": 1.8, "cost_bias":  0.06, "damage_p": 0.030},
    "DirectMove":  {"delay_mu":  2.2, "delay_sigma": 2.5, "cost_bias":  0.12, "damage_p": 0.055},
}

# --Helpers ──────────────────────────────────────────────────────────────────

def seasonal_delay_bonus(month: int) -> float:
    """Extra delay days added in Nov/Dec (holiday season)."""
    return {11: 1.2, 12: 1.8}.get(month, 0.0)


def direction_delay_bonus(direction: str) -> float:
    """Inbound shipments face more customs / receiving dock congestion."""
    return 0.4 if direction == "inbound" else 0.0


def make_lane_id(origin: str, destination: str) -> str:
    o = origin.split(",")[1].strip()
    d = destination.split(",")[1].strip()
    return f"{o}-{d}"


# --Table 1: Shipments ────────────────────────────────────────────────────────

def generate_shipments(n: int = 3000) -> pd.DataFrame:
    rows = []
    start_date = pd.Timestamp("2023-01-01")
    end_date   = pd.Timestamp("2024-12-31")
    date_range = (end_date - start_date).days

    for i in range(n):
        date      = start_date + pd.Timedelta(days=int(np.random.randint(0, date_range)))
        month     = date.month
        direction = np.random.choice(["inbound", "outbound"], p=[0.45, 0.55])
        carrier   = np.random.choice(CARRIERS)
        origin    = np.random.choice(ORIGINS)
        dest      = np.random.choice([d for d in DESTINATIONS if d != origin])
        lane_id   = make_lane_id(origin, dest)
        sku_cat   = np.random.choice(SKU_CATEGORIES)

        weight_kg  = round(np.random.lognormal(mean=4.5, sigma=0.8), 1)   # ~90–2000 kg range
        quantity   = int(np.random.lognormal(mean=3.8, sigma=0.9))

        promised   = int(np.random.choice([2, 3, 5, 7, 10, 14], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05]))

        prof = CARRIER_PROFILES[carrier]
        raw_delay = np.random.normal(
            loc=prof["delay_mu"] + seasonal_delay_bonus(month) + direction_delay_bonus(direction),
            scale=prof["delay_sigma"],
        )
        delay_days = round(raw_delay, 1)
        actual     = max(1, promised + int(round(delay_days)))
        is_delayed = int(delay_days > 0)

        base_rate   = 2.80 + weight_kg * 0.012 + quantity * 0.05
        freight_usd = round(base_rate * np.random.uniform(0.9, 1.1), 2)
        cost_factor = 1 + prof["cost_bias"] + np.random.normal(0, 0.04)
        invoiced    = round(freight_usd * cost_factor, 2)
        cost_var    = round(invoiced - freight_usd, 2)

        # Damage correlates with cost overruns and DirectMove
        damage_p = prof["damage_p"] * (1.5 if cost_var > 15 else 1.0)
        damage_flag = int(np.random.random() < damage_p)

        rows.append({
            "shipment_id":        f"SHP-{i+1:05d}",
            "date":               date.date(),
            "direction":          direction,
            "carrier":            carrier,
            "origin":             origin,
            "destination":        dest,
            "lane_id":            lane_id,
            "sku_category":       sku_cat,
            "weight_kg":          weight_kg,
            "quantity":           quantity,
            "promised_lead_days": promised,
            "actual_lead_days":   actual,
            "delay_days":         delay_days,
            "is_delayed":         is_delayed,
            "freight_cost_usd":   freight_usd,
            "invoiced_cost_usd":  invoiced,
            "cost_variance_usd":  cost_var,
            "damage_flag":        damage_flag,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# --Table 2: Carrier Scorecard ────────────────────────────────────────────────

def generate_carrier_scorecard(shipments: pd.DataFrame) -> pd.DataFrame:
    df = shipments.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    agg = (
        df.groupby(["carrier", "month"])
        .agg(
            avg_delay_days    = ("delay_days",        "mean"),
            on_time_rate      = ("is_delayed",        lambda x: round(1 - x.mean(), 4)),
            avg_cost_variance = ("cost_variance_usd", "mean"),
            damage_rate       = ("damage_flag",       "mean"),
            total_shipments   = ("shipment_id",       "count"),
        )
        .reset_index()
    )

    agg["avg_delay_days"]    = agg["avg_delay_days"].round(3)
    agg["avg_cost_variance"] = agg["avg_cost_variance"].round(2)
    agg["damage_rate"]       = agg["damage_rate"].round(4)
    return agg.sort_values(["carrier", "month"]).reset_index(drop=True)


# --Table 3: Lane Risk ────────────────────────────────────────────────────────

def generate_lane_risk(shipments: pd.DataFrame) -> pd.DataFrame:
    df = shipments.copy()

    agg = (
        df.groupby(["lane_id", "origin", "destination", "direction"])
        .agg(
            avg_delay_days    = ("delay_days",        "mean"),
            delay_rate        = ("is_delayed",        "mean"),
            avg_cost_variance = ("cost_variance_usd", "mean"),
            shipment_count    = ("shipment_id",       "count"),
        )
        .reset_index()
    )

    # Filter out lanes with very few shipments for reliability
    agg = agg[agg["shipment_count"] >= 5].copy()

    agg["avg_delay_days"]    = agg["avg_delay_days"].round(3)
    agg["delay_rate"]        = agg["delay_rate"].round(4)
    agg["avg_cost_variance"] = agg["avg_cost_variance"].round(2)

    # Risk tier based on delay_rate percentiles
    p33 = agg["delay_rate"].quantile(0.33)
    p66 = agg["delay_rate"].quantile(0.66)

    def risk_tier(rate):
        if rate <= p33:
            return "Low"
        elif rate <= p66:
            return "Medium"
        return "High"

    agg["risk_tier"] = agg["delay_rate"].apply(risk_tier)
    agg = agg.drop(columns=["shipment_count"])
    return agg.sort_values("lane_id").reset_index(drop=True)


# --Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    print("Generating shipments (3000 rows)...")
    shipments = generate_shipments(3000)
    shipments.to_csv(out_dir / "shipments.csv", index=False)

    print("Generating carrier scorecard...")
    scorecard = generate_carrier_scorecard(shipments)
    scorecard.to_csv(out_dir / "carrier_scorecard.csv", index=False)

    print("Generating lane risk table...")
    lane_risk = generate_lane_risk(shipments)
    lane_risk.to_csv(out_dir / "lane_risk.csv", index=False)

    # --Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nshipments.csv          {len(shipments):>6,} rows × {len(shipments.columns)} cols")
    print(f"carrier_scorecard.csv  {len(scorecard):>6,} rows × {len(scorecard.columns)} cols")
    print(f"lane_risk.csv          {len(lane_risk):>6,} rows × {len(lane_risk.columns)} cols")

    print("\n-- Carrier Performance (avg delay days, on-time rate) --")
    perf = (
        shipments.groupby("carrier")
        .agg(
            avg_delay  = ("delay_days",  "mean"),
            on_time    = ("is_delayed",  lambda x: 1 - x.mean()),
            shipments  = ("shipment_id", "count"),
            damage_pct = ("damage_flag", "mean"),
        )
        .sort_values("avg_delay")
    )
    for c, r in perf.iterrows():
        print(f"  {c:<14} delay={r['avg_delay']:+.2f}d  on-time={r['on_time']:.1%}  "
              f"damage={r['damage_pct']:.1%}  n={int(r['shipments'])}")

    print("\n--Seasonal Delay Pattern (avg delay_days by month) --")
    shipments["month"] = pd.to_datetime(shipments["date"]).dt.month
    monthly = shipments.groupby("month")["delay_days"].mean()
    for m, v in monthly.items():
        bar = "#" * int(max(0, v) * 3)
        print(f"  Month {m:>2}: {v:+.2f}d  {bar}")

    print("\n--Direction Comparison --")
    dir_stats = shipments.groupby("direction").agg(
        avg_delay  = ("delay_days",  "mean"),
        on_time    = ("is_delayed",  lambda x: 1 - x.mean()),
    )
    for d, r in dir_stats.iterrows():
        print(f"  {d:<10} avg_delay={r['avg_delay']:+.2f}d  on_time={r['on_time']:.1%}")

    print("\n--Lane Risk Distribution --")
    print(lane_risk["risk_tier"].value_counts().to_string())

    print("\n--Cost Variance Stats --")
    cv = shipments["cost_variance_usd"]
    print(f"  mean={cv.mean():.2f}  median={cv.median():.2f}  "
          f"p95={cv.quantile(0.95):.2f}  max={cv.max():.2f}")

    print("\nAll CSVs saved to data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
