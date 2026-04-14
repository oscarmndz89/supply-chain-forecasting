"""
Supply Chain Intelligence Dashboard
4-page Streamlit app — sidebar radio navigation
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────

DATA = Path(__file__).parent.parent / "data"

CARRIERS_ORDER = [
    "FastFreight", "RelayEx", "PrimeHaul",
    "SwiftLog", "CargoLink", "NorthStar", "DirectMove",
]

# FastFreight = green, DirectMove = red, others = neutral slate
CARRIER_COLORS = {
    "FastFreight": "#27ae60",
    "RelayEx":     "#7f8c8d",
    "PrimeHaul":   "#7f8c8d",
    "SwiftLog":    "#7f8c8d",
    "CargoLink":   "#7f8c8d",
    "NorthStar":   "#7f8c8d",
    "DirectMove":  "#c0392b",
}

RISK_COLORS = {"Low": "#27ae60", "Medium": "#e67e22", "High": "#c0392b"}

ON_TIME_THRESHOLD = 0.70   # 70 % threshold for Page 1 chart

# ── Data loaders (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(DATA / "scored_shipments_full.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_scorecard() -> pd.DataFrame:
    sc = pd.read_csv(DATA / "carrier_scorecard.csv")
    sc["month_dt"] = pd.to_datetime(sc["month"])
    return sc


@st.cache_data
def load_lane_risk() -> pd.DataFrame:
    return pd.read_csv(DATA / "lane_risk.csv")


# ── Shared style helpers ───────────────────────────────────────────────────────

def carrier_color_seq(carriers: list[str]) -> list[str]:
    return [CARRIER_COLORS.get(c, "#7f8c8d") for c in carriers]


def _plotly_defaults(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    return fig


# ── Page 1 — Carrier Risk Scorecard ───────────────────────────────────────────

def page_carrier_scorecard() -> None:
    st.title("📊 Carrier Risk Scorecard")
    st.caption("Performance benchmarks across all 7 carriers — Jan 2023 to Dec 2024")

    df = load_scored()
    sc = load_scorecard()

    # ── KPI row ────────────────────────────────────────────────────────────────
    total_shipments  = len(df)
    overall_on_time  = 1 - df["is_delayed"].mean()
    avg_cost_var     = df["cost_variance_usd"].mean()
    total_damage     = int(df["damage_flag"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Shipments",     f"{total_shipments:,}")
    c2.metric("Overall On-Time Rate", f"{overall_on_time:.1%}",
              delta=f"{overall_on_time - ON_TIME_THRESHOLD:+.1%} vs 70% target")
    c3.metric("Avg Cost Variance",    f"${avg_cost_var:.2f}")
    c4.metric("Damage Incidents",     f"{total_damage:,}")

    st.divider()

    # ── Per-carrier stats ──────────────────────────────────────────────────────
    carrier_stats = (
        df.groupby("carrier")
        .agg(
            on_time_rate  = ("is_delayed",        lambda x: 1 - x.mean()),
            avg_delay_days= ("delay_days",         "mean"),
            damage_rate   = ("damage_flag",        "mean"),
            n_shipments   = ("shipment_id",        "count"),
        )
        .reindex(CARRIERS_ORDER)
        .reset_index()
    )

    col_colors = carrier_color_seq(carrier_stats["carrier"].tolist())

    st.subheader("Carrier Comparison")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # On-Time Rate
    with chart_col1:
        fig = go.Figure(go.Bar(
            x=carrier_stats["carrier"],
            y=carrier_stats["on_time_rate"],
            marker_color=col_colors,
            text=[f"{v:.0%}" for v in carrier_stats["on_time_rate"]],
            textposition="outside",
        ))
        fig.add_hline(
            y=ON_TIME_THRESHOLD,
            line_dash="dot",
            line_color="#e74c3c",
            annotation_text="70% target",
            annotation_position="top right",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        fig.update_layout(title="On-Time Rate")
        st.plotly_chart(_plotly_defaults(fig), width="stretch")

    # Avg Delay Days
    with chart_col2:
        fig = go.Figure(go.Bar(
            x=carrier_stats["carrier"],
            y=carrier_stats["avg_delay_days"],
            marker_color=col_colors,
            text=[f"{v:+.2f}d" for v in carrier_stats["avg_delay_days"]],
            textposition="outside",
        ))
        fig.update_layout(title="Avg Delay Days")
        st.plotly_chart(_plotly_defaults(fig), width="stretch")

    # Damage Rate
    with chart_col3:
        fig = go.Figure(go.Bar(
            x=carrier_stats["carrier"],
            y=carrier_stats["damage_rate"],
            marker_color=col_colors,
            text=[f"{v:.1%}" for v in carrier_stats["damage_rate"]],
            textposition="outside",
        ))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title="Damage Rate")
        st.plotly_chart(_plotly_defaults(fig), width="stretch")

    st.divider()

    # ── Carrier drilldown ──────────────────────────────────────────────────────
    st.subheader("Monthly Carrier Drilldown")
    selected_carrier = st.selectbox(
        "Select a carrier",
        CARRIERS_ORDER,
        index=0,
    )

    carrier_monthly = (
        sc[sc["carrier"] == selected_carrier]
        .sort_values("month_dt")
    )

    drill_col1, drill_col2 = st.columns(2)
    line_color = CARRIER_COLORS.get(selected_carrier, "#3498db")

    with drill_col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=carrier_monthly["month_dt"],
            y=carrier_monthly["avg_delay_days"],
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(size=5),
            name="Avg Delay Days",
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(line_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        fig.update_layout(title=f"{selected_carrier} — Monthly Avg Delay")
        fig.update_xaxes(title="Month")
        fig.update_yaxes(title="Days")
        st.plotly_chart(_plotly_defaults(fig), width="stretch")

    with drill_col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=carrier_monthly["month_dt"],
            y=carrier_monthly["avg_cost_variance"],
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(size=5),
            name="Avg Cost Variance",
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(line_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        fig.update_layout(title=f"{selected_carrier} — Monthly Cost Variance")
        fig.update_xaxes(title="Month")
        fig.update_yaxes(title="USD", tickprefix="$")
        st.plotly_chart(_plotly_defaults(fig), width="stretch")


# ── Page 2 — Delay Forecast ────────────────────────────────────────────────────

def page_delay_forecast() -> None:
    st.title("⏱ Delay Forecast")
    st.caption("Predicted delay probability from the XGBoost classifier")

    df = load_scored()

    # ── Sidebar filters ────────────────────────────────────────────────────────
    st.sidebar.subheader("Filters")

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=[date_min, date_max],
        min_value=date_min,
        max_value=date_max,
        key="delay_dates",
    )
    # Guard: date_input may return 1 or 2 dates while user is editing
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_min

    carrier_sel = st.sidebar.multiselect(
        "Carriers",
        options=CARRIERS_ORDER,
        default=CARRIERS_ORDER,
        key="delay_carriers",
    )
    if not carrier_sel:
        carrier_sel = CARRIERS_ORDER

    # Apply filters
    mask = (
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
        & (df["carrier"].isin(carrier_sel))
    )
    dff = df[mask].copy()

    if dff.empty:
        st.warning("No shipments match the current filters.")
        return

    # ── KPI cards ──────────────────────────────────────────────────────────────
    overall_baseline = df["is_delayed"].mean()
    filtered_pred_rate = dff["pred_is_delayed"].mean()

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Filtered Shipments",      f"{len(dff):,}")
    kc2.metric("Predicted Delay Rate",    f"{filtered_pred_rate:.1%}",
               delta=f"{filtered_pred_rate - overall_baseline:+.1%} vs overall baseline")
    kc3.metric("Overall Baseline",        f"{overall_baseline:.1%}")
    kc4.metric("Avg Predicted Probability", f"{dff['pred_delay_proba'].mean():.1%}")

    st.divider()

    # ── Time series of predicted delay probability ─────────────────────────────
    st.subheader("Predicted Delay Probability Over Time")

    daily = (
        dff.groupby("date")["pred_delay_proba"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    daily["rolling_7d"] = daily["pred_delay_proba"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["pred_delay_proba"],
        mode="lines",
        line=dict(color="#bdc3c7", width=1),
        name="Daily avg",
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling_7d"],
        mode="lines",
        line=dict(color="#2c3e50", width=2.5),
        name="7-day rolling avg",
    ))
    fig.add_hline(
        y=overall_baseline,
        line_dash="dot",
        line_color="#e74c3c",
        annotation_text=f"Baseline {overall_baseline:.0%}",
        annotation_position="bottom right",
    )
    fig.update_yaxes(tickformat=".0%", title="Predicted Delay Probability", range=[0, 1])
    fig.update_xaxes(title="Date")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(_plotly_defaults(fig, height=360), width="stretch")

    st.divider()

    # ── Top 20 highest-risk shipments ──────────────────────────────────────────
    st.subheader("Top 20 Highest-Risk Shipments")

    top20 = (
        dff.sort_values("pred_delay_proba", ascending=False)
        .head(20)[["shipment_id", "date", "carrier", "origin", "destination",
                   "sku_category", "promised_lead_days",
                   "pred_delay_proba", "pred_is_delayed"]]
        .reset_index(drop=True)
    )
    top20.index += 1
    top20["pred_delay_proba"] = top20["pred_delay_proba"].map("{:.1%}".format)
    top20["pred_is_delayed"]  = top20["pred_is_delayed"].map({1: "🔴 Delayed", 0: "✅ On-time"})
    top20["date"] = top20["date"].dt.date
    st.dataframe(top20, width="stretch")


# ── Page 3 — Cost Variance Alerts ─────────────────────────────────────────────

def page_cost_variance_alerts() -> None:
    st.title("💸 Cost Variance Alerts")
    st.caption("Predicted freight billing overruns — flag shipments before the invoice arrives")

    df = load_scored()

    threshold_p75 = df["pred_cost_variance"].quantile(0.75)
    excess_exposure = df.loc[df["pred_cost_variance"] > 0, "pred_cost_variance"].sum()
    high_risk_count = int((df["pred_cost_variance"] >= threshold_p75).sum())

    # ── KPI row ────────────────────────────────────────────────────────────────
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("75th Pctile Threshold",    f"${threshold_p75:.2f}")
    kc2.metric("Excess Cost Exposure",     f"${excess_exposure:,.2f}",
               help="Sum of predicted cost variance where pred > 0")
    kc3.metric("High-Risk Shipments (≥p75)", f"{high_risk_count:,}")

    st.divider()

    left_col, right_col = st.columns([1, 1])

    # ── Histogram ──────────────────────────────────────────────────────────────
    with left_col:
        st.subheader("Distribution of Predicted Cost Variance")
        fig = px.histogram(
            df, x="pred_cost_variance",
            nbins=60,
            color_discrete_sequence=["#3498db"],
        )
        fig.add_vline(
            x=threshold_p75,
            line_dash="dash",
            line_color="#e74c3c",
            line_width=2,
            annotation_text=f"p75 = ${threshold_p75:.2f}",
            annotation_position="top right",
        )
        fig.add_vline(x=0, line_dash="dot", line_color="#888", line_width=1)
        fig.update_xaxes(title="Predicted Cost Variance (USD)")
        fig.update_yaxes(title="Shipment Count")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(_plotly_defaults(fig, height=350), width="stretch")

    # ── Scatter ────────────────────────────────────────────────────────────────
    with right_col:
        st.subheader("Freight Cost vs Predicted Variance by Carrier")
        fig = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="freight_cost_usd",
            y="pred_cost_variance",
            color="carrier",
            color_discrete_map=CARRIER_COLORS,
            opacity=0.55,
            hover_data=["shipment_id", "carrier", "date"],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        fig.add_hline(
            y=threshold_p75,
            line_dash="dot",
            line_color="#e74c3c",
            line_width=1.5,
            annotation_text="p75",
            annotation_position="top right",
        )
        fig.update_xaxes(title="Freight Cost (USD)", tickprefix="$")
        fig.update_yaxes(title="Predicted Cost Variance (USD)", tickprefix="$")
        st.plotly_chart(_plotly_defaults(fig, height=350), width="stretch")

    st.divider()

    # ── Top 20 overrun candidates ──────────────────────────────────────────────
    st.subheader("Top 20 Shipments — Highest Predicted Cost Overrun")

    top20 = (
        df.sort_values("pred_cost_variance", ascending=False)
        .head(20)[["shipment_id", "date", "carrier", "origin", "destination",
                   "weight_kg", "freight_cost_usd",
                   "pred_cost_variance", "pred_delay_proba"]]
        .reset_index(drop=True)
    )
    top20.index += 1
    top20["date"]             = top20["date"].dt.date
    top20["freight_cost_usd"] = top20["freight_cost_usd"].map("${:.2f}".format)
    top20["pred_cost_variance"]= top20["pred_cost_variance"].map("${:.3f}".format)
    top20["pred_delay_proba"] = top20["pred_delay_proba"].map("{:.1%}".format)
    st.dataframe(top20, width="stretch")


# ── Page 4 — Lane Intelligence ─────────────────────────────────────────────────

def page_lane_intelligence() -> None:
    st.title("🗺 Lane Intelligence")
    st.caption("Delay risk and cost profiles by shipping lane")

    lr = load_lane_risk()

    # ── Sidebar filter ─────────────────────────────────────────────────────────
    st.sidebar.subheader("Filters")
    direction_filter = st.sidebar.radio(
        "Direction",
        ["All", "inbound", "outbound"],
        key="lane_direction",
    )

    if direction_filter != "All":
        lr = lr[lr["direction"] == direction_filter]

    if lr.empty:
        st.warning("No lanes match the current filter.")
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Lanes Shown",       f"{len(lr):,}")
    kc2.metric("Avg Delay Rate",    f"{lr['delay_rate'].mean():.1%}")
    kc3.metric("Avg Cost Variance", f"${lr['avg_cost_variance'].mean():.2f}")

    st.divider()

    # ── Top 20 highest-delay-rate lanes ───────────────────────────────────────
    st.subheader("Top 20 Highest-Risk Lanes")

    top20_lanes = (
        lr.sort_values("delay_rate", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # Build label: "TX-TN (inbound)"
    top20_lanes["lane_label"] = (
        top20_lanes["lane_id"] + "  (" + top20_lanes["direction"] + ")"
    )

    fig = go.Figure(go.Bar(
        x=top20_lanes["delay_rate"],
        y=top20_lanes["lane_label"],
        orientation="h",
        marker_color=[RISK_COLORS[t] for t in top20_lanes["risk_tier"]],
        text=[f"{v:.0%}" for v in top20_lanes["delay_rate"]],
        textposition="outside",
    ))
    fig.update_xaxes(tickformat=".0%", title="Delay Rate", range=[0, 1.15])
    fig.update_yaxes(title="")
    fig.update_layout(
        title="Top 20 Lanes by Delay Rate",
        yaxis=dict(autorange="reversed"),
    )

    # Legend patches via invisible scatter traces
    for tier, color in RISK_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=f"{tier} Risk",
        ))

    st.plotly_chart(_plotly_defaults(fig, height=520), width="stretch")

    st.divider()

    # ── Summary table ──────────────────────────────────────────────────────────
    st.subheader("Lane Summary Table")

    display_cols = ["lane_id", "origin", "destination", "direction",
                    "delay_rate", "avg_delay_days", "avg_cost_variance", "risk_tier"]
    tbl = (
        lr[display_cols]
        .sort_values("delay_rate", ascending=False)
        .reset_index(drop=True)
    )
    tbl.index += 1

    # Colour-code risk_tier text for display
    tbl_display = tbl.copy()
    tbl_display["delay_rate"]        = tbl_display["delay_rate"].map("{:.1%}".format)
    tbl_display["avg_delay_days"]    = tbl_display["avg_delay_days"].map("{:+.2f}d".format)
    tbl_display["avg_cost_variance"] = tbl_display["avg_cost_variance"].map("${:.2f}".format)

    st.dataframe(
        tbl_display,
        width="stretch",
        height=420,
        column_config={
            "risk_tier": st.column_config.TextColumn("Risk Tier"),
            "delay_rate": st.column_config.TextColumn("Delay Rate"),
        },
    )


# ── Sidebar navigation ─────────────────────────────────────────────────────────

st.sidebar.title("📦 Supply Chain")
st.sidebar.caption("Intelligence Dashboard")
st.sidebar.divider()

PAGES = {
    "📊  Carrier Risk Scorecard": page_carrier_scorecard,
    "⏱  Delay Forecast":         page_delay_forecast,
    "💸  Cost Variance Alerts":   page_cost_variance_alerts,
    "🗺  Lane Intelligence":       page_lane_intelligence,
}

page_name = st.sidebar.radio("Navigate to", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Data: 3,000 shipments | Jan 2023 – Dec 2024")

PAGES[page_name]()
