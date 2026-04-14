# Project Brief — Supply Chain Forecasting

**One-page summary for hiring managers and operations directors**

---

## The Problem

Logistics teams managing multi-carrier freight networks consistently face three
costly blind spots: shipments that arrive late without warning, freight invoices
that exceed estimates by the time they're reconciled, and recurring damage
incidents on specific lanes. Each problem is discovered after the fact, when the
cost has already landed. Reactive management of these issues ties up analyst time,
erodes carrier relationships, and makes it nearly impossible to hold partners
accountable to agreed service levels.

---

## The Solution

This project builds a full data science pipeline — from raw shipment records to a
live operations dashboard — that converts each reactive blind spot into a
forward-looking prediction.

Three models are trained on two years of shipment history across seven carriers
and 281 origin-destination lanes:

- A **delay classifier** that predicts whether a shipment will arrive late, using
  only information available at dispatch time (carrier performance history, lane
  risk, seasonality, and shipment characteristics).

- A **cost variance model** that estimates how much a shipment's final invoice will
  exceed the freight estimate before the invoice arrives, enabling pre-authorisation
  reviews and supplier dispute prioritisation.

- A **lead-time forecaster** that projects each carrier's average delivery time
  over the next 30 days, providing a forward-looking input for inventory and
  safety-stock planning.

All three models feed a four-page Streamlit dashboard that an operations team can
use without any data science background.

---

## Results

| Outcome | Impact |
|---|---|
| Delay prediction accuracy (AUC) | **0.767** — correctly ranks at-risk shipments in 77 % of cases |
| F1 score on held-out test set | **0.865** — strong balance of precision and recall |
| Cost variance prediction error | **$0.36 RMSE** — flags overrun candidates within cents |
| Carrier performance gap surfaced | Best carrier (FastFreight) runs **3.6× the on-time rate** of the worst (DirectMove) |
| High-risk lanes identified | **84 of 281 active lanes** flagged for proactive carrier assignment |
| Seasonal planning window | Nov/Dec delays average **+2.7 days** above baseline — quantified for inventory planning |

---

## What This Demonstrates

- **End-to-end ML ownership** — data generation, feature engineering, model
  training with cross-validated hyperparameter search, SHAP explainability, and
  deployment as an interactive dashboard, all in a single coherent codebase.

- **Business-first modelling** — every design choice (pre-delivery features only,
  time-based train/test split, actionable SHAP summaries) is grounded in how the
  predictions would actually be used operationally.

- **Production thinking** — the architecture is structured for real extension: the
  `FeatureEngineer` and model classes are importable libraries, not notebook
  scripts; data loading is cached; and a written roadmap addresses ERP integration,
  automated retraining, alerting, and SLA tracking.

- **Communication** — the dashboard is designed for an operations audience, not a
  data science one: plain language labels, colour-coded risk tiers, and metric
  cards with deltas against baselines.
