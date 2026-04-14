# Supply Chain Forecasting

**End-to-end data science portfolio project** — predictive models and an operations
dashboard for a simulated freight network of seven carriers across 60+ lanes.

---

## Business Case

For operations teams managing a carrier network of four to ten partners, late
deliveries, invoice surprises, and damage incidents represent the three most
controllable sources of cost and customer-satisfaction risk. Yet most logistics
teams fly blind: on-time performance is reviewed monthly in spreadsheets, freight
invoices are reconciled after the fact, and high-risk lanes are identified only
after problems repeat. This project demonstrates how a purpose-built machine
learning stack can shift each of those three problems from reactive to predictive —
flagging likely delays before a shipment departs, surfacing cost overruns before
the invoice arrives, and ranking every lane by its empirical risk tier — giving a
VP of Operations the forward-looking visibility needed to negotiate carrier SLAs,
pre-position buffer stock, and reduce unplanned freight spend.

---

## Architecture

```
Raw Data                  Feature Engineering           Models
────────────────          ──────────────────────        ──────────────────────────────
data/                     src/features.py               src/models.py
  shipments.csv    ──▶    FeatureEngineer               DelayClassifier
  carrier_                  • carrier rolling 30d  ──▶    XGBoost · AUC 0.767
    scorecard.csv           • lane risk merge             F1 0.865
  lane_risk.csv             • temporal flags       ──▶  CostVarianceRegressor
                            • weight/cost ratios          XGBoost · RMSE $0.360
                            • delay ratio          ──▶  LeadTimeForecaster
                                                          Prophet · 30-day / carrier
                                  │
                                  ▼
                        data/shipments_featured.csv
                        data/scored_shipments_full.csv
                                  │
                                  ▼
                        Streamlit Dashboard
                        ──────────────────────────────
                        app/streamlit_app.py
                          Page 1 · Carrier Risk Scorecard
                          Page 2 · Delay Forecast
                          Page 3 · Cost Variance Alerts
                          Page 4 · Lane Intelligence
```

---

## Models

| Model | Type | Target | Key Metric | Top Predictive Feature |
|---|---|---|---|---|
| `DelayClassifier` | Binary classification | `is_delayed` | AUC **0.767** · F1 **0.865** | `lane_delay_rate` |
| `CostVarianceRegressor` | Regression | `cost_variance_usd` | RMSE **$0.360** · R² **0.463** | `carrier_enc` |
| `LeadTimeForecaster` | Time-series (Prophet) | Daily avg lead time | 30-day horizon per carrier | Trend + yearly/weekly seasonality |

Both supervised models use only **pre-delivery features** (carrier history, lane
risk, shipment characteristics, temporal signals) — no information from the
actual delivery is included, ensuring the predictions are actionable at dispatch
time.

---

## Dataset (Simulated)

| Table | Rows | Description |
|---|---|---|
| `shipments.csv` | 3,000 | Individual shipments — Jan 2023 to Dec 2024 |
| `carrier_scorecard.csv` | 168 | Monthly KPIs per carrier (24 months × 7 carriers) |
| `lane_risk.csv` | 281 | Risk tier per origin-destination-direction lane |
| `shipments_featured.csv` | 3,000 | Shipments + 15 engineered features |
| `scored_shipments_full.csv` | 3,000 | Features + model predictions |

Injected patterns: FastFreight best performer (50 % on-time vs 14 % for
DirectMove), Nov/Dec seasonal delay spike (+1.9 d → +2.7 d above baseline),
inbound lanes 40 % more delayed than outbound, 84 high-risk lanes out of 281.

---

## Project Structure

```
supply-chain-forecasting/
├── data/                     # CSVs (generated) + saved figures
│   ├── generate_data.py      # Synthetic dataset generator
│   ├── shipments.csv
│   ├── carrier_scorecard.csv
│   ├── lane_risk.csv
│   ├── shipments_featured.csv
│   └── scored_shipments_full.csv
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb         # All three models
├── src/
│   ├── features.py           # FeatureEngineer class
│   └── models.py             # DelayClassifier, CostVarianceRegressor,
│                             #   LeadTimeForecaster
├── app/
│   ├── streamlit_app.py      # 4-page dashboard
│   └── test_app.py           # AppTest headless test
├── docs/
│   └── project_brief.md
└── requirements.txt
```

---

## Setup & Run

### 1. Clone and create environment

```bash
git clone https://github.com/oscarmndz89/supply-chain-forecasting.git
cd supply-chain-forecasting

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python data/generate_data.py
```

### 4. (Optional) Run the notebooks in order

```bash
jupyter lab
# open notebooks/01_eda.ipynb → 02_feature_engineering.ipynb → 03_modeling.ipynb
```

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## Key Results

| Finding | Metric |
|---|---|
| Delay classifier test AUC | **0.767** |
| Delay classifier F1 score | **0.865** |
| Cost variance RMSE | **$0.360** |
| FastFreight vs DirectMove on-time gap | **50.4 % vs 13.8 %** |
| Nov/Dec delay spike vs baseline | **+1.9 d → +2.7 d** |
| High-risk lanes identified | **84 of 281 (30 %)** |
| Predicted excess cost exposure | **$975** across scored shipments |
| Lead-time forecast horizon | **30 days per carrier** |

---

## Production Roadmap

- **ERP / TMS integration** — connect to SAP, Oracle TMS, or a 3PL API to replace
  the simulated CSV pipeline with live shipment data; retrain on a rolling 90-day
  window rather than a static dataset.

- **Automated retraining pipeline** — schedule weekly retrains via Airflow or
  Prefect; log model versions and feature drift to MLflow; auto-rollback if
  test AUC drops below threshold.

- **Proactive alerting system** — push Slack / email alerts when `pred_delay_proba`
  exceeds a configurable threshold for in-transit shipments, giving ops teams 24–48 h
  to arrange contingency stock or expedited re-routing.

- **Carrier SLA tracking module** — aggregate model predictions and actuals into a
  monthly scorecard with contractual penalty triggers; surface carriers trending
  toward SLA breach before the month closes, enabling proactive renegotiation.

---

## Stack

Python 3.11 · XGBoost · Prophet · scikit-learn · SHAP · Streamlit · Plotly ·
pandas · NumPy · Jupyter
