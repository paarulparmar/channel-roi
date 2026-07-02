# Marketing Channel ROI Prediction & Budget Optimization

An end-to-end machine learning system that predicts whether a newly acquired
customer will become **high-value** and uses those predictions to **reallocate
marketing budget across acquisition channels to maximize ROI**.

Built for Product, Marketing, Growth, Finance, and Business Analytics use cases.
It pairs rigorous ML methodology (leakage-safe pipelines, imbalance handling,
cross-validated model selection, SHAP explainability) with a decision layer that
turns model output into a concrete, dollar-denominated budget recommendation and
a before/after ROI simulation.

> **The pipeline runs on real data by default.** The primary dataset is Olist's
> publicly released, anonymized commercial data (Marketing Funnel + Brazilian
> E-Commerce). A fully synthetic generator is retained as a high-volume fallback.

---

## 1. Data: Real by Default

### Primary source — Olist (REAL, anonymized commercial data)
The unit of analysis is an **acquired seller**: a marketing-qualified lead (MQL)
that converted to a closed deal. This is the entity Olist actually spends to
acquire through each channel, and it links cleanly to downstream revenue.

Pipeline: **MQL** (`origin` = acquisition channel) → **closed deal**
(`seller_id`, `won_date`, business attributes) → **order items** (revenue) →
**orders** (timestamps for behavioural features).

| Project field | Source | Real? |
|---|---|---|
| `acquisition_channel` | MQL `origin` (organic_search, paid_search, social, email, referral, direct, display, …) | ✅ real |
| `signup_date` | deal `won_date` | ✅ real |
| `business_type` | seller business type (reseller/manufacturer) | ✅ real |
| `first_order_value` | value of seller's first order | ✅ real |
| `days_to_second_purchase` | gap between 1st and 2nd order | ✅ real |
| `total_revenue_generated` | summed order-item revenue | ✅ real |
| `is_high_value` (target) | top-tier revenue (data-driven threshold) | ✅ real |
| `acquisition_cost` | **per-channel CAC benchmark overlay** | ⚠️ assumption |

Everything is real Olist data **except `acquisition_cost`** — Olist never
published per-lead ad spend, so a per-channel CAC benchmark is overlaid (earned
channels cheap, paid channels expensive) and clearly labelled as an assumption.
Leads that closed but generated **no** orders are kept as the genuine
low-value / churn-risk class (revenue = 0) — exactly the outcome the model
exists to predict.

Result: **842 acquired sellers**, **9 real channels**, **~28% high-value**
(realistic class imbalance). The Olist CSVs are cached under `data/olist_raw/`
and re-downloaded automatically if absent.

### Why not a single ready-made table? (options considered)
There is no public dataset that already contains *acquisition channel + cost +
early behaviour + lifetime revenue + a high-value label* per customer — that
combination is a company's core unit economics and is rarely released. Options
evaluated:

| Option | Channel? | Revenue? | Verdict |
|---|---|---|---|
| **Olist Marketing Funnel + E-Commerce** | ✅ real `origin` | ✅ real | **Chosen** — richest real channel→value link |
| Kaggle *Customer Acquisition Data* | ✅ | ✅ | Real but tiny (~500 rows), few features |
| Google Analytics Merchandise Store | ✅ real channelGrouping | ✅ | Great fit but BigQuery/GCP-gated |
| UCI Online Retail II / CDNow | ❌ | ✅ | Real behaviour, **no channel or CAC** |
| Telco Customer Churn | ❌ | ~ | No acquisition channel |

Olist wins because the acquisition channel and the downstream revenue are both
genuinely present and joinable. CAC is the one field no public source exposes,
so it is overlaid from benchmarks in every option.

### Synthetic fallback
`src/data_generation.py` produces a 4,200-row simulated dataset with
channel-specific CAC/quality and non-linear behavioural interactions. Switch via
`DATA_SOURCE = "synthetic"` in `src/config.py` when a larger volume is needed for
modelling demonstrations.

---

## 2. Business Problem

A growth team spends a fixed monthly budget acquiring customers across several
channels that differ sharply in **cost** (CAC) and **quality** (how often they
produce loyal, high-revenue customers). Spending equally — or spending most where
volume is cheapest — leaves money on the table.

**Goal:** score each customer's probability of becoming high-value from
acquisition-time and early-behaviour signals, roll the scores up to the channel
level, and shift budget toward channels with the best quality-adjusted return.

---

## 3. Methodology

- **EDA (`src/eda.py`)** — per-channel unit economics: CAC, average revenue,
  ROI = (revenue − spend) / spend, and high-value rate.
- **Feature engineering (`src/preprocessing.py`)** — derived acquisition-time
  features (`value_per_cost`, `fast_repeat_flag`, `signup_month`,
  `signup_dayofweek`); numeric standardized + categoricals one-hot encoded in a
  scikit-learn `ColumnTransformer` applied identically at train/test/score time.
  `total_revenue_generated` is **excluded** from features to prevent label
  leakage.
- **Modelling (`src/modeling.py`)** — Logistic Regression, Random Forest, and
  XGBoost trained in the shared pipeline, compared by **5-fold stratified
  cross-validated ROC-AUC**, with class imbalance handled explicitly
  (`class_weight="balanced"` / `scale_pos_weight`). Best model chosen on held-out
  test ROC-AUC; explainability via native importance and a **SHAP** summary.
- **Business layer (`src/business_layer.py`)** — score every customer, build a
  channel scorecard, run a guard-railed budget reallocation proportional to each
  channel's efficiency (expected revenue per acquisition dollar), and simulate
  customers / revenue / ROI before vs after.

---

## 4. Results (real Olist data)

### Model comparison (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (best)** | 0.896 | 0.744 | 0.967 | 0.841 | **0.952** |
| Logistic Regression | 0.901 | 0.741 | 1.000 | 0.851 | 0.951 |
| XGBoost | 0.886 | 0.773 | 0.850 | 0.810 | 0.951 |

Early order behaviour (first-order value, repeat speed, value-per-cost) is
strongly predictive of a seller's eventual value, so all three models score
highly; Random Forest edges out on test ROC-AUC. Recall is favoured — missing a
future high-value account is costlier than a false positive.

### Channel scorecard

| Channel | CAC | ROI | Predicted HV % | Recommendation |
|---|---|---|---|---|
| Email | $14.66 | 37.6x | 27% | **Scale Up** |
| Referral | $17.91 | 40.6x | 33% | **Scale Up** |
| Direct | $21.53 | 17.2x | 39% | **Scale Up** |
| Organic Search | $25.29 | 29.2x | 32% | Maintain |
| Unknown | $39.55 | 29.2x | 35% | Maintain |
| Other | $44.52 | 7.7x | 23% | Maintain |
| Social | $54.60 | 9.6x | 30% | **Reduce / Optimize** |
| Paid Search | $69.30 | 10.3x | 38% | **Reduce / Optimize** |
| Display | $59.43 | 1.6x | 23% | **Reduce / Optimize** |

### Budget optimization impact (simulated, $250K budget)

| Scenario | Customers acquired | Expected revenue | ROI |
|---|---|---|---|
| Before (current mix) | ~6,100 | ~$1.37M | 4.5x |
| After (optimized) | ~10,900 | ~$2.38M | 8.5x |
| **Lift** | **+80%** | **+74%** | **+90%** |

> Shifting spend away from high-CAC Paid Search / Social / Display toward
> low-CAC Email, Referral, Direct and Organic Search **~1.9x's expected revenue**
> on the same total budget.

*(Figures are reproducible from the fixed seed; exact values shift slightly if
the CAC overlay or threshold is changed in `config.py`.)*

---

## 5. Business Recommendations

1. **Scale Up Email, Referral & Direct** — lowest CAC with solid predicted
   quality give the best return per dollar; primary destinations for reallocated
   budget.
2. **Protect Organic Search** — largest volume channel at healthy ROI; maintain
   investment and SEO.
3. **Reduce / Optimize Paid Search, Social & Display** — high CAC drags ROI even
   where predicted quality is decent (Paid Search). Restructure targeting/creative
   or reallocate before scaling.
4. **Investigate "Unknown"** — meaningful volume and revenue with untracked
   attribution; fixing tracking would sharpen the whole allocation.
5. **Re-score each cycle** — quality signals drift; re-run scoring and let the
   scorecard drive the next allocation.

---

## 6. Project Structure

```
channel_roi_ml/
├── main.py                     # Run the entire pipeline
├── requirements.txt
├── README.md
├── data/
│   ├── customers.csv           # Built dataset (project schema)
│   └── olist_raw/              # Cached real Olist CSVs
├── models/
│   └── best_model.joblib       # Trained best model (full pipeline)
├── outputs/
│   ├── model_metrics.csv
│   ├── business_scorecard.csv
│   ├── budget_optimization.csv
│   ├── scored_customers.csv
│   └── 01..12_*.png            # All visualizations
└── src/
    ├── config.py               # Central config + DATA_SOURCE switch
    ├── real_data.py            # REAL Olist loader (default)
    ├── data_generation.py      # Synthetic fallback generator
    ├── eda.py                  # Channel unit economics
    ├── preprocessing.py        # Feature engineering + ColumnTransformer
    ├── modeling.py             # Train / compare / evaluate / select
    ├── business_layer.py       # Scoring, scorecard, budget optimization
    └── visualization.py        # All figures
```

---

## 7. How to Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Runs on real Olist data by default (auto-downloads/caches the CSVs, or uses the
bundled cache under `data/olist_raw/`). To use the synthetic fallback, set
`DATA_SOURCE = "synthetic"` in `src/config.py`. To rebuild from scratch, delete
`data/customers.csv`.

The persisted `models/best_model.joblib` is the full pipeline (preprocessing +
estimator), so it scores raw customer rows directly.

---

## 8. Visualizations (`outputs/`)

CAC by channel · average revenue by channel · ROI by channel · high-value rate by
channel · confusion matrix · ROC curves · model comparison · feature importance ·
SHAP summary · predicted vs actual high-value rate · budget before vs after ·
ROI/revenue simulation.

---

## 9. Methodological Notes

- **No label leakage:** `total_revenue_generated` is never a model feature.
- **Imbalance handled** at the estimator level, not by resampling the test set.
- **Reproducible:** a single `RANDOM_STATE` seeds the build, splits, and models.
- **Deployment-shaped:** the saved pipeline scores raw rows end-to-end.
- **Honest about the one overlay:** channel CAC is a documented benchmark; all
  channel, behaviour, and revenue data is real Olist data.
- **Fully open-source:** numpy, pandas, scikit-learn, XGBoost, SHAP, matplotlib,
  seaborn.

---

## 10. Extending

- Swap the CAC overlay for your own channel spend to make ROI fully real.
- Point `real_data.py` at Google Analytics exports for customer-side channels.
- Add `GridSearchCV` hyperparameter search around the estimators.
- Replace the proportional allocator with a constrained optimizer under
  diminishing returns (`scipy.optimize`).
- Serve `best_model.joblib` behind a FastAPI endpoint for real-time scoring.

---

*Data credit: Olist (Brazilian E-Commerce Public Dataset & Marketing Funnel),
released publicly and anonymized. `acquisition_cost` is an illustrative benchmark
overlay, not Olist's actual spend.*
