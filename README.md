# Low-Review Risk Prediction — Plexe FDE Take-Home

Binary classification model that predicts the probability an order will receive a low review (≤2 stars) using only information available before or at shipment time. Built on the Brazilian E-Commerce (Olist) dataset and served via FastAPI with SHAP-based explanations.

---

## Problem framing

The customer (a mid-size online marketplace) reported: *"We're growing fast but our margins are getting squeezed. Sellers are complaining, buyers leave bad reviews, and we don't really know where to focus. We've heard ML can help but we don't know what to build first."*

They did not specify what to predict. The task was to explore the data, choose 1–2 high-impact ML problems, justify the choice, build and evaluate a model, and deploy it behind a REST API.

---

## Why this problem was chosen

**Primary problem: Predict Low Review Risk (≤ 2 stars) per order.**

- **Customer pain:** Bad reviews drive churn, seller dissatisfaction, and support cost; they also correlate with delivery issues, seller quality, and category/geography.
- **Actionability:** Predicting which orders are at high risk of a low review enables proactive outreach, seller quality monitoring, and targeted interventions (e.g. prioritising high-risk orders for support or compensation).
- **Data fit:** The Olist dataset has orders, items, payments, customers, sellers, products, reviews (1–5 stars), and geolocation, so we can define a target from `review_score` and use only pre-shipment (or at-shipment) features to avoid leakage.

The target is **binary**: `target = 1` if `review_score ≤ 2`, else `target = 0`. Only orders with a review are included.

---

## Alternatives considered

- **Predicting review score (regression/multiclass):** Less directly tied to “bad experience” and harder to act on than a binary “high risk / low risk” flag.
- **Predicting delivery delay or churn:** Valuable but secondary; low reviews are the stated pain and are observable in the data.
- **Multiple models (e.g. separate models per category):** Rejected for scope; one well-built order-level model with error analysis by segment is preferred.

---

## Feature engineering decisions

Only features **available before or at shipment time** are used (no post-delivery or post-review information).

| Group | Features |
|-------|----------|
| **Order-level** | `order_value`, `freight_value`, `num_items`, `payment_type`, `payment_installments` |
| **Seller-level (historical)** | `seller_avg_rating`, `seller_order_volume`, `seller_late_delivery_rate`, `seller_tenure_days` (all from past orders only) |
| **Product-level** | `product_category`, `product_weight`, product price (from order items) |
| **Logistics** | Estimated delivery time, distance (seller–customer), delivery delay when available at prediction time |

Categoricals (e.g. `payment_type`, `product_category`) are encoded (one-hot or target/label encoding) before training. Historical seller aggregates are computed with a strict time-based cutoff to avoid leakage.

---

## Evaluation methodology

- **Split:** Time-based (train: before July 2018; validation: July–August 2018; test: September 2018+). No random split, to reflect production use on future orders.
- **Primary metric:** **PR-AUC** (appropriate for imbalanced “low review” class).
- **Secondary metrics:** ROC-AUC, Precision@10%, Recall@10%, and probability calibration.
- **Error analysis:** Performance by product category, region, seller segment, and order value bucket; results translated into business language (e.g. “Intervening on top 10% highest-risk orders captures X% of all 1–2 star reviews”).
- **Models compared:** Baseline = Logistic Regression; primary = LightGBM or XGBoost; experiments logged in `experiments/`.

---

## Model limitations

- **Cold start:** New sellers or products have little or no history; predictions may be less reliable.
- **Concept drift:** Marketplace and logistics change over time; the model should be retrained and monitored periodically.
- **Causality:** The model is associative (risk prediction), not causal; interventions should be validated via experiments.
- **Scope:** Single order-level model; no explicit multi-item or sequence modeling within an order.

---

## What I'd improve with more time

- Richer seller/product embeddings or segment-specific models for cold start.
- Automated retraining and validation (e.g. on a schedule or on data drift).
- A/B test framework to measure impact of acting on predicted risk.
- Full calibration plots and threshold selection tied to business cost/benefit.
- CI/CD and tests around the API and feature pipeline.

---

## Setup and deployment

### Prerequisites

- Python 3.10+
- [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) downloaded and extracted

### 1. Clone and install

```bash
git clone <repo-url>
cd FDE-PlexeAI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data

Place the 9 Olist CSV files in `data/` (see `data/README.md` for the exact list). Then build the modeling dataframe:

```bash
python -m src.data_loader
```

Output: `data/modeling.parquet` (one row per order with target and joined attributes). Optional: set `OLIST_DATA_DIR` and `OLIST_MODELING_PATH` if your CSVs live elsewhere or you want a different output path.

### 3. EDA and training

- Run `notebooks/01_eda.ipynb` for exploration and to justify the problem.
- Run `notebooks/02_modeling.ipynb` (or `src/train.py`) to train with a time-based split; the best model and metrics are written to `experiments/`.

### 4. Run the API locally

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/predict` with a JSON body (see API spec below). The response includes `risk_probability`, `risk_level`, and `top_features` (SHAP-based).

**Example curl (once `/predict` is implemented):**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"order_value": 120.5, "freight_value": 15.2, "num_items": 2, "payment_type": "credit_card", "seller_id": "abc123", "product_category": "electronics", "customer_state": "SP"}'
```

### 5. Docker

```bash
docker build -t low-review-risk-api .
docker run -p 8000:8000 low-review-risk-api
```

The container serves the FastAPI app on port 8000. Mount a volume if you need to load a specific model artifact from the host.

---

## Repository structure

```text
├── README.md
├── DEV.md
├── requirements.txt
├── Dockerfile
├── data/
│   ├── README.md          # Where to put Olist CSVs
│   └── modeling.parquet    # Built by src.data_loader
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── serve.py
│   └── schemas.py
├── experiments/
│   ├── results.csv
│   └── metrics.json
└── ai_chat_logs/
    └── chatgpt_export.md
```

---

## API specification

- **Endpoint:** `POST /predict`
- **Response:** `risk_probability` (float), `risk_level` (e.g. `"high"`), `top_features` (list of `{feature, impact}` from SHAP).

See `DEV.md` for full request/response examples and feature list.
