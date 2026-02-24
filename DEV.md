# Plexe FDE Take-Home — Development Execution Guide

**Summary:** Predict low-review risk (≤2 stars) per order using pre-shipment data → binary classification, time-based split, LightGBM/XGBoost, FastAPI + SHAP. Target: 4–6 hours; ship by 6h.

---

## Quick start

1. **Setup** → venv, install deps, download Olist data from Kaggle.
2. **Data** → Load CSVs, join tables, validate, build clean modeling dataframe.
3. **EDA** → Low-review %, delay vs rating, seller/category/region patterns (5–7 insights).
4. **Features** → Order, seller historical, product, logistics; encode; final dataset.
5. **Train & evaluate** → Time-based split, baseline + boosted model, metrics, error analysis, save best model.
6. **Serve** → FastAPI `POST /predict`, load model, SHAP top features, test with curl.
7. **Docker** → Slim image, port 8000, test locally.

---

## 1. Assignment Overview

You are simulating the role of a **Forward Deployed Engineer (FDE)**.

The customer says:

> "We're growing fast but our margins are getting squeezed. Sellers are complaining, buyers leave bad reviews, and we don't really know where to focus. We've heard ML can help but we don't know what to build first. Can you look at our data and tell us what would actually make a difference?"

They have not specified what to predict. Your job is to:

- Identify 1–2 high-impact ML problems
- Justify your choice using data exploration
- Build and evaluate a model properly
- Deploy it behind a REST API
- Provide AI tool usage logs

This is a **business framing + modeling judgment + production-readiness** evaluation.

---

## 2. Dataset Summary

- **Dataset:** Brazilian E-Commerce Public Dataset (Olist)
- **Size:** ~100,000 orders across 9 linked tables

| Table | Description |
|-------|-------------|
| `orders` | Order metadata |
| `order_items` | Line items per order |
| `order_payments` | Payment info |
| `customers` | Customer info |
| `sellers` | Seller info |
| `products` | Product details |
| `product_category` | Category names |
| `reviews` | 1–5 stars + free text |
| `geolocation` | Zip → lat/lng |

This is a **relational marketplace dataset**.

---

## 3. Selected ML Problem

### Primary problem: Predict Low Review Risk (≤ 2 Stars)

**Business motivation**

- Customer complaints: buyers leave bad reviews, sellers unhappy, margins shrinking.
- Low reviews likely correlate with: late delivery, seller quality issues, high freight costs, certain product categories, geographic delivery challenges.
- Predicting low-review risk enables: proactive customer support, seller quality monitoring, targeted interventions, operational cost reduction.

---

## 4. Final Problem Statement

Build a supervised ML model that:

- **Predicts** the probability that an order will receive a low review (≤ 2 stars).
- **Uses only** information available before or at shipment time.

This is a **binary classification** task.

---

## 5. Target Definition

```text
target = 1  if review_score <= 2
target = 0  if review_score >= 3
```

---

## 6. Train / Validation / Test Strategy

Use **time-based splitting** (NOT random split).

| Split | Period |
|-------|--------|
| Train | Orders before July 2018 |
| Validation | July–August 2018 |
| Test | September 2018+ |

**Reason:** Production systems predict future data. Random splitting causes temporal leakage.

---

## 7. Feature Engineering Plan

Only include features available **before** the review is written.

### Order-level

- `order_value`
- `freight_value`
- `number_of_items`
- `payment_type`
- `installment_count`

### Seller-level (historical only)

- `seller_avg_rating` (past orders only)
- `seller_order_volume`
- `seller_late_delivery_rate`
- `seller_tenure_days`

### Product-level

- `product_category`
- `product_weight`
- `product_price`

### Logistics

- `estimated_delivery_time`
- Distance between seller and customer
- Delivery delay (if predicting post-delivery)

---

## 8. Modeling Plan

| Role | Model |
|------|--------|
| **Baseline** | Logistic Regression |
| **Primary** | LightGBM or XGBoost |

Track experiments and compare:

- ROC-AUC
- PR-AUC
- Precision@Top-K

---

## 9. Evaluation Framework

### Primary metrics

- **PR-AUC** (for class imbalance)

### Secondary metrics

- ROC-AUC
- Precision@10%
- Recall@10%
- **Calibration** — check probability calibration

### Error analysis

Investigate:

- Performance by product category
- Performance by region
- Performance by seller segment
- Performance by order value bucket

Translate results into business value, e.g.:

> "If we intervene on the top 10% highest-risk orders, we capture 45% of all 1-star reviews."

---

## 10. API Specification

- **Framework:** FastAPI
- **Endpoint:** `POST /predict`

### Example input

```json
{
  "order_value": 120.5,
  "freight_value": 15.2,
  "num_items": 2,
  "payment_type": "credit_card",
  "seller_id": "abc123",
  "product_category": "electronics",
  "customer_state": "SP"
}
```

### Example output

```json
{
  "risk_probability": 0.72,
  "risk_level": "high",
  "top_features": [
    {"feature": "delivery_delay", "impact": 0.31},
    {"feature": "seller_avg_rating", "impact": 0.22}
  ]
}
```

Use **SHAP** for feature attribution.

---

## 11. Repository Structure

```text
plexe-fde-assignment/
├── README.md
├── DEV.md
├── requirements.txt
├── Dockerfile
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

## 12. Development Execution Plan (4–6 Hours)

### Phase 1 — Setup (30 min)

- Create virtual environment
- Install dependencies:

```text
pandas
numpy
scikit-learn
lightgbm
shap
fastapi
uvicorn
pydantic
```

### Phase 2 — Data Integration (45–60 min)

- Load all CSVs
- Join relational tables carefully
- Validate row counts
- Handle missing values
- Create clean modeling dataframe

### Phase 3 — EDA (1–1.5 hours)

Answer:

- % of low reviews
- Correlation between delay and rating
- Seller-level rating variance
- Category-level rating trends
- Region-level patterns
- Generate **5–7 clear insights**

### Phase 4 — Feature Pipeline (1–1.5 hours)

- Create reusable feature functions
- Build historical aggregates carefully
- Encode categorical variables
- Create final modeling dataset

### Phase 5 — Train & Evaluate (1–1.5 hours)

- Implement time-based split
- Train baseline model
- Train boosted model
- Compare metrics
- Perform error analysis
- Save best model

### Phase 6 — Serve Model (45–60 min)

- Implement FastAPI service
- Load trained model
- Add SHAP explanations
- Test endpoint
- Write sample curl command

### Phase 7 — Dockerize (30–45 min)

- Use slim Python image
- Install dependencies
- Copy project files
- Expose port 8000
- Test locally

---

## 13. README Must Include

- Problem framing
- Why this problem was chosen
- Alternatives considered
- Feature engineering decisions
- Evaluation methodology
- Model limitations
- What you'd improve with more time
- Deployment instructions

---

## 14. What This Assignment Is Actually Testing

This is **NOT** a Kaggle competition. It evaluates:

- Business reasoning
- Problem prioritization
- Model selection judgment
- Evaluation depth
- Production readiness
- Responsible AI usage

---

## 15. Stop Rule

**If you cross 6 hours: ship.**

Sharp thinking on one well-built solution beats over-engineering multiple half-finished models.
