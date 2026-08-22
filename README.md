# Flipkart Support Capstone

## Project Overview

This repository contains the three-part Flipkart Support Capstone:

- Part 1 — Return-risk prediction
- Part 2 — Product-category image classification
- Part 3 — Retrieval-augmented support agent using LangGraph

The three parts form one connected system: Part 1 saves the return-risk model, Part 2 saves the product-image classifier, and Part 3 loads both saved artifacts as real callable tools while answering grounded policy questions from a local knowledge base.

---

# Part 1 — Return Risk

## Main files

- `generate_orders.py`
- `orders_dataset.csv`
- `train_return_risk.py`
- `models/return_risk_model.pkl`

## Dataset verification

- Rows: `6000`
- Columns: `13`
- Overall return rate: `0.2275`
- Missing `rating_given` rate: `0.1305`

### Return rate by product category

| Product category | Return rate |
|---|---:|
| Apparel | 0.2643 |
| Beauty | 0.2003 |
| Electronics | 0.1869 |
| Footwear | 0.2596 |
| Home | 0.1915 |

### Return rate by payment method

| Payment method | Return rate |
|---|---:|
| COD | 0.3075 |
| Prepaid_Card | 0.1682 |
| Prepaid_UPI | 0.1692 |
| Wallet | 0.1785 |

## Missingness analysis

The `rating_given` missingness is classified as **MAR** because the missingness depends on the observed `payment_method` column.

- COD missing-rate: `0.2283`
- Non-COD missing-rate: `0.0606`
- Missing-rate gap: `0.1677`

The dependency on the observed payment method means the pattern is not MCAR. It is not MNAR because the missingness mechanism is not defined by the unobserved rating value itself.

## Train/test split

- Training rows: `4800`
- Test rows: `1200`

## Baseline

- DummyClassifier accuracy: `0.7725`
- DummyClassifier F1 for `returned=1`: `0.0`

The high baseline accuracy is misleading because the classifier predicts the majority class and therefore has zero recall for returned orders.

## Logistic Regression

Default threshold:

`0.50`

Metrics:

- Accuracy: `0.5917`
- F1: `0.3921`
- Recall: `0.5788`
- Precision: `0.2964`
- ROC-AUC: `0.6253`

### Threshold sweep

Best Logistic Regression threshold:

`t* = 0.44`

At this threshold:

- F1: `0.4091`
- Recall: `0.7582`
- Precision: `0.2801`

Lowering the threshold makes the system more willing to flag possible returns, increasing recall while accepting more false positives and therefore reducing precision.

## Random Forest

Grid-search space included:

- `n_estimators`: `[100, 200]`
- `max_depth`: `[6, 10, None]`
- Scoring: `roc_auc`
- Cross-validation: 5-fold `StratifiedKFold`

Best parameters:

```text
classifier__max_depth = 6
classifier__n_estimators = 100
