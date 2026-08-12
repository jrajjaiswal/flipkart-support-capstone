import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("orders_dataset.csv")


# ============================================================
# DATASET VERIFICATION
# ============================================================

print("DATASET VERIFICATION")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Overall return rate:", round(df["returned"].mean(), 4))
print("Missing rating rate:", round(df["rating_given"].isna().mean(), 4))


print("\nRETURN RATE BY PRODUCT CATEGORY")
print(df.groupby("product_category")["returned"].mean().round(4))


print("\nRETURN RATE BY PAYMENT METHOD")
print(df.groupby("payment_method")["returned"].mean().round(4))


# ============================================================
# MISSINGNESS ANALYSIS
# ============================================================

cod_missing_rate = df.loc[
    df["payment_method"] == "COD", "rating_given"
].isna().mean()

non_cod_missing_rate = df.loc[
    df["payment_method"] != "COD", "rating_given"
].isna().mean()

missing_rate_gap = cod_missing_rate - non_cod_missing_rate

print("\nMISSINGNESS ANALYSIS")
print("COD missing rating rate:", round(cod_missing_rate, 4))
print("Non-COD missing rating rate:", round(non_cod_missing_rate, 4))
print("Missing-rate gap:", round(missing_rate_gap, 4))
print(
    "Missingness classification: MAR, because missingness depends "
    "on the observed payment_method column."
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

target = "returned"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nSPLIT SIZES")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
]

categorical_features = [
    "product_category",
    "payment_method"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# BASELINE MODEL
# ============================================================

print("\nBASELINE MODEL")

dummy_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(
        strategy="most_frequent",
        random_state=42
    ))
])

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_test)

dummy_accuracy = accuracy_score(
    y_test,
    dummy_predictions
)

dummy_f1 = f1_score(
    y_test,
    dummy_predictions,
    pos_label=1,
    zero_division=0
)

print("Dummy accuracy:", round(dummy_accuracy, 4))
print("Dummy F1 for returned=1:", round(dummy_f1, 4))
print(
    "The baseline can have high accuracy but zero recall "
    "because it predicts the majority class."
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nLOGISTIC REGRESSION")

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

logistic_default_predictions = (
    logistic_probabilities >= 0.5
).astype(int)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_default_predictions
)

logistic_f1 = f1_score(
    y_test,
    logistic_default_predictions,
    pos_label=1,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_default_predictions,
    pos_label=1,
    zero_division=0
)

logistic_precision = precision_score(
    y_test,
    logistic_default_predictions,
    pos_label=1,
    zero_division=0
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print("Default threshold: 0.50")
print("Accuracy:", round(logistic_accuracy, 4))
print("F1:", round(logistic_f1, 4))
print("Recall:", round(logistic_recall, 4))
print("Precision:", round(logistic_precision, 4))
print("ROC-AUC:", round(logistic_auc, 4))


# ============================================================
# LOGISTIC REGRESSION THRESHOLD SWEEP
# ============================================================

thresholds = np.arange(0.10, 0.901, 0.01)

threshold_results = []

for threshold in thresholds:

    predictions = (
        logistic_probabilities >= threshold
    ).astype(int)

    threshold_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    threshold_precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "f1": threshold_f1,
        "recall": threshold_recall,
        "precision": threshold_precision
    })

threshold_table = pd.DataFrame(threshold_results)

best_logistic_row = threshold_table.loc[
    threshold_table["f1"].idxmax()
]

best_logistic_threshold = float(
    best_logistic_row["threshold"]
)

best_logistic_f1 = float(
    best_logistic_row["f1"]
)

best_logistic_recall = float(
    best_logistic_row["recall"]
)

best_logistic_precision = float(
    best_logistic_row["precision"]
)

print("\nLOGISTIC REGRESSION THRESHOLD SWEEP")
print(threshold_table.to_string(index=False))

print(
    "\nBest Logistic threshold:",
    round(best_logistic_threshold, 2)
)

print(
    "Best Logistic F1:",
    round(best_logistic_f1, 4)
)

print(
    "Recall at best threshold:",
    round(best_logistic_recall, 4)
)

print(
    "Precision at best threshold:",
    round(best_logistic_precision, 4)
)

logistic_precision_drop = logistic_precision - best_logistic_precision

print(
    "Precision drop from default threshold:",
    round(logistic_precision_drop, 4)
)

print(
    "Business trade-off: lowering the threshold makes it easier "
    "to flag possible returns, increasing recall while accepting "
    "more false positives and therefore lower precision."
)


# ============================================================
# RANDOM FOREST GRID SEARCH
# ============================================================

print("\nRANDOM FOREST GRID SEARCH")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

rf_probabilities = best_rf_model.predict_proba(X_test)[:, 1]

rf_test_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print(
    "Best parameters:",
    grid_search.best_params_
)

print(
    "Best cross-validated ROC-AUC:",
    round(grid_search.best_score_, 4)
)

print(
    "Test ROC-AUC:",
    round(rf_test_auc, 4)
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\nRANDOM FOREST FEATURE IMPORTANCE")

fitted_preprocessor = (
    best_rf_model.named_steps["preprocessor"]
)

fitted_rf = (
    best_rf_model.named_steps["classifier"]
)

feature_names = (
    fitted_preprocessor.get_feature_names_out()
)

importances = fitted_rf.feature_importances_

importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(
    "importance",
    ascending=False
)

top_5 = importance_table.head(5).copy()

print("\nTop 5 features:")
print(top_5.to_string(index=False))


# ============================================================
# TOP 5 FEATURE INTERPRETATION
# ============================================================

print("\nTOP 5 FEATURE INTERPRETATION")

feature_interpretations = {
    "payment_method_COD": (
        "COD payment method is strongly associated with return risk "
        "in this dataset, so the model uses COD status as an important "
        "signal when estimating whether an order may be returned."
    ),
    "price_inr": (
        "Product price may affect return risk because higher-value "
        "purchases can involve different customer expectations "
        "and return behavior."
    ),
    "customer_tenure_days": (
        "Customer tenure can reflect familiarity and loyalty, "
        "which may be associated with different return behavior."
    ),
    "delivery_distance_km": (
        "Longer delivery distances may affect return risk through "
        "delivery experience and customer expectations."
    ),
    "discount_pct": (
        "Higher discounts may influence purchase decisions and "
        "customer expectations, which can affect return behavior."
    ),
    "num_previous_orders": (
        "Previous order history can indicate customer experience "
        "and purchasing behavior."
    ),
    "num_previous_returns": (
        "Previous returns can indicate a customer's historical "
        "tendency to return products."
    ),
    "delivery_days": (
        "Longer delivery times may increase dissatisfaction and "
        "therefore influence return risk."
    ),
    "rating_given": (
        "Customer rating information may provide a signal of "
        "satisfaction, although missing ratings are imputed."
    ),
    "product_category": (
        "Product category can affect return risk because different "
        "product types have different fit, expectation, and usage "
        "characteristics."
    )
}

for _, row in top_5.iterrows():

    feature = row["feature"]

    if "__" in feature:
        base_feature = feature.split(
            "__",
            1
        )[1]
    else:
        base_feature = feature

    interpretation = feature_interpretations.get(
        base_feature,
        "This feature contributes to the model's prediction "
        "of return risk."
    )

    print(
        f"- {feature}: {interpretation}"
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\nPERMUTATION IMPORTANCE")

X_test_transformed = (
    fitted_preprocessor.transform(X_test)
)

permutation_result = permutation_importance(
    fitted_rf,
    X_test_transformed,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_table = pd.DataFrame({
    "feature": feature_names,
    "permutation_importance":
        permutation_result.importances_mean
})

permutation_table = permutation_table[
    permutation_table["feature"].isin(
        top_5["feature"]
    )
].sort_values(
    "permutation_importance",
    ascending=False
)

comparison_table = top_5.merge(
    permutation_table,
    on="feature"
)

print(
    comparison_table.to_string(index=False)
)


# ============================================================
# PERMUTATION IMPORTANCE COMPARISON
# ============================================================

print("\nFEATURE IMPORTANCE COMPARISON")

for _, row in comparison_table.iterrows():

    feature = row["feature"]
    impurity = row["importance"]
    permutation = row["permutation_importance"]

    if permutation > 0:

        interpretation = (
            "Permutation importance is positive, so shuffling "
            "this feature reduced ROC-AUC and provides evidence "
            "that the feature contributes useful predictive "
            "information."
        )

    elif permutation < 0:

        interpretation = (
            "Permutation importance is negative, so shuffling "
            "this feature did not improve the model's held-out "
            "ROC-AUC; its impurity importance may therefore "
            "overstate its independent predictive contribution."
        )

    else:

        interpretation = (
            "Permutation importance is approximately zero, "
            "suggesting limited independent contribution to "
            "held-out ROC-AUC."
        )

    print(
        f"- {feature}: "
        f"impurity importance={impurity:.6f}, "
        f"permutation importance={permutation:.6f}. "
        f"{interpretation}"
    )

print(
    "\nComparison rationale: impurity-based feature importance "
    "measures how much tree splits reduce impurity during "
    "training, whereas permutation importance measures the "
    "change in held-out ROC-AUC after shuffling a feature. "
    "Therefore, a large impurity importance with a small or "
    "negative permutation importance can indicate that the "
    "feature was useful for training splits but has weaker "
    "independent predictive value on unseen data."
)


# ============================================================
# RANDOM FOREST THRESHOLD SWEEP
# ============================================================

print("\nRANDOM FOREST THRESHOLD SWEEP")

rf_threshold_results = []

for threshold in thresholds:

    predictions = (
        rf_probabilities >= threshold
    ).astype(int)

    threshold_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    threshold_precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    rf_threshold_results.append({
        "threshold": threshold,
        "f1": threshold_f1,
        "recall": threshold_recall,
        "precision": threshold_precision
    })

rf_threshold_table = pd.DataFrame(
    rf_threshold_results
)

best_rf_row = rf_threshold_table.loc[
    rf_threshold_table["f1"].idxmax()
]

t_rf = float(
    best_rf_row["threshold"]
)

t_rf_f1 = float(
    best_rf_row["f1"]
)

t_rf_recall = float(
    best_rf_row["recall"]
)

t_rf_precision = float(
    best_rf_row["precision"]
)

print(
    "t*_rf:",
    round(t_rf, 2)
)

print(
    "F1 at t*_rf:",
    round(t_rf_f1, 4)
)

print(
    "Recall at t*_rf:",
    round(t_rf_recall, 4)
)

print(
    "Precision at t*_rf:",
    round(t_rf_precision, 4)
)


# ============================================================
# TUNED RF PREDICTIONS
# ============================================================

rf_tuned_predictions = (
    rf_probabilities >= t_rf
).astype(int)


# ============================================================
# SUBGROUP ANALYSIS
# ============================================================

print("\nSUBGROUP ANALYSIS")

overall_recall = recall_score(
    y_test,
    rf_tuned_predictions,
    pos_label=1,
    zero_division=0
)

overall_precision = precision_score(
    y_test,
    rf_tuned_predictions,
    pos_label=1,
    zero_division=0
)

print(
    "Overall recall:",
    round(overall_recall, 4)
)

print(
    "Overall precision:",
    round(overall_precision, 4)
)


# ============================================================
# PRODUCT CATEGORY PERFORMANCE
# ============================================================

category_results = []

for category in sorted(
    X_test["product_category"].unique()
):

    mask = (
        X_test["product_category"] == category
    )

    category_recall = recall_score(
        y_test[mask],
        rf_tuned_predictions[mask],
        pos_label=1,
        zero_division=0
    )

    category_precision = precision_score(
        y_test[mask],
        rf_tuned_predictions[mask],
        pos_label=1,
        zero_division=0
    )

    category_results.append({
        "product_category": category,
        "recall": category_recall,
        "precision": category_precision
    })

category_table = pd.DataFrame(
    category_results
)

print("\nProduct category performance:")
print(
    category_table.to_string(index=False)
)


# ============================================================
# PAYMENT METHOD PERFORMANCE
# ============================================================

payment_results = []

for payment in sorted(
    X_test["payment_method"].unique()
):

    mask = (
        X_test["payment_method"] == payment
    )

    payment_recall = recall_score(
        y_test[mask],
        rf_tuned_predictions[mask],
        pos_label=1,
        zero_division=0
    )

    payment_precision = precision_score(
        y_test[mask],
        rf_tuned_predictions[mask],
        pos_label=1,
        zero_division=0
    )

    payment_results.append({
        "payment_method": payment,
        "recall": payment_recall,
        "precision": payment_precision
    })

payment_table = pd.DataFrame(
    payment_results
)

print("\nPayment method performance:")
print(
    payment_table.to_string(index=False)
)


# ============================================================
# SUBGROUP ROOT-CAUSE ANALYSIS
# ============================================================

print("\nSUBGROUP ROOT-CAUSE ANALYSIS")

category_candidates = category_table.copy()

category_candidates["subgroup_type"] = (
    "product_category"
)

category_candidates["subgroup"] = (
    category_candidates["product_category"]
)


payment_candidates = payment_table.copy()

payment_candidates["subgroup_type"] = (
    "payment_method"
)

payment_candidates["subgroup"] = (
    payment_candidates["payment_method"]
)


subgroup_candidates = pd.concat(
    [
        category_candidates[
            [
                "subgroup_type",
                "subgroup",
                "recall",
                "precision"
            ]
        ],
        payment_candidates[
            [
                "subgroup_type",
                "subgroup",
                "recall",
                "precision"
            ]
        ]
    ],
    ignore_index=True
)


weakest_subgroup = subgroup_candidates.sort_values(
    ["recall", "precision"],
    ascending=[True, True]
).iloc[0]


print(
    "Weakest subgroup type:",
    weakest_subgroup["subgroup_type"]
)

print(
    "Weakest subgroup:",
    weakest_subgroup["subgroup"]
)

print(
    "Recall:",
    round(weakest_subgroup["recall"], 4)
)

print(
    "Precision:",
    round(weakest_subgroup["precision"], 4)
)

print(
    "Root-cause interpretation: this subgroup has the "
    "lowest recall among the evaluated subgroups, meaning "
    "the model is missing a larger share of actual returns "
    "for this group."
)

print(
    "Concrete next step: evaluate a subgroup-specific "
    "probability threshold for this subgroup and compare "
    "its recall, precision, and F1 against the current "
    "overall threshold before deployment."
)


# ============================================================
# CONFUSION MATRIX AT SELECTED RF THRESHOLD
# ============================================================

print("\nCONFUSION MATRIX")

rf_confusion = confusion_matrix(
    y_test,
    rf_tuned_predictions
)

print(rf_confusion)

print(
    "Confusion matrix threshold:",
    round(t_rf, 2)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_rf_model,
    "models/return_risk_model.pkl"
)

print("\nMODEL SAVED")
print("models/return_risk_model.pkl")
print(
    "Saved model type:",
    type(best_rf_model).__name__
)
