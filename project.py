
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier


# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("data/creditcard.csv")

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print(df["Class"].value_counts(normalize=True) * 100)


# -------------------------------
# EXPLORATORY DATA ANALYSIS
# -------------------------------

class_counts = df["Class"].value_counts().sort_index()

class_counts.plot(kind="bar")
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()


amount_df = df[["Amount", "Class"]].copy()

amount_df["Transaction Type"] = amount_df["Class"].map({
    0: "Legitimate",
    1: "Fraudulent"
})

# Verify both classes are included
print("\nTransactions plotted by class:")
print(amount_df["Transaction Type"].value_counts())


plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount",
    order=[0, 1]
)

plt.xticks([0, 1], ["Legitimate", "Fraudulent"])
plt.yscale("symlog", linthresh=1)
plt.title("Transaction Amount Comparison")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Amount (symlog scale)")

plt.tight_layout()
plt.savefig(
    "amount_comparison_symlog.png",
    dpi=300,
    bbox_inches="tight"
)
import os
print("Chart saved at:", os.path.abspath("amount_comparison_symlog.png"))
plt.close()



print("\nLEGITIMATE TRANSACTION AMOUNTS")
print(df[df["Class"] == 0]["Amount"].describe())

print("\nFRAUDULENT TRANSACTION AMOUNTS")
print(df[df["Class"] == 1]["Amount"].describe())


# -------------------------------
# SPLIT DATA
# -------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING AND TESTING DATA")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# -------------------------------
# LOGISTIC REGRESSION
# -------------------------------

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))


# -------------------------------
# RANDOM FOREST
# -------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))


# -------------------------------
# XGBOOST
# -------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

print("\nXGBOOST")
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))
print(classification_report(y_test, xgb_predictions, zero_division=0))


# -------------------------------
# ISOLATION FOREST
# -------------------------------

from sklearn.ensemble import IsolationForest

iso_model = IsolationForest(
    n_estimators=200,
    contamination=492 / 284807,
    random_state=42,
    n_jobs=-1
)

iso_model.fit(X_train)

# Isolation Forest: -1 = anomaly, 1 = normal
iso_raw_predictions = iso_model.predict(X_test)

# Convert predictions to our labels:
# 1 = potential fraud, 0 = legitimate
iso_predictions = (iso_raw_predictions == -1).astype(int)

print("\nISOLATION FOREST")
print("Confusion Matrix:")
print(confusion_matrix(y_test, iso_predictions))

print(classification_report(
    y_test,
    iso_predictions,
    zero_division=0
))

print("Isolation Forest Accuracy:",
      accuracy_score(y_test, iso_predictions))

print("Isolation Forest Fraud Precision:",
      precision_score(y_test, iso_predictions, zero_division=0))

print("Isolation Forest Fraud Recall:",
      recall_score(y_test, iso_predictions, zero_division=0))

print("Isolation Forest Fraud F1-score:",
      f1_score(y_test, iso_predictions, zero_division=0))

joblib.dump(iso_model, "isolation_forest_model.pkl")

print("Isolation Forest model saved: isolation_forest_model.pkl")

# -------------------------------
# MODEL COMPARISON
# -------------------------------


predictions = {
    "Logistic Regression": y_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_predictions,
    "Isolation Forest": iso_predictions
}

comparison = []

for model_name, model_predictions in predictions.items():
    comparison.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, model_predictions),
        "Fraud Precision": precision_score(
            y_test, model_predictions, zero_division=0
        ),
        "Fraud Recall": recall_score(
            y_test, model_predictions, zero_division=0
        ),
        "Fraud F1-Score": f1_score(
            y_test, model_predictions, zero_division=0
        )
    })

results = pd.DataFrame(comparison)

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))


# -------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTOP 10 IMPORTANT FEATURES")
print(feature_importance.head(10).to_string(index=False))

top_10 = feature_importance.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(9, 6))
plt.barh(top_10["Feature"], top_10["Importance"])
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()


# -------------------------------
# SAVE MODELS AND RESULTS
# -------------------------------

joblib.dump(rf_model, "fraud_model.pkl")
joblib.dump(xgb_model, "xgboost_fraud_model.pkl")
joblib.dump(lr_model, "logistic_regression_model.pkl")

results.to_csv("model_comparison.csv", index=False)

print("\nRandom Forest model saved: fraud_model.pkl")
print("XGBoost model saved: xgboost_fraud_model.pkl")
print("Model comparison saved: model_comparison.csv")


# -------------------------------
# CONFUSION MATRIX HEATMAPS
# -------------------------------

# Reuse predictions already generated above
predictions = {
    "Logistic Regression": y_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_predictions,
    "Isolation Forest": iso_predictions
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (model_name, model_predictions) in zip(
    axes, predictions.items()
):
    cm = confusion_matrix(
        y_test,
        model_predictions,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        ax=ax
    )

    ax.set_title(model_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.suptitle(
    "FraudLens - Confusion Matrix Analysis",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    "confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nConfusion matrix heatmaps saved: confusion_matrices.png")
