"""
STEP 3: Preprocessing + Model Training + Evaluation
Trains 3 models, compares them, picks the best by ROC-AUC.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

os.makedirs("outputs", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

df = pd.read_csv("./bank_churn.csv")

# ---------------- Feature engineering ----------------
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
df["CreditScoreBand"] = pd.cut(
    df["CreditScore"], bins=[300, 580, 670, 740, 800, 850],
    labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"]
)

y = df["Exited"]
X = df.drop(columns=["Exited", "CustomerId"])

cat_cols = ["Geography", "Gender", "CreditScoreBand"]
num_cols = [c for c in X.columns if c not in cat_cols]

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
feature_names = X_encoded.columns.tolist()

# ---------------- Train/test split (stratified: keep churn ratio equal) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Scale numeric features ----------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------- Handle class imbalance with SMOTE (train set only) ----------------
print("Before SMOTE:", y_train.value_counts().to_dict())
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE:", y_train_bal.value_counts().to_dict())

# ---------------- Train models ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
}

results = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted_models[name] = model
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))
    print("ROC-AUC:", round(results[name]["roc_auc"], 4))

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print("\n===== MODEL COMPARISON =====")
print(results_df.round(4))
results_df.to_csv("outputs/model_comparison.csv")

best_model_name = results_df.index[0]
best_model = fitted_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------- ROC curves ----------------
plt.figure(figsize=(7, 6))
for name, model in fitted_models.items():
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=130)

# ---------------- Confusion matrix for best model ----------------
best_proba = best_model.predict_proba(X_test_scaled)[:, 1]
best_pred = (best_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5.5, 4.5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=130)

# ---------------- Feature importance ----------------
plt.figure(figsize=(8, 7))
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
elif hasattr(best_model, "coef_"):
    importances = pd.Series(np.abs(best_model.coef_[0]), index=feature_names)
importances = importances.sort_values(ascending=True).tail(15)
importances.plot(kind="barh", color="#4C72B0")
plt.title(f"Top 15 Feature Importances - {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=130)

# ---------------- Save artifacts for the risk-scoring step ----------------
joblib.dump(best_model, "outputs/best_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump({"feature_names": feature_names, "num_cols": num_cols, "cat_cols": cat_cols,
             "best_model_name": best_model_name},
            "outputs/metadata.pkl")

print("\nAll artifacts saved to outputs/")
