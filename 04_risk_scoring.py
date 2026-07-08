"""
STEP 4: Risk Scoring
Converts churn probabilities into a 0-100 Risk Score and Low/Medium/High/
Critical risk bands, then produces a prioritized outreach list for the
retention team.
"""
import os
import pandas as pd
import numpy as np
import joblib

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("./bank_churn.csv")
best_model = joblib.load("outputs/best_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")
meta = joblib.load("outputs/metadata.pkl")

feature_names, num_cols, cat_cols = meta["feature_names"], meta["num_cols"], meta["cat_cols"]

# Recreate the exact same feature engineering as step 3
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
df["CreditScoreBand"] = pd.cut(
    df["CreditScore"], bins=[300, 580, 670, 740, 800, 850],
    labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"]
)

X = df.drop(columns=["Exited", "CustomerId"])
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)
X_encoded[num_cols] = scaler.transform(X_encoded[num_cols])

# ---- Score every customer in the book ----
churn_proba = best_model.predict_proba(X_encoded)[:, 1]

risk_df = df[["CustomerId", "Geography", "Gender", "Age", "Tenure",
              "Balance", "NumOfProducts", "IsActiveMember", "CreditScore", "Exited"]].copy()
risk_df["ChurnProbability"] = churn_proba
risk_df["RiskScore"] = (churn_proba * 100).round(1)


def risk_band(score):
    if score < 25:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Critical"


risk_df["RiskBand"] = risk_df["RiskScore"].apply(risk_band)

# Estimated revenue-at-risk: use balance as a simple proxy for account value
risk_df["EstimatedValueAtRisk"] = (risk_df["Balance"] * risk_df["ChurnProbability"]).round(2)


def recommended_action(row):
    if row["RiskBand"] == "Critical":
        return "Immediate call from relationship manager + retention offer"
    elif row["RiskBand"] == "High":
        return "Proactive outreach within 7 days, personalized offer"
    elif row["RiskBand"] == "Medium":
        return "Add to targeted email/loyalty campaign"
    else:
        return "Standard monitoring, no action needed"


risk_df["RecommendedAction"] = risk_df.apply(recommended_action, axis=1)

risk_df = risk_df.sort_values("RiskScore", ascending=False)

print("Risk band distribution:")
print(risk_df["RiskBand"].value_counts())
print("\nTop 10 highest-risk customers:")
print(risk_df.head(10)[["CustomerId", "RiskScore", "RiskBand", "EstimatedValueAtRisk"]])

print("\nTotal estimated value at risk across book: ${:,.2f}".format(
    risk_df["EstimatedValueAtRisk"].sum()))
print("Value at risk in Critical band alone: ${:,.2f}".format(
    risk_df.loc[risk_df["RiskBand"] == "Critical", "EstimatedValueAtRisk"].sum()))

risk_df.to_csv("outputs/customer_risk_scores.csv", index=False)
print("\nSaved: outputs/customer_risk_scores.csv")
