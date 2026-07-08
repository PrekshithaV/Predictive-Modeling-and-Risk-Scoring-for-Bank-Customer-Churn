"""
Streamlit app: Bank Customer Churn Prediction & Risk Scoring
Self-contained — generates data and trains the model on first load (cached),
so it deploys with zero external file dependencies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Bank Churn Risk Scoring", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------
# Data generation + model training (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def generate_data(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    geography = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age = np.clip(rng.normal(38, 10, n), 18, 92).astype(int)
    tenure = rng.integers(0, 11, n)
    credit_score = np.clip(rng.normal(650, 96, n), 350, 850).astype(int)
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.5, 0.4, 0.07, 0.03])
    has_cr_card = rng.choice([0, 1], size=n, p=[0.3, 0.7])
    is_active_member = rng.choice([0, 1], size=n, p=[0.48, 0.52])
    estimated_salary = np.round(rng.uniform(11, 200000, n), 2)
    zero_balance_mask = rng.random(n) < 0.36
    balance = np.where(zero_balance_mask, 0.0,
                        np.round(np.clip(rng.normal(97000, 45000, n), 0, 250000), 2))

    df = pd.DataFrame({
        "CustomerId": np.arange(15000000, 15000000 + n),
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member, "EstimatedSalary": estimated_salary,
    })

    z = (-1.15 + 0.045 * (df["Age"] - 38)
         + 0.85 * (df["Geography"] == "Germany").astype(int)
         - 0.20 * (df["Geography"] == "France").astype(int)
         - 0.10 * df["Tenure"] - 0.90 * df["IsActiveMember"]
         + 0.55 * (df["NumOfProducts"] >= 3).astype(int)
         - 0.35 * (df["NumOfProducts"] == 2).astype(int)
         + 0.9 * (df["CreditScore"] < 500).astype(int)
         + 0.35 * (df["Gender"] == "Female").astype(int)
         + 0.0000018 * df["Balance"]
         + rng.normal(0, 0.55, n))
    prob = 1 / (1 + np.exp(-z))
    df["Exited"] = rng.binomial(1, prob)
    return df


def engineer_features(df):
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["CreditScoreBand"] = pd.cut(
        df["CreditScore"], bins=[300, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"])
    return df


@st.cache_resource
def train_model():
    df = generate_data()
    df_fe = engineer_features(df)
    y = df_fe["Exited"]
    X = df_fe.drop(columns=["Exited", "CustomerId"])
    cat_cols = ["Geography", "Gender", "CreditScoreBand"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_names = X_enc.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_s[num_cols] = scaler.transform(X_test[num_cols])

    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
    model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                    random_state=42, n_jobs=-1)
    model.fit(X_train_bal, y_train_bal)

    test_proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    cm = confusion_matrix(y_test, (test_proba >= 0.5).astype(int))

    return {
        "model": model, "scaler": scaler, "feature_names": feature_names,
        "num_cols": num_cols, "cat_cols": cat_cols, "auc": auc,
        "fpr": fpr, "tpr": tpr, "cm": cm, "df_raw": df,
    }


def score_customers(df_raw, bundle):
    df_fe = engineer_features(df_raw)
    X = df_fe.drop(columns=["Exited", "CustomerId"])
    X_enc = pd.get_dummies(X, columns=bundle["cat_cols"], drop_first=True)
    X_enc = X_enc.reindex(columns=bundle["feature_names"], fill_value=0)
    X_enc[bundle["num_cols"]] = bundle["scaler"].transform(X_enc[bundle["num_cols"]])
    proba = bundle["model"].predict_proba(X_enc)[:, 1]

    out = df_raw[["CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance",
                   "NumOfProducts", "IsActiveMember", "CreditScore"]].copy()
    out["ChurnProbability"] = proba
    out["RiskScore"] = (proba * 100).round(1)
    out["RiskBand"] = pd.cut(out["RiskScore"], bins=[-1, 25, 50, 75, 101],
                              labels=["Low", "Medium", "High", "Critical"])
    out["EstimatedValueAtRisk"] = (out["Balance"] * out["ChurnProbability"]).round(2)
    action_map = {
        "Critical": "Immediate relationship-manager call + retention offer",
        "High": "Proactive outreach within 7 days",
        "Medium": "Add to targeted retention campaign",
        "Low": "Standard monitoring",
    }
    out["RecommendedAction"] = out["RiskBand"].map(action_map)
    return out.sort_values("RiskScore", ascending=False)


def score_single_customer(inputs, bundle):
    df_single = pd.DataFrame([{**inputs, "CustomerId": 0, "Exited": 0}])
    scored = score_customers(df_single, bundle)
    return scored.iloc[0]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏦 Bank Customer Churn Prediction & Risk Scoring")
st.caption("Random Forest model trained on 10,000 customer records, with SMOTE-balanced "
           "training and a business-facing 0–100 risk score.")

with st.spinner("Training model (first load only, cached afterward)..."):
    bundle = train_model()
    risk_df = score_customers(bundle["df_raw"], bundle)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Score a Customer", "📋 Full Risk List"])

# ---- TAB 1: Dashboard ----
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model ROC-AUC", f"{bundle['auc']:.3f}")
    col2.metric("Overall Churn Rate", f"{bundle['df_raw']['Exited'].mean()*100:.1f}%")
    col3.metric("Critical-Risk Customers", int((risk_df["RiskBand"] == "Critical").sum()))
    col4.metric("Total Value at Risk", f"${risk_df['EstimatedValueAtRisk'].sum():,.0f}")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Band Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        band_counts = risk_df["RiskBand"].value_counts().reindex(["Low", "Medium", "High", "Critical"])
        colors = ["#4C72B0", "#CCB974", "#DD8452", "#C44E52"]
        band_counts.plot(kind="bar", ax=ax, color=colors)
        ax.set_ylabel("Customers")
        st.pyplot(fig)

    with c2:
        st.subheader("Churn Rate by Geography")
        fig, ax = plt.subplots(figsize=(5, 4))
        bundle["df_raw"].groupby("Geography")["Exited"].mean().sort_values(ascending=False).plot(
            kind="bar", ax=ax, color="#55A868")
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(bundle["fpr"], bundle["tpr"], label=f"Random Forest (AUC={bundle['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    with c4:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(bundle["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"], ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)

# ---- TAB 2: Score a single customer ----
with tab2:
    st.subheader("Enter customer details to get a live risk score")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0, step=1000.0)
    with c3:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active = st.selectbox("Active Member", ["Yes", "No"])
        salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 60000.0, step=1000.0)

    if st.button("Calculate Risk Score", type="primary"):
        inputs = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_products, "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0, "EstimatedSalary": salary,
        }
        result = score_single_customer(inputs, bundle)

        band_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
        st.metric("Risk Score", f"{result['RiskScore']} / 100",
                   f"{band_colors[result['RiskBand']]} {result['RiskBand']} Risk")
        st.info(f"**Recommended action:** {result['RecommendedAction']}")
        st.caption(f"Estimated value at risk: ${result['EstimatedValueAtRisk']:,.2f}")

# ---- TAB 3: Full risk-scored customer list ----
with tab3:
    st.subheader("Full customer book, ranked by risk")
    band_filter = st.multiselect("Filter by risk band", ["Low", "Medium", "High", "Critical"],
                                  default=["Critical", "High"])
    filtered = risk_df[risk_df["RiskBand"].isin(band_filter)] if band_filter else risk_df
    st.dataframe(filtered, use_container_width=True, height=450)
    st.download_button("Download filtered list as CSV",
                        filtered.to_csv(index=False), "customer_risk_scores.csv", "text/csv")
