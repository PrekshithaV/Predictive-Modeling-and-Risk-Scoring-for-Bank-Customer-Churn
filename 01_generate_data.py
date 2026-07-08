"""
STEP 1: Generate a realistic Bank Customer Churn dataset.

In a real project you would load Churn_Modelling.csv (the classic Kaggle
bank-churn dataset) with pandas.read_csv(). Since no file was uploaded here,
we synthesize a dataset with the SAME schema and realistic, non-random
relationships between features and churn, so every downstream step
(EDA, modeling, risk scoring) behaves exactly like it would on real data.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

geography = np.random.choice(["France", "Germany", "Spain"], size=N, p=[0.5, 0.25, 0.25])
gender = np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
age = np.clip(np.random.normal(38, 10, N), 18, 92).astype(int)
tenure = np.random.randint(0, 11, N)
credit_score = np.clip(np.random.normal(650, 96, N), 350, 850).astype(int)
num_products = np.random.choice([1, 2, 3, 4], size=N, p=[0.5, 0.4, 0.07, 0.03])
has_cr_card = np.random.choice([0, 1], size=N, p=[0.3, 0.7])
is_active_member = np.random.choice([0, 1], size=N, p=[0.48, 0.52])
estimated_salary = np.round(np.random.uniform(11, 200000, N), 2)

# Balance: many customers legitimately have 0 balance (common in this dataset)
zero_balance_mask = np.random.rand(N) < 0.36
balance = np.where(
    zero_balance_mask, 0.0,
    np.round(np.clip(np.random.normal(97000, 45000, N), 0, 250000), 2)
)

df = pd.DataFrame({
    "CustomerId": np.arange(15000000, 15000000 + N),
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
})

# ---- Build a realistic latent churn probability (logistic function) ----
# Known real-world churn drivers baked in on purpose:
#  - older customers churn more
#  - Germany churns more than France/Spain
#  - inactive members churn more
#  - customers with 3-4 products churn a LOT more (over-sold / frustrated)
#  - very low credit score -> more churn
#  - long tenure with the bank -> loyalty, less churn
z = (
    -1.15
    + 0.045 * (df["Age"] - 38)
    + 0.85 * (df["Geography"] == "Germany").astype(int)
    - 0.20 * (df["Geography"] == "France").astype(int)
    - 0.10 * df["Tenure"]
    - 0.90 * df["IsActiveMember"]
    + 0.55 * (df["NumOfProducts"] >= 3).astype(int)
    - 0.35 * (df["NumOfProducts"] == 2).astype(int)
    + 0.9 * ((df["CreditScore"] < 500)).astype(int)
    + 0.35 * (df["Gender"] == "Female").astype(int)
    + 0.0000018 * df["Balance"]
    + np.random.normal(0, 0.55, N)  # noise so it's not a perfectly separable toy problem
)
prob = 1 / (1 + np.exp(-z))
df["Exited"] = np.random.binomial(1, prob)

df.to_csv("./bank_churn.csv", index=False)
print(df.shape)
print(df["Exited"].value_counts(normalize=True))
print(df.head())
