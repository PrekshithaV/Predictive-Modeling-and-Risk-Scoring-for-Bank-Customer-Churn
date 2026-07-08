"""
STEP 2: Exploratory Data Analysis (EDA)
Produces charts saved to outputs/ that explain WHY customers churn.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")
df = pd.read_csv("./bank_churn.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn rate: {:.2f}%".format(df["Exited"].mean() * 100))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Overall churn distribution
df["Exited"].value_counts().rename({0: "Stayed", 1: "Churned"}).plot(
    kind="bar", ax=axes[0, 0], color=["#4C72B0", "#C44E52"])
axes[0, 0].set_title("Overall Churn Distribution")
axes[0, 0].set_ylabel("Number of Customers")

# 2. Churn rate by Geography
df.groupby("Geography")["Exited"].mean().sort_values(ascending=False).plot(
    kind="bar", ax=axes[0, 1], color="#55A868")
axes[0, 1].set_title("Churn Rate by Geography")
axes[0, 1].set_ylabel("Churn Rate")

# 3. Churn rate by Number of Products
df.groupby("NumOfProducts")["Exited"].mean().plot(
    kind="bar", ax=axes[0, 2], color="#8172B2")
axes[0, 2].set_title("Churn Rate by Number of Products")
axes[0, 2].set_ylabel("Churn Rate")

# 4. Age distribution by churn
sns.kdeplot(data=df, x="Age", hue="Exited", fill=True, ax=axes[1, 0], common_norm=False)
axes[1, 0].set_title("Age Distribution: Churned vs Retained")

# 5. Churn rate by Active Membership
df.groupby("IsActiveMember")["Exited"].mean().rename({0: "Inactive", 1: "Active"}).plot(
    kind="bar", ax=axes[1, 1], color="#CCB974")
axes[1, 1].set_title("Churn Rate: Active vs Inactive Members")
axes[1, 1].set_ylabel("Churn Rate")

# 6. Correlation heatmap (numeric features)
numeric_df = df.select_dtypes(include="number").drop(columns=["CustomerId"])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2])
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("outputs/eda_overview.png", dpi=130)
print("\nSaved: outputs/eda_overview.png")

# Quick tabular summary that will drive the business story
summary = df.groupby("Geography")["Exited"].agg(["mean", "count"]).rename(
    columns={"mean": "churn_rate", "count": "n_customers"})
print("\nChurn by Geography:\n", summary)
