# Bank Customer Churn Prediction & Risk Scoring

An end-to-end machine learning pipeline that predicts which bank customers are likely to churn and converts those predictions into business-ready risk scores for a retention team to act on.

📄 **Research Paper:** [Bank_Churn_Research_Paper.pdf](./Bank_Churn_Research_Paper.pdf)
🚀 **Live Demo:** [Add your Streamlit Cloud URL here after deploying](#deployment)

## Overview

Customer churn is one of the costliest problems for retail banks — acquiring a new customer costs far more than retaining an existing one. This project builds a full pipeline that:

1. Explores and understands churn drivers in the customer base
2. Trains and compares multiple classification models
3. Converts model probabilities into an interpretable **0–100 risk score**
4. Segments customers into **Low / Medium / High / Critical** risk bands
5. Produces a prioritized, actionable list for the retention team, complete with estimated value at risk

## Project Structure

```
.
├── 01_generate_data.py       # Dataset creation (swap with your own CSV if you have real data)
├── 02_eda.py                 # Exploratory data analysis + charts
├── 03_modeling.py            # Preprocessing, model training & evaluation
├── 04_risk_scoring.py        # Converts predictions into risk scores & action list
├── app.py                    # Streamlit app: interactive dashboard + live risk scoring
├── requirements.txt          # Dependencies for the Streamlit app
├── Bank_Churn_Research_Paper.pdf  # Short research paper describing this project
├── bank_churn.csv            # Generated dataset (created by step 1)
├── outputs/
│   ├── eda_overview.png              # EDA charts (churn distribution, correlations, etc.)
│   ├── roc_curves.png                # ROC curve comparison across models
│   ├── confusion_matrix.png          # Confusion matrix for the best model
│   ├── feature_importance.png        # Top drivers of churn
│   ├── model_comparison.csv          # Accuracy / precision / recall / F1 / ROC-AUC per model
│   ├── customer_risk_scores.csv      # Final prioritized, risk-scored customer list
└── README.md
```

## Dataset

The pipeline uses a bank customer dataset with the following schema:

| Column | Description |
|---|---|
| `CustomerId` | Unique customer identifier |
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | Target variable — 1 if the customer churned, 0 otherwise |

> No dataset was provided for this project, so `01_generate_data.py` synthesizes 10,000 realistic customer records with genuine churn signal baked in (age, geography, product count, activity level, and credit score all influence churn probability, plus noise so it isn't trivially separable). **To use your own data**, replace this step with `pd.read_csv("your_file.csv")` as long as the column names match, or update the column references in `03_modeling.py` and `04_risk_scoring.py`.

## Methodology

### 1. Exploratory Data Analysis
Checked for missing values, then visualized churn distribution, churn rate by geography and product count, age distribution split by churn, active-vs-inactive churn rates, and a correlation heatmap.

### 2. Feature Engineering
- `BalanceSalaryRatio` — balance relative to income
- `IsZeroBalance` — flag for zero-balance accounts
- `CreditScoreBand` — binned credit score categories

### 3. Preprocessing
- One-hot encoding for categorical features
- `StandardScaler` for numeric features
- Stratified 80/20 train/test split (preserves churn ratio in both sets)
- **SMOTE** oversampling applied to the training set only, to correct for class imbalance without leaking synthetic samples into the test set

### 4. Model Training & Evaluation
Three models were trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble, handles non-linear interactions |
| XGBoost | Gradient-boosted trees, typically strongest performer |

Evaluated on accuracy, precision, recall, F1, and ROC-AUC. The best model (by ROC-AUC) is automatically selected and saved for the risk-scoring step.

### 5. Risk Scoring
The winning model's predicted churn probability is converted into:
- **Risk Score** (0–100)
- **Risk Band**: Low / Medium / High / Critical
- **Estimated Value at Risk** (account balance × churn probability)
- **Recommended Action** per band (e.g., Critical → immediate relationship-manager outreach)

## Results

- Baseline churn rate in the dataset: ~18%
- Best model selected automatically by ROC-AUC (see `outputs/model_comparison.csv` for exact numbers on your run)
- Final output: a prioritized `customer_risk_scores.csv` ranking every customer by churn risk, ready to hand to a retention/marketing team

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- joblib

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn joblib
```

## Usage

Run the scripts in order from the project root:

```bash
python 01_generate_data.py    # creates bank_churn.csv
python 02_eda.py              # creates outputs/eda_overview.png
python 03_modeling.py         # trains models, creates outputs/ charts + saved model
python 04_risk_scoring.py     # creates outputs/customer_risk_scores.csv
```

Each script auto-creates the `outputs/` folder if it doesn't exist, and all paths are relative — clone the repo anywhere and it works out of the box.

## Deployment

The project includes a self-contained Streamlit app (`app.py`) that trains the model in-memory on load (cached), so it needs no pre-saved model files to deploy.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the dashboard in your browser at `http://localhost:8501`, with three tabs: a metrics dashboard, a live single-customer risk calculator, and the full risk-scored customer list with CSV download.

### Deploy for free on Streamlit Community Cloud
1. Push this repo (including `app.py` and `requirements.txt`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select this repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. The first build takes a few minutes to install dependencies.
5. Copy the resulting `https://<your-app-name>.streamlit.app` URL and paste it into the **Live Demo** link at the top of this README.
