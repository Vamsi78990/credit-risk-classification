
import pandas as pd
import numpy as np
from pathlib import Path

# Random generator
np.random.seed(42)

# Number of customers
n = 5000

# Generate customer information
person_age = np.random.randint(21, 65, n)

person_income = np.random.randint(
    20000, 200000, n
)

person_emp_length = np.random.randint(
    0, 25, n
)

loan_amnt = np.random.randint(
    5000, 50000, n
)

loan_int_rate = np.random.uniform(
    5, 25, n
).round(2)

credit_history_length = np.random.randint(
    0, 25, n
)

home_ownership = np.random.choice(
    ["RENT", "OWN", "MORTGAGE", "OTHER"],
    n,
    p=[0.40, 0.20, 0.35, 0.05]
)

loan_intent = np.random.choice(
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ],
    n
)

# Calculate financial ratios
loan_percent_income = (
    loan_amnt / person_income
).round(3)

debt_to_income = np.random.uniform(
    0.05, 0.80, n
).round(3)

# Calculate credit default risk probability
risk_score = (
    -2.0
    + 3.0 * loan_percent_income
    + 2.5 * debt_to_income
    + 0.12 * (loan_int_rate - 10)
    - 0.08 * credit_history_length
    - 0.000015 * (person_income - 50000)
    - 0.035 * person_emp_length
)

# Convert risk score into probability
default_probability = 1 / (
    1 + np.exp(-risk_score)
)

# Generate target variable
credit_risk = np.random.binomial(
    1,
    default_probability
)

# Create DataFrame
df = pd.DataFrame({
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "credit_history_length": credit_history_length,
    "debt_to_income": debt_to_income,
    "home_ownership": home_ownership,
    "loan_intent": loan_intent,
    "credit_risk": credit_risk
})

# Save dataset
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset" / "credit_risk_dataset.csv"

df.to_csv(DATA_PATH, index=False)

# Display results
print("Credit Risk Dataset Created Successfully!")

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nCredit Risk Distribution:")
print(df["credit_risk"].value_counts())

print("\nDefault Rate:")
print(round(df["credit_risk"].mean() * 100, 2), "%")

