# 💳 Credit Risk Classification Engine

A Machine Learning project that predicts whether a loan applicant belongs to a **Low Risk** or **High Risk** credit category based on financial and personal information.

The project uses Python, Pandas, NumPy, Scikit-learn, and Streamlit to build an end-to-end machine learning application.

## 🎯 Project Objective

The main objective is to develop a machine learning classification system that analyzes loan applicant information and predicts their credit risk category.

This project demonstrates the complete machine learning workflow, including:
- Dataset creation
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation and comparison
- Model serialization
- Streamlit application development

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Streamlit | Web application |
| Jupyter Notebook | Exploratory Data Analysis |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
creditrisk/
│
├── dataset/
│   ├── create_dataset.py
│   └── credit_risk_dataset.csv
│
├── models/
│   └── credit_risk_model.pkl
│
├── notebook/
│   ├── eda.ipynb
│   └── complete_credit_risk_dataset.csv
│
├── src/
│   ├── app.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── readme.md
│   └── requirement.txt
│
├── model_comparison.csv
└── README.md
```

## 📊 Dataset Information

The project uses a synthetic credit risk dataset containing **5,000 records**.

It includes the following features:

| Feature | Description |
|---|---|
| person_age | Applicant's age |
| person_income | Applicant's annual income |
| person_emp_length | Employment length |
| loan_amnt | Loan amount |
| loan_int_rate | Loan interest rate |
| loan_percent_income | Loan amount as a proportion of income |
| credit_history_length | Length of credit history |
| debt_to_income | Debt-to-income ratio |
| home_ownership | Home ownership status |
| loan_intent | Purpose of the loan |
| credit_risk | Target variable |

### Target Variable

- `0` – Low Risk
- `1` – High Risk

## 🤖 Machine Learning Models

The project explores classification algorithms such as:

- Logistic Regression
- Random Forest Classifier

Models are evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The selected model is saved as a `.pkl` file using Joblib.

## 🔍 Exploratory Data Analysis

EDA includes:

- Dataset overview
- Missing value analysis
- Duplicate detection
- Credit risk distribution
- Income and credit risk analysis
- Loan amount distribution
- Interest rate analysis
- Correlation heatmap
- Categorical feature analysis
- Outlier detection

## 🌐 Streamlit Application

The project includes a Streamlit web application that accepts applicant information and displays a predicted credit risk category.

### Application Features

- Applicant financial information input
- Loan information input
- Credit risk prediction
- Low Risk / High Risk classification
- Prediction probability display

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vamsi78990/credit-risk-classification-engine.git
```

### 2. Navigate to the Project

```bash
cd credit-risk-classification-engine
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r src/requirement.txt
```

### 6. Run the Streamlit Application

```bash
python -m streamlit run src/app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

**Note:** Train and save the model using the current preprocessing pipeline before launching the application. The saved model must expect the same input features supplied by `app.py`.

## 📈 Future Improvements

- Experiment with additional classification algorithms.
- Improve model performance through hyperparameter tuning.
- Add cross-validation.
- Implement explainable AI techniques.
- Improve the Streamlit user interface.
- Deploy the application online.
- Explore real-world credit datasets.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes using synthetic data.

It is not intended for actual loan approval or financial decision-making.

## 👨‍💻 Author

**Vamsi Krishnal**

B.Tech – Computer Science and Engineering (Data Science)

GitHub: [Vamsi78990](https://github.com/Vamsi78990)

---

⭐ If you find this project interesting, consider giving it a star!
