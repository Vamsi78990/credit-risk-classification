
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import build_preprocessor


# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset" / "credit_risk_dataset.csv"

MODEL_PATH = ROOT / "models" / "credit_risk_model.pkl"

TARGET = "credit_risk"


def main():

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset Loaded Successfully!")
    print("Dataset Shape:", df.shape)

    # 2. Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data:", X_test.shape)

    # 4. Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # 5. Logistic Regression Model
    logistic_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)

    # 6. Random Forest Model
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # 7. Hyperparameter tuning
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("\nTraining Random Forest Model...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    rf_predictions = best_model.predict(X_test)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    # 8. Model evaluation
    models = {
        "Logistic Regression": logistic_predictions,
        "Random Forest": rf_predictions
    }

    results = []

    for name, predictions in models.items():

        print("\n==============================")
        print(name)
        print("==============================")

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # 9. Save model comparison
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        ROOT / "model_comparison.csv",
        index=False
    )

    # 10. Save best model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    print("\nBest Random Forest Model Saved Successfully!")

    print("\nFinal Model Comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
