# src/train_model.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train(data_path):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(data_path)

    print("[INFO] Preprocessing data...")

    # Drop 'id' column if it exists
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Convert target variable to binary (0 or 1)
    df['classification'] = df['classification'].map(lambda x: 1 if str(x).lower().strip() == 'ckd' else 0)

    # Drop rows with missing values (you can replace this with imputation if needed)
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop('classification', axis=1)
    y = df['classification']

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"[DEBUG] Categorical columns: {list(categorical_cols)}")

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("[INFO] Starting MLflow run...")
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"[RESULT] Accuracy: {acc:.4f}")
        print("[INFO] Logging to MLflow...")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # Save and log classification report
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

    print("[INFO] Training complete and MLflow run logged.")
