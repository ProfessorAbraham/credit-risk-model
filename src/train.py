import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def evaluate_model(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
    return metrics

def main():
    mlflow.set_experiment("Credit_Risk_Probability_Model")

    data_path = os.getenv("DATA_PATH", "data/processed/processed_data.csv")
    df = load_data(data_path)

    # Assume 'is_high_risk' is target and drop IDs and non-features
    X = df.drop(columns=["is_high_risk", "CustomerId", "TransactionId", "BatchId", "SubscriptionId", "CurrencyCode", "CountryCode", "ProviderId", "ProductId", "ProductCategory", "ChannelId", "PricingStrategy", "FraudResult", "TransactionStartTime"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "GradientBoosting": GradientBoostingClassifier()
    }

    params = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"]
        },
        "GradientBoosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }

    best_model = None
    best_score = 0
    best_name = None

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        grid = GridSearchCV(model, params[model_name], cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)

        print(f"Best params for {model_name}: {grid.best_params_}")

        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)
        print(f"Evaluation metrics for {model_name}: {metrics}")

        mlflow.start_run(run_name=model_name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        mlflow.end_run()

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = grid.best_estimator_
            best_name = model_name

    print(f"Best model: {best_name} with ROC AUC: {best_score}")

    # Save best model locally
    model_path = f"models/{best_name}_best_model.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    main()
