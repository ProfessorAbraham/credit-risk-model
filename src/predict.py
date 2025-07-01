import os
import sys
import pandas as pd
import joblib
import argparse

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    return model

def predict(model, X):
    proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (high risk)
    return proba

def main(input_csv, model_name):
    model_path = f"models/{model_name}_best_model.pkl"
    model = load_model(model_path)

    X = pd.read_csv(input_csv)
    preds = predict(model, X)

    output_df = X.copy()
    output_df['risk_probability'] = preds
    output_path = "predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict credit risk probability.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV file with input features")
    parser.add_argument("--model_name", type=str, default="GradientBoosting", help="Model name to load")

    args = parser.parse_args()
    main(args.input_csv, args.model_name)
