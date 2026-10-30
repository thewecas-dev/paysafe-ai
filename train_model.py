"""Train and evaluate fraud risk model using Kaggle dataset or synthetic baseline."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

def load_data():
    kaggle_path = "data/fraud_dataset.csv"
    fallback_path = "data/transactions.csv"

    if os.path.exists(kaggle_path):
        print(f"Loading Kaggle UPI Fraud Dataset from {kaggle_path}...")
        df = pd.read_csv(kaggle_path)

        # Feature Engineering for UPI risk detection:
        # Night transaction: hours 22, 23, 0, 1, 2, 3, 4, 5
        df["night_transaction"] = df["transaction_time_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        # New payee: receiver has 0 prior transaction history
        df["new_payee"] = (df["receiver_transaction_history"] == 0).astype(int)
        df["new_device"] = df["unusual_device_flag"].astype(int)
        df["unusual_location"] = df["unusual_location_flag"].astype(int)
        df["failed_attempts"] = df["failed_transaction_count"].astype(int)
        df["velocity"] = df["transaction_velocity"].astype(int)
        df["time_pressure"] = (df["time_pressure_indicators"] > 0).astype(int)
        df["receiver_account_age"] = df["receiver_account_age"].astype(float)

        features = [
            "amount", "velocity", "new_payee", "receiver_account_age",
            "new_device", "unusual_location", "night_transaction",
            "failed_attempts", "time_pressure"
        ]
        dataset_name = f"Kaggle UPI Fraud Dataset ({len(df):,} records)"
        return df[features], df["is_fraud"].astype(int), features, dataset_name

    elif os.path.exists(fallback_path):
        print(f"Loading synthetic dataset from {fallback_path}...")
        df = pd.read_csv(fallback_path)
        features = [
            "amount", "txns_last_hour", "new_payee", "new_device",
            "night_transaction", "failed_attempts"
        ]
        dataset_name = f"Synthetic Dataset ({len(df):,} records)"
        return df[features], df["is_fraud"].astype(int), features, dataset_name
    else:
        raise SystemExit("Dataset not found. Place fraud_dataset.csv in data/ or run generate_data.py")

def main():
    X, y, features, dataset_name = load_data()
    print(f"Dataset: {dataset_name}")
    print(f"Class distribution: Genuine = {(y == 0).sum():,}, Fraud = {(y == 1).sum():,} ({(y == 1).mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nEvaluation on held-out test data:")
    print(classification_report(y_test, predictions, zero_division=0, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    auc = None
    if y_test.nunique() == 2:
        auc = round(roc_auc_score(y_test, probabilities), 4)
        print("ROC-AUC:", auc)

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": model,
        "features": features,
        "dataset_name": dataset_name,
        "auc": auc
    }
    joblib.dump(bundle, "models/fraud_model.joblib")
    print(f"\nSaved model to models/fraud_model.joblib")

if __name__ == "__main__":
    main()
