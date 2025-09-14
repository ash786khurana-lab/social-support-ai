import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

MODEL_PATH = "eligibility_model.pkl"

# --- Step 1: Generate synthetic training data ---
def generate_synthetic_data(n=500):
    np.random.seed(42)
    data = {
        "income": np.random.randint(2000, 15000, n),
        "employment_years": np.random.randint(0, 20, n),
        "age": np.random.randint(18, 65, n),
        "net_worth": np.random.randint(0, 500000, n),
        "family_size": np.random.randint(1, 7, n),
    }
    df = pd.DataFrame(data)

    # Rule-based label (mock ground truth)
    # Approve if low income but big family, else decline
    df["label"] = ((df["income"] < 6000) & (df["family_size"] >= 3)).astype(int)

    return df


# --- Step 2: Train model ---
def train_model():
    df = generate_synthetic_data(1000)
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print(" Training Complete. Evaluation on test set:")
    print(classification_report(y_test, clf.predict(X_test)))

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")


# --- Step 3: Predict eligibility from features ---
def ml_check_eligibility(features, model_path=MODEL_PATH):
    clf = joblib.load(model_path)
    X = [[
        features.get("income", 0),
        features.get("employment_years", 0),
        features.get("age", 0),
        features.get("net_worth", 0),
        features.get("family_size", 0),
    ]]
    pred = clf.predict(X)[0]
    return "APPROVED" if pred == 1 else "SOFT DECLINE"


# --- Run training if executed directly ---
if __name__ == "__main__":
    train_model()
