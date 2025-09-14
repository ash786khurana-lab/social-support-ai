import json
import datetime
import os
from sklearn.inspection import permutation_importance
import joblib
import numpy as np

# Load ML Model
MODEL_PATH = "eligibility_model.pkl"
model = joblib.load(MODEL_PATH)

# --- Explainability ---
def explain_decision(features):
    """
    Generate explainability report:
    - Feature contributions
    - Why decision is taken
    """
    X = np.array([[
        features.get("income", 0),
        features.get("employment_years", 0),
        features.get("age", 0),
        features.get("net_worth", 0),
        features.get("family_size", 0),
    ]])

    # Predict
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    # Get feature importances (global model level)
    global_importance = model.feature_importances_
    feature_names = ["Income", "Employment Years", "Age", "Net Worth", "Family Size"]

    # Pair feature names with values & importance
    explain_dict = {
        "prediction": int(prediction),
        "confidence": round(float(prob), 3),
        "features": {
            name: {
                "value": float(val),
                "importance": round(float(imp), 3),
            }
            for name, val, imp in zip(feature_names, X[0], global_importance)
        }
    }

    # Human-readable reason
    if prediction == 1:
        explain_dict["reason"] = "Application approved. Key positive factors: " + \
            ", ".join([f for f, imp in sorted(zip(feature_names, global_importance), key=lambda x: -x[1])[:2]])
    else:
        explain_dict["reason"] = "Application declined. Weak areas in: " + \
            ", ".join([f for f, imp in sorted(zip(feature_names, global_importance), key=lambda x: -x[1])[-2:]])

    return explain_dict


# --- Monitoring ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_application(user_id, features, decision, explain_report):
    """
    Save logs for monitoring & admin check
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "features": features,
        "decision": decision,
        "explainability": explain_report,
    }

    log_path = os.path.join(LOG_DIR, f"{user_id}_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=4)

    print(f" Log saved at {log_path}")
    return log_path


# --- Example Run ---
if __name__ == "__main__":
    # Mock features for testing
    mock_features = {
        "income": 5000,
        "employment_years": 3,
        "age": 30,
        "net_worth": 100000,
        "family_size": 4
    }
    user_id = "demo_user_123"

    # Explain
    report = explain_decision(mock_features)
    print(" Explainability Report:", json.dumps(report, indent=4))

    # Log
    log_application(user_id, mock_features, "Approved", report)
