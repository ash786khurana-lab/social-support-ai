import joblib
import numpy as np
import re
from datetime import datetime

# -------- Build Features --------
def build_features(data):
    """Convert ingested raw data into numeric features"""
    features = {}

    # --- Income (from bank statement PDF text) ---
    features["income"] = 0
    if "pdf_bank" in data and "Salary" in str(data["pdf_bank"]):
        try:
            salary_line = [line for line in str(data["pdf_bank"]).split("\n") if "Salary" in line][0]
            # Capture numbers with optional + or - sign
            matches = re.findall(r"[+-]?\d{3,}", salary_line)
            if matches:
                features["income"] = int(matches[-1].replace("+", ""))
        except Exception as e:
            print(" Income parse error:", e)

    # --- Employment years (from resume DOCX) ---
    features["employment_years"] = 0
    if "docx_resume" in data and "Experience" in str(data["docx_resume"]):
        match = re.search(r"(\d+)\s+years", str(data["docx_resume"]), re.IGNORECASE)
        if match:
            features["employment_years"] = int(match.group(1))

    # --- Age (from Emirates ID OCR → DOB) ---
    features["age"] = None
    if "image_id" in data:
        # Match clean YYYY-MM-DD only
        dob_match = re.search(r"\b(19\d{2}|20\d{2})-(\d{2})-(\d{2})\b", str(data["image_id"]))
        if dob_match:
            try:
                dob_str = dob_match.group(0)  # e.g., "1990-05-12"
                dob = datetime.strptime(dob_str, "%Y-%m-%d")
                today = datetime.today()
                features["age"] = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except Exception as e:
                print(" DOB parse error:", e)

    # --- Net worth (from assets excel list) ---
    features["net_worth"] = 0
    if "excel_assets" in data:
        try:
            total = 0
            for item in data["excel_assets"]:
                val = item.get("Value (AED)", 0)
                if isinstance(val, str):
                    val = val.replace(",", "").replace(" ", "")
                    if val.lstrip("-").isdigit():
                        val = int(val)
                    else:
                        val = 0
                total += val
            features["net_worth"] = total
        except Exception as e:
            print(" Net worth parse error:", e)

    # --- Family size (default / mock) ---
    features["family_size"] = 3

    return features


# -------- ML Eligibility --------
model = joblib.load("eligibility_model.pkl")

def ml_check_eligibility(features):
    if features.get("age") is None:
        return " ID Card Not Valid (DOB missing in OCR)"

    X = np.array([[ 
        features.get("income", 0),
        features.get("employment_years", 0),
        features.get("age", 0),
        features.get("net_worth", 0),
        features.get("family_size", 0),
    ]])

    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if prediction == 1:
        return f" Approved (ML Model, Confidence: {prob:.2f})"
    else:
        return f" Declined (ML Model, Confidence: {prob:.2f})"


# -------- Wrapper --------
def check_eligibility(features):
    return ml_check_eligibility(features)
