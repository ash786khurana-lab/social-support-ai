from langsmith.evaluation import evaluate

def eligibility_check(example, prediction):
    decision = prediction.get("decision", "")
    features = prediction.get("features", {})

    if decision == "APPROVED" and features.get("loan", 0) < -100000:
        return {"score": 0, "reason": "Loan too high but approved"}
    return {"score": 1, "reason": "Decision seems valid"}

if __name__ == "__main__":
    evaluate(
        project_name="default",
        evaluator=eligibility_check
    )
