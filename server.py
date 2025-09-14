from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import uvicorn
from ingestion import ingest_all
from eligibility import build_features, check_eligibility
from recommendations import generate_recommendations
from agents_orchestration import ollama_reasoning, check_ollama_server

app = FastAPI(title="Social Support AI API")

# Request schema
class EvalRequest(BaseModel):
    excel_path: str
    pdf_path: str
    img_path: str
    resume_path: str

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    # Step 1: Ingest
    data = ingest_all(
        excel_path=req.excel_path,
        pdf_path=req.pdf_path,
        img_path=req.img_path,
        resume_path=req.resume_path,
    )

    # Step 2: Features
    features = build_features(data)

    # Step 3: Eligibility
    decision = check_eligibility(features)

    # Step 4: Reasoning
    if not check_ollama_server():
        reasoning = "[LLM Error] Ollama not running"
    else:
        prompt = f"User data: {data}. System decision: {decision}. Explain briefly why."
        reasoning = ollama_reasoning(prompt, model="gemma:2b")

    # Step 5: Recommendations
    recs = generate_recommendations(features, decision)

    return {
        "data": data,
        "features": features,
        "decision": decision,
        "reasoning": reasoning,
        "recommendations": recs,
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
