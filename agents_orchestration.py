# agents_orchestration.py
from langgraph.graph import StateGraph, END
from ingestion import ingest_all
from eligibility import build_features, check_eligibility
from recommendations import generate_recommendations
import ollama
import requests
import datetime

# --- State ---
class AppState(dict):
    pass

# --- Ollama Health Check ---
def check_ollama_server():
    """Check if Ollama server is running on localhost:11434"""
    try:
        r = requests.get("http://localhost:11434")
        return r.status_code == 200
    except Exception:
        return False

# --- LLM Reasoning via Ollama ---
def ollama_reasoning(prompt: str, model="gemma:2b") -> str:
    if not check_ollama_server():
        log_manual_check("Ollama server not running")
        return "[LLM Error] Ollama server not running"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are an eligibility reasoning assistant. Think step by step (ReAct style)."},
                {"role": "user", "content": prompt}
            ],
            options={"timeout": 120}
        )
        return response["message"]["content"]
    except Exception as e:
        log_manual_check(f"Ollama error: {str(e)}")
        return f"[LLM Error] {str(e)}"

# --- Manual Check Logger ---
def log_manual_check(issue: str):
    """Log manual check issues into a file for admin review"""
    with open("manual_checks.log", "a") as f:
        f.write(f"{datetime.datetime.now()} - {issue}\n")

# --- Agents ---
def ingestion_agent(state: AppState):
    print("Running IngestionAgent...")
    state["data"] = ingest_all()
    return state

def reasoning_agent(state: AppState):
    print("Running ReasoningAgent (Ollama ReAct)...")
    prompt = f"User data: {state.get('data',{})}. What factors should I check for eligibility?"
    reasoning = ollama_reasoning(prompt, model="gemma:2b")  #  use small model
    state["reasoning"] = reasoning
    return state

def eligibility_agent(state: AppState):
    print("Running EligibilityAgent...")
    features = build_features(state["data"])
    decision = check_eligibility(features)
    state["features"] = features
    state["decision"] = decision
    return state

def decision_agent(state: AppState):
    print("Running DecisionAgent...")
    decision = state.get("decision", "UNKNOWN")
    features = state.get("features", {})
    recs = generate_recommendations(features, decision)
    state["recommendations"] = recs
    return state

# --- Workflow ---
workflow = StateGraph(AppState)
workflow.add_node("ingestion", ingestion_agent)
workflow.add_node("reasoning", reasoning_agent)
workflow.add_node("eligibility", eligibility_agent)
workflow.add_node("decision", decision_agent)

workflow.set_entry_point("ingestion")
workflow.add_edge("ingestion", "reasoning")   # ingestion → reasoning
workflow.add_edge("reasoning", "eligibility") # reasoning → eligibility
workflow.add_edge("eligibility", "decision")
workflow.add_edge("decision", END)

app = workflow.compile()

if __name__ == "__main__":
    final_state = app.invoke(AppState())
    print("\n Reasoning Trace:\n", final_state.get("reasoning"))
    print("\n Features:\n", final_state.get("features"))
    print("\n Decision:\n", final_state.get("decision"))
    print("\n Recommendations:\n", final_state.get("recommendations"))
