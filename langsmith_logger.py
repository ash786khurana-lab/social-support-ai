import os
from langsmith import Client

# LangSmith client using API key from env
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

def log_trace(user_id: str, step: str, state: dict):
    """
    Log pipeline steps to LangSmith for observability
    """
    try:
        client.create_run(
            name="EligibilityPipeline",
            inputs={"user_id": user_id, "step": step},
            outputs={"state": state},
            metadata={"project": "social-support-ai"},
            run_type="chain"   # Required field (could be "llm" / "tool" also)
        )
        print(f" Logged to LangSmith: {step}")
    except Exception as e:
        print(f" LangSmith log failed: {e}")
