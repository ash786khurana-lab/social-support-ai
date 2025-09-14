import streamlit as st
import uuid
from ingestion import ingest_all
from eligibility import build_features, check_eligibility
from recommendations import generate_recommendations
from agents_orchestration import ollama_reasoning, check_ollama_server, log_manual_check
from langsmith_logger import log_trace

st.title(" Social Support AI Chatbot (LangGraph + Ollama)")

# Generate unique user session ID
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

user_id = st.session_state["user_id"]
st.write(f"Your Session ID: {user_id}")

st.write(" Upload your documents for eligibility check:")

uploaded_excel = st.file_uploader(" Upload Assets/Liabilities Excel", type=["xlsx"])
uploaded_pdf = st.file_uploader(" Upload Bank Statement", type=["pdf"])
uploaded_img = st.file_uploader(" Upload Emirates ID (Image)", type=["png", "jpg", "jpeg"])
uploaded_docx = st.file_uploader(" Upload Resume", type=["docx"])

if st.button("Run Evaluation"):
    st.write(" Button clicked, starting pipeline...")

    if uploaded_excel and uploaded_pdf and uploaded_img and uploaded_docx:
        st.info(" Files uploaded successfully")

        # Save files locally with unique session ID
        excel_path = f"data/{user_id}_assets.xlsx"
        pdf_path = f"data/{user_id}_bank.pdf"
        img_path = f"data/{user_id}_id.png"
        resume_path = f"data/{user_id}_resume.docx"

        with open(excel_path, "wb") as f:
            f.write(uploaded_excel.getbuffer())
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        with open(img_path, "wb") as f:
            f.write(uploaded_img.getbuffer())
        with open(resume_path, "wb") as f:
            f.write(uploaded_docx.getbuffer())

        # Step 1: Ingestion
        st.subheader("Step 1:  Raw Data Preview")
        data = ingest_all(
            excel_path=excel_path,
            pdf_path=pdf_path,
            img_path=img_path,
            resume_path=resume_path,
        )
        st.json(data)
        log_trace(user_id, "ingestion", data)

        # Step 2: Features
        st.subheader("Step 2:  Extracted Features")
        features = build_features(data)
        st.json(features)
        log_trace(user_id, "eligibility_features", features)

        # Step 3: Eligibility
        st.subheader("Step 3:  Decision")
        decision = check_eligibility(features)
        st.success(decision)
        log_trace(user_id, "decision", decision)

        # Step 4: LLM Reasoning
        st.subheader("Step 4:  LLM Reasoning Trace")
        if not check_ollama_server():
            st.error(" Ollama server not running. Please start with `ollama serve` in another CMD window.")
            log_manual_check("Ollama not running at evaluation time")
        else:
            prompt = f"User data: {data}. System decision: {decision}. Explain briefly why."
            reasoning = ollama_reasoning(prompt, model="gemma:2b")
            if reasoning.startswith("[LLM Error]"):
                st.error(reasoning)
            else:
                st.write(reasoning)
                log_trace(user_id, "reasoning", reasoning)

        # Step 5: Recommendations
        st.subheader("Step 5:  Recommendations")
        recs = generate_recommendations(features, decision)
        for r in recs:
            st.write("- " + r)
        log_trace(user_id, "recommendations", recs)

    else:
        st.warning(" Please upload all 4 documents before running.")
