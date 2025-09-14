📘 Social Support AI – Multimodal Eligibility System

This project implements a multimodal AI pipeline that ingests documents (PDF, Excel, DOCX, Image), extracts features, runs ML eligibility checks + LLM reasoning, and provides recommendations through a Streamlit chatbot interface.

⚙️ 1. Prerequisites

Python 3.10+ (recommended: 3.10 or 3.11)

pip (Python package installer)

Git (to clone repo)

Ollama (for local LLM hosting, e.g., gemma:2b or llama2)

LangSmith account (optional, for observability)

🏗️ 2. Setup Virtual Environment
# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Activate venv (Linux/Mac)
source venv/bin/activate

📦 3. Install Required Libraries

Install all dependencies:

pip install -r requirements.txt


If you don’t have a requirements.txt yet, create one with these libraries:

streamlit
pandas
numpy
scikit-learn
joblib
python-docx
pytesseract
pillow
openpyxl
PyMuPDF   # (fitz for PDFs)
langgraph
ollama
requests
langsmith

📂 4. Project Structure
social-support-ai/
│── agents_orchestration.py   # LangGraph pipeline orchestration
│── chatbot_demo.py            # Streamlit chatbot UI
│── eligibility.py             # Feature extraction + ML eligibility
│── ingestion.py               # Data ingestion (Excel, PDF, Image, DOCX)
│── ml_model.py                # Training RandomForest & saving eligibility_model.pkl
│── recommendations.py         # Rule-based recommendations
│── run_eligibility.py         # CLI runner
│── langsmith_logger.py        # LangSmith observability logger
│── requirements.txt
│── README.md
│
├── data/                      # All input files go here
│   ├── excel/                 # Upload assets/liabilities excel
│   ├── pdf/                   # Bank statements
│   ├── images/                # Emirates ID images
│   ├── resumes/               # DOCX resumes
│
├── models/
│   └── eligibility_model.pkl  # Trained ML model
│
└── manual_checks.log          # Log for admin/manual checks

🧑‍🏫 5. Train the Model

If you want to retrain the eligibility model:

python ml_model.py


This will generate eligibility_model.pkl inside models/.

🚀 6. Run the Application
6.1 Start Ollama server (new terminal)
ollama serve

6.2 Run Streamlit chatbot
streamlit run chatbot_demo.py


Open the app at http://localhost:8501

🔗 7. Using the Application

Upload 4 files:

Excel (Assets & Liabilities)

PDF (Bank Statement)

Image (Emirates ID)

DOCX (Resume)

The system will:

Ingest and parse data

Extract features

Run ML eligibility model

Generate LLM reasoning trace (via Ollama)

Provide decision + recommendations

📊 8. Observability with LangSmith

To enable LangSmith logging:

export LANGSMITH_API_KEY="your_api_key_here"


Windows PowerShell:

setx LANGSMITH_API_KEY "your_api_key_here"


Traces will appear in your LangSmith dashboard.
