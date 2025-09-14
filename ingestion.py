import pandas as pd
import fitz  # PyMuPDF for PDF
import pytesseract
from PIL import Image
import docx
import re

# --- Set Tesseract Path (Windows default installation) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Excel Reader ---
def read_excel(path):
    try:
        df = pd.read_excel(path)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

# --- PDF Reader ---
def read_pdf(path):
    try:
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"PDF read error: {e}"

# --- Image Reader (OCR) ---
def read_image(path):
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Image OCR failed: {e}"

# --- DOCX Reader ---
def read_docx(path):
    try:
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"DOCX read error: {e}"

# --- Ingest All ---
def ingest_all(excel_path=None, pdf_path=None, img_path=None, resume_path=None):
    data = {}
    if excel_path:
        data["excel_assets"] = read_excel(excel_path)
    if pdf_path:
        data["pdf_bank"] = read_pdf(pdf_path)
    if img_path:
        data["image_id"] = read_image(img_path)
    if resume_path:
        data["docx_resume"] = read_docx(resume_path)
    return data
