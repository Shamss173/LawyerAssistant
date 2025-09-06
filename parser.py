# backend/utils/parser.py
import io
from typing import Tuple
import pdfplumber
from docx import Document

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from file bytes based on filename extension (.pdf, .docx, .txt).
    Returns a string (extracted text).
    """
    fname = filename.lower().strip()

    # PDF
    if fname.endswith(".pdf"):
        texts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        return "\n\n".join(texts).strip()

    # DOCX
    if fname.endswith(".docx"):
        bio = io.BytesIO(file_bytes)
        doc = Document(bio)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs).strip()

    # TXT
    if fname.endswith(".txt"):
        # try utf-8 then fallback to latin-1
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1").strip()

    raise ValueError("Unsupported file type. Only .pdf, .docx, .txt are allowed.")
