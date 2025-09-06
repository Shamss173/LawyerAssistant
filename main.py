# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

from utils.vector_store import VectorStore
from utils.llm import analyze_legal_issue
from utils.parser import extract_text_from_bytes

app = FastAPI(title="Lawyer Assistant API")

# CORS — during dev allow all origins (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize vector store (heavy operation at startup)
vector_store = VectorStore()

class Query(BaseModel):
    query: str

def _try_parse_llm_json(text: str) -> dict:
    """
    Try to safely extract JSON from LLM output.
    If parsing fails, return a fallback dict with 'raw_output'.
    """
    if not isinstance(text, str):
        return {"issues": [], "references": [], "raw_output": str(text)}

    # strip code fence blocks if present
    content = text.strip()
    if content.startswith("```"):
        # take between first pair of ```
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            # if content starts with "json", drop that
            if content.strip().lower().startswith("json"):
                content = content.split("\n", 1)[1] if "\n" in content else content

    try:
        return json.loads(content)
    except Exception:
        # fallback: return raw text under 'raw_output'
        return {"issues": [], "references": [], "raw_output": text}

@app.post("/query")
async def process_query(q: Query):
    try:
        similar_cases = vector_store.search(q.query)
        llm_output = analyze_legal_issue(q.query, similar_cases)
        parsed = _try_parse_llm_json(llm_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return {
        "issues": parsed.get("issues", []),
        "cases": similar_cases,
        "references": parsed.get("references", []),
        "links": [c.get("link") for c in similar_cases],
        "raw_llm": parsed.get("raw_output", None)
    }

@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """
    Accepts only .pdf, .docx, .txt, extracts text, runs vector search + LLM analysis,
    and returns the same JSON schema as /query.
    """
    # Validate filename extension
    filename = (file.filename or "").lower()
    if not filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf, .docx or .txt files are allowed.")

    # limit file size (optional) - e.g., 20 MB
    contents = await file.read()
    max_bytes = 20 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB allowed.")

    # Extract text
    try:
        extracted_text = extract_text_from_bytes(contents, file.filename)
        if not extracted_text or not extracted_text.strip():
            raise ValueError("No readable text found in uploaded file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract file text: {str(e)}")

    # Run vector search + LLM analysis
    try:
        similar_cases = vector_store.search(extracted_text)
        llm_output = analyze_legal_issue(extracted_text, similar_cases)
        parsed = _try_parse_llm_json(llm_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return {
        "issues": parsed.get("issues", []),
        "cases": similar_cases,
        "references": parsed.get("references", []),
        "links": [c.get("link") for c in similar_cases],
        "raw_llm": parsed.get("raw_output", None)
    }

# Optional root message
@app.get("/")
def root():
    return {"message": "Lawyer Assistant backend running. Use /query (POST) or /upload (POST file)."}
