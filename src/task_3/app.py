import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.task_3.citations import answer_with_citations

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

app = FastAPI(title="Policy QA Assistant")

APP_API_KEY = os.getenv("APP_API_KEY")

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    x_api_key: str = Header(...)
):
    if not APP_API_KEY or not secrets.compare_digest(
        x_api_key, APP_API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    result = answer_with_citations(request.question)

    return {
        "question": request.question,
        "answer": result["answer"],
        "citations": result["citations"],
        "citation_valid": result["valid"],
        "sources": result["sources"]
    }