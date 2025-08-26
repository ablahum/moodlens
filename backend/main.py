from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

sentiment_pipeline = pipeline("sentiment-analysis")

class JournalInput(BaseModel):
    text: str

# Endpoint sederhana
@app.post("/analyze")
def analyze_journal(input: JournalInput):
    result = sentiment_pipeline(input.text)[0]
    return {
        "label": result["label"],
        "score": float(result["score"])
    }
