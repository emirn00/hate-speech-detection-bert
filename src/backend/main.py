from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import os
import re

app = FastAPI(title="Hate Speech Detection API")

# Load Baseline Model for fallback or comparison
try:
    baseline_model = joblib.load('models/baseline_model.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
except:
    baseline_model = None
    tfidf_vectorizer = None

# Load BERT Model
MODEL_PATH = 'models/bert_hate_speech'
if os.path.exists(MODEL_PATH):
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    bert_model.eval()
else:
    tokenizer = None
    bert_model = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    model_used: str

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

LABEL_MAP = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    cleaned = clean_text(request.text)
    
    # Try BERT first
    if bert_model and tokenizer:
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            return PredictResponse(
                text=request.text,
                prediction=LABEL_MAP[pred_idx.item()],
                confidence=float(confidence.item()),
                model_used="BERT"
            )
    
    # Fallback to Baseline
    if baseline_model and tfidf_vectorizer:
        vec = tfidf_vectorizer.transform([cleaned])
        pred_idx = baseline_model.predict(vec)[0]
        probs = baseline_model.predict_proba(vec)[0]
        confidence = probs[pred_idx]
        
        return PredictResponse(
            text=request.text,
            prediction=LABEL_MAP[pred_idx],
            confidence=float(confidence),
            model_used="Baseline (Logistic Regression)"
        )
    
    raise HTTPException(status_code=503, detail="Models not loaded")

@app.get("/")
def read_root():
    return {"message": "Hate Speech Detection API is running"}
