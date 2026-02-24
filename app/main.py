import os
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config.config import CONFIG
from src.features.tokenizer import TextTokenizer
from src.data.cleaning import DataCleaning
from src.utils.common import load_json
from src.explain.shap_text import ShapTextExplainer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend", "index.html")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    text: str


MODEL = None
TOKENIZER = None
CLEANER = None
LABELS = None
EXPLAINER = None


@app.on_event("startup")
def load_assets():

    global MODEL, TOKENIZER, CLEANER, LABELS, EXPLAINER

    model_path = os.path.join(CONFIG["model_dir"], "model.keras")
    labels_path = os.path.join(CONFIG["model_dir"], "label_maps.json")
    shap_path = os.path.join(CONFIG["shap_dir"], "shap_models.joblib")
    tok_cfg = os.path.join(CONFIG["tokenizer_dir"], "vectorizer_config.json")
    tok_vocab = os.path.join(CONFIG["tokenizer_dir"], "vocab.json")

    missing = []
    for p in [model_path, labels_path, shap_path, tok_cfg, tok_vocab]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise RuntimeError("Missing artifacts: " + " | ".join(missing))

    MODEL = tf.keras.models.load_model(model_path)
    TOKENIZER = TextTokenizer().load()
    CLEANER = DataCleaning()
    LABELS = load_json(labels_path)
    EXPLAINER = ShapTextExplainer().load()


def ensure_loaded():
    if MODEL is None or TOKENIZER is None or CLEANER is None or LABELS is None or EXPLAINER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def predict_text(text):

    ensure_loaded()

    clean = CLEANER.clean(text)
    X = TOKENIZER.transform([clean])

    p_cat, p_pri, p_sent = MODEL.predict(X, verbose=0)

    return {
        "category": LABELS["category"][int(np.argmax(p_cat))],
        "priority": LABELS["priority"][int(np.argmax(p_pri))],
        "sentiment": LABELS["sentiment"][int(np.argmax(p_sent))]
    }


def suggest_reply(preds):

    if preds["category"] == "billing":
        return "We are checking your billing issue."

    if preds["category"] == "technical":
        return "Please restart the application."

    if preds["category"] == "delivery":
        return "Delivery status will update soon."

    if preds["category"] == "refund":
        return "Refund request is being processed."

    if preds["category"] == "account":
        return "Please reset your password."

    return "Support team will contact you."


@app.get("/")
def frontend():
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {"status": "running", "note": "frontend/index.html not found"}


@app.get("/health")
def health():
    loaded = MODEL is not None
    return {"status": "running", "model_loaded": loaded}


@app.post("/predict")
def predict(req: TicketRequest):

    preds = predict_text(req.text)
    preds["reply"] = suggest_reply(preds)
    return preds


@app.post("/explain")
def explain(req: TicketRequest):

    ensure_loaded()
    return EXPLAINER.explain(req.text)
