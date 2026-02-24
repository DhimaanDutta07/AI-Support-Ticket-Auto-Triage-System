import os
import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
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


print("Loading model...")

model = tf.keras.models.load_model(
    os.path.join(CONFIG["model_dir"], "model.keras")
)

tokenizer = TextTokenizer().load()

cleaner = DataCleaning()

labels = load_json(
    os.path.join(CONFIG["model_dir"], "label_maps.json")
)

explainer = ShapTextExplainer().load()

print("Model loaded")


def predict_text(text):

    clean = cleaner.clean(text)

    X = tokenizer.transform([clean])

    p_cat, p_pri, p_sent = model.predict(X, verbose=0)

    preds = {
        "category": labels["category"][int(np.argmax(p_cat))],
        "priority": labels["priority"][int(np.argmax(p_pri))],
        "sentiment": labels["sentiment"][int(np.argmax(p_sent))]
    }

    return preds


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
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict(req: TicketRequest):

    preds = predict_text(req.text)

    preds["reply"] = suggest_reply(preds)

    return preds


@app.post("/explain")
def explain(req: TicketRequest):

    return explainer.explain(req.text)