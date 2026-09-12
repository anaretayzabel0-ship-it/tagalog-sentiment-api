import gc
import os

import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

app = FastAPI()

MODEL_NAME = "dost-asti/RoBERTa-tl-sentiment-analysis"

# Load with low_cpu_mem_usage to avoid a memory spike while loading, then
# quantize the linear layers to int8. This roughly quarters the model's
# memory footprint (the bulk of a transformer's weights live in its linear
# layers), which is what makes this fit inside a 512MB instance.
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, low_cpu_mem_usage=True
)
_model.eval()
_model = torch.quantization.quantize_dynamic(
    _model, {torch.nn.Linear}, dtype=torch.qint8
)

# Free the pre-quantization references so the original fp32 weights can be
# garbage-collected rather than sitting in memory alongside the quantized copy.
gc.collect()

classifier = pipeline(
    "text-classification",
    model=_model,
    tokenizer=_tokenizer,
    top_k=None,  # return all 3 class scores so we can pick the top one
    device=-1,  # force CPU
)

# The model's config.json only exposes generic LABEL_0/1/2 -- this is the
# documented/community-confirmed mapping for this model.
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

# Shared secret so random visitors can't spend your free CPU quota.
# Set this as a "Secret" (not a plain variable) in the Space's Settings tab.
SPACE_API_KEY = os.environ.get("SPACE_API_KEY", "")


def verify_key(authorization: str = Header(default="")):
    if SPACE_API_KEY and authorization != f"Bearer {SPACE_API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


class TextIn(BaseModel):
    text: str


@app.get("/")
def health():
    # Lets you (or a monitor) check the Space is awake without spending
    # a model inference call.
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextIn, _=Depends(verify_key)):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    scores = classifier(text)[0]  # [{"label": "LABEL_0", "score": 0.02}, ...]
    top = max(scores, key=lambda x: x["score"])

    return {
        "sentiment": LABEL_MAP.get(top["label"], top["label"]),
        "confidence": round(float(top["score"]), 4),
        "raw": scores,
    }
