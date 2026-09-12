import os

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Loaded ONCE when the container starts (not per-request). After the first
# request wakes a sleeping free-tier Space, every request after that is
# fast -- only the wake-up itself is slow.
classifier = pipeline(
    "text-classification",
    model="dost-asti/RoBERTa-tl-sentiment-analysis",
    top_k=None,  # return all 3 class scores so we can pick the top one
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
