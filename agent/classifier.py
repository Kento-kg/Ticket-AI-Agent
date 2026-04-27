from functools import lru_cache
from typing import TypedDict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agents.config import cfg
from agents.mappings import CATEGORY_TO_TEAM


class ClassificationResult(TypedDict):
    category: str
    confidence: float
    team: str


@lru_cache(maxsize=1)
def load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """Load the fine-tuned classifier once. Cached for the process lifetime."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.hf_model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def classify(text: str) -> ClassificationResult:
    """Classify a ticket text. Returns category, confidence, and the team derived
    from CATEGORY_TO_TEAM (1-to-1 mapping verified empirically in the dataset)."""
    tokenizer, model, device = load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_length,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id])
    category = model.config.id2label[pred_id]

    return {
        "category": category,
        "confidence": confidence,
        "team": CATEGORY_TO_TEAM[category],
    }