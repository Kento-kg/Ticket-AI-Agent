from functools import lru_cache
from typing import TypedDict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from agent.config import cfg
from agent.mapping import CATEGORY_TO_TEAM

class ClassificationResult(TypedDict):
    category: str
    confidence: float
    team: str

@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Clasificador fine-tuned con pipeline gestionando tokenizacion, device, softmax,batching..."""
    return pipeline(
        "text-classification",
        model=cfg.hf_model_id,
        tokenizer=cfg.hf_model_id,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=cfg.max_length,
        top_k=1
    )

def classify(text: str) -> ClassificationResult:
    """Classify a ticket text. Returns category, confidence, and the team derived
    from CATEGORY_TO_TEAM (1-to-1 mapping)."""
    result = load_pipeline(text)[0][0]
    category = result["label"]
    return {
        "category": category,
        "confidence": float(result["score"]),
        "team": CATEGORY_TO_TEAM[category],
    }