import wandb
from agent.classifier import _load as _load_classifier
from agent.graph import build_graph, run_triage
from pathlib import Path
from datasets import Dataset
import evaluate as hf_evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.json"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "holdout.json"

def load_holdout(sample: int | None = None):
    with HOLDOUT_PATH.open() as f:
        holdout = json.load(f)
    if sample:
        holdout = holdout[:sample]
    return holdout

def run_agent_on_holdout(holdout: Dataset):
    results = []
    for i, ticket in enumerate(holdout):
        results.append(run_triage(ticket["text"]))
    return results