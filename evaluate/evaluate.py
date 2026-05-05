import json
from pathlib import Path

import wandb
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error

from agent.config import cfg
from agent.graph import run_triage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.json"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "holdout.json"

URGENCY_ORDER = {"low": 0, "medium": 1, "high": 2}

DRAFT_JUDGE_PROMPT = 
"""
You are evaluating a customer-facing support reply.
Ticket: {ticket_text}
Draft reply: {response}
Score professionalism 1–5: Is the tone empathetic, clear, and appropriate for a customer?
"""

ESCALATE_JUDGE_PROMPT = 
"""You are evaluating an internal escalation summary sent to the on-call team.
Ticket: {ticket_text}
Escalation summary: {response}
Score actionability 1–5: Does it give the on-call team concrete, immediate next steps?
"""


class DraftJudge(BaseModel):
    professionalism: int = Field(ge=1, le=5, description="1=very unprofessional, 5=exemplary")


class EscalateJudge(BaseModel):
    actionability: int = Field(ge=1, le=5, description="1=no concrete steps, 5=very actionable")
    
def load_holdout(sample: int | None = None):
    with HOLDOUT_PATH.open() as f:
        holdout = json.load(f)
    if sample:
        holdout = holdout[:sample]
    return holdout

def run_agent_on_holdout(holdout: list[dict]) -> list:
    results = []
    for ticket in(holdout):
        results.append(run_triage(ticket["text"]))
    return results

def compute_hard_metrics(holdout: list[dict], predictions: list) -> dict:
    cat_pred = [pred.category for pred in predictions]
    cat_real = [ticket["category"] for ticket in holdout]
    cat_f1 = f1_score(cat_real, cat_pred, average="macro")

    urg_pred = [URGENCY_ORDER[pred.urgency] for pred in predictions]
    urg_real = [URGENCY_ORDER[row["urgency"]] for row in holdout]
    urg_mae = mean_absolute_error(urg_real, urg_pred)

    return {
        "category_f1_macro": cat_f1,
        "urgency_mae": urg_mae,
    }

def llm_as_judge(holdout: list[dict], predictions: list) -> dict:
    draft_llm = ChatAnthropic(model=cfg.claude_haiku_model, temperature=0).with_structured_output(DraftJudge)
    escalate_llm = ChatAnthropic(model=cfg.claude_haiku_model, temperature=0).with_structured_output(EscalateJudge)

    draft_scores = []
    escalate_scores = []

    for ticket, pred in zip(holdout, predictions):
        if pred.action == "draft":
            jr = draft_llm.invoke(DRAFT_JUDGE_PROMPT.format(ticket_text=ticket["text"], response=pred.response))
            draft_scores.append(jr.professionalism)
        else:
            jr = escalate_llm.invoke(ESCALATE_JUDGE_PROMPT.format(ticket_text=ticket["text"], response=pred.response))
            escalate_scores.append(jr.actionability)

    metrics = {}
    if draft_scores:
        metrics["judge_professionalism"] = sum(draft_scores) / len(draft_scores)
    if escalate_scores:
        metrics["judge_actionability"] = sum(escalate_scores) / len(escalate_scores)
    return metrics

def main():
    holdout = load_holdout
    predictions = run_agent_on_holdout(holdout)
    hard_metrics = compute_hard_metrics(holdout, predictions)
    llm_metrics = llm_as_judge(holdout, predictions)
    return {
        "hard_metrics": hard_metrics,
        "llm_metrics": llm_metrics
    }


if __name__ == "__main__":
    main()