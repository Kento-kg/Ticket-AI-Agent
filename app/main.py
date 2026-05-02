import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.classifier import _load as _load_classifier
from agent.graph import build_graph, run_triage
from agent.schemas import TicketTriageResult

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm el clasificador y el grafo para evitar cold start en la primera request
    _load_classifier()
    build_graph()
    yield


app = FastAPI(title="Ticket Triage Agent", lifespan=lifespan)


class TriageRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TicketTriageResult)
def triage(request: TriageRequest) -> TicketTriageResult:
    """Clasifica el ticket, evalúa urgencia, busca similares y genera draft o escalación."""
    logger.info(f"Triage request: {request.text[:80]!r}")
    try:
        return run_triage(request.text)
    except Exception as e:
        logger.exception("Triage failed")
        raise HTTPException(status_code=500, detail=str(e))
