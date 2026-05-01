from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from agent.schemas import TicketTriageResult
app = FastAPI()


class Ticket(BaseModel):
    id: int = Field(alias = "")
    text: str = Field(alias = "Body")
    team: str = Field(alias = "Department")
    urgency: str = Field(alias = "Priority")
    tags: list[str] = Field(alias = "Tags")

@app.get("/")
def home():
    return {"health_check": "OK"}
