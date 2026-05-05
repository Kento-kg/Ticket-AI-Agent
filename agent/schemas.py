from typing import Literal

from pydantic import BaseModel, Field

CategoryLiteral = Literal["account", "biling", "general", "technical"]
UrgencyLiteral = Literal["low", "medium", "high"]
TeamLiteral = Literal[
    "account-team", "biling-team", "general-support", "tech-support"
]


class UrgencyAssessment(BaseModel):
    urgency: UrgencyLiteral = Field(description="Assessed urgency level.")
    reasoning: str = Field(description="Brief justification for the assigned level.")


class EscalationSummary(BaseModel):
    summary: str = Field(description="One- or two-sentence summary of the issue.")
    urgency_reason: str = Field(description="Why this ticket is urgent.")
    recommended_actions: list[str] = Field(
        description="Two or three concrete next steps for the on-call team."
    )
    oncall_team: str = Field(description="Team that should handle the escalation.")


class SimilarTicket(BaseModel):
    text: str
    category: str
    urgency: str
    team: str
    score: float


class TicketTriageResult(BaseModel):
    category: CategoryLiteral
    urgency: UrgencyLiteral
    team: TeamLiteral
    similar_tickets: list[SimilarTicket]
    action: Literal["draft", "escalate"]
    response: str = Field(
        description="Draft response (action=draft) or escalation summary (action=escalate)."
    )
