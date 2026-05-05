import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

from agent.classifier import classify
from agent.config import cfg
from agent.retrieval import search
from agent.schemas import EscalationSummary, UrgencyAssessment

logger = logging.getLogger(__name__)

@tool
def classify_ticket(text: str) -> dict:
    """Classify the ticket into one of: account, biling, general, technical.
    Returns a dict with category, model confidence and the responsible team.
    Low-confidence predictions (out-of-domain) are routed to 'general'."""
    return classify(text)

@tool
def assess_urgency(text: str) -> dict:
    """Assess the urgency of a ticket: low, medium, or high.

    Use 'high' for outages, security issues, blockers affecting many users,
    data loss, or production-critical problems.
    Use 'medium' for individual user blockers without broader impact.
    Use 'low' for general questions or minor inconveniences."""
    llm = ChatAnthropic(model=cfg.claude_haiku_model, temperature=0)
    structured = llm.with_structured_output(UrgencyAssessment)
    result: UrgencyAssessment = structured.invoke(
        [
            SystemMessage(
                content=(
                    "You assess IT support ticket urgency. Output 'low', 'medium' or 'high' based on impact and urgency cues in the ticket."
                )
            ),
            HumanMessage(content=text),
        ]
    )
    return result.model_dump()


@tool
def search_similar_tickets(
    text: str, category: str | None = None, k: int = 3
) -> list[dict]:
    """Retrieve the top-k most similar past tickets using semantic search.

    Filters by category via ChromaDB's native metadata filter when provided.
    Returns text, category, urgency, team, and similarity score for each."""
    return search(text, k=k, category=category)


@tool
def generate_draft_response(
    ticket_text: str,
    category: str,
    similar_tickets: list[str],
) -> str:
    """Generate a customer-facing draft reply for a non-critical ticket.

    similar_tickets: texts of similar resolved tickets, used as few-shot reference."""
    similar_block = "\n\n".join(f"- {t}" for t in similar_tickets) or "(none)"
    system = (
        "You write helpful, friendly customer-facing draft replies for IT support tickets.\n"
        f"Ticket category: {category}.\n\n"
        "Use the following similar past tickets only as reference for tone and structure "
        "(do not copy verbatim):\n\n"
        f"{similar_block}\n\n"
        "Keep the draft to 3-6 sentences and make it actionable."
    )
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0.3)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=ticket_text)])
    return str(response.content)


@tool
def escalate_ticket(
    ticket_text: str,
    category: str,
    urgency: str,
    team: str,
) -> dict:
    """Produce a structured escalation summary for a high-urgency ticket.

    Returns a dict with summary, urgency_reason, recommended_actions, oncall_team."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0)
    structured = llm.with_structured_output(EscalationSummary)
    result: EscalationSummary = structured.invoke(
        [
            SystemMessage(
                content=(
                    "You are escalating a high-urgency IT ticket to the on-call team. "
                    "Produce a structured summary with the issue, why it is urgent, "
                    "recommended next steps, and the responsible team."
                )
            ),
            HumanMessage(
                content=(
                    f"Category: {category}\n"
                    f"Urgency: {urgency}\n"
                    f"Assigned team: {team}\n\n"
                    f"Ticket:\n{ticket_text}"
                )
            ),
        ]
    )
    return result.model_dump()


TOOLS = [
    classify_ticket,
    assess_urgency,
    search_similar_tickets,
    generate_draft_response,
    escalate_ticket,
]
