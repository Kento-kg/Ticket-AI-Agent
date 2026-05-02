import logging
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.config import cfg
from agent.prompts import EXTRACT_OUTPUT_PROMPT, SYSTEM_PROMPT
from agent.schemas import TicketTriageResult
from agent.tools import TOOLS

logger = logging.getLogger(__name__)


class TriageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_text: str
    final_output: TicketTriageResult | None


def _agent_node(state: TriageState) -> dict:
    """Invoke Claude with bound tools. The LLM decides which tool to call next
    or stops by emitting an AIMessage with no tool_calls."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0).bind_tools(TOOLS)
    return {"messages": [llm.invoke(state["messages"])]}


def _should_continue(state: TriageState) -> str:
    """Route to tools if the agent requested any, else to extract_output."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "extract_output"


def _extract_output_node(state: TriageState) -> dict:
    """Final node: distill the conversation into a Pydantic TicketTriageResult."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0).with_structured_output(
        TicketTriageResult
    )
    extraction_messages = list(state["messages"]) + [
        HumanMessage(content=EXTRACT_OUTPUT_PROMPT)
    ]
    result: TicketTriageResult = llm.invoke(extraction_messages)
    return {"final_output": result}


@lru_cache(maxsize=1)
def build_graph():
    """Compile the StateGraph once and cache it for the process lifetime."""
    graph = StateGraph(TriageState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("extract_output", _extract_output_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "extract_output": "extract_output"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("extract_output", END)
    return graph.compile()


def run_triage(text: str) -> TicketTriageResult:
    """Run the full triage flow on a single ticket. Returns the structured result."""
    graph = build_graph()
    initial_state: TriageState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=text),
        ],
        "ticket_text": text,
        "final_output": None,
    }
    result = graph.invoke(initial_state)
    return result["final_output"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    text = sys.argv[1] if len(sys.argv) > 1 else "el wifi se desconecta cada 10 min"
    output = run_triage(text)
    print(output.model_dump_json(indent=2))
