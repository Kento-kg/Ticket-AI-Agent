from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from agent.config import cfg
from agent.schemas import TicketTriageResult
from agent.tools import TOOLS
from agent.prompts import STRUCTURED_OUTPUT, SYSTEM_PROMPT

class TriageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_text: str
    final_output: TicketTriageResult

def agent_node(state: TriageState) -> dict:
    """Invoke Claude with bound tools. The LLM decides which tool to call next
    or stops by emitting an AIMessage with no tool_calls."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0).bind_tools(TOOLS)
    return {"messages": [llm.invoke(state["messages"])]}

def output(state: TriageState) -> dict:
    """Final node: output the conversation into a Pydantic TicketTriageResult."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0).with_structured_output(TicketTriageResult)
    result = llm.invoke(
        state["messages"] + HumanMessage(content=STRUCTURED_OUTPUT)
    )
    return {"final_output": result}

def should_continue(state: TriageState):
    """Route to 'tools' if the agent requested any, else to 'extract_output'."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "usar_tool"
    else:
        return "paso_final"

@lru_cache(maxsize=1)
def build_graph():
    """Compile the StateGraph once and cache it for the process lifetime."""
    graph = StateGraph(TriageState)

    graph.add_node("agent", agent_node) 
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("extract_output", output)

    graph.add_edge(START, "node_1")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"usar_tools": "tools", "paso_final": "extract_output"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("extract_output", END)
    return graph.compile()

def run_ticket_triage(text: str) -> TicketTriageResult:
    """Run the full triage flow on a single ticket. Returns the structured result."""
    graph = build_graph()
    initial_state: TriageState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)],
        "ticket_text": text,
        "final_output": None
    }
    result = graph.invoke(initial_state)
    return result["final_output"]

if __name__ == "__main__":
    result = run_ticket_triage(text)
    print(output.model_dump_json)