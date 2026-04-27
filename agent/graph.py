from langgraph.graph import StateGraph, START
from langchain_core.messages import AnyMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from agent.schemas import TicketTriageResult
from agent.tools import TOOLS
from agent.prompts import STRUCTURED_OUTPUT, SYSTEM_PROMPT

class TriageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_text: str
    final_output: TicketTriageResult

def agent_node(state: TriageState):
    """Invoke Claude with tools. The LLM decides which tools to use depending on the state."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0).bind_tools(TOOLS)
    return {"messages": [llm.invoke([SYSTEM_PROMPT] + state["messages"])]}

def output(state: TriageState):
    """Generate the output in the TicketTriageResult PyDantic state."""
    llm = ChatAnthropic(model=cfg.claude_main_model, temperature=0).with_structured_output(TicketTriageResult)
    result = structured.invoke(
        state["messages"] + HumanMessage(content=STRUCTURED_OUTPUT)
    )
    return {"result": result}

graph = StateGraph(TriageState)
graph.add_node("agent", agent_node) 
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("node_3", output)

graph.add_edge(START, "node_1")
graph.add_conditional_edges(
    "agent",
    tools_condition
)
graph.add_edge("tools", "agent")

react_graph = builder.compile()
