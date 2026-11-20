import os
import random
from typing import Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph


load_dotenv()

os.environ.setdefault("LANGSMITH_PROJECT", "langgraph-agentic-systems")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
else:
    os.environ["LANGSMITH_TRACING"] = "false"

class State(TypedDict):
    graph_state: str

def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" yo"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" you!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" me!"}

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()


def run_agent(input_text: str = "Hi, this is Lance.") -> dict:
    """Run the LangGraph agent and send its trace to LangSmith."""
    return graph.invoke(
        {"graph_state": input_text},
        config={
            "tags": ["langgraph-agentic-systems"],
            "metadata": {
                "agent_id": os.getenv("LANGSMITH_AGENT_ID", "agent"),
                "environment": os.getenv(
                    "LANGSMITH_AGENT_ENVIRONMENT",
                    "development",
                ),
            },
        },
    )


if __name__ == "__main__":
    print(run_agent())