"""
LangGraph pipeline for the Requirements Quality Analyzer.

Graph structure:
START → ambiguity → consistency → completeness → testability → reporter → validator
                                                                               ↓          ↓
                                                                           (invalid)   (valid)
                                                                               ↓          ↓
                                                                           reporter     END
"""

from langgraph.graph import StateGraph, START, END
from agents.ambiguity_agent import ambiguity_agent
from agents.consistency_agent import consistency_agent
from agents.completeness_agent import completeness_agent
from agents.testability_agent import testability_agent
from agents.reporter_agent import reporter_agent
from utils.models import GraphState


# ---------------------------------------------------------------------------
# Validator Node — conditional edge logic
# ---------------------------------------------------------------------------

def validator_node(state: GraphState) -> dict:
    """
    Checks if the final report was generated successfully.
    Returns a routing key used by the conditional edge.
    """
    if (
        state.final_report is None
        or len(state.final_report.strip()) < 50
    ):
        print("[Validator] Report is missing or too short. Routing back to reporter.")
        return {"final_report": None}  # reset so reporter reruns cleanly
    
    print("[Validator] Report looks good. Routing to END.")
    return {}


def route_after_validation(state: GraphState) -> str:
    """
    Routing function for the conditional edge after validator.
    Returns the name of the next node.
    """
    if state.final_report is None or len(state.final_report.strip()) < 50:
        return "reporter"  # retry
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    # Register all nodes
    graph.add_node("ambiguity", ambiguity_agent)
    graph.add_node("consistency", consistency_agent)
    graph.add_node("completeness", completeness_agent)
    graph.add_node("testability", testability_agent)
    graph.add_node("reporter", reporter_agent)
    graph.add_node("validator", validator_node)

    # Linear edges
    graph.add_edge(START, "ambiguity")
    graph.add_edge("ambiguity", "consistency")
    graph.add_edge("consistency", "completeness")
    graph.add_edge("completeness", "testability")
    graph.add_edge("testability", "reporter")
    graph.add_edge("reporter", "validator")

    # Conditional edge — the key LangGraph feature
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "reporter": "reporter",  # retry path
            END: END                 # success path
        }
    )

    return graph.compile()


# Compiled graph — imported by app.py
pipeline = build_graph()