from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import SupportAgentState
from src.nodes.classifier import classify_query
from src.nodes.rewriter import rewrite_query
from src.nodes.retrieval_nodes import retrieve_policy, retrieve_product_manual, handle_order_actions
from src.nodes.grader import grade_documents
from src.nodes.generator import generate_response, handle_chitchat
from src.nodes.escalation import escalate_to_human
from src.config import MAX_RETRIES

# Conditional Routers

def route_after_classification(state: SupportAgentState) -> Literal["chitchat", "escalator", "handle_order", "rewriter"]:
    category = state.get("category", "policy_inquiry")
    sentiment = state.get("sentiment", "neutral")
    
    if category == "human_escalation" or sentiment == "frustrated":
        return "escalator"
    elif category == "general_chitchat":
        return "chitchat"
    elif category in ["order_tracking", "return_request"]:
        return "handle_order"
    else:
        return "rewriter"

def route_retrieval_domain(state: SupportAgentState) -> Literal["retrieve_policy", "retrieve_product"]:
    category = state.get("category", "policy_inquiry")
    if category == "product_inquiry":
        return "retrieve_product"
    return "retrieve_policy"

def route_after_grading(state: SupportAgentState) -> Literal["generator", "rewriter", "escalator"]:
    grader_result = state.get("grader_result") or {}
    is_relevant = grader_result.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)

    if is_relevant:
        return "generator"
    elif retry_count < 1:
        # Loop back once for CRAG self-correction
        return "rewriter"
    else:
        # Unresolved case routed to human support
        return "escalator"

def create_support_graph():
    """Build and compile the LangGraph workflow for customer support."""
    workflow = StateGraph(SupportAgentState)

    # Add Nodes
    workflow.add_node("classifier", classify_query)
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("retrieve_product", retrieve_product_manual)
    workflow.add_node("handle_order", handle_order_actions)
    workflow.add_node("grader", grade_documents)
    workflow.add_node("generator", generate_response)
    workflow.add_node("chitchat", handle_chitchat)
    workflow.add_node("escalator", escalate_to_human)

    # Connect Entry Point
    workflow.add_edge(START, "classifier")

    # Conditional Branching after Classification
    workflow.add_conditional_edges(
        "classifier",
        route_after_classification,
        {
            "chitchat": "chitchat",
            "escalator": "escalator",
            "handle_order": "handle_order",
            "rewriter": "rewriter"
        }
    )

    # Route Rewriter to Domain-Specific Retriever
    workflow.add_conditional_edges(
        "rewriter",
        route_retrieval_domain,
        {
            "retrieve_policy": "retrieve_policy",
            "retrieve_product": "retrieve_product"
        }
    )

    # Connect Retrievers & Tool Actions to Grader
    workflow.add_edge("retrieve_policy", "grader")
    workflow.add_edge("retrieve_product", "grader")
    workflow.add_edge("handle_order", "grader")

    # Conditional Branching after Grading (CRAG loop or Escalate)
    workflow.add_conditional_edges(
        "grader",
        route_after_grading,
        {
            "generator": "generator",
            "rewriter": "rewriter",
            "escalator": "escalator"
        }
    )

    # Terminal Edges
    workflow.add_edge("generator", END)
    workflow.add_edge("chitchat", END)
    workflow.add_edge("escalator", END)

    # In-memory checkpointer for conversational sessions & HITL
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

# Singleton compiled graph
_support_app = None

def get_support_graph():
    global _support_app
    if _support_app is None:
        _support_app = create_support_graph()
    return _support_app
