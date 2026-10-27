from typing import Dict, Any, List
from src.state import SupportAgentState, ExecutionStep, RetrievedDocument
from src.retriever import get_retriever
from src.tools import lookup_order, check_return_eligibility

def retrieve_policy(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Retrieve relevant documents from company policy manuals."""
    query = state.get("rewritten_query") or state.get("raw_query", "")
    retriever = get_retriever()
    docs = retriever.search(query, category="policy", top_k=3)

    avg_score = sum(d["score"] for d in docs) / len(docs) if docs else 0.0

    step: ExecutionStep = {
        "node": "retrieve_policy",
        "title": "Policy Document Retrieval",
        "details": {
            "search_query": query,
            "chunks_found": len(docs),
            "top_sources": [f"{d['title']} - {d['section']} (Score: {d['score']})" for d in docs]
        },
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "retrieved_docs": docs,
        "confidence_score": avg_score,
        "execution_steps": steps
    }

def retrieve_product_manual(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Retrieve relevant sections from product manuals & specifications."""
    query = state.get("rewritten_query") or state.get("raw_query", "")
    retriever = get_retriever()
    docs = retriever.search(query, category="manual", top_k=3)

    avg_score = sum(d["score"] for d in docs) / len(docs) if docs else 0.0

    step: ExecutionStep = {
        "node": "retrieve_product_manual",
        "title": "Product Manual Retrieval",
        "details": {
            "search_query": query,
            "chunks_found": len(docs),
            "top_sources": [f"{d['title']} - {d['section']} (Score: {d['score']})" for d in docs]
        },
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "retrieved_docs": docs,
        "confidence_score": avg_score,
        "execution_steps": steps
    }

def handle_order_actions(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Executes transactional mock tools for order tracking or returns."""
    category = state.get("category", "")
    order_id = state.get("order_id")
    raw_query = state.get("raw_query", "")

    tool_result = {}
    docs: List[RetrievedDocument] = []

    if not order_id:
        tool_result = {
            "action": "missing_order_id",
            "message": "Please provide your Order ID (e.g., ORD-98214 or ORD-76150) so I can look up the details for you."
        }
    elif category == "order_tracking":
        res = lookup_order(order_id)
        tool_result = res
        if res["found"]:
            ord_info = res["order"]
            doc_content = (
                f"Order ID: {ord_info['order_id']}\n"
                f"Status: {ord_info['status']}\n"
                f"Carrier: {ord_info.get('carrier', 'N/A')}\n"
                f"Tracking Number: {ord_info.get('tracking_number', 'N/A')}\n"
                f"Estimated / Actual Delivery: {ord_info.get('estimated_delivery') or ord_info.get('delivery_date')}\n"
                f"Shipping Address: {ord_info.get('shipping_address')}\n"
                f"Items: {', '.join([i['product_name'] for i in ord_info.get('items', [])])}"
            )
            docs.append({
                "id": f"order_{order_id}",
                "title": f"Order {order_id} Details",
                "section": "Live Order Status",
                "category": "order_record",
                "source_file": "orders_db.json",
                "content": doc_content,
                "score": 1.0
            })
    elif category == "return_request":
        res = check_return_eligibility(order_id)
        tool_result = res
        ord_info = res.get("details") or {}
        doc_content = (
            f"Order ID: {order_id}\n"
            f"Return Eligible: {res['eligible']}\n"
            f"Reasoning: {res['reason']}\n"
            f"Items: {', '.join([i['product_name'] for i in ord_info.get('items', [])]) if ord_info else 'N/A'}"
        )
        docs.append({
            "id": f"return_check_{order_id}",
            "title": f"Order {order_id} Return Check",
            "section": "Eligibility Decision",
            "category": "order_record",
            "source_file": "orders_db.json",
            "content": doc_content,
            "score": 1.0
        })

    step: ExecutionStep = {
        "node": "handle_order_actions",
        "title": "E-Commerce Tool Execution",
        "details": {
            "order_id": order_id or "Not provided",
            "tool_action": category,
            "tool_summary": tool_result.get("message") or tool_result.get("reason", "Action processed.")
        },
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "retrieved_docs": docs,
        "confidence_score": 1.0 if docs else 0.4,
        "execution_steps": steps
    }
