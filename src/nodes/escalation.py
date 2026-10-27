from typing import Dict, Any
from src.state import SupportAgentState, ExecutionStep
from src.tools import create_escalation_ticket

def escalate_to_human(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Human-in-the-Loop Escalation and Ticket Generation."""
    query = state.get("raw_query", "")
    sentiment = state.get("sentiment", "neutral")
    order_id = state.get("order_id")
    category = state.get("category", "unresolved")
    retry_count = state.get("retry_count", 0)
    grader_result = state.get("grader_result") or {}

    # Determine reason
    if category == "human_escalation":
        reason = "Customer explicitly requested a human specialist or expressed frustration."
    elif retry_count > 0 or not grader_result.get("is_relevant"):
        reason = f"Automated knowledge retrieval failed to find verified documentation (Confidence: {state.get('confidence_score', 0.0):.2f})."
    else:
        reason = state.get("escalation_reason", "General support escalation required.")

    ticket_id = create_escalation_ticket(
        customer_query=query,
        reason=reason,
        sentiment=sentiment,
        order_id=order_id
    )

    escalation_message = (
        f"🚨 **Your inquiry has been escalated to our Human Support Specialist Team.**\n\n"
        f"A priority ticket has been created on your behalf:\n"
        f"* **Ticket ID:** `{ticket_id}`\n"
        f"* **Status:** Assigned to Live Human Agent\n"
        f"* **Escalation Reason:** {reason}\n"
        f"*(A representative has received your full chat transcript and will respond directly shortly.)*"
    )

    step: ExecutionStep = {
        "node": "escalate_to_human",
        "title": "Human Support Escalation (HITL)",
        "details": {
            "ticket_id": ticket_id,
            "escalation_reason": reason,
            "customer_sentiment": sentiment,
            "order_id": order_id or "N/A"
        },
        "status": "warning"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "is_escalated": True,
        "escalation_reason": reason,
        "ticket_id": ticket_id,
        "response": escalation_message,
        "citations": [],
        "execution_steps": steps
    }
