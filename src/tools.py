import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.config import ORDERS_FILE, TICKETS_FILE

def load_orders() -> Dict[str, Any]:
    if not ORDERS_FILE.exists():
        return {}
    try:
        with open(ORDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def lookup_order(order_id: str) -> Dict[str, Any]:
    """Look up order details from the mock database."""
    orders = load_orders()
    clean_id = order_id.strip().upper()
    
    if clean_id in orders:
        order = orders[clean_id]
        return {
            "found": True,
            "order": order,
            "message": f"Order {clean_id} found. Status: {order.get('status')}."
        }
    
    return {
        "found": False,
        "order": None,
        "message": f"Order ID '{clean_id}' was not found in our records."
    }

def check_return_eligibility(order_id: str) -> Dict[str, Any]:
    """Evaluate return eligibility based on delivery status and 30-day window."""
    order_data = lookup_order(order_id)
    if not order_data["found"]:
        return {
            "eligible": False,
            "reason": f"Order {order_id} could not be located in our system.",
            "details": None
        }

    order = order_data["order"]
    if order.get("return_eligible"):
        return {
            "eligible": True,
            "reason": f"Order is within the standard 30-day return window. (Delivered on {order.get('delivery_date')}, deadline is {order.get('return_window_deadline')}).",
            "details": order
        }
    else:
        return {
            "eligible": False,
            "reason": order.get("return_ineligibility_reason", "Item does not meet current return policy criteria."),
            "details": order
        }

def load_tickets() -> List[Dict[str, Any]]:
    if not TICKETS_FILE.exists():
        return []
    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_tickets(tickets: List[Dict[str, Any]]):
    TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)

def create_escalation_ticket(
    customer_query: str,
    reason: str,
    sentiment: str = "neutral",
    order_id: Optional[str] = None
) -> str:
    """Creates a human support ticket and persists it to JSON."""
    ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    tickets = load_tickets()
    
    new_ticket = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_query": customer_query,
        "reason": reason,
        "sentiment": sentiment,
        "order_id": order_id,
        "status": "OPEN",
        "agent_response": None
    }
    
    tickets.insert(0, new_ticket)
    save_tickets(tickets)
    return ticket_id

def resolve_ticket(ticket_id: str, agent_response: str) -> bool:
    """Updates an escalated ticket with human agent notes/response."""
    tickets = load_tickets()
    found = False
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            t["status"] = "RESOLVED"
            t["agent_response"] = agent_response
            t["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if found:
        save_tickets(tickets)
    return found
