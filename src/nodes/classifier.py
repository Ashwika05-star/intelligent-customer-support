import re
from typing import Dict, Any
from src.state import SupportAgentState, ExecutionStep
from src.llm import get_llm_service
from src.config import CATEGORIES

SYSTEM_INSTRUCTION = """You are an expert customer support classifier for an e-commerce platform called NovaMart.
Your job is to analyze the customer's latest query along with conversation context and classify:
1. "category": Choose exactly ONE from:
   - "policy_inquiry": Questions about return windows, refund timelines, shipping rates, warranties, damaged goods policies.
   - "product_inquiry": Technical specifications, battery life, dimensions, setup instructions, error codes, troubleshooting for products (AuraSound headphones, ApexClean vacuum, ErgoComfort chair).
   - "order_tracking": Customer asking about the location, delivery date, carrier, or status of an existing order.
   - "return_request": Customer explicitly wanting to initiate a return or exchange for a purchased order.
   - "human_escalation": Customer demanding a human representative, expressing severe anger/threats, or reporting a critical unresolvable issue.
   - "general_chitchat": Greetings, thanks, farewells, or pleasantries.
2. "sentiment": Choose ONE: "positive", "neutral", "frustrated".
3. "order_id": Extract any mentioned Order ID (format typically ORD-XXXXX), or null if none.
4. "reasoning": A brief 1-sentence rationale.
"""

def extract_order_id_regex(text: str) -> str:
    match = re.search(r'\b(ORD-?\d{4,6})\b', text, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        if not val.startswith("ORD-"):
            val = "ORD-" + val.replace("ORD", "")
        return val
    return ""

def classify_query(state: SupportAgentState) -> Dict[str, Any]:
    """Node 1: Intent, Category, and Sentiment Classifier."""
    raw_query = state.get("raw_query", "")
    history = state.get("messages", [])
    
    # Check regex for order ID first
    detected_order_id = extract_order_id_regex(raw_query)

    llm = get_llm_service()
    category = "policy_inquiry"
    sentiment = "neutral"
    order_id = detected_order_id
    reasoning = "Classified based on query analysis."

    if llm.is_configured():
        context_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-3:]])
        prompt = f"Recent History:\n{context_str}\n\nLatest Customer Message: \"{raw_query}\""
        
        result = llm.generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
        if result and "category" in result:
            category = result.get("category", "policy_inquiry")
            sentiment = result.get("sentiment", "neutral")
            if not order_id and result.get("order_id"):
                order_id = str(result.get("order_id"))
            reasoning = result.get("reasoning", reasoning)
    else:
        # High quality rule-based heuristic fallback if API key not provided yet
        def has_word(words):
            return any(re.search(r'\b' + re.escape(w) + r'\b', raw_query, re.IGNORECASE) for w in words)

        if has_word(["human", "agent", "person", "representative", "manager", "lawyer", "speak to someone"]):
            category = "human_escalation"
            sentiment = "frustrated"
        elif has_word(["track", "tracking", "status", "where is my order", "shipped yet"]) or detected_order_id:
            category = "order_tracking"
        elif has_word(["start return", "want to return", "exchange my", "send back"]):
            category = "return_request"
        elif has_word(["spec", "specs", "specification", "battery", "bluetooth", "pair", "reset", "dimension", "dimensions", "vacuum", "headphone", "headphones", "chair", "hepa", "filter"]):
            category = "product_inquiry"
        elif has_word(["hello", "hi", "hey", "good morning", "good evening", "thank you", "thanks", "bye", "goodbye"]):
            category = "general_chitchat"
        else:
            category = "policy_inquiry"

        if has_word(["angry", "terrible", "worst", "ridiculous", "scam", "unacceptable", "furious", "hate"]):
            sentiment = "frustrated"

    step: ExecutionStep = {
        "node": "intent_classifier",
        "title": "Intent & Sentiment Classification",
        "details": {
            "category": category,
            "sentiment": sentiment,
            "order_id": order_id or "None",
            "reasoning": reasoning
        },
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "category": category,
        "sentiment": sentiment,
        "order_id": order_id if order_id else state.get("order_id"),
        "execution_steps": steps
    }
