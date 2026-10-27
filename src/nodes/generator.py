from typing import Dict, Any, List
from src.state import SupportAgentState, ExecutionStep
from src.llm import get_llm_service

GENERATOR_SYSTEM_PROMPT = """You are NovaMart's friendly, professional AI Customer Support Assistant.
Your mission is to provide accurate, clear, and reassuring answers based SOLELY on the verified documents provided below.

Guidelines:
1. Grounding: Answer strictly using facts from the provided context. Never speculate, invent store rules, or guess product specs.
2. Tone: Helpful, concise, and courteous.
3. Citations: At the end of your answer, list the exact sources used in the format:
   **Sources:**
   - [Document Title - Section Name]
4. If an order status or return check is provided, clearly state the current status, dates, and next steps for the customer.
"""

CHITCHAT_SYSTEM_PROMPT = """You are NovaMart's AI Support Assistant.
Respond warmly and briefly to general greetings, thanks, or goodbyes.
Remind the customer that you can help with product manuals, return/refund policies, shipping times, or tracking existing orders.
"""

def generate_response(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Grounded answer generator with markdown citations."""
    query = state.get("raw_query", "")
    rewritten = state.get("rewritten_query") or query
    docs = state.get("retrieved_docs", [])
    history = state.get("messages", [])

    llm = get_llm_service()
    citations: List[str] = []

    # Compile context and citations
    context_blocks = []
    for d in docs:
        cite_str = f"{d['title']} - {d['section']}"
        if cite_str not in citations:
            citations.append(cite_str)
        context_blocks.append(f"Source: [{cite_str}]\n{d['content']}")

    full_context = "\n\n---\n\n".join(context_blocks)

    if llm.is_configured():
        history_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-3:]])
        prompt = (
            f"Conversation History:\n{history_text}\n\n"
            f"User Question: {query}\n"
            f"Understood Subject: {rewritten}\n\n"
            f"Verified Company Context:\n{full_context}\n\n"
            "Please provide the support response with citations:"
        )
        response_text = llm.generate(prompt, system_instruction=GENERATOR_SYSTEM_PROMPT, temperature=0.2)
    else:
        # High-quality fallback template when offline
        if docs and docs[0].get("category") == "order_record":
            response_text = (
                f"Here is the verified information for your request:\n\n"
                f"{docs[0]['content']}\n\n"
                f"If you need further assistance with this order, please let me know!"
            )
        elif docs:
            top_doc = docs[0]
            response_text = (
                f"Based on our official **{top_doc['title']}** ({top_doc['section']}):\n\n"
                f"{top_doc['content'][:400]}...\n\n"
                f"**Sources:**\n" + "\n".join([f"- [{c}]" for c in citations])
            )
        else:
            response_text = "I reviewed our records but could not find specific documentation for your request."

    step: ExecutionStep = {
        "node": "response_generator",
        "title": "Grounded Response Generation",
        "details": {
            "sources_cited": len(citations),
            "citations": citations,
            "char_count": len(response_text)
        },
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "response": response_text,
        "citations": citations,
        "execution_steps": steps
    }

def handle_chitchat(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Handles greetings, pleasantries, and polite closings."""
    query = state.get("raw_query", "")
    llm = get_llm_service()

    if llm.is_configured():
        response_text = llm.generate(query, system_instruction=CHITCHAT_SYSTEM_PROMPT, temperature=0.7)
    else:
        query_lower = query.lower()
        if any(w in query_lower for w in ["thank", "thanks"]):
            response_text = "You're very welcome! Feel free to ask if you have questions about NovaMart products, shipping, returns, or order tracking. Have a great day!"
        else:
            response_text = "Hello! Welcome to NovaMart Support. I'm your AI assistant. How can I help you today? You can ask me about product specs, return & refund policies, shipping times, or track an existing order with your Order ID!"

    step: ExecutionStep = {
        "node": "chitchat_node",
        "title": "Conversational Greeting / Chitchat",
        "details": {"response_length": len(response_text)},
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "response": response_text,
        "citations": [],
        "confidence_score": 1.0,
        "execution_steps": steps
    }
