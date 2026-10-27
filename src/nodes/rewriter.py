from typing import Dict, Any
from src.state import SupportAgentState, ExecutionStep
from src.llm import get_llm_service

REWRITE_SYSTEM_PROMPT = """You are a Contextual Query Rewriter for an e-commerce support system.
Your job is to examine the conversation history and the user's latest query, and rewrite it into a clear, fully self-contained search query suitable for semantic vector retrieval.

Rules:
1. Resolve any pronouns like "it", "they", "this", "that product" using the prior messages.
   - Example: Prior: "Tell me about the AuraSound Pro headphones." Latest: "Is it waterproof?" -> Rewritten: "Is the AuraSound Pro headphones water resistant or waterproof?"
2. If the user query is already self-contained and clear, keep it essentially as-is.
3. Keep the query concise and focused on the key retrieval terms (product name, feature, or policy term).
4. Output ONLY the rewritten query text. Do not add explanations, quotes, or preambles.
"""

def rewrite_query(state: SupportAgentState) -> Dict[str, Any]:
    """Node 2: Contextual Query Rewriter for handling follow-up turns."""
    raw_query = state.get("raw_query", "")
    history = state.get("messages", [])
    retry_count = state.get("retry_count", 0)

    # If there is no previous history and this isn't a retry, raw_query is sufficient
    if len(history) <= 1 and retry_count == 0:
        rewritten = raw_query
    else:
        llm = get_llm_service()
        if llm.is_configured():
            history_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-4:]])
            prompt = (
                f"Conversation History:\n{history_text}\n\n"
                f"Latest Query: {raw_query}\n"
                f"Retry Count: {retry_count}\n"
                f"Rewrite this query into a standalone search query:"
            )
            generated = llm.generate(prompt, system_instruction=REWRITE_SYSTEM_PROMPT, temperature=0.1)
            rewritten = generated.strip().strip('"') if generated else raw_query
        else:
            # Fallback heuristic: look for product names in past assistant messages
            rewritten = raw_query
            past_text = " ".join([m.get("content", "") for m in history[-2:]]).lower()
            if "aurasound" in past_text and "aurasound" not in raw_query.lower():
                rewritten = f"AuraSound Pro AS-900 {raw_query}"
            elif "apexclean" in past_text and "apexclean" not in raw_query.lower():
                rewritten = f"ApexClean X9 robot vacuum {raw_query}"
            elif "ergocomfort" in past_text and "ergocomfort" not in raw_query.lower():
                rewritten = f"ErgoComfort Max chair {raw_query}"

    step: ExecutionStep = {
        "node": "query_rewriter",
        "title": "Contextual Query Rewriter",
        "details": {
            "original_query": raw_query,
            "rewritten_query": rewritten,
            "resolved_followup": rewritten != raw_query
        },
        "status": "completed"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return {
        "rewritten_query": rewritten,
        "execution_steps": steps
    }
