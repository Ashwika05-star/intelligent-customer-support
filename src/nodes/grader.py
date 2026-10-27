from typing import Dict, Any
from src.state import SupportAgentState, ExecutionStep
from src.llm import get_llm_service
from src.config import CONFIDENCE_THRESHOLD

GRADER_SYSTEM_PROMPT = """You are an objective document relevance evaluator for a RAG support system.
Your job is to assess whether the retrieved context passages contain enough facts to answer the user query.

Output a valid JSON object with:
1. "is_relevant": true or false
2. "confidence_score": float between 0.0 and 1.0
3. "reasoning": brief 1-sentence evaluation of whether the text directly answers the question
"""

def grade_documents(state: SupportAgentState) -> Dict[str, Any]:
    """Node: Evaluates retrieved documents for factual relevance (Corrective RAG)."""
    query = state.get("rewritten_query") or state.get("raw_query", "")
    docs = state.get("retrieved_docs", [])
    
    # If no documents retrieved at all
    if not docs:
        step: ExecutionStep = {
            "node": "document_grader",
            "title": "Document Relevance Grader",
            "details": {
                "is_relevant": False,
                "confidence_score": 0.0,
                "reasoning": "No documents were found in the knowledge base."
            },
            "status": "warning"
        }
        steps = list(state.get("execution_steps", []))
        steps.append(step)
        return {
            "grader_result": {"is_relevant": False, "confidence_score": 0.0, "reasoning": "No documents found."},
            "confidence_score": 0.0,
            "retry_count": state.get("retry_count", 0) + 1,
            "execution_steps": steps
        }

    # If this was an order record from mock tools, it's inherently 100% relevant
    if docs and docs[0].get("category") == "order_record":
        step: ExecutionStep = {
            "node": "document_grader",
            "title": "Document Relevance Grader",
            "details": {
                "is_relevant": True,
                "confidence_score": 1.0,
                "reasoning": "Verified live operational database record."
            },
            "status": "completed"
        }
        steps = list(state.get("execution_steps", []))
        steps.append(step)
        return {
            "grader_result": {"is_relevant": True, "confidence_score": 1.0, "reasoning": "Verified database record."},
            "confidence_score": 1.0,
            "execution_steps": steps
        }

    llm = get_llm_service()
    is_relevant = False
    confidence = 0.5
    reasoning = "Evaluated via similarity threshold."

    # Combine top document snippets
    context_text = "\n\n".join([f"[{d['title']} - {d['section']}]:\n{d['content']}" for d in docs[:2]])

    if llm.is_configured():
        prompt = (
            f"User Question: \"{query}\"\n\n"
            f"Retrieved Document Context:\n{context_text}\n\n"
            "Evaluate if the context contains information to answer the question:"
        )
        res = llm.generate_json(prompt, system_instruction=GRADER_SYSTEM_PROMPT)
        if res and "is_relevant" in res:
            is_relevant = bool(res.get("is_relevant"))
            confidence = float(res.get("confidence_score", 0.5))
            reasoning = res.get("reasoning", reasoning)
    else:
        # Fallback heuristic: check if top chunk similarity score meets threshold
        top_score = docs[0].get("score", 0.0) if docs else 0.0
        if top_score >= CONFIDENCE_THRESHOLD:
            is_relevant = True
            confidence = top_score
            reasoning = f"Chunk score {top_score:.2f} met the confidence threshold of {CONFIDENCE_THRESHOLD}."
        else:
            is_relevant = False
            confidence = top_score
            reasoning = f"Chunk score {top_score:.2f} was below the required threshold of {CONFIDENCE_THRESHOLD}."

    step: ExecutionStep = {
        "node": "document_grader",
        "title": "Document Relevance Grader (CRAG)",
        "details": {
            "is_relevant": is_relevant,
            "confidence_score": round(confidence, 2),
            "reasoning": reasoning
        },
        "status": "completed" if is_relevant else "warning"
    }

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    current_retry = state.get("retry_count", 0)
    new_retry = current_retry + (0 if is_relevant else 1)

    return {
        "grader_result": {"is_relevant": is_relevant, "confidence_score": confidence, "reasoning": reasoning},
        "confidence_score": confidence,
        "retry_count": new_retry,
        "execution_steps": steps
    }
