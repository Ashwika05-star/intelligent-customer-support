import time
from src.graph import get_support_graph

print("1. Loading graph...")
t0 = time.time()
graph = get_support_graph()
print(f"Graph loaded in {time.time() - t0:.2f}s")

test_queries = [
    "What is the return policy window?",
    "Where is my order ORD-98214?",
    "I demand to speak to a real human manager right now!"
]

for q in test_queries:
    print(f"\n--- Testing Query: '{q}' ---")
    state = {
        "messages": [{"role": "user", "content": q}],
        "raw_query": q,
        "rewritten_query": "",
        "category": "",
        "sentiment": "neutral",
        "order_id": None,
        "retrieved_docs": [],
        "grader_result": None,
        "confidence_score": 0.0,
        "retry_count": 0,
        "response": "",
        "citations": [],
        "is_escalated": False,
        "escalation_reason": None,
        "ticket_id": None,
        "execution_steps": []
    }
    t_start = time.time()
    res = graph.invoke(state, config={"configurable": {"thread_id": "test_thread"}})
    elapsed = time.time() - t_start
    print(f"Executed in {elapsed:.2f}s")
    print("Category:", res.get("category"))
    print("Is Escalated:", res.get("is_escalated"))
    print("Citations:", res.get("citations"))
    print("Steps:", [s["node"] for s in res.get("execution_steps", [])])
    print("Response Preview:", res.get("response", "")[:120], "...\n")
