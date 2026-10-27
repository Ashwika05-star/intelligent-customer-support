from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class RetrievedDocument(TypedDict):
    id: str
    title: str
    section: str
    category: str
    source_file: str
    content: str
    score: float

class ExecutionStep(TypedDict):
    node: str
    title: str
    details: Dict[str, Any]
    status: str

class SupportAgentState(TypedDict):
    # Conversation input & history
    messages: List[Dict[str, str]]
    raw_query: str
    rewritten_query: str
    
    # Classification & intent
    category: str
    sentiment: str
    order_id: Optional[str]
    
    # Retrieval & Quality Grading
    retrieved_docs: List[RetrievedDocument]
    grader_result: Optional[Dict[str, Any]]
    confidence_score: float
    retry_count: int
    
    # Response & Citations
    response: str
    citations: List[str]
    
    # Human Escalation (HITL)
    is_escalated: bool
    escalation_reason: Optional[str]
    ticket_id: Optional[str]
    
    # UI Traceability
    execution_steps: List[ExecutionStep]
