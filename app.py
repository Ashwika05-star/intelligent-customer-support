import os
import streamlit as st
import json
from pathlib import Path
from dotenv import load_dotenv

# Load local environment if available
load_dotenv()

from src.graph import get_support_graph
from src.llm import get_llm_service
from src.tools import load_tickets, resolve_ticket, load_orders
from src.config import DEFAULT_GEMINI_MODEL, CATEGORIES

# Page Configuration
st.set_page_config(
    page_title="NovaMart Support Agent | LangGraph RAG",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
        margin-bottom: 8px;
    }
    .citation-tag {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0px;
    }
    .step-box {
        border-left: 3px solid #4f46e5;
        padding-left: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_steps" not in st.session_state:
    st.session_state.last_steps = []
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = "customer_session_001"

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ Agent Controls")
    
    # API Key Configuration
    st.markdown("### 🔑 LLM Provider")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Paste your Google AI Studio Gemini API Key here. If blank, smart offline fallback rules apply."
    )
    
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        llm = get_llm_service(api_key=api_key_input)
        if llm.is_configured():
            st.success("🟢 Gemini API: Connected", icon="✅")
        else:
            st.warning("🟡 Gemini API: Initializing...")
    else:
        st.info("ℹ️ Running in **Smart Simulation Mode** (No API Key). Enter an API key above for full LLM generation.", icon="💡")

    st.markdown("---")
    
    # Quick Test Presets
    st.markdown("### 🧪 Quick Test Scenarios")
    quick_queries = [
        ("📦 Policy Query", "What is the return window and how much is the return shipping fee?"),
        ("🎧 Product Specs", "What is the battery life and water resistance of the AuraSound Pro headphones?"),
        ("🔄 Multi-turn Follow-up", "Does it support multipoint Bluetooth pairing?"),
        ("🚚 Track Order (In Transit)", "Where is my order ORD-98214?"),
        ("🏷️ Return Request (Eligible)", "I want to return my order ORD-76150."),
        ("🚨 Human Escalation", "This service is terrible and I demand to speak to a human manager immediately!")
    ]

    for label, query_text in quick_queries:
        if st.button(label, use_container_width=True):
            st.session_state.current_prompt = query_text

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_steps = []
        st.rerun()

# ----------------- MAIN INTERFACE -----------------
st.title("🛍️ NovaMart Intelligent Support Assistant")
st.caption("Powered by **LangGraph State Machine**, **Self-Corrective RAG (CRAG)**, and **Human-in-the-Loop Escalation**")

tab_chat, tab_graph, tab_human = st.tabs([
    "💬 Customer Support Chat",
    "🧠 LangGraph Workflow & State Inspector",
    "👨‍💼 Human Agent Desk (HITL)"
])

# ----------------- TAB 1: CUSTOMER CHAT -----------------
with tab_chat:
    # Display message history
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            
            # Show citations if present
            if msg.get("citations"):
                with st.expander("📚 Verified Sources Cited"):
                    for c in msg["citations"]:
                        st.markdown(f"<span class='citation-tag'>📄 {c}</span>", unsafe_allow_html=True)
            
            # Show escalation alert if present
            if msg.get("is_escalated"):
                st.warning(f"⚠️ **Escalated to Human Support** | Ticket: `{msg.get('ticket_id', 'N/A')}`")

    # Chat input
    prompt_to_send = None
    if "current_prompt" in st.session_state and st.session_state.current_prompt:
        prompt_to_send = st.session_state.current_prompt
        st.session_state.current_prompt = None
    else:
        chat_val = st.chat_input("Type your question (e.g., 'How long does shipping take?' or 'Where is ORD-98214?')...")
        if chat_val:
            prompt_to_send = chat_val

    if prompt_to_send:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Prepare Graph State
        graph = get_support_graph()
        thread_config = {"configurable": {"thread_id": st.session_state.session_thread_id}}

        initial_state = {
            "messages": st.session_state.messages,
            "raw_query": prompt_to_send,
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

        with st.chat_message("assistant"):
            with st.spinner("Agent orchestrating LangGraph workflow..."):
                final_state = graph.invoke(initial_state, config=thread_config)
                
                resp_text = final_state.get("response", "I'm sorry, I could not process that request.")
                citations = final_state.get("citations", [])
                is_escalated = final_state.get("is_escalated", False)
                ticket_id = final_state.get("ticket_id")

                st.markdown(resp_text)
                
                if citations:
                    with st.expander("📚 Verified Sources Cited"):
                        for c in citations:
                            st.markdown(f"<span class='citation-tag'>📄 {c}</span>", unsafe_allow_html=True)
                
                if is_escalated:
                    st.warning(f"⚠️ **Escalated to Human Support** | Ticket ID: `{ticket_id}`")

                # Store assistant response and execution steps
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp_text,
                    "citations": citations,
                    "is_escalated": is_escalated,
                    "ticket_id": ticket_id
                })
                st.session_state.last_steps = final_state.get("execution_steps", [])
                st.rerun()

# ----------------- TAB 2: LANGGRAPH WORKFLOW INSPECTOR -----------------
with tab_graph:
    st.subheader("🔍 Real-time LangGraph Execution Trace")
    st.markdown("Inspect how the LangGraph State Machine routed the query, rewrote multi-turn context, retrieved documents, and graded relevance.")

    if not st.session_state.last_steps:
        st.info("No query has been executed yet. Send a message in the Chat tab or click a Quick Test Scenario to see the live graph trace.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📋 Step-by-Step Node Execution")
            for idx, step in enumerate(st.session_state.last_steps):
                status_icon = "✅" if step["status"] == "completed" else "⚠️"
                with st.expander(f"{status_icon} Step {idx+1}: **{step['title']}** (`{step['node']}`)", expanded=True):
                    for k, v in step.get("details", {}).items():
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** `{v}`")

        with col2:
            st.markdown("### 🕸️ LangGraph Architecture Diagram")
            st.markdown("""
```mermaid
flowchart TD
    Start([User Input]) --> Classifier[Intent & Sentiment Classifier]
    
    Classifier -->|Chitchat| Chitchat[Conversational Node]
    Classifier -->|Escalation / Anger| Escalator[Human Escalation HITL]
    Classifier -->|Order / Return| OrderAction[Order DB Tool]
    Classifier -->|Product / Policy| Rewriter[Contextual Query Rewriter]
    
    Rewriter --> RouteDomain{Domain Router}
    RouteDomain -->|Policy Inquiry| RetPolicy[Policy Retriever]
    RouteDomain -->|Product Specs| RetProduct[Product Manual Retriever]
    
    RetPolicy --> Grader[Document Relevance Grader CRAG]
    RetProduct --> Grader
    OrderAction --> Grader
    
    Grader -->|Relevant & Confident| Generator[Grounded Generator + Citations]
    Grader -->|Low Relevance / Missing| RetryCheck{Retry < 1?}
    
    RetryCheck -->|Yes| Rewriter
    RetryCheck -->|No| Escalator
    
    Generator --> End([Customer Response])
    Chitchat --> End
    Escalator --> End
```
            """)

# ----------------- TAB 3: HUMAN AGENT DESK (HITL) -----------------
with tab_human:
    st.subheader("👨‍💼 Human Support Escalation Portal")
    st.markdown("Unresolved queries, low-confidence retrievals, and frustrated customer messages are automatically placed in this priority inbox.")

    tickets = load_tickets()
    
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        st.metric("Total Escalated Tickets", len(tickets))
        open_count = sum(1 for t in tickets if t.get("status") == "OPEN")
        st.metric("Open Pending Reviews", open_count)

    with col_t2:
        if not tickets:
            st.success("🎉 No escalated tickets currently pending! The automated agent is successfully resolving incoming queries.")
        else:
            for ticket in tickets:
                status_color = "🔴" if ticket.get("status") == "OPEN" else "🟢"
                with st.expander(f"{status_color} Ticket `{ticket['ticket_id']}` - Status: **{ticket.get('status')}** ({ticket.get('created_at')})", expanded=(ticket.get('status') == "OPEN")):
                    st.markdown(f"**Customer Query:** *\"{ticket.get('customer_query')}\"*")
                    st.markdown(f"**Escalation Reason:** {ticket.get('reason')}")
                    st.markdown(f"**Sentiment Detected:** `{ticket.get('sentiment')}`")
                    if ticket.get("order_id"):
                        st.markdown(f"**Related Order:** `{ticket.get('order_id')}`")

                    if ticket.get("status") == "OPEN":
                        st.markdown("---")
                        human_note = st.text_area(f"Human Specialist Response ({ticket['ticket_id']})", placeholder="Type response or action taken...")
                        if st.button(f"Mark Resolved & Dispatch ({ticket['ticket_id']})", key=f"btn_{ticket['ticket_id']}"):
                            if human_note.strip():
                                resolve_ticket(ticket["ticket_id"], human_note)
                                st.success("Ticket marked as RESOLVED!")
                                st.rerun()
                            else:
                                st.error("Please enter response notes before resolving.")
                    else:
                        st.info(f"**Resolution Notes:** {ticket.get('agent_response', 'No notes.')} *(Resolved at {ticket.get('resolved_at')})*")
