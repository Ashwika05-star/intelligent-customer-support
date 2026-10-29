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

# Custom Styling using exact Palette:
# Midnight: #0b1957 | Sapphire: #426bc2 | Petal: #e8d9ed | Lace: #f7f4ed | Lavender: #d2b3db | Bridal Blue: #e9f3ff
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --midnight: #0b1957;
        --sapphire: #426bc2;
        --petal: #e8d9ed;
        --lace: #f7f4ed;
        --lavender: #d2b3db;
        --bridal-blue: #e9f3ff;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--midnight);
    }

    /* Background canvas */
    .stApp {
        background-color: var(--lace);
        background-image: 
            radial-gradient(at 0% 0%, rgba(233, 243, 255, 0.75) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(232, 217, 237, 0.55) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f3efe6;
        border-right: 1.5px solid var(--lavender);
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: var(--midnight) !important;
        font-weight: 700;
    }

    /* Top Hero Header Card */
    .hero-header {
        background: linear-gradient(135deg, #0b1957 0%, #172d79 50%, #426bc2 100%);
        border-radius: 18px;
        padding: 24px 30px;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(11, 25, 87, 0.18);
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.90);
        margin-top: 6px;
        margin-bottom: 14px;
        line-height: 1.4;
    }
    .pill-badges-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-pill {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(8px);
        color: #ffffff;
    }
    .hero-pill.glow {
        background: rgba(233, 243, 255, 0.25);
        border-color: #e9f3ff;
        color: #ffffff;
    }

    /* Tab Bar Styling */
    div[data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.75);
        border-radius: 14px;
        padding: 6px;
        border: 1.5px solid var(--lavender);
        gap: 6px;
        margin-bottom: 18px;
    }
    div[data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        color: var(--midnight);
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s ease;
    }
    div[data-baseweb="tab"]:hover {
        background-color: var(--petal);
        color: var(--midnight);
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--sapphire) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(66, 107, 194, 0.3);
    }

    /* Chat Message Styling */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(11, 25, 87, 0.04);
    }
    /* User Message: Petal palette */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: var(--petal) !important;
        border: 1px solid var(--lavender) !important;
        color: var(--midnight) !important;
    }
    /* Assistant Message: Bridal Blue palette with Sapphire accent bar */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: var(--bridal-blue) !important;
        border-left: 4px solid var(--sapphire) !important;
        border-top: 1px solid #dbeafe !important;
        border-right: 1px solid #dbeafe !important;
        border-bottom: 1px solid #dbeafe !important;
        color: var(--midnight) !important;
    }

    /* Citation Tags */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ffffff;
        color: var(--midnight);
        border: 1.5px solid var(--lavender);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 6px 4px 0px;
        box-shadow: 0 1px 3px rgba(11, 25, 87, 0.06);
        transition: all 0.2s ease;
    }
    .citation-tag:hover {
        border-color: var(--sapphire);
        color: var(--sapphire);
        transform: translateY(-1px);
    }

    /* Metric Cards */
    .custom-metric-card {
        background: #ffffff;
        border: 1.5px solid var(--lavender);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(11, 25, 87, 0.04);
        margin-bottom: 14px;
    }
    .custom-metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--sapphire);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .custom-metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--midnight);
        margin-top: 4px;
        line-height: 1.1;
    }

    /* Buttons in Sidebar & App */
    .stButton > button {
        border-radius: 10px;
        border: 1.5px solid var(--lavender);
        background-color: #ffffff;
        color: var(--midnight);
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 8px 14px;
    }
    .stButton > button:hover {
        background-color: var(--sapphire);
        color: #ffffff !important;
        border-color: var(--midnight);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(66, 107, 194, 0.25);
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        border-radius: 16px;
        border: 1.5px solid var(--lavender);
        background-color: #ffffff;
        box-shadow: 0 4px 14px rgba(11, 25, 87, 0.06);
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--sapphire);
        box-shadow: 0 0 0 3px rgba(66, 107, 194, 0.18);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: var(--midnight) !important;
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid var(--lavender);
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(11, 25, 87, 0.03);
        margin-bottom: 12px;
    }

    /* Quick Feature Cards */
    .feature-card {
        background: #ffffff;
        border: 1.5px solid var(--lavender);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(11, 25, 87, 0.04);
        margin-bottom: 10px;
    }
    .feature-card h4 {
        margin: 0 0 6px 0;
        color: var(--midnight);
        font-size: 0.98rem;
    }
    .feature-card p {
        margin: 0;
        color: #4a5568;
        font-size: 0.85rem;
        line-height: 1.4;
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
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 14px;'>
        <div style='background: #0b1957; color: white; width: 36px; height: 36px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;'>🛍️</div>
        <div>
            <div style='font-weight: 800; font-size: 1.15rem; color: #0b1957; line-height: 1.1;'>NovaMart</div>
            <div style='font-size: 0.78rem; color: #426bc2; font-weight: 600;'>Agent Control Center</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
        ("📦 Policy Return Window", "What is the return window and how much is the return shipping fee?"),
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
st.markdown("""
<div class='hero-header'>
    <div class='hero-title'>
        <span>🛍️ NovaMart Intelligent Support Assistant</span>
    </div>
    <div class='hero-subtitle'>
        Enterprise-grade customer care powered by a stateful <b>LangGraph cyclic state machine</b>, <b>Corrective RAG (CRAG)</b> verification, and <b>Human-in-the-Loop</b> escalation.
    </div>
    <div class='pill-badges-row'>
        <span class='hero-pill glow'>⚡ LangGraph Cyclic Graph</span>
        <span class='hero-pill glow'>🛡️ CRAG Relevance Gate (τ ≥ 0.30)</span>
        <span class='hero-pill glow'>🔍 Dual-Partitioned Retrieval</span>
        <span class='hero-pill'>👨‍💼 Human-in-the-Loop Escalation</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_graph, tab_human = st.tabs([
    "💬 Customer Support Chat",
    "🧠 LangGraph Workflow & State Inspector",
    "👨‍💼 Human Agent Desk (HITL)"
])

# ----------------- TAB 1: CUSTOMER CHAT -----------------
with tab_chat:
    # If no messages yet, show nice welcome guidance cards
    if not st.session_state.messages:
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown("""
            <div class='feature-card'>
                <h4>📦 Policy & Shipping</h4>
                <p>Ask about return policies, 30-day coverage, warranty terms, and international freight rules.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_w2:
            with st.container():
                st.markdown("""
                <div class='feature-card'>
                    <h4>🎧 Product Specs & Specs</h4>
                    <p>Troubleshoot hardware, battery endurance, pairing, and error codes for NovaMart electronics.</p>
                </div>
                """, unsafe_allow_html=True)
        with col_w3:
            st.markdown("""
            <div class='feature-card'>
                <h4>🚚 Live Order Operations</h4>
                <p>Provide an order ID like <code>ORD-98214</code> to track live transit or verify return eligibility.</p>
            </div>
            """, unsafe_allow_html=True)

    # Display message history
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            
            # Show citations if present
            if msg.get("citations"):
                with st.expander("📚 Verified Source Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"<span class='citation-tag'>📄 {c}</span>", unsafe_allow_html=True)
            
            # Show escalation alert if present
            if msg.get("is_escalated"):
                st.warning(f"⚠️ **Escalated to Human Support** | Ticket ID: `{msg.get('ticket_id', 'N/A')}`")

    # Chat input
    prompt_to_send = None
    if "current_prompt" in st.session_state and st.session_state.current_prompt:
        prompt_to_send = st.session_state.current_prompt
        st.session_state.current_prompt = None
    else:
        chat_val = st.chat_input("Type your question (e.g., 'What is the return window for headphones?' or 'Where is ORD-98214?')...")
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
                    with st.expander("📚 Verified Source Citations"):
                        for c in citations:
                            st.markdown(f"<span class='citation-tag'>📄 {c}</span>", unsafe_allow_html=True)
                
                if is_escalated:
                    st.warning(f"⚠️ **Escalated to Human Support** | Priority Ticket ID: `{ticket_id}`")

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
    st.subheader("🔍 Real-Time LangGraph Execution Trace")
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
    st.markdown("Unresolved queries, low-confidence retrievals, and frustrated customer messages are automatically routed here for live human triage.")

    tickets = load_tickets()
    
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        open_count = sum(1 for t in tickets if t.get("status") == "OPEN")
        
        st.markdown(f"""
        <div class='custom-metric-card'>
            <div class='custom-metric-title'>Total Escalated Tickets</div>
            <div class='custom-metric-value'>{len(tickets)}</div>
        </div>
        <div class='custom-metric-card'>
            <div class='custom-metric-title'>Open Pending Reviews</div>
            <div class='custom-metric-value' style='color: {'#dc2626' if open_count > 0 else '#16a34a'};'>{open_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        if not tickets:
            st.success("🎉 No escalated tickets currently pending! The automated agent is successfully resolving incoming customer queries.")
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
