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

# Page Configuration - Clean Professional, Zero Emojis
st.set_page_config(
    page_title="NovaMart Support Agent | LangGraph RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Exact 6-Color Palette:
# Deep Navy: #253A82 | Periwinkle Blue: #88A2FF | Neon Lime: #E3FC87 | Bright Pink: #FFB2F7 | Light Sky Blue: #C0E0FF | Rich Violet: #AB9DFF
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --deep-navy: #253A82;
        --periwinkle: #88A2FF;
        --neon-lime: #E3FC87;
        --bright-pink: #FFB2F7;
        --light-sky: #C0E0FF;
        --rich-violet: #AB9DFF;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--deep-navy);
    }

    /* Canvas background - Soft Elegant Lavender */
    .stApp {
        background-color: #f5f2f9 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(171, 157, 255, 0.18) 0px, transparent 55%),
            radial-gradient(at 100% 100%, rgba(255, 178, 247, 0.14) 0px, transparent 55%) !important;
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #eef5fc !important;
        border-right: 2px solid var(--periwinkle) !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: var(--deep-navy) !important;
        font-weight: 800;
        letter-spacing: -0.01em;
    }

    /* Top Hero Header Card */
    .hero-container {
        background: var(--deep-navy);
        border-radius: 16px;
        padding: 24px 28px;
        color: #ffffff;
        box-shadow: 0 12px 28px -6px rgba(37, 58, 130, 0.28);
        margin-bottom: 20px;
        border: 2px solid var(--periwinkle);
    }
    .hero-title-text {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-sub-text {
        font-size: 0.95rem;
        color: var(--light-sky);
        margin-top: 6px;
        margin-bottom: 16px;
        line-height: 1.45;
        font-weight: 500;
    }
    .badge-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .brand-badge {
        font-size: 0.76rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-lime {
        background-color: var(--neon-lime);
        color: var(--deep-navy);
        border: 1px solid var(--neon-lime);
    }
    .badge-periwinkle {
        background-color: var(--periwinkle);
        color: #ffffff;
        border: 1px solid var(--periwinkle);
    }
    .badge-pink {
        background-color: var(--bright-pink);
        color: var(--deep-navy);
        border: 1px solid var(--bright-pink);
    }
    .badge-violet {
        background-color: var(--rich-violet);
        color: #ffffff;
        border: 1px solid var(--rich-violet);
    }

    /* Tab Navigation */
    div[data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
        padding: 6px !important;
        border: 2px solid var(--periwinkle) !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }
    div[data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 22px !important;
        color: var(--deep-navy) !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="tab"]:hover {
        background-color: var(--light-sky) !important;
        color: var(--deep-navy) !important;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--deep-navy) !important;
        color: var(--neon-lime) !important;
        border: 1px solid var(--deep-navy) !important;
        box-shadow: 0 4px 14px rgba(37, 58, 130, 0.3) !important;
    }

    /* Chat Message Bubbles - Explicit Dual Shading */
    .user-bubble {
        background-color: var(--bright-pink);
        border: 2px solid var(--rich-violet);
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin-bottom: 14px;
        color: var(--deep-navy);
        box-shadow: 0 2px 8px rgba(255, 178, 247, 0.25);
    }
    .user-label {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        color: var(--deep-navy);
        margin-bottom: 4px;
        letter-spacing: 0.05em;
    }
    .user-text {
        font-size: 0.95rem;
        color: var(--deep-navy);
        font-weight: 500;
        line-height: 1.5;
    }

    .assistant-bubble {
        background-color: var(--light-sky);
        border: 2px solid var(--periwinkle);
        border-left: 6px solid var(--deep-navy);
        border-radius: 16px 16px 16px 4px;
        padding: 16px 20px;
        margin-bottom: 14px;
        color: var(--deep-navy);
        box-shadow: 0 4px 12px rgba(37, 58, 130, 0.08);
    }
    .assistant-label {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        color: var(--deep-navy);
        margin-bottom: 6px;
        letter-spacing: 0.05em;
    }
    .assistant-text {
        font-size: 0.95rem;
        color: var(--deep-navy);
        line-height: 1.55;
    }

    /* Citation Tags */
    .citation-chip {
        display: inline-block;
        background-color: #ffffff;
        color: var(--deep-navy);
        border: 1.5px solid var(--deep-navy);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 4px 6px 4px 0px;
        box-shadow: 0 2px 4px rgba(37, 58, 130, 0.08);
        transition: all 0.2s ease;
    }
    .citation-chip:hover {
        background-color: var(--neon-lime);
        border-color: var(--deep-navy);
        transform: translateY(-1px);
    }

    /* Escalation Banner */
    .escalation-banner {
        background-color: var(--bright-pink);
        border: 2px solid var(--deep-navy);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 10px;
        color: var(--deep-navy);
        font-weight: 700;
        font-size: 0.88rem;
    }

    /* Metrics Cards */
    .metric-box-sky {
        background-color: var(--light-sky);
        border: 2px solid var(--periwinkle);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .metric-box-pink {
        background-color: var(--bright-pink);
        border: 2px solid var(--rich-violet);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .metric-label-text {
        font-size: 0.80rem;
        font-weight: 800;
        color: var(--deep-navy);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-number-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--deep-navy);
        margin-top: 2px;
        line-height: 1.1;
    }

    /* Buttons */
    .stButton > button {
        background-color: #ffffff !important;
        border: 2px solid var(--periwinkle) !important;
        color: var(--deep-navy) !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: var(--neon-lime) !important;
        border-color: var(--deep-navy) !important;
        color: var(--deep-navy) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 58, 130, 0.2) !important;
    }

    /* Chat Input */
    div[data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 2px solid var(--periwinkle) !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(37, 58, 130, 0.08) !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--deep-navy) !important;
        box-shadow: 0 0 0 3px rgba(136, 162, 255, 0.35) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 2px solid var(--light-sky) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }

    /* Welcome guidance feature cards */
    .guide-card {
        background-color: #ffffff;
        border: 2px solid var(--periwinkle);
        border-top: 5px solid var(--deep-navy);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
        box-shadow: 0 3px 10px rgba(37, 58, 130, 0.06);
    }
    .guide-title {
        font-weight: 800;
        font-size: 0.95rem;
        color: var(--deep-navy);
        margin-bottom: 6px;
    }
    .guide-desc {
        font-size: 0.85rem;
        color: #3b4d82;
        line-height: 1.45;
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
    <div style='padding-bottom: 12px; border-bottom: 2px solid #88A2FF; margin-bottom: 16px;'>
        <div style='font-weight: 800; font-size: 1.25rem; color: #253A82; letter-spacing: -0.02em;'>NOVAMART</div>
        <div style='display: inline-block; background-color: #E3FC87; color: #253A82; font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 12px; margin-top: 4px; border: 1px solid #253A82;'>CONTROL CONSOLE</div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Configuration
    st.markdown("### LLM Provider Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Google AI Studio Gemini API Key. If left blank, offline heuristic logic executes."
    )
    
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        llm = get_llm_service(api_key=api_key_input)
        if llm.is_configured():
            st.markdown("<div style='background: #E3FC87; color: #253A82; padding: 6px 10px; border-radius: 8px; font-weight: 700; font-size: 0.82rem; border: 1px solid #253A82;'>STATUS: GEMINI API CONNECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background: #FFB2F7; color: #253A82; padding: 6px 10px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;'>STATUS: INITIALIZING API...</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background: #C0E0FF; color: #253A82; padding: 6px 10px; border-radius: 8px; font-weight: 600; font-size: 0.82rem; border: 1px solid #88A2FF;'>MODE: SIMULATION (OFFLINE VERIFIED)</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Test Presets
    st.markdown("### Benchmark Test Scenarios")
    quick_queries = [
        ("Policy: Return Window & Fees", "What is the return window and how much is the return shipping fee?"),
        ("Specs: AuraSound Pro Specifications", "What is the battery life and water resistance of the AuraSound Pro headphones?"),
        ("Multi-Turn: Bluetooth Multipoint Follow-up", "Does it support multipoint Bluetooth pairing?"),
        ("Order Tracking: ORD-98214 Status", "Where is my order ORD-98214?"),
        ("Return Check: ORD-76150 Eligibility", "I want to return my order ORD-76150."),
        ("Escalation: Request Human Specialist", "This service is terrible and I demand to speak to a human manager immediately!")
    ]

    for label, query_text in quick_queries:
        if st.button(label, use_container_width=True):
            st.session_state.current_prompt = query_text

    st.markdown("---")
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_steps = []
        st.rerun()

# ----------------- MAIN INTERFACE HERO -----------------
st.markdown("""
<div class='hero-container'>
    <div class='hero-title-text'>NovaMart Intelligent Support Agent</div>
    <div class='hero-sub-text'>
        Stateful LangGraph State Machine &bull; Corrective RAG (CRAG) Verification Gate &bull; Deterministic Human-in-the-Loop Escalation
    </div>
    <div class='badge-strip'>
        <span class='brand-badge badge-lime'>LangGraph Cyclic Graph</span>
        <span class='brand-badge badge-periwinkle'>CRAG Relevance Gate (Score &ge; 0.30)</span>
        <span class='brand-badge badge-pink'>Dual-Partitioned Retrieval</span>
        <span class='brand-badge badge-violet'>Human Operator Desk</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_graph, tab_human = st.tabs([
    "Customer Support Chat",
    "LangGraph Execution Inspector",
    "Human Agent Desk (HITL)"
])

# ----------------- TAB 1: CUSTOMER CHAT -----------------
with tab_chat:
    # If no messages yet, show clean guide cards
    if not st.session_state.messages:
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("""
            <div class='guide-card'>
                <div class='guide-title'>Corporate Policies & Terms</div>
                <div class='guide-desc'>Query return windows, fee deductions, international freight rules, and limited warranty coverage.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_g2:
            st.markdown("""
            <div class='guide-card'>
                <div class='guide-title'>Hardware Technical Specs</div>
                <div class='guide-desc'>Lookup specifications, battery limits, pairing protocols, and error codes across NovaMart products.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_g3:
            st.markdown("""
            <div class='guide-card'>
                <div class='guide-title'>Transactional Operations</div>
                <div class='guide-desc'>Track shipment status via order IDs (e.g., ORD-98214) or calculate 30-day statutory return eligibility.</div>
            </div>
            """, unsafe_allow_html=True)

    # Render Chat History with visible contrasting shades
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        is_escalated = msg.get("is_escalated", False)
        ticket_id = msg.get("ticket_id")

        if role == "user":
            st.markdown(f"""
            <div class='user-bubble'>
                <div class='user-label'>Customer Query</div>
                <div class='user-text'>{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            citations_html = ""
            if citations:
                chips = "".join([f"<span class='citation-chip'>Source: {c}</span>" for c in citations])
                citations_html = f"<div style='margin-top: 10px; padding-top: 8px; border-top: 1px solid #88A2FF;'><div style='font-size: 0.72rem; font-weight: 800; color: #253A82; text-transform: uppercase; margin-bottom: 4px;'>Verified Source Citations:</div>{chips}</div>"

            escalation_html = ""
            if is_escalated:
                escalation_html = f"""
                <div class='escalation-banner'>
                    [ESCALATION TRIGGERED] Customer inquiry escalated to Human Operator Desk. Priority Ticket ID: <code>{ticket_id}</code>
                </div>
                """

            st.markdown(f"""
            <div class='assistant-bubble'>
                <div class='assistant-label'>NovaMart Support Agent (Grounded Synthesis)</div>
                <div class='assistant-text'>{content}</div>
                {citations_html}
                {escalation_html}
            </div>
            """, unsafe_allow_html=True)

    # Chat Input Box
    prompt_to_send = None
    if "current_prompt" in st.session_state and st.session_state.current_prompt:
        prompt_to_send = st.session_state.current_prompt
        st.session_state.current_prompt = None
    else:
        chat_val = st.chat_input("Enter customer inquiry or operational order query (e.g., 'Where is ORD-98214?')...")
        if chat_val:
            prompt_to_send = chat_val

    if prompt_to_send:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})

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

        with st.spinner("Executing LangGraph state machine..."):
            final_state = graph.invoke(initial_state, config=thread_config)
            
            resp_text = final_state.get("response", "Unable to process the request.")
            citations = final_state.get("citations", [])
            is_escalated = final_state.get("is_escalated", False)
            ticket_id = final_state.get("ticket_id")

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
    st.subheader("Real-Time LangGraph Execution Trace")
    st.markdown("Direct inspection of state transitions, entity rewriting, document retrieval confidence, and CRAG relevance verification.")

    if not st.session_state.last_steps:
        st.info("No query has been executed yet. Send a query in the Chat tab or click a Benchmark Test Scenario in the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Node Execution Sequence")
            for idx, step in enumerate(st.session_state.last_steps):
                status_label = "[COMPLETED]" if step["status"] == "completed" else "[ATTENTION]"
                with st.expander(f"Step {idx+1}: {step['title']} ({step['node']}) {status_label}", expanded=True):
                    for k, v in step.get("details", {}).items():
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** `{v}`")

        with col2:
            st.markdown("### LangGraph Decision Graph")
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
    st.subheader("Human Support Escalation Portal")
    st.markdown("Automated priority queue for low-confidence retrievals, out-of-domain queries, and frustrated customer interactions.")

    tickets = load_tickets()
    
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        open_count = sum(1 for t in tickets if t.get("status") == "OPEN")
        
        st.markdown(f"""
        <div class='metric-box-sky'>
            <div class='metric-label-text'>Total Escalated Tickets</div>
            <div class='metric-number-text'>{len(tickets)}</div>
        </div>
        <div class='metric-box-pink'>
            <div class='metric-label-text'>Open Pending Reviews</div>
            <div class='metric-number-text'>{open_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        if not tickets:
            st.markdown("<div style='background: #E3FC87; color: #253A82; padding: 14px; border-radius: 10px; font-weight: 700; border: 1.5px solid #253A82;'>NO PENDING ESCALATIONS: All customer inquiries resolved autonomously.</div>", unsafe_allow_html=True)
        else:
            for ticket in tickets:
                is_open = (ticket.get("status") == "OPEN")
                status_badge = "<span style='background: #E3FC87; color: #253A82; padding: 2px 8px; border-radius: 10px; font-weight: 800; border: 1px solid #253A82;'>OPEN</span>" if is_open else "<span style='background: #88A2FF; color: white; padding: 2px 8px; border-radius: 10px; font-weight: 700;'>RESOLVED</span>"
                
                with st.expander(f"Ticket {ticket['ticket_id']} &bull; Status: {ticket.get('status')} ({ticket.get('created_at')})", expanded=is_open):
                    st.markdown(f"**Customer Query:** *\"{ticket.get('customer_query')}\"*")
                    st.markdown(f"**Escalation Reason:** `{ticket.get('reason')}`")
                    st.markdown(f"**Customer Sentiment:** `{ticket.get('sentiment')}`")
                    if ticket.get("order_id"):
                        st.markdown(f"**Associated Order:** `{ticket.get('order_id')}`")

                    if is_open:
                        st.markdown("---")
                        human_note = st.text_area(f"Specialist Resolution Notes ({ticket['ticket_id']})", placeholder="Enter resolution notes or dispatched customer message...")
                        if st.button(f"Mark Resolved & Dispatch ({ticket['ticket_id']})", key=f"btn_{ticket['ticket_id']}"):
                            if human_note.strip():
                                resolve_ticket(ticket["ticket_id"], human_note)
                                st.success("Ticket status updated to RESOLVED.")
                                st.rerun()
                            else:
                                st.error("Resolution notes are required prior to ticket closure.")
                    else:
                        st.info(f"Resolution Notes: {ticket.get('agent_response', 'No notes recorded.')} (Resolved at {ticket.get('resolved_at')})")
