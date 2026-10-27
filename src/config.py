import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"
MANUALS_DIR = DATA_DIR / "manuals"
MOCK_DB_DIR = DATA_DIR / "mock_db"
ORDERS_FILE = MOCK_DB_DIR / "orders.json"
TICKETS_FILE = MOCK_DB_DIR / "escalated_tickets.json"

# Models & Parameters
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_GEMINI_MODEL = "gemini-1.5-flash"

# Confidence & Thresholds
RETRIEVAL_TOP_K = 3
CONFIDENCE_THRESHOLD = 0.30
MAX_RETRIES = 1

# Categories
CATEGORIES = [
    "policy_inquiry",      # Returns, refunds, shipping, warranty
    "product_inquiry",     # Specs, setup, compatibility, troubleshooting
    "order_tracking",      # Where is my order, status
    "return_request",      # Initiating a return for an order
    "human_escalation",    # Explicit human request or extreme anger
    "general_chitchat"     # Greetings, thank you, bye
]
