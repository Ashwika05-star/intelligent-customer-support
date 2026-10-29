import os
import sys

# Render defaults to: gunicorn your_application.wsgi
# Intercept and replace process with Streamlit server on $PORT
port = os.environ.get("PORT", "8501")
cmd = [
    "streamlit",
    "run",
    "app.py",
    "--server.port",
    str(port),
    "--server.address",
    "0.0.0.0",
    "--browser.gatherUsageStats",
    "false"
]
print(f"Launching Streamlit on port {port}...")
os.execvp("streamlit", cmd)
