# 🚀 Deployment Guide: Intelligent Customer Support Agent

This guide explains how to deploy the web application live to the internet for free so that professors, evaluators, and users can access it via a public URL.

---

## Option 1: Streamlit Community Cloud (Recommended & 100% Free)
Streamlit Cloud offers free public hosting directly connected to your GitHub repository with automatic updates on `git push`.

### Step-by-Step Instructions:
1. **Push your code to GitHub:**
   ```bash
   git remote add origin https://github.com/<your-username>/intelligent-customer-support.git
   git branch -M main
   git push -u origin main
   ```
2. **Sign in to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Deploy New App:**
   - Click **"New app"**.
   - Select your repository: `<your-username>/intelligent-customer-support`.
   - Branch: `main`.
   - Main file path: `app.py`.
4. **Configure Secrets / API Keys (Optional):**
   - Click **Advanced Settings** $\rightarrow$ **Secrets**.
   - Add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key_here"
     ```
   *(Note: If left blank, the app will still run in Smart Simulation / Offline Mode without crashing!)*
5. **Click "Deploy!":**
   - In 1 to 2 minutes, you will receive a public live URL (e.g., `https://novamart-support.streamlit.app`).

---

## Option 2: Render.com (Free Web Service)
1. Go to [render.com](https://render.com) and create an account.
2. Click **"New +"** $\rightarrow$ **"Web Service"** $\rightarrow$ Connect your GitHub repo.
3. Configure settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port 10000 --server.address 0.0.0.0`
4. Add Environment Variable:
   - `GEMINI_API_KEY`: `your_key`
5. Click **"Create Web Service"**.

---

## Option 3: Local or Docker Container Run
To run via Docker locally or on a VPS:
```bash
# Build the Docker image
docker build -t intelligent-customer-support .

# Run the container
docker run -p 8501:8501 -e GEMINI_API_KEY="your_key" intelligent-customer-support
```
Access at `http://localhost:8501`.
