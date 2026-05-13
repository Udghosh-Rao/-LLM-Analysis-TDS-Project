#  Nexus AI Agent

> **Autonomous AI Financial Research & Data Science Dashboard**  
> A premium, full-stack AI platform combining LangGraph task routing, real-time financial data, machine learning anomaly detection, and a high-end React glassmorphic UI.

---

##  Overview

Nexus AI Agent is a modern, production-ready AI product built to demonstrate advanced Data Science and AI Engineering concepts with a sleek, startup-quality aesthetic.

Instead of a terminal-heavy or purely academic UI, this project features a **Premium React Frontend** that dynamically generates real-time market analysis, beautiful animated charts, risk assessments, and natural-language explanations powered by **Groq (LLaMA 3)**.

### Core Philosophy
- **Compute First, Explain Second:** Indicators, ML risk scores, and sentiment are computed deterministically, *then* passed to the LLM to generate narrative explanations.
- **Demo-Ready Polish:** A high-contrast, visually stunning UI that makes complex quant models easy to understand for managers, HR, and interviewers within seconds.

---

## 🎨 Features & UI Elements

1. **Hero Dashboard:** Complete glassmorphic UI with animated metric cards, risk gauges, and a recommendation engine.
2. **AI Chat Assistant Sidebar:** A mini-ChatGPT embedded directly in the application that uses your active stock selection as context.
3. **Interactive Charts:** Smooth, dynamic area charts using Recharts for visualizing price action and moving averages.
4. **Sentiment Analysis:** Real-time news ingestion evaluated via Hugging Face `FinBERT` transformers.
5. **Machine Learning Pipeline:** `scikit-learn` Isolation Forests running in the background to detect market anomalies and output calibrated risk scores.

---

## 🏗️ Architecture Stack

| Layer | Technology |
|---|---|
| **Frontend UI** | React, Vite, Tailwind CSS, Framer Motion, Recharts, Lucide Icons |
| **Backend API** | FastAPI, Uvicorn, Python 3 |
| **AI / Orchestration**| LangChain, LangGraph, Groq API |
| **ML & Data Science** | Pandas, Numpy, Scikit-Learn (Isolation Forests), Hugging Face Transformers |
| **Data Ingestion** | `yfinance`, Playwright (Web Scraping) |

---

## 💻 How to Run Locally

You can run both the Backend Server and the Frontend UI easily on your local machine.

### Prerequisites
- Python 3.11+
- Node.js & npm (for the frontend)
- A [Groq API key](https://console.groq.com)

### 1. Setup Environment
First, clone the repository and set up your backend credentials:

```bash
git clone https://github.com/Udghosh-Rao/nexus-agent.git
cd nexus-agent/nexus-agent-main

# Copy the example env file
cp .env.example .env
```
*Open `.env` and add your `GROQ_API_KEY`.*

### 2. The One-Click Start Script
If you are on Mac/Linux, you can launch everything at once using the provided bash script:

```bash
chmod +x start.sh
./start.sh
```
This will automatically activate your environment, start the Python API, and launch the React UI.

---

### 3. Running Manually (Separate Terminals)

If you prefer to see the logs for both servers separately, use two terminal tabs:

**Terminal 1 (Backend API):**
```bash
cd nexus-agent-main
source venv/bin/activate  # (Or use your system python if no venv)
pip install -r requirements.txt
python3 -m uvicorn app.api.app:app --host 0.0.0.0 --port 7860
```
*Backend runs on `http://localhost:7860`*

**Terminal 2 (Frontend React App):**
```bash
cd nexus-agent-main/frontend
npm install
npm run dev
```
*Frontend runs on `http://localhost:5173`*

**Access the UI:** Open [http://localhost:5173](http://localhost:5173) in your browser!

---

## ⚙️ Core API Endpoints

If you wish to interact with the backend programmatically:

- **`GET /dashboard/{ticker}`**
  Returns the complete aggregated data for the UI (prices, ML risk scores, Groq explanations, sentiment).
- **`GET /chart/{ticker}`**
  Returns OHLCV and moving average time-series data for charting.
- **`POST /chat`**
  Conversational endpoint that handles the AI Assistant sidebar context.
- **`POST /analyze/finance`**
  Run the pure quant analytics pipeline on a ticker.

---

## ⚖️ Disclaimer & License

This project is built for **portfolio demonstration and educational purposes**.
The machine learning models are unsupervised and should not be used as actual financial or investment advice. Use at your own risk.
