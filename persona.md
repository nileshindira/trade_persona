# Trade Persona Analyzer Architecture

This document describes the architecture, file structure, and technical requirements of the Trade Persona Analyzer project, providing a roadmap for rebuilding or extending the application.

## 🏗️ System Architecture

The application is a Python-based trading analysis engine that processes historical trade data and generates a "Trading Persona" report using advanced metrics, pattern detection, and LLM-based insights.

### Core Components

1.  **Data Processor (`src/data_processor.py`)**
    *   Handles data ingestion from various sources (CSV, Excel, Databases).
    *   Cleans and standardizes trade data (Normalizes symbols, dates, and transaction types).
2.  **Metrics Calculator (`src/metrics_calculator.py`)**
    *   Computes over 50+ quantitative trading metrics.
    *   Includes: net PnL, win rate, Sharpe ratio, Sortino ratio, max drawdown, VaR (Value at Risk), etc.
3.  **Pattern Detector (`src/pattern_detector.py`)**
    *   Identifies behavioral patterns like overtrading, revenge trading, and trend following.
    *   Analyzes holding periods and execution quality.
4.  **LLM Analyzer (`src/llm_analyzer.py`)**
    *   Interfaces with local or remote LLMs (e.g., Ollama, ChatGPT).
    *   Drives psychological profiling and actionable recommendations based on quantitative data.
5.  **Report Generator (`src/report_generator.py`)**
    *   Orchestrates the export of findings into structured JSON and beautifully styled HTML.
    *   Uses Jinja2 templates and Chart.js for visualization.

---

## 📂 File Structure

```text
trade_persona/
├── main.py                 # CLI Entry point
├── config.yaml             # Application configuration
├── requirements.txt        # Python dependencies
├── src/                    # Core source code
│   ├── data_processor.py   # Data cleaning/loading
│   ├── metrics_calculator.py# Quant metrics engine
│   ├── pattern_detector.py  # Behavioral pattern logic
│   ├── llm_analyzer.py      # LLM integration logic
│   ├── report_generator.py  # Report exporter (Jinja2)
│   ├── templates/          # Jinja2 templates
│   └── static/             # CSS and Assets
├── webapp/                 # NEW Web Application Root
│   ├── backend/           # FastAPI Application
│   │   ├── main.py        # API Entrypoint (Wraps analyzer)
│   │   └── requirements.txt
│   ├── frontend/          # Next.js Application (Tailwind + Bootstrap)
│   │   ├── src/app/       # App Router logic & pages
│   │   ├── src/styles/    # Ported CSS from original src/static
│   │   └── package.json
│   ├── uploads/           # Temporary trade file storage
│   ├── reports/           # Generated specific report storage
│   └── run_webapp.sh      # Unified runner script
```

---

## 🎨 UI & UX Design System

The application uses a **"Surgical Diagnostic Engine"** theme, characterized by:
- **Design Aesthetic**: Premium Dark Mode, high-contrast text, and blueprint-like styling inspired by specialized audit reports.
- **Foundations**: Bootstrap 5 with custom overrides in `theme.css`.
- **Charts**: Chart.js for interactive visualizations (Radar Charts for behavior, Line Charts for PnL).
- **Typography**: `Inter` and `Outfit` Google Fonts for a modern, crisp feel.

---

## 🛠️ Rebuilding Requirements

To recreate the full environment:

### Core Analysis Engine
1.  **Python 3.10+**: Runtime.
2.  **Pandas / NumPy / Scipy**: For quantitative analysis.
3.  **Jinja2**: For report rendering logic.
4.  **Ollama**: Set up your local LLM (e.g., `gpt-oss:20b-32k`) as defined in `config.yaml`.

### Web Application Layer
1.  **FastAPI**: Modern Python web framework for performance.
2.  **Next.js 14+ (App Router)**: React-based frontend framework.
3.  **Uvicorn**: Server to run the backend.
4.  **Axios**: For frontend-to-backend communication.
5.  **React-Chartjs-2**: To port Chart.js visualizations into the component-based UI.

---

## 🚀 How to Run

1.  **Backend Setup**: 
    - `pip install fastapi uvicorn python-multipart`
    - Run: `python webapp/backend/main.py`
2.  **Frontend Setup**:
    - `cd webapp/frontend && npm install`
    - Run: `npm run dev`
3.  **Unified Launch**:
    - Use provided runner: `./webapp/run_webapp.sh`
