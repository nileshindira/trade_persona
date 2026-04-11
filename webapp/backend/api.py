import os
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import json
import yaml
import logging
import pandas as pd

# Add project root to sys.path for importing src modules
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Explicitly import from the root main.py file
from main import TradingPersonaAnalyzer
from src.report_generator import ReportGenerator

app = FastAPI(title="Trade Persona Analyzer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration and setup
CONFIG_PATH = os.path.join(project_root, "config.yaml")
UPLOAD_DIR = os.path.join(project_root, "webapp", "uploads")
REPORT_DIR = os.path.join(project_root, "webapp", "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webapp_backend")

# Mount static files for reports access
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")

@app.get("/health")
def health_check():
    return {"status": "ok"}

def sanitize_data(data):
    """Recursively replace NaN/Inf values and convert non-serializable types for JSON compliance."""
    import math
    import numpy as np

    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, (float, np.float32, np.float64)):
        val = float(data)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, (bool, np.bool_)):
        return bool(data)
    elif data is None:
        return None
    return data

async def save_upload_files(trader_name: str, trade_file: UploadFile, pnl_file: UploadFile):
    trade_ext = Path(trade_file.filename).suffix
    if trade_ext.lower() not in ['.csv', '.xlsx', '.xls']:
        raise HTTPException(status_code=400, detail="Invalid trade file format. Upload CSV or Excel.")

    trade_path = os.path.join(UPLOAD_DIR, f"{trader_name}_{trade_file.filename}")
    with open(trade_path, "wb") as buffer:
        shutil.copyfileobj(trade_file.file, buffer)

    pnl_path = False
    if pnl_file and pnl_file.filename:
        pnl_path = os.path.join(UPLOAD_DIR, f"pnl_{trader_name}_{pnl_file.filename}")
        with open(pnl_path, "wb") as buffer:
            shutil.copyfileobj(pnl_file.file, buffer)
    
    return trade_path, pnl_path

@app.post("/analyze")
async def analyze_trades(
    trader_name: str = Form(...), 
    trade_file: UploadFile = File(...),
    pnl_file: UploadFile = File(None)
):
    trade_path, pnl_path = await save_upload_files(trader_name, trade_file, pnl_file)

    try:
        analyzer = TradingPersonaAnalyzer(config_path=CONFIG_PATH)
        run_output_dir = os.path.join(REPORT_DIR, trader_name)
        os.makedirs(run_output_dir, exist_ok=True)

        report = analyzer.analyze(
            data_filepath=trade_path,
            trader_name=trader_name,
            output_dir=run_output_dir,
            include_ema=True,
            pnl_csv=pnl_path
        )

        return sanitize_data(report)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse

@app.post("/analyze-stream")
async def analyze_trades_stream(
    trader_name: str = Form(...), 
    trade_file: UploadFile = File(...),
    pnl_file: UploadFile = File(None)
):
    trade_path, pnl_path = await save_upload_files(trader_name, trade_file, pnl_file)

    async def event_generator():
        try:
            analyzer = TradingPersonaAnalyzer(config_path=CONFIG_PATH)
            run_output_dir = os.path.join(REPORT_DIR, trader_name)
            os.makedirs(run_output_dir, exist_ok=True)

            # --- Step 1: Load and Process ---
            yield json.dumps({"type": "status", "stage": "loading", "message": "Ingesting trade journal and P&L data...", "progress": 10}) + "\n"
            df = analyzer.data_processor.load_data(trade_path)
            
            nifty_chart_data = {}
            if pnl_path and str(pnl_path).lower() != 'false':
                pnl_csv_df = pd.read_csv(pnl_path)
                date_col = next((c for c in pnl_csv_df.columns if c.lower() == 'date'), 'Date')
                pnl_csv_df['Date'] = pd.to_datetime(pnl_csv_df[date_col], errors='coerce')
                nifty_chart_data = analyzer.data_processor.get_nifty_data(pnl_csv_df)

            is_valid, missing_cols = analyzer.data_processor.validate_data(df)
            if not is_valid:
                yield json.dumps({"type": "error", "message": f"Missing columns: {missing_cols}"}) + "\n"
                return

            df = analyzer.data_processor.clean_data(df)
            df = analyzer.data_processor.pair_trades(df, output_dir=run_output_dir, trader_name=trader_name)
            
            # --- Step 2: Detect Patterns (MUST GO BEFORE METRICS in main.py) ---
            yield json.dumps({"type": "status", "stage": "patterns", "message": "Identifying behavioral fingerprints...", "progress": 30}) + "\n"
            patterns = analyzer.pattern_detector.detect_all_patterns(df)

            # --- Step 3: Calculate Metrics (now with patterns for trace sheet) ---
            yield json.dumps({"type": "status", "stage": "auditing", "message": "Performing clinical performance audit...", "progress": 50}) + "\n"
            metrics = analyzer.metrics_calculator.calculate_all_metrics(df, pnl_path, nifty_data=nifty_chart_data, patterns=patterns)
            
            # Send partial metrics to UI
            yield json.dumps({"type": "data", "metrics": sanitize_data(metrics)}) + "\n"
            
            # --- Step 4: AI Analysis ---
            yield json.dumps({"type": "status", "stage": "ai_analysis", "message": "Synthesizing AI behavioral narrative...", "progress": 70}) + "\n"
            analysis = analyzer.llm_analyzer.generate_analysis(metrics, patterns, df, trader_name=trader_name)
            
            # --- Normalization logic from main.py ---
            pnl_vals = analysis['web_data']['charts']['pnl_timeline']['values']
            if pnl_vals:
                start = pnl_vals[0]
                norm_pnl_vals = [(v - start) + 100 for v in pnl_vals]
            else:
                norm_pnl_vals = []

            nifty_vals = list(nifty_chart_data.values())
            if nifty_vals and nifty_vals[0] != 0:
                start_idx = nifty_vals[0]
                norm_nifty_vals = [(v / start_idx) * 100 for v in nifty_vals]
            else:
                norm_nifty_vals = [100] * len(nifty_vals) if nifty_vals else []

            analysis['web_data']['charts']['nifty_pnl_timeline'] = {
                'dates': list(nifty_chart_data.keys()),
                'values': list(nifty_chart_data.values())
            }
            analysis['web_data']['charts']['relative_chart'] = {
                "dates": list(nifty_chart_data.keys()),
                "pnl_normalized": norm_pnl_vals,
                "nifty_normalized": norm_nifty_vals,
            }

            # --- Step 4.5: Load Proposed Trade File (from main.py) ---
            proposed_trades = []
            proposed_csv_path = Path(project_root) / "TRADE_FILES/Proposed_Trade_File.csv"
            if proposed_csv_path.exists():
                try:
                    pt_df = pd.read_csv(proposed_csv_path)
                    if 'pnl' not in pt_df.columns and not df.empty:
                        df_lookup = df.copy()
                        df_lookup['dt_match'] = pd.to_datetime(df_lookup['trade_date']).dt.strftime('%d-%m-%Y %H:%M')
                        pnl_map = df_lookup.groupby(['symbol', 'dt_match'])['pnl'].sum().to_dict()
                        
                        def lookup_pnl(row):
                            key = (row.get('symbol'), str(row.get('trade_date')).strip())
                            return pnl_map.get(key, 0.0)
                        
                        pt_df['pnl'] = pt_df.apply(lookup_pnl, axis=1)
                    
                    if 'pnl' not in pt_df.columns:
                        pt_df['pnl'] = 0.0
                    proposed_trades = pt_df.to_dict('records')
                except Exception as e:
                    logger.error(f"Error loading Proposed_Trade_File: {e}")

            # --- Step 5: Generate & Export Report ---
            yield json.dumps({"type": "status", "stage": "finalizing", "message": "Assembling surgical report...", "progress": 90}) + "\n"
            report = analyzer.report_generator.generate_report(metrics, patterns, analysis, trader_name)
            report["appendix"]["proposed_trades"] = proposed_trades

            # Export exactly like main.py
            json_path = os.path.join(run_output_dir, f"{trader_name}_report.json")
            analyzer.report_generator.export_json(report, json_path)
            
            html_path = os.path.join(run_output_dir, f"{trader_name}_report.html")
            analyzer.report_generator.export_html(report, html_path)
            
            trace_data = metrics.get('consolidated_trace', [])
            if trace_data:
                trace_df = pd.DataFrame(trace_data)
                trace_csv_path = os.path.join(run_output_dir, f"{trader_name}_consolidated_trace.csv")
                trace_df.to_csv(trace_csv_path, index=False)

            yield json.dumps({"type": "complete", "report": sanitize_data(report)}) + "\n"

        except Exception as e:
            import traceback
            logger.error(f"Stream error: {str(e)}\n{traceback.format_exc()}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
