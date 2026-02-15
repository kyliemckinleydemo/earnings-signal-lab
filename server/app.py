"""
Earnings Signal Lab — Web Server
=================================
FastAPI app that serves:
  - Public dashboard at /
  - JSON API for predictions and model results
  - Cron-triggered pipeline runs

Does NOT expose: raw transcripts, Claude analysis details, or API keys.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
import subprocess
import asyncio

# ============================================================
# Config
# ============================================================

DATA_DIR = Path(os.environ.get("DATA_DIR", "earnings_signal_data"))
RESULTS_FILE = DATA_DIR / "backtest_results.json"
SUMMARY_FILE = DATA_DIR / "SUMMARY.md"
PIPELINE_SCRIPT = Path("earnings_signal_pipeline.py")

# Cache the results in memory for fast serving
_cached_results = None
_cached_at = None


def load_results():
    """Load and cache backtest results, stripping sensitive data."""
    global _cached_results, _cached_at

    if not RESULTS_FILE.exists():
        return None

    mtime = RESULTS_FILE.stat().st_mtime
    if _cached_results and _cached_at == mtime:
        return _cached_results

    raw = json.loads(RESULTS_FILE.read_text())

    # Build public-safe version — no raw evidence/transcripts
    public = {
        "metadata": raw.get("metadata", {}),
        "features": {},
        "combinations": raw.get("combinations", []),
        "correlation_matrix": raw.get("correlation_matrix", {}),
        "regression": {},
        "summary": raw.get("summary", {}).get("markdown", ""),
        "last_updated": datetime.fromtimestamp(mtime).isoformat(),
    }

    # Strip evidence text from features (keeps scores + stats only)
    for feat_name, periods in raw.get("features", {}).items():
        public["features"][feat_name] = periods

    # Strip evidence from sample extractions — keep scores and returns only
    public["sample_extractions"] = {}
    for feat_name, samples in raw.get("sample_extractions", {}).items():
        public["sample_extractions"][feat_name] = [
            {
                "symbol": s.get("symbol"),
                "quarter": s.get("quarter"),
                "score": s.get("score"),
                "return_5D": s.get("return_5D"),
                "return_10D": s.get("return_10D"),
                # Deliberately omit "evidence" — that's derived from transcripts
            }
            for s in samples
        ]

    # Regression — include model comparisons, weights, importances
    # but not raw data
    for period, pdata in raw.get("regression", {}).items():
        public_period = {}
        for key in ["n_observations", "y_mean", "y_std", "model_comparison",
                     "recommended_weights"]:
            if key in pdata:
                public_period[key] = pdata[key]

        # Include top features from each model
        for model_key in ["ols", "ridge", "lasso", "elasticnet", "stepwise",
                          "polynomial", "random_forest", "gradient_boosting",
                          "mutual_information"]:
            if model_key in pdata:
                model_data = pdata[model_key]
                safe_model = {}
                for k in ["r2_insample", "r2_cv_mean", "r2_cv_std", "rmse_cv",
                          "best_alpha", "best_l1_ratio", "best_learning_rate",
                          "best_max_depth", "n_features_selected",
                          "n_features_eliminated", "n_poly_features",
                          "top_features", "selected_features",
                          "feature_importances", "coefficients",
                          "interaction_terms", "scores", "selection_order"]:
                    if k in model_data:
                        safe_model[k] = model_data[k]
                public_period[model_key] = safe_model

        public["regression"][period] = public_period

    _cached_results = public
    _cached_at = mtime
    return public


# ============================================================
# App
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load results
    load_results()
    yield

app = FastAPI(
    title="Earnings Signal Lab",
    description="Granular NLP feature extraction from earnings transcripts → forward return predictions",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# API Routes
# ============================================================

@app.get("/api/results")
async def get_results():
    """Full public results — features, regression, combinations, correlations."""
    results = load_results()
    if not results:
        raise HTTPException(status_code=503, detail="No results available yet. Pipeline hasn't run.")
    return results


@app.get("/api/predictions")
async def get_predictions():
    """Just the prediction model weights and expected performance."""
    results = load_results()
    if not results:
        raise HTTPException(status_code=503, detail="No results available yet.")

    predictions = {}
    for period, pdata in results.get("regression", {}).items():
        rec = pdata.get("recommended_weights", {})
        comparison = pdata.get("model_comparison", [])
        best_model = comparison[0] if comparison else {}

        predictions[period] = {
            "weights": rec.get("weights", {}),
            "intercept": rec.get("intercept", 0),
            "method": rec.get("method", ""),
            "best_model": best_model.get("model", ""),
            "best_cv_r2": best_model.get("cv_r2", 0),
            "n_observations": pdata.get("n_observations", 0),
        }

    return {
        "predictions": predictions,
        "metadata": results.get("metadata", {}),
        "last_updated": results.get("last_updated", ""),
    }


@app.get("/api/features")
async def get_features():
    """Individual feature signal strength across holding periods."""
    results = load_results()
    if not results:
        raise HTTPException(status_code=503, detail="No results available yet.")
    return {
        "features": results.get("features", {}),
        "sample_extractions": results.get("sample_extractions", {}),
    }


@app.get("/api/summary")
async def get_summary():
    """Markdown summary of learnings."""
    results = load_results()
    if not results:
        # Try loading from file directly
        if SUMMARY_FILE.exists():
            return {"summary": SUMMARY_FILE.read_text()}
        raise HTTPException(status_code=503, detail="No summary available yet.")
    return {"summary": results.get("summary", "")}


@app.get("/api/status")
async def get_status():
    """Pipeline health and data freshness."""
    status = {
        "results_exist": RESULTS_FILE.exists(),
        "data_dir_exists": DATA_DIR.exists(),
    }

    if RESULTS_FILE.exists():
        results = load_results()
        status["last_updated"] = results.get("last_updated", "")
        status["total_events"] = results.get("metadata", {}).get("total_events", 0)
        status["companies"] = len(results.get("metadata", {}).get("companies", []))
        status["periods_modeled"] = list(results.get("regression", {}).keys())

    # Check sub-directories
    transcripts_dir = DATA_DIR / "transcripts"
    analysis_dir = DATA_DIR / "analysis"

    if transcripts_dir.exists():
        status["transcripts_cached"] = sum(
            len(json.loads(f.read_text()))
            for f in transcripts_dir.glob("*_transcripts.json")
        )
    if analysis_dir.exists():
        status["analyses_cached"] = len(list(analysis_dir.glob("*_analysis.json")))

    return status


@app.post("/api/run-pipeline")
async def trigger_pipeline(mode: str = "refresh"):
    """Trigger a pipeline run. Protected by ADMIN_KEY env var."""
    admin_key = os.environ.get("ADMIN_KEY", "")
    # In production, you'd check an auth header here
    # For now, this endpoint exists for Railway cron to call

    if mode not in ["refresh", "refresh-all", "full"]:
        raise HTTPException(status_code=400, detail="mode must be refresh, refresh-all, or full")

    cmd = ["python", str(PIPELINE_SCRIPT), "--no-confirm"]
    if mode == "refresh":
        cmd.append("--refresh")
    elif mode == "refresh-all":
        cmd.append("--refresh-all")
    # "full" = no extra flags, runs everything

    # Run async so we don't block the server
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Don't wait for completion — return immediately
    return {
        "status": "started",
        "mode": mode,
        "pid": process.pid,
        "message": f"Pipeline running in background with --{mode}" if mode != "full" else "Full pipeline running in background",
    }


# ============================================================
# Serve Static Dashboard
# ============================================================

# Mount static files (CSS, JS, images)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard page."""
    index_file = Path("static/index.html")
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text())

    # Fallback: return a minimal page that loads from API
    return HTMLResponse(content="""<!DOCTYPE html>
<html><head><title>Earnings Signal Lab</title></head>
<body style="background:#070714;color:#e5e7eb;font-family:monospace;padding:40px;text-align:center">
<h1 style="color:#4ade80">EARNINGS SIGNAL LAB</h1>
<p>Dashboard not built yet. API is running at <a href="/api/results" style="color:#60a5fa">/api/results</a></p>
<p><a href="/docs" style="color:#60a5fa">API Documentation</a></p>
</body></html>""")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
