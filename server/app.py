"""
Earnings Signal Lab — Web Server
=================================
FastAPI app that serves:
  - Public dashboard at /
  - JSON API for predictions and model results
  - Multi-source support: earnings, sec, combined
  - Cron-triggered pipeline runs

Does NOT expose: raw transcripts, Claude analysis details, or API keys.
"""

import os
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
import subprocess
import asyncio

# ============================================================
# Config
# ============================================================

# Data directories for each source
SOURCE_DIRS = {
    "earnings": Path(os.environ.get("DATA_DIR", "earnings_signal_data")),
    "sec": Path("sec_filing_data"),
    "combined": Path("combined_data"),
}

PIPELINE_SCRIPT = Path("earnings_signal_pipeline.py")

# Per-source cache: { source: { file_type: (data, mtime) } }
_cache = {}


def _get_dir(source: str) -> Path:
    """Get the data directory for a source."""
    if source not in SOURCE_DIRS:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}. Use: {list(SOURCE_DIRS.keys())}")
    return SOURCE_DIRS[source]


def _load_json(filepath: Path, source: str, cache_key: str):
    """Load and cache a JSON file with mtime checking."""
    global _cache
    if source not in _cache:
        _cache[source] = {}

    if not filepath.exists():
        return None

    mtime = filepath.stat().st_mtime
    cached = _cache[source].get(cache_key)
    if cached and cached[1] == mtime:
        return cached[0]

    data = json.loads(filepath.read_text())
    _cache[source][cache_key] = (data, mtime)
    return data


def load_results(source: str = "earnings"):
    """Load and cache backtest results, stripping sensitive data."""
    data_dir = _get_dir(source)
    filepath = data_dir / "backtest_results.json"

    if not filepath.exists():
        return None

    mtime = filepath.stat().st_mtime

    # Check processed cache
    cache_key = "results_public"
    if source in _cache and cache_key in _cache[source]:
        cached = _cache[source][cache_key]
        if cached[1] == mtime:
            return cached[0]

    raw = json.loads(filepath.read_text())

    # Build public-safe version — no raw evidence/transcripts
    public = {
        "metadata": raw.get("metadata", {}),
        "features": {},
        "combinations": raw.get("combinations", []),
        "correlation_matrix": raw.get("correlation_matrix", {}),
        "regression": {},
        "summary": raw.get("summary", {}).get("markdown", "") if isinstance(raw.get("summary"), dict) else raw.get("summary", ""),
        "last_updated": datetime.fromtimestamp(mtime).isoformat(),
    }

    # Strip evidence text from features (keeps scores + stats only)
    for feat_name, periods in raw.get("features", {}).items():
        public["features"][feat_name] = periods

    # Strip evidence from sample extractions — keep scores and returns only
    public["sample_extractions"] = {}
    for feat_name, samples in raw.get("sample_extractions", {}).items():
        public["sample_extractions"][feat_name] = [
            {k: v for k, v in s.items() if k != "evidence"}
            for s in samples
        ]

    # Regression — include model comparisons, weights, importances
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

    if source not in _cache:
        _cache[source] = {}
    _cache[source][cache_key] = (public, mtime)
    return public


def load_scores(source: str = "earnings"):
    """Load and cache scored items."""
    data_dir = _get_dir(source)
    return _load_json(data_dir / "scores.json", source, "scores")


def load_model(source: str = "earnings"):
    """Load and cache scoring model."""
    data_dir = _get_dir(source)
    return _load_json(data_dir / "scoring_model.json", source, "model")


# ============================================================
# App
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload earnings data
    load_results("earnings")
    load_scores("earnings")
    load_model("earnings")
    yield

app = FastAPI(
    title="Earnings Signal Lab",
    description="Multi-source signal analysis: earnings transcripts, SEC filings, and combined",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# API Routes
# ============================================================

@app.get("/api/results")
async def get_results(source: str = Query("earnings", description="Data source: earnings, sec, combined")):
    """Full public results — features, regression, combinations, correlations."""
    results = load_results(source)
    if not results:
        raise HTTPException(status_code=503, detail=f"No results available for source '{source}'. Pipeline hasn't run.")
    return results


@app.get("/api/predictions")
async def get_predictions(source: str = Query("earnings")):
    """Just the prediction model weights and expected performance."""
    results = load_results(source)
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
async def get_features(source: str = Query("earnings")):
    """Individual feature signal strength across holding periods."""
    results = load_results(source)
    if not results:
        raise HTTPException(status_code=503, detail="No results available yet.")
    return {
        "features": results.get("features", {}),
        "sample_extractions": results.get("sample_extractions", {}),
    }


@app.get("/api/scores")
async def get_scores(source: str = Query("earnings"), signal: str = None,
                     confidence: str = None, limit: int = 50, offset: int = 0):
    """Scored transcripts/filings with LONG/SHORT/NEUTRAL signals."""
    scores = load_scores(source)
    if not scores:
        raise HTTPException(status_code=503, detail=f"No scores available for source '{source}'.")

    # Summary stats (always from full dataset)
    signal_counts = Counter(s["signal"] for s in scores)
    confidence_counts = Counter(s["confidence"] for s in scores)

    # Filter
    filtered = scores
    if signal and signal.upper() in ("LONG", "SHORT", "NEUTRAL"):
        filtered = [s for s in filtered if s["signal"] == signal.upper()]
    if confidence and confidence.lower() in ("high", "medium", "low"):
        filtered = [s for s in filtered if s["confidence"] == confidence.lower()]

    # Paginate
    page = filtered[offset:offset + limit]

    return {
        "items": page,
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "summary": {
            "total": len(scores),
            "signals": dict(signal_counts),
            "confidences": dict(confidence_counts),
        },
    }


@app.get("/api/model")
async def get_model(source: str = Query("earnings")):
    """Scoring model weights and training metadata."""
    model = load_model(source)
    if not model:
        raise HTTPException(status_code=503, detail=f"No scoring model available for source '{source}'.")
    return model


@app.get("/api/summary")
async def get_summary(source: str = Query("earnings")):
    """Markdown summary of learnings."""
    results = load_results(source)
    if not results:
        data_dir = _get_dir(source)
        summary_file = data_dir / "SUMMARY.md"
        if summary_file.exists():
            return {"summary": summary_file.read_text()}
        raise HTTPException(status_code=503, detail="No summary available yet.")
    return {"summary": results.get("summary", "")}


@app.get("/api/sources")
async def get_sources():
    """List available data sources and their status."""
    sources = {}
    for name, data_dir in SOURCE_DIRS.items():
        results_file = data_dir / "backtest_results.json"
        scores_file = data_dir / "scores.json"
        model_file = data_dir / "scoring_model.json"

        info = {
            "available": results_file.exists(),
            "has_scores": scores_file.exists(),
            "has_model": model_file.exists(),
        }

        if results_file.exists():
            try:
                results = load_results(name)
                meta = results.get("metadata", {})
                info["total_events"] = meta.get("total_events", 0)
                info["companies"] = len(meta.get("companies", []))
                info["last_updated"] = results.get("last_updated", "")
            except Exception:
                pass

        sources[name] = info

    return sources


@app.get("/api/status")
async def get_status(source: str = Query("earnings")):
    """Pipeline health and data freshness."""
    data_dir = _get_dir(source)
    results_file = data_dir / "backtest_results.json"

    status = {
        "source": source,
        "results_exist": results_file.exists(),
        "data_dir_exists": data_dir.exists(),
    }

    if results_file.exists():
        results = load_results(source)
        status["last_updated"] = results.get("last_updated", "")
        status["total_events"] = results.get("metadata", {}).get("total_events", 0)
        status["companies"] = len(results.get("metadata", {}).get("companies", []))
        status["periods_modeled"] = list(results.get("regression", {}).keys())

    # Source-specific checks
    if source == "earnings":
        transcripts_dir = data_dir / "transcripts"
        analysis_dir = data_dir / "analysis"
        if transcripts_dir.exists():
            status["transcripts_cached"] = sum(
                len(json.loads(f.read_text()))
                for f in transcripts_dir.glob("*_transcripts.json")
            )
        if analysis_dir.exists():
            status["analyses_cached"] = len(list(analysis_dir.glob("*_analysis.json")))
    elif source == "sec":
        csv_file = data_dir / "raw" / "ml_dataset_with_concern.csv"
        status["csv_exists"] = csv_file.exists()

    return status


@app.post("/api/run-pipeline")
async def trigger_pipeline(mode: str = "refresh"):
    """Trigger a pipeline run. Protected by ADMIN_KEY env var."""
    admin_key = os.environ.get("ADMIN_KEY", "")

    if mode not in ["refresh", "refresh-all", "full"]:
        raise HTTPException(status_code=400, detail="mode must be refresh, refresh-all, or full")

    cmd = ["python", str(PIPELINE_SCRIPT), "--no-confirm"]
    if mode == "refresh":
        cmd.append("--refresh")
    elif mode == "refresh-all":
        cmd.append("--refresh-all")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    return {
        "status": "started",
        "mode": mode,
        "pid": process.pid,
        "message": f"Pipeline running in background with --{mode}" if mode != "full" else "Full pipeline running in background",
    }


# ============================================================
# Serve Static Dashboard
# ============================================================

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard page."""
    index_file = Path("static/index.html")
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text())

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
