"""
Earnings Transcript Signal Analyzer - Real Data Pipeline
=========================================================
Loads S&P 500 earnings transcripts from HuggingFace (glopardo/sp500-earnings-transcripts),
analyzes them with Claude API for 16 granular NLP features, and backtests against real
price data.

Requirements:
    pip install datasets yfinance pandas numpy anthropic scipy

Setup:
    1. Get a Claude API key at https://console.anthropic.com
    2. Set environment variable:
       export ANTHROPIC_API_KEY="your_claude_key"

Usage:
    python earnings_signal_pipeline.py                  # Full pipeline
    python earnings_signal_pipeline.py --refresh        # Update incomplete prices & re-backtest
    python earnings_signal_pipeline.py --refresh-all    # Re-fetch ALL prices & re-backtest
    python earnings_signal_pipeline.py --status         # Show what's cached
    python earnings_signal_pipeline.py --step pull      # Only load transcripts from HuggingFace
    python earnings_signal_pipeline.py --step analyze   # Only run Claude analysis
    python earnings_signal_pipeline.py --step prices    # Only fetch price data
    python earnings_signal_pipeline.py --step backtest  # Only re-run backtest on existing data
    python earnings_signal_pipeline.py --no-confirm     # Skip prompts (for cron jobs)

Caching:
    All data is cached aggressively. Re-runs only process new/incomplete data.
    - Transcripts:  earnings_signal_data/transcripts/
    - Analyses:     earnings_signal_data/analysis/
    - Price data:   earnings_signal_data/price_data.json
    - Results:      earnings_signal_data/backtest_results.json
"""

import os
import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from anthropic import Anthropic

# Max trading days needed for longest holding period to fully mature
MAX_HOLDING_CALENDAR_DAYS = 35  # ~21 trading days + weekends/holidays buffer

# ============================================================
# CONFIGURATION
# ============================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY_HERE")

# HuggingFace dataset: ~20.7k transcripts across ~496 S&P 500 tickers (2014-Nov 2025)
HF_DATASET = "glopardo/sp500-earnings-transcripts"

# Populated dynamically from the HuggingFace dataset during pull step
COMPANIES = []

# How many quarters back to analyze per company (0 = use all available data)
QUARTERS_BACK = 0

# Year range filter (set via --years CLI arg; None = no filter)
YEAR_RANGE = None  # e.g. (2023, 2025)

# Concurrent API calls for analysis step
CONCURRENCY = 10

# Holding periods to test (trading days)
HOLDING_PERIODS = {
    "1D": 1,
    "5D": 5,
    "10D": 10,
    "21D": 21,
}

# Output paths
DATA_DIR = Path("earnings_signal_data")
DATA_DIR.mkdir(exist_ok=True)

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = DATA_DIR / "backtest_results.json"

# ============================================================
# FEATURE EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = """You are a quantitative analyst extracting granular NLP features from an earnings call transcript.

Analyze this transcript and extract EXACTLY the following 16 features. For each feature, provide:
- score: float 0.0 to 1.0 (where the meaning depends on the feature)
- evidence: 1-2 specific quotes or observations from the transcript
- section: whether this came from "prepared" remarks or "qa" section or "both"

FEATURES TO EXTRACT:

1. mgmt_hedging (0=very definitive, 1=very hedged)
   Count hedging words: "approximately", "roughly", "we believe", "potentially", "subject to", "could", "might", "may"
   vs definitive words: "we will", "revenue will exceed", "we are committed to"

2. mgmt_deflection (0=direct answers, 1=constant deflection)
   In Q&A: does management directly answer questions or redirect, pivot, give non-answers?

3. mgmt_specificity (0=vague/qualitative, 1=very specific numbers)
   Does guidance use specific numbers, ranges, or vague qualitative language?

4. mgmt_confidence_shift (0=consistent confidence, 1=large drop from prepared to Q&A)
   Compare tone/confidence in scripted remarks vs live Q&A responses.

5. analyst_skepticism (0=accepting, 1=highly skeptical)
   Do analysts ask challenging follow-ups, express doubt, re-ask questions differently?

6. analyst_surprise (0=expected info, 1=lots of surprises)
   Language indicating unexpected information: clarification requests, pauses, rephrasing.

7. analyst_focus_cluster (0=dispersed questions, 1=all on same topic)
   Are multiple analysts asking about the same specific concern/topic?
   If clustered, what topic? Score higher if 3+ analysts converge.

8. guidance_revision_dir (0=lowered, 0.5=maintained, 1=raised)
   Direction of guidance changes relative to prior quarter/consensus.

9. guidance_qualifiers (0=clean guidance, 1=heavily qualified)
   Density of conditional language: "assuming", "if", "barring", "dependent on", "subject to"

10. new_risk_mention (0=no new risks, 1=multiple new risks)
    Risk factors mentioned for the first time vs prior calls.

11. macro_blame (0=takes ownership, 1=blames external factors)
    Attributing misses or weakness to macro, FX, supply chain, weather, regulation.

12. capex_language (0=cautious pullback, 0.5=disciplined, 1=aggressive expansion)
    Tone around capital expenditure and investment plans.

13. hiring_language (0=cutting/restructuring, 0.5=flat, 1=aggressive hiring)
    References to headcount, hiring, restructuring, efficiency.

14. competitive_mentions (0=dismissive/no mention, 0.5=neutral, 1=concerned/defensive)
    How management discusses competitors - unprompted mentions suggest concern.

15. customer_language (0=softening/normalizing, 0.5=stable, 1=record/robust)
    Specific demand descriptors: pipeline, backlog, "robust", "record", "softening", "normalizing"

16. pricing_power (0=under pressure/discounting, 0.5=stable, 1=raising prices easily)
    References to price increases, pricing pressure, mix shifts, discounting.

Respond ONLY with valid JSON in this exact format (no markdown, no backticks):
{
  "features": {
    "mgmt_hedging": {"score": 0.0, "evidence": "...", "section": "prepared|qa|both"},
    "mgmt_deflection": {"score": 0.0, "evidence": "...", "section": "qa"},
    ... (all 16 features)
  },
  "qa_vs_prepared_summary": "Brief note on key differences between prepared remarks and Q&A",
  "most_notable_signal": "The single most important finding from this transcript"
}
"""

# ============================================================
# STEP 1: Load Transcripts from HuggingFace Dataset
# ============================================================

def pull_all_transcripts():
    """Load transcripts from the HuggingFace S&P 500 earnings dataset."""
    global COMPANIES

    print("=" * 60)
    print("STEP 1: Loading Transcripts from HuggingFace Dataset")
    print(f"        Dataset: {HF_DATASET}")
    print("=" * 60)

    print("\nLoading dataset (cached locally after first download)...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"  Dataset loaded: {len(ds)} records")

    # Filter to recent N quarters if QUARTERS_BACK > 0
    if QUARTERS_BACK > 0:
        now = datetime.now()
        current_q = (now.month - 1) // 3 + 1
        current_y = now.year
        cutoff_quarters = set()
        q, y = current_q, current_y
        for _ in range(QUARTERS_BACK):
            q -= 1
            if q == 0:
                q = 4
                y -= 1
            cutoff_quarters.add((q, y))
        print(f"  Filtering to last {QUARTERS_BACK} quarters")

    # Group records by ticker
    by_ticker = defaultdict(list)
    skipped_null = 0
    skipped_empty = 0
    skipped_quarter_filter = 0

    for row in ds:
        ticker = row.get("ticker")
        year = row.get("year")
        quarter = row.get("quarter")
        transcript = row.get("transcript", "")

        # Skip records with missing required fields
        if not ticker or year is None or quarter is None:
            skipped_null += 1
            continue
        if not transcript or len(transcript.strip()) < 100:
            skipped_empty += 1
            continue

        # Apply quarter filter if set
        if QUARTERS_BACK > 0 and (quarter, year) not in cutoff_quarters:
            skipped_quarter_filter += 1
            continue

        entry = {
            "symbol": ticker,
            "quarter": int(quarter),
            "year": int(year),
            "content": transcript,
            "date": row.get("earnings_date"),
        }
        by_ticker[ticker].append(entry)

    print(f"  Valid records: {sum(len(v) for v in by_ticker.values())}")
    if skipped_null > 0:
        print(f"  Skipped (null ticker/year/quarter): {skipped_null}")
    if skipped_empty > 0:
        print(f"  Skipped (empty transcript): {skipped_empty}")
    if skipped_quarter_filter > 0:
        print(f"  Skipped (outside quarter filter): {skipped_quarter_filter}")

    # Derive COMPANIES from the dataset
    COMPANIES = sorted(by_ticker.keys())
    print(f"  Tickers found: {len(COMPANIES)}")

    # Merge with existing cache and write per-ticker JSON files
    total_new = 0
    total_existing = 0

    for ticker in COMPANIES:
        cache_file = TRANSCRIPTS_DIR / f"{ticker}_transcripts.json"

        # Load existing cache
        if cache_file.exists():
            existing = json.loads(cache_file.read_text())
        else:
            existing = []

        existing_keys = {(t["quarter"], t["year"]) for t in existing if "quarter" in t and "year" in t}

        new_count = 0
        for entry in by_ticker[ticker]:
            key = (entry["quarter"], entry["year"])
            if key not in existing_keys:
                existing.append(entry)
                existing_keys.add(key)
                new_count += 1

        total_new += new_count
        total_existing += len(existing) - new_count

        _save_transcripts(cache_file, existing)

    print(f"\nDone: {total_new} new transcripts cached, {total_existing} already existed")
    print(f"  {len(COMPANIES)} tickers across {sum(len(json.loads(f.read_text())) for f in TRANSCRIPTS_DIR.glob('*_transcripts.json'))} total transcripts")
    return total_new


def _save_transcripts(cache_file, transcripts):
    cache_file.write_text(json.dumps(transcripts, indent=2))


# ============================================================
# STEP 2: Analyze Transcripts with Claude
# ============================================================

def analyze_transcript(client: Anthropic, symbol: str, quarter: int, year: int, content: str) -> dict | None:
    """Use Claude to extract 16 granular features from a transcript."""
    analysis_file = ANALYSIS_DIR / f"{symbol}_Q{quarter}_{year}_analysis.json"

    # Return cached analysis if exists
    if analysis_file.exists():
        return json.loads(analysis_file.read_text())

    # Truncate very long transcripts to stay within context
    # Most transcripts are 15-40k chars, Claude can handle this
    if len(content) > 80000:
        # Keep first 30k (prepared) + last 50k (Q&A, which is more important)
        content = content[:30000] + "\n\n[... middle section truncated ...]\n\n" + content[-50000:]

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"Here is the earnings call transcript for {symbol} Q{quarter} {year}:\n\n{content}\n\n{EXTRACTION_PROMPT}"
                }
            ],
        )

        text = response.content[0].text.strip()
        # Clean potential markdown formatting
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        analysis = json.loads(text)
        analysis["symbol"] = symbol
        analysis["quarter"] = quarter
        analysis["year"] = year

        # Save to cache
        analysis_file.write_text(json.dumps(analysis, indent=2))
        print(f"  ✓ {symbol} Q{quarter} {year} analyzed")
        return analysis

    except json.JSONDecodeError as e:
        print(f"  ✗ {symbol} Q{quarter} {year} - JSON parse error: {e}")
        # Save raw response for debugging
        debug_file = ANALYSIS_DIR / f"{symbol}_Q{quarter}_{year}_raw.txt"
        debug_file.write_text(text if 'text' in dir() else "no response")
        return None
    except Exception as e:
        print(f"  ✗ {symbol} Q{quarter} {year} - Error: {e}")
        return None


def analyze_all_transcripts():
    """Run Claude analysis on all cached transcripts using concurrent workers."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Analyzing Transcripts with Claude API ({CONCURRENCY} workers)")
    print("=" * 60)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build work queue: collect all (symbol, quarter, year, content) needing analysis
    work_items = []
    total_cached = 0

    for cache_file in sorted(TRANSCRIPTS_DIR.glob("*_transcripts.json")):
        symbol = cache_file.stem.replace("_transcripts", "")
        transcripts = json.loads(cache_file.read_text())

        for t in transcripts:
            content = t.get("content", "")
            if not content or len(content) < 500:
                continue

            q = t.get("quarter")
            y = t.get("year")

            # Apply year range filter
            if YEAR_RANGE and y is not None and not (YEAR_RANGE[0] <= y <= YEAR_RANGE[1]):
                continue

            # Check if already analyzed
            analysis_file = ANALYSIS_DIR / f"{symbol}_Q{q}_{y}_analysis.json"
            if analysis_file.exists():
                total_cached += 1
                continue

            work_items.append((symbol, q, y, content))

    print(f"\n  {len(work_items)} transcripts to analyze, {total_cached} already cached")

    if not work_items:
        print("  Nothing to do.")
        return

    total_analyzed = 0
    total_failed = 0

    def _analyze_one(item):
        symbol, q, y, content = item
        return analyze_transcript(client, symbol, q, y, content)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_analyze_one, item): item for item in work_items}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                total_analyzed += 1
            else:
                total_failed += 1
            if i % 50 == 0:
                print(f"  ... {i}/{len(work_items)} done ({total_analyzed} ok, {total_failed} failed)")

    print(f"\nDone: {total_analyzed} newly analyzed, {total_failed} failed, {total_cached} cached")


# ============================================================
# STEP 3: Get Price Data
# ============================================================

def get_earnings_date_from_transcript(symbol: str, quarter: int, year: int) -> str | None:
    """Try to find the actual earnings date from transcript metadata."""
    cache_file = TRANSCRIPTS_DIR / f"{symbol}_transcripts.json"
    if not cache_file.exists():
        return None

    transcripts = json.loads(cache_file.read_text())
    for t in transcripts:
        if t.get("quarter") == quarter and t.get("year") == year:
            date_str = t.get("date", "")
            if date_str:
                try:
                    return date_str[:10]  # YYYY-MM-DD
                except:
                    pass
    return None


def get_price_data(symbol: str, start_date: str, days_after: int = 30) -> pd.DataFrame | None:
    """Get daily price data around an earnings date."""
    try:
        start = pd.Timestamp(start_date) - timedelta(days=5)  # A few days before
        end = pd.Timestamp(start_date) + timedelta(days=days_after + 10)  # Buffer

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            return None

        return hist
    except Exception as e:
        print(f"  Error getting price for {symbol}: {e}")
        return None


def compute_returns(symbol: str, earnings_date: str) -> dict | None:
    """Compute forward returns at various horizons from earnings date."""
    prices = get_price_data(symbol, earnings_date)
    if prices is None or len(prices) < 5:
        return None

    earnings_ts = pd.Timestamp(earnings_date)
    # Match timezone of Yahoo Finance index if needed
    if prices.index.tz is not None:
        earnings_ts = earnings_ts.tz_localize(prices.index.tz)

    # Find the first trading day on or after earnings date
    future_prices = prices[prices.index >= earnings_ts]
    if len(future_prices) < 2:
        return None

    # Use close on earnings day (or next trading day) as base
    base_price = future_prices.iloc[0]["Close"]

    returns = {}
    for label, days in HOLDING_PERIODS.items():
        if len(future_prices) > days:
            future_price = future_prices.iloc[days]["Close"]
            ret = (future_price - base_price) / base_price * 100
            returns[label] = round(float(ret), 4)
        else:
            returns[label] = None

    # Also get pre-earnings close for earnings day return
    pre_prices = prices[prices.index < earnings_ts]
    if len(pre_prices) > 0:
        pre_close = pre_prices.iloc[-1]["Close"]
        earnings_day_return = (base_price - pre_close) / pre_close * 100
        returns["earnings_day"] = round(float(earnings_day_return), 4)

    return returns


def gather_all_price_data(refresh=False):
    """Get price data for all analyzed transcripts.
    
    Args:
        refresh: If True, re-fetch price data for events that have:
                 - Any None returns (holding period hadn't elapsed yet)
                 - Earnings date within the last MAX_HOLDING_CALENDAR_DAYS
                 If False, only fetch price data for events not yet in the cache.
    """
    print("\n" + "=" * 60)
    print(f"STEP 3: Gathering Price Data from Yahoo Finance {'(REFRESH MODE)' if refresh else ''}")
    print("=" * 60)

    # Load existing price data cache
    price_file = DATA_DIR / "price_data.json"
    if price_file.exists():
        existing_price_data = json.loads(price_file.read_text())
    else:
        existing_price_data = {}

    price_data = dict(existing_price_data)  # Start with existing
    total_new = 0
    total_refreshed = 0
    total_skipped = 0
    today = datetime.now()

    for analysis_file in ANALYSIS_DIR.glob("*_analysis.json"):
        analysis = json.loads(analysis_file.read_text())
        symbol = analysis.get("symbol")
        quarter = analysis.get("quarter")
        year = analysis.get("year")

        if not all([symbol, quarter, year]):
            continue

        # Apply year range filter
        if YEAR_RANGE and not (YEAR_RANGE[0] <= year <= YEAR_RANGE[1]):
            continue

        key = f"{symbol}_Q{quarter}_{year}"

        # Get earnings date
        earnings_date = get_earnings_date_from_transcript(symbol, quarter, year)
        if not earnings_date:
            month_map = {1: 4, 2: 7, 3: 10, 4: 1}
            est_month = month_map.get(quarter, 4)
            est_year = year + 1 if quarter == 4 else year
            earnings_date = f"{est_year}-{est_month:02d}-15"

        # Decide whether to fetch/refresh
        needs_fetch = False
        reason = ""

        if key not in existing_price_data:
            needs_fetch = True
            reason = "new"
        elif refresh:
            cached = existing_price_data[key]
            cached_returns = cached.get("returns", {})

            # Check 1: Any None returns (period hadn't elapsed when last fetched)
            has_incomplete = any(v is None for k, v in cached_returns.items() if k != "earnings_day")
            
            # Check 2: Earnings date is recent enough that returns may have changed
            try:
                ed = pd.Timestamp(cached.get("earnings_date", earnings_date))
                days_since = (today - ed).days
                is_recent = days_since < MAX_HOLDING_CALENDAR_DAYS
            except:
                is_recent = False

            if has_incomplete:
                needs_fetch = True
                reason = "incomplete returns"
            elif is_recent:
                needs_fetch = True
                reason = f"recent ({days_since}d ago)"

        if not needs_fetch:
            total_skipped += 1
            continue

        returns = compute_returns(symbol, earnings_date)
        if returns:
            price_data[key] = {
                "symbol": symbol,
                "quarter": quarter,
                "year": year,
                "earnings_date": earnings_date,
                "returns": returns,
                "last_updated": today.isoformat(),
            }
            if reason == "new":
                total_new += 1
            else:
                total_refreshed += 1
            
            # Show what changed for refreshed entries
            if reason != "new" and key in existing_price_data:
                old_rets = existing_price_data[key].get("returns", {})
                changes = []
                for period in HOLDING_PERIODS:
                    old_val = old_rets.get(period)
                    new_val = returns.get(period)
                    if old_val is None and new_val is not None:
                        changes.append(f"{period}: None→{new_val:+.1f}%")
                    elif old_val is not None and new_val is not None and abs(old_val - new_val) > 0.01:
                        changes.append(f"{period}: {old_val:+.1f}%→{new_val:+.1f}%")
                change_str = ", ".join(changes) if changes else "no change"
                print(f"  ↻ {key} [{reason}]: {change_str}")
            else:
                print(f"  ✓ {key}: 1D={returns.get('1D', 'N/A')}%, 5D={returns.get('5D', 'N/A')}%")
        else:
            print(f"  ✗ {key}: no price data")

        time.sleep(0.1)  # Be nice to Yahoo

    # Save price data
    price_file.write_text(json.dumps(price_data, indent=2))
    print(f"\nDone: {total_new} new, {total_refreshed} refreshed, {total_skipped} unchanged")
    return price_data


# ============================================================
# STEP 4: Backtest - Combine Features + Returns
# ============================================================

def run_backtest():
    """Combine NLP features with price data to test signal strength."""
    print("\n" + "=" * 60)
    print("STEP 4: Running Backtest")
    print("=" * 60)

    # Load price data
    price_file = DATA_DIR / "price_data.json"
    if not price_file.exists():
        print("No price data found. Run step 3 first.")
        return

    price_data = json.loads(price_file.read_text())

    # Build dataset: each row = one earnings event with features + returns
    rows = []

    for key, pdata in price_data.items():
        symbol = pdata["symbol"]
        quarter = pdata["quarter"]
        year = pdata["year"]
        returns = pdata["returns"]

        # Load corresponding analysis
        analysis_file = ANALYSIS_DIR / f"{symbol}_Q{quarter}_{year}_analysis.json"
        if not analysis_file.exists():
            continue

        analysis = json.loads(analysis_file.read_text())
        features = analysis.get("features", {})

        row = {
            "symbol": symbol,
            "quarter": quarter,
            "year": year,
            "earnings_date": pdata["earnings_date"],
        }

        # Add feature scores
        for feat_name, feat_data in features.items():
            if isinstance(feat_data, dict):
                row[f"feat_{feat_name}"] = feat_data.get("score", None)
                row[f"evidence_{feat_name}"] = feat_data.get("evidence", "")
                row[f"section_{feat_name}"] = feat_data.get("section", "")

        # Add returns
        for period, ret in returns.items():
            row[f"return_{period}"] = ret

        rows.append(row)

    if not rows:
        print("No matched data found.")
        return

    df = pd.DataFrame(rows)
    print(f"\nDataset: {len(df)} earnings events × {len(df.columns)} columns")
    print(f"Companies: {df['symbol'].nunique()}")
    print(f"Date range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")

    # ============================================================
    # SIGNAL ANALYSIS FOR EACH FEATURE
    # ============================================================

    # Only use the 16 known features (ignore any extras hallucinated by the model)
    KNOWN_FEATURES = [
        "mgmt_hedging", "mgmt_deflection", "mgmt_specificity", "mgmt_confidence_shift",
        "analyst_skepticism", "analyst_surprise", "analyst_focus_cluster",
        "guidance_revision_dir", "guidance_qualifiers",
        "new_risk_mention", "macro_blame", "capex_language", "hiring_language",
        "competitive_mentions", "customer_language", "pricing_power",
    ]
    feature_cols = [f"feat_{f}" for f in KNOWN_FEATURES if f"feat_{f}" in df.columns]
    return_cols = [c for c in df.columns if c.startswith("return_") and c != "return_earnings_day"]

    results = {
        "metadata": {
            "total_events": len(df),
            "companies": sorted(df["symbol"].unique().tolist()),
            "date_range": [df["earnings_date"].min(), df["earnings_date"].max()],
            "generated_at": datetime.now().isoformat(),
        },
        "features": {},
        "sample_extractions": {},
    }

    print("\n--- Feature Signal Analysis ---\n")
    print(f"{'Feature':<30} {'Period':<8} {'IC':<10} {'Accuracy':<10} {'Sharpe':<10} {'p-value':<10} {'n':<6}")
    print("-" * 90)

    for feat_col in feature_cols:
        feat_name = feat_col.replace("feat_", "")
        feat_results = {}

        for ret_col in return_cols:
            period = ret_col.replace("return_", "")

            # Drop NaN
            valid = df[[feat_col, ret_col]].dropna()
            if len(valid) < 20:
                continue

            scores = valid[feat_col].values
            returns = valid[ret_col].values

            # Information Coefficient (rank correlation)
            from scipy import stats as scipy_stats
            ic, ic_pvalue = scipy_stats.spearmanr(scores, returns)

            # Directional accuracy
            # For bearish features (hedging, deflection, etc.) high score = negative return expected
            # For bullish features (specificity, customer_language) high score = positive return
            bearish_features = [
                "mgmt_hedging", "mgmt_deflection", "mgmt_confidence_shift",
                "analyst_skepticism", "guidance_qualifiers", "new_risk_mention",
                "macro_blame", "competitive_mentions",
            ]

            if feat_name in bearish_features:
                # High score should predict negative returns
                predictions = scores > 0.5
                actuals = returns < 0
            else:
                # High score should predict positive returns
                predictions = scores > 0.5
                actuals = returns > 0

            accuracy = np.mean(predictions == actuals) if len(predictions) > 0 else 0.5

            # Win rate: returns when signal is triggered
            signal_on = returns[scores > 0.6]
            signal_off = returns[scores < 0.4]

            if feat_name in bearish_features:
                signal_returns = -signal_on  # Flip sign for short signal
            else:
                signal_returns = signal_on

            if len(signal_returns) > 1:
                avg_ret = np.mean(signal_returns)
                std_ret = np.std(signal_returns)
                sharpe = avg_ret / std_ret * np.sqrt(4) if std_ret > 0 else 0  # Annualize quarterly
                win_rate = np.mean(signal_returns > 0)
            else:
                avg_ret = 0
                sharpe = 0
                win_rate = 0.5

            feat_results[period] = {
                "ic": round(float(ic), 4),
                "ic_pvalue": round(float(ic_pvalue), 4),
                "accuracy": round(float(accuracy), 4),
                "sharpe": round(float(sharpe), 4),
                "avg_return_pct": round(float(avg_ret), 4),
                "win_rate": round(float(win_rate), 4),
                "n_observations": len(valid),
                "n_signal_triggered": len(signal_returns),
            }

            print(f"{feat_name:<30} {period:<8} {ic:>8.4f}  {accuracy:>8.1%}  {sharpe:>8.2f}  {ic_pvalue:>8.4f}  {len(valid):<6}")

        results["features"][feat_name] = feat_results

        # Gather sample extractions for this feature
        samples = []
        valid_rows = df[df[feat_col].notna()].sort_values(feat_col, ascending=False)
        for _, row in valid_rows.head(3).iterrows():  # Top 3 highest scores
            samples.append({
                "symbol": row["symbol"],
                "quarter": f"Q{row['quarter']} {row['year']}",
                "score": round(float(row[feat_col]), 2),
                "evidence": row.get(f"evidence_{feat_name}", ""),
                "return_5D": row.get("return_5D"),
                "return_10D": row.get("return_10D"),
            })
        for _, row in valid_rows.tail(3).iterrows():  # Bottom 3 lowest scores
            samples.append({
                "symbol": row["symbol"],
                "quarter": f"Q{row['quarter']} {row['year']}",
                "score": round(float(row[feat_col]), 2),
                "evidence": row.get(f"evidence_{feat_name}", ""),
                "return_5D": row.get("return_5D"),
                "return_10D": row.get("return_10D"),
            })
        results["sample_extractions"][feat_name] = samples

    # ============================================================
    # CORRELATION MATRIX BETWEEN FEATURES
    # ============================================================

    feat_df = df[feature_cols].dropna(axis=1, how="all")
    if len(feat_df.columns) > 1:
        corr = feat_df.corr(method="spearman")
        results["correlation_matrix"] = {
            c1.replace("feat_", ""): {
                c2.replace("feat_", ""): round(float(corr.loc[c1, c2]), 3)
                for c2 in corr.columns
            }
            for c1 in corr.index
        }
    else:
        results["correlation_matrix"] = {}

    # ============================================================
    # MULTI-FEATURE COMBINATIONS
    # ============================================================

    print("\n--- Multi-Feature Combinations ---\n")

    combos = [
        {
            "name": "Confidence Gap + Analyst Skepticism",
            "bullish_when": "both low",
            "features": ["mgmt_confidence_shift", "analyst_skepticism"],
            "bearish_threshold": [0.6, 0.6],
        },
        {
            "name": "Guidance Raise + Low Hedging + Strong Demand",
            "bullish_when": "guidance high, hedging low, demand high",
            "features": ["guidance_revision_dir", "mgmt_hedging", "customer_language"],
            "bullish_threshold": [0.7, 0.3, 0.7],  # dir > 0.7, hedging < 0.3, demand > 0.7
        },
        {
            "name": "Deflection + Question Clustering",
            "bullish_when": "both low",
            "features": ["mgmt_deflection", "analyst_focus_cluster"],
            "bearish_threshold": [0.6, 0.7],
        },
        {
            "name": "New Risks + External Blame + Guidance Qualifiers",
            "bullish_when": "all low",
            "features": ["new_risk_mention", "macro_blame", "guidance_qualifiers"],
            "bearish_threshold": [0.6, 0.6, 0.6],
        },
        {
            "name": "Pricing Power + CapEx Confidence",
            "bullish_when": "both high",
            "features": ["pricing_power", "capex_language"],
            "bullish_threshold": [0.6, 0.6],
        },
    ]

    combo_results = []
    for combo in combos:
        feat_names = combo["features"]
        cols = [f"feat_{f}" for f in feat_names]

        if not all(c in df.columns for c in cols):
            continue

        valid = df[cols + ["return_5D", "return_10D"]].dropna()
        if len(valid) < 15:
            continue

        # Simple signal: check thresholds
        # This is a simplified version - a real backtest would be more rigorous
        if "bearish_threshold" in combo:
            thresholds = combo["bearish_threshold"]
            signal = pd.Series(True, index=valid.index)
            for col, thresh in zip(cols, thresholds):
                signal &= valid[col] > thresh
            # Bearish signal: short
            signal_returns_5d = -valid.loc[signal, "return_5D"]
            signal_returns_10d = -valid.loc[signal, "return_10D"]
        elif "bullish_threshold" in combo:
            # Custom logic for mixed thresholds
            thresholds = combo["bullish_threshold"]
            signal = pd.Series(True, index=valid.index)
            for col, thresh in zip(cols, thresholds):
                if thresh < 0.5:
                    signal &= valid[col] < thresh
                else:
                    signal &= valid[col] > thresh
            signal_returns_5d = valid.loc[signal, "return_5D"]
            signal_returns_10d = valid.loc[signal, "return_10D"]

        n_triggered = signal.sum()
        if n_triggered < 3:
            continue

        for period_label, sig_rets in [("5D", signal_returns_5d), ("10D", signal_returns_10d)]:
            if len(sig_rets) < 3:
                continue
            avg = sig_rets.mean()
            std = sig_rets.std()
            sharpe = avg / std * np.sqrt(4) if std > 0 else 0
            win = (sig_rets > 0).mean()

            combo_results.append({
                "name": combo["name"],
                "features": feat_names,
                "period": period_label,
                "sharpe": round(float(sharpe), 3),
                "avg_return": round(float(avg), 3),
                "win_rate": round(float(win), 3),
                "n_triggered": int(n_triggered),
                "n_total": len(valid),
            })

            print(f"  {combo['name']:<45} {period_label}  Sharpe={sharpe:.2f}  Win={win:.1%}  n={n_triggered}")

    results["combinations"] = combo_results

    # ============================================================
    # STEP 5: Regression Modeling & Feature Weighting
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 5: Regression Modeling & Optimal Weightings")
    print("=" * 60)

    from scipy import stats as scipy_stats

    try:
        from sklearn.linear_model import (
            LinearRegression, Ridge, Lasso, ElasticNet
        )
        from sklearn.ensemble import (
            RandomForestRegressor, GradientBoostingRegressor
        )
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.model_selection import (
            TimeSeriesSplit, cross_val_score, cross_val_predict
        )
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.feature_selection import (
            SequentialFeatureSelector, mutual_info_regression
        )
        HAS_SKLEARN = True
    except ImportError:
        print("\n⚠️  scikit-learn not installed. Installing...")
        os.system("pip install scikit-learn --break-system-packages 2>/dev/null || pip install scikit-learn")
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler, PolynomialFeatures
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_regression
            HAS_SKLEARN = True
        except ImportError:
            print("  Could not install scikit-learn. Skipping regression analysis.")
            HAS_SKLEARN = False

    if HAS_SKLEARN:
        regression_results = {}

        for ret_col in return_cols:
            period = ret_col.replace("return_", "")
            print(f"\n{'='*50}")
            print(f"  Modeling: {period} forward returns")
            print(f"{'='*50}")

            # Build clean feature matrix
            valid = df[feature_cols + [ret_col]].dropna()
            if len(valid) < 30:
                print(f"  Skipping — only {len(valid)} observations (need 30+)")
                continue

            X = valid[feature_cols].values
            y = valid[ret_col].values
            feature_names = [c.replace("feat_", "") for c in feature_cols]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Time-series aware cross-validation
            # Sort by date to respect temporal ordering
            valid_sorted = valid.copy()
            valid_sorted["_date"] = df.loc[valid.index, "earnings_date"]
            valid_sorted = valid_sorted.sort_values("_date")
            X_sorted = scaler.transform(valid_sorted[feature_cols].values)
            y_sorted = valid_sorted[ret_col].values
            n_splits = min(5, max(2, len(valid) // 20))
            tscv = TimeSeriesSplit(n_splits=n_splits)

            period_results = {
                "n_observations": len(valid),
                "y_mean": round(float(y.mean()), 4),
                "y_std": round(float(y.std()), 4),
            }

            # --------------------------------------------------------
            # 5a. OLS Linear Regression (baseline)
            # --------------------------------------------------------
            print("\n  --- OLS Linear Regression ---")
            ols = LinearRegression()
            ols.fit(X_scaled, y)

            # In-sample metrics
            y_pred_ols = ols.predict(X_scaled)
            r2_in = r2_score(y, y_pred_ols)

            # Cross-validated metrics
            cv_scores_ols = cross_val_score(ols, X_sorted, y_sorted, cv=tscv,
                                            scoring="neg_mean_squared_error")
            cv_r2_ols = cross_val_score(ols, X_sorted, y_sorted, cv=tscv,
                                        scoring="r2")

            ols_weights = dict(zip(feature_names, [round(float(c), 4) for c in ols.coef_]))
            ols_sorted = sorted(ols_weights.items(), key=lambda x: abs(x[1]), reverse=True)

            print(f"  In-sample R²: {r2_in:.4f}")
            print(f"  CV R² (mean): {cv_r2_ols.mean():.4f} ± {cv_r2_ols.std():.4f}")
            print(f"  CV RMSE:      {np.sqrt(-cv_scores_ols.mean()):.4f}")
            print(f"  Top features by |coefficient|:")
            for name, weight in ols_sorted[:5]:
                direction = "↑ bullish" if weight > 0 else "↓ bearish"
                print(f"    {name:<35} {weight:>8.4f}  ({direction})")

            period_results["ols"] = {
                "r2_insample": round(float(r2_in), 4),
                "r2_cv_mean": round(float(cv_r2_ols.mean()), 4),
                "r2_cv_std": round(float(cv_r2_ols.std()), 4),
                "rmse_cv": round(float(np.sqrt(-cv_scores_ols.mean())), 4),
                "intercept": round(float(ols.intercept_), 4),
                "coefficients": ols_weights,
                "top_features": [{"feature": n, "weight": w} for n, w in ols_sorted[:8]],
            }

            # --------------------------------------------------------
            # 5b. Ridge Regression (L2 regularization)
            # --------------------------------------------------------
            print("\n  --- Ridge Regression (L2) ---")
            best_ridge = None
            best_ridge_score = -np.inf
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                ridge = Ridge(alpha=alpha)
                scores = cross_val_score(ridge, X_sorted, y_sorted, cv=tscv, scoring="r2")
                if scores.mean() > best_ridge_score:
                    best_ridge_score = scores.mean()
                    best_ridge = alpha

            ridge = Ridge(alpha=best_ridge)
            ridge.fit(X_scaled, y)
            cv_r2_ridge = cross_val_score(ridge, X_sorted, y_sorted, cv=tscv, scoring="r2")
            ridge_weights = dict(zip(feature_names, [round(float(c), 4) for c in ridge.coef_]))
            ridge_sorted = sorted(ridge_weights.items(), key=lambda x: abs(x[1]), reverse=True)

            print(f"  Best alpha:   {best_ridge}")
            print(f"  CV R² (mean): {cv_r2_ridge.mean():.4f} ± {cv_r2_ridge.std():.4f}")
            print(f"  Top features:")
            for name, weight in ridge_sorted[:5]:
                print(f"    {name:<35} {weight:>8.4f}")

            period_results["ridge"] = {
                "best_alpha": best_ridge,
                "r2_cv_mean": round(float(cv_r2_ridge.mean()), 4),
                "r2_cv_std": round(float(cv_r2_ridge.std()), 4),
                "coefficients": ridge_weights,
                "top_features": [{"feature": n, "weight": w} for n, w in ridge_sorted[:8]],
            }

            # --------------------------------------------------------
            # 5c. Lasso Regression (L1 — automatic feature selection)
            # --------------------------------------------------------
            print("\n  --- Lasso Regression (L1 — sparse) ---")
            best_lasso = None
            best_lasso_score = -np.inf
            for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
                lasso = Lasso(alpha=alpha, max_iter=10000)
                scores = cross_val_score(lasso, X_sorted, y_sorted, cv=tscv, scoring="r2")
                if scores.mean() > best_lasso_score:
                    best_lasso_score = scores.mean()
                    best_lasso = alpha

            lasso = Lasso(alpha=best_lasso, max_iter=10000)
            lasso.fit(X_scaled, y)
            cv_r2_lasso = cross_val_score(lasso, X_sorted, y_sorted, cv=tscv, scoring="r2")
            lasso_weights = {name: round(float(c), 4) for name, c in zip(feature_names, lasso.coef_) if abs(c) > 1e-6}
            lasso_sorted = sorted(lasso_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            n_selected = sum(1 for c in lasso.coef_ if abs(c) > 1e-6)
            n_eliminated = len(feature_names) - n_selected

            print(f"  Best alpha:      {best_lasso}")
            print(f"  CV R² (mean):    {cv_r2_lasso.mean():.4f} ± {cv_r2_lasso.std():.4f}")
            print(f"  Features kept:   {n_selected}/{len(feature_names)} ({n_eliminated} eliminated)")
            print(f"  Selected features:")
            for name, weight in lasso_sorted:
                print(f"    {name:<35} {weight:>8.4f}")

            period_results["lasso"] = {
                "best_alpha": best_lasso,
                "r2_cv_mean": round(float(cv_r2_lasso.mean()), 4),
                "r2_cv_std": round(float(cv_r2_lasso.std()), 4),
                "n_features_selected": n_selected,
                "n_features_eliminated": n_eliminated,
                "selected_features": lasso_weights,
                "top_features": [{"feature": n, "weight": w} for n, w in lasso_sorted],
            }

            # --------------------------------------------------------
            # 5d. ElasticNet (L1+L2 blend)
            # --------------------------------------------------------
            print("\n  --- ElasticNet (L1+L2 blend) ---")
            best_en = None
            best_en_score = -np.inf
            for alpha in [0.01, 0.1, 0.5, 1.0]:
                for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                    scores = cross_val_score(en, X_sorted, y_sorted, cv=tscv, scoring="r2")
                    if scores.mean() > best_en_score:
                        best_en_score = scores.mean()
                        best_en = (alpha, l1_ratio)

            en = ElasticNet(alpha=best_en[0], l1_ratio=best_en[1], max_iter=10000)
            en.fit(X_scaled, y)
            cv_r2_en = cross_val_score(en, X_sorted, y_sorted, cv=tscv, scoring="r2")
            en_weights = {name: round(float(c), 4) for name, c in zip(feature_names, en.coef_) if abs(c) > 1e-6}

            print(f"  Best alpha={best_en[0]}, l1_ratio={best_en[1]}")
            print(f"  CV R² (mean): {cv_r2_en.mean():.4f}")

            period_results["elasticnet"] = {
                "best_alpha": best_en[0],
                "best_l1_ratio": best_en[1],
                "r2_cv_mean": round(float(cv_r2_en.mean()), 4),
                "selected_features": en_weights,
            }

            # --------------------------------------------------------
            # 5e. Stepwise Feature Selection (forward)
            # --------------------------------------------------------
            print("\n  --- Forward Stepwise Selection ---")
            if len(valid) >= 50:  # Need enough data for this
                base_model = Ridge(alpha=best_ridge or 1.0)
                max_feats = min(8, len(feature_names))
                try:
                    sfs = SequentialFeatureSelector(
                        base_model, n_features_to_select=max_feats,
                        direction="forward", cv=tscv, scoring="r2"
                    )
                    sfs.fit(X_sorted, y_sorted)
                    selected_mask = sfs.get_support()
                    selected_names = [feature_names[i] for i, s in enumerate(selected_mask) if s]

                    # Fit final model on selected features only
                    X_selected = X_scaled[:, selected_mask]
                    X_selected_sorted = X_sorted[:, selected_mask]
                    final_model = Ridge(alpha=best_ridge or 1.0)
                    final_model.fit(X_selected, y)
                    cv_r2_sfs = cross_val_score(final_model, X_selected_sorted, y_sorted,
                                                cv=tscv, scoring="r2")

                    stepwise_weights = dict(zip(selected_names,
                                                [round(float(c), 4) for c in final_model.coef_]))
                    stepwise_sorted = sorted(stepwise_weights.items(),
                                            key=lambda x: abs(x[1]), reverse=True)

                    print(f"  Selected {len(selected_names)} features:")
                    print(f"  CV R² (mean): {cv_r2_sfs.mean():.4f} ± {cv_r2_sfs.std():.4f}")
                    for name, weight in stepwise_sorted:
                        print(f"    {name:<35} {weight:>8.4f}")

                    period_results["stepwise"] = {
                        "n_selected": len(selected_names),
                        "r2_cv_mean": round(float(cv_r2_sfs.mean()), 4),
                        "r2_cv_std": round(float(cv_r2_sfs.std()), 4),
                        "selected_features": stepwise_weights,
                        "selection_order": selected_names,
                    }
                except Exception as e:
                    print(f"  Stepwise selection failed: {e}")
            else:
                print(f"  Skipping — need 50+ observations, have {len(valid)}")

            # --------------------------------------------------------
            # 5f. Non-linear: Polynomial Features (degree 2)
            # --------------------------------------------------------
            print("\n  --- Polynomial Regression (degree 2) ---")
            # Use only top features from Lasso to avoid explosion
            top_feat_names = [n for n, _ in lasso_sorted[:6]] if lasso_sorted else \
                             [n for n, _ in ols_sorted[:6]]
            top_feat_indices = [feature_names.index(n) for n in top_feat_names if n in feature_names]

            if len(top_feat_indices) >= 2 and len(valid) >= 50:
                X_top = X_scaled[:, top_feat_indices]
                X_top_sorted = X_sorted[:, top_feat_indices]

                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                X_poly = poly.fit_transform(X_top)
                X_poly_sorted = poly.transform(X_top_sorted)
                poly_names = poly.get_feature_names_out([top_feat_names[i] for i in range(len(top_feat_indices))])

                # Use Ridge on polynomial features to prevent overfitting
                poly_model = Ridge(alpha=10.0)
                poly_model.fit(X_poly, y)
                cv_r2_poly = cross_val_score(poly_model, X_poly_sorted, y_sorted,
                                            cv=tscv, scoring="r2")
                r2_in_poly = r2_score(y, poly_model.predict(X_poly))

                # Find most important polynomial terms
                poly_weights = dict(zip(poly_names, poly_model.coef_))
                poly_sorted = sorted(poly_weights.items(), key=lambda x: abs(x[1]), reverse=True)

                # Identify interaction terms specifically
                interaction_terms = {n: w for n, w in poly_sorted
                                    if " " in n and abs(w) > 0.01}

                print(f"  Base features:      {len(top_feat_names)}")
                print(f"  Polynomial features: {X_poly.shape[1]}")
                print(f"  In-sample R²:       {r2_in_poly:.4f}")
                print(f"  CV R² (mean):       {cv_r2_poly.mean():.4f} ± {cv_r2_poly.std():.4f}")
                print(f"  Top polynomial/interaction terms:")
                for name, weight in poly_sorted[:8]:
                    marker = " [interaction]" if " " in name else (" [squared]" if "^2" in name else "")
                    print(f"    {name:<40} {weight:>8.4f}{marker}")

                period_results["polynomial"] = {
                    "base_features": top_feat_names,
                    "n_poly_features": X_poly.shape[1],
                    "r2_insample": round(float(r2_in_poly), 4),
                    "r2_cv_mean": round(float(cv_r2_poly.mean()), 4),
                    "r2_cv_std": round(float(cv_r2_poly.std()), 4),
                    "top_terms": [{"term": n, "weight": round(float(w), 4)}
                                  for n, w in poly_sorted[:12]],
                    "interaction_terms": {n: round(float(w), 4)
                                          for n, w in interaction_terms.items()},
                }
            else:
                print(f"  Skipping — insufficient features or data")

            # --------------------------------------------------------
            # 5g. Non-linear: Random Forest
            # --------------------------------------------------------
            print("\n  --- Random Forest ---")
            if len(valid) >= 40:
                rf = RandomForestRegressor(
                    n_estimators=200, max_depth=5, min_samples_leaf=5,
                    random_state=42, n_jobs=-1
                )
                cv_r2_rf = cross_val_score(rf, X_sorted, y_sorted, cv=tscv, scoring="r2")
                rf.fit(X_scaled, y)

                # Feature importances
                rf_importances = dict(zip(feature_names,
                                          [round(float(i), 4) for i in rf.feature_importances_]))
                rf_sorted = sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)

                print(f"  CV R² (mean): {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}")
                print(f"  Feature importances:")
                for name, imp in rf_sorted[:8]:
                    bar = "█" * int(imp * 100)
                    print(f"    {name:<35} {imp:.4f}  {bar}")

                period_results["random_forest"] = {
                    "r2_cv_mean": round(float(cv_r2_rf.mean()), 4),
                    "r2_cv_std": round(float(cv_r2_rf.std()), 4),
                    "feature_importances": rf_importances,
                    "top_features": [{"feature": n, "importance": i} for n, i in rf_sorted[:10]],
                }
            else:
                print(f"  Skipping — need 40+ observations")

            # --------------------------------------------------------
            # 5h. Non-linear: Gradient Boosting
            # --------------------------------------------------------
            print("\n  --- Gradient Boosting ---")
            if len(valid) >= 40:
                best_gb = None
                best_gb_score = -np.inf
                for lr in [0.01, 0.05, 0.1]:
                    for depth in [2, 3, 4]:
                        gb = GradientBoostingRegressor(
                            n_estimators=200, learning_rate=lr, max_depth=depth,
                            min_samples_leaf=5, subsample=0.8, random_state=42
                        )
                        scores = cross_val_score(gb, X_sorted, y_sorted, cv=tscv, scoring="r2")
                        if scores.mean() > best_gb_score:
                            best_gb_score = scores.mean()
                            best_gb = (lr, depth)

                gb = GradientBoostingRegressor(
                    n_estimators=200, learning_rate=best_gb[0], max_depth=best_gb[1],
                    min_samples_leaf=5, subsample=0.8, random_state=42
                )
                cv_r2_gb = cross_val_score(gb, X_sorted, y_sorted, cv=tscv, scoring="r2")
                gb.fit(X_scaled, y)

                gb_importances = dict(zip(feature_names,
                                          [round(float(i), 4) for i in gb.feature_importances_]))
                gb_sorted = sorted(gb_importances.items(), key=lambda x: x[1], reverse=True)

                print(f"  Best lr={best_gb[0]}, depth={best_gb[1]}")
                print(f"  CV R² (mean): {cv_r2_gb.mean():.4f} ± {cv_r2_gb.std():.4f}")
                print(f"  Feature importances:")
                for name, imp in gb_sorted[:8]:
                    bar = "█" * int(imp * 100)
                    print(f"    {name:<35} {imp:.4f}  {bar}")

                period_results["gradient_boosting"] = {
                    "best_learning_rate": best_gb[0],
                    "best_max_depth": best_gb[1],
                    "r2_cv_mean": round(float(cv_r2_gb.mean()), 4),
                    "r2_cv_std": round(float(cv_r2_gb.std()), 4),
                    "feature_importances": gb_importances,
                    "top_features": [{"feature": n, "importance": i} for n, i in gb_sorted[:10]],
                }

            # --------------------------------------------------------
            # 5i. Mutual Information (non-linear dependency detection)
            # --------------------------------------------------------
            print("\n  --- Mutual Information (non-linear dependency) ---")
            mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
            mi_dict = dict(zip(feature_names, [round(float(s), 4) for s in mi_scores]))
            mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
            print(f"  Non-linear dependency scores:")
            for name, score in mi_sorted[:8]:
                bar = "█" * int(score * 50)
                print(f"    {name:<35} {score:.4f}  {bar}")

            period_results["mutual_information"] = {
                "scores": mi_dict,
                "top_features": [{"feature": n, "mi_score": s} for n, s in mi_sorted[:10]],
            }

            # --------------------------------------------------------
            # 5j. Model Comparison Summary
            # --------------------------------------------------------
            print("\n  --- Model Comparison ---")
            model_summary = []
            for model_name, key in [
                ("OLS", "ols"), ("Ridge", "ridge"), ("Lasso", "lasso"),
                ("ElasticNet", "elasticnet"), ("Stepwise+Ridge", "stepwise"),
                ("Polynomial", "polynomial"), ("Random Forest", "random_forest"),
                ("Gradient Boost", "gradient_boosting"),
            ]:
                if key in period_results:
                    r2 = period_results[key].get("r2_cv_mean", 0)
                    model_summary.append({"model": model_name, "cv_r2": r2})
                    marker = " ← BEST" if r2 == max(m.get("r2_cv_mean", -999)
                                                      for m in period_results.values()
                                                      if isinstance(m, dict) and "r2_cv_mean" in m) else ""
                    print(f"    {model_name:<25} CV R²: {r2:>8.4f}{marker}")

            period_results["model_comparison"] = sorted(model_summary,
                                                        key=lambda x: x["cv_r2"], reverse=True)

            # --------------------------------------------------------
            # 5k. Final Recommended Weights
            # --------------------------------------------------------
            # Blend: average the Ridge and Lasso coefficients for a robust estimate
            # (Ridge handles multicollinearity, Lasso prunes noise)
            print("\n  --- Recommended Feature Weights (Ridge-Lasso blend) ---")
            final_weights = {}
            for name in feature_names:
                ridge_w = ridge_weights.get(name, 0)
                lasso_w = lasso_weights.get(name, 0) if name in lasso_weights else 0
                # If Lasso eliminated it, downweight heavily
                if lasso_w == 0:
                    blended = ridge_w * 0.2  # Keep small ridge signal
                else:
                    blended = (ridge_w + lasso_w) / 2
                if abs(blended) > 0.001:
                    final_weights[name] = round(float(blended), 4)

            final_sorted = sorted(final_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"  {len(final_weights)} features with non-zero weight:")
            for name, weight in final_sorted:
                direction = "↑" if weight > 0 else "↓"
                print(f"    {direction} {name:<35} {weight:>8.4f}")

            period_results["recommended_weights"] = {
                "method": "Ridge-Lasso blend (avg, Lasso-zero features downweighted 80%)",
                "weights": dict(final_sorted),
                "intercept": round(float(ols.intercept_), 4),
            }

            regression_results[period] = period_results

        results["regression"] = regression_results

    # ============================================================
    # STEP 6: Generate Summary of Learnings
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 6: Synthesizing Learnings")
    print("=" * 60)

    summary = generate_modeling_summary(results, df, feature_cols, return_cols)
    results["summary"] = summary

    # Save final results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results saved to {RESULTS_FILE}")
    print(f"   Load this file into the React dashboard to visualize.")

    # Also save a CSV for easy exploration
    csv_file = DATA_DIR / "backtest_dataset.csv"
    df.to_csv(csv_file, index=False)
    print(f"   Raw dataset saved to {csv_file}")

    # Also save summary as standalone markdown
    summary_file = DATA_DIR / "SUMMARY.md"
    summary_file.write_text(summary.get("markdown", ""))
    print(f"   Summary saved to {summary_file}")

    return results


def generate_modeling_summary(results: dict, df: pd.DataFrame,
                              feature_cols: list, return_cols: list) -> dict:
    """Use Claude to synthesize all modeling results into actionable learnings."""

    regression = results.get("regression", {})
    features_data = results.get("features", {})
    correlations = results.get("correlation_matrix", {})
    combos = results.get("combinations", [])
    metadata = results.get("metadata", {})

    # Build a condensed data package for Claude to analyze
    analysis_package = {
        "dataset": {
            "n_events": metadata.get("total_events", 0),
            "n_companies": len(metadata.get("companies", [])),
            "date_range": metadata.get("date_range", []),
        },
        "individual_feature_signals": {},
        "regression_by_period": {},
        "feature_correlations_notable": {},
        "combinations": combos,
    }

    # Condense individual feature results
    for feat_name, periods in features_data.items():
        best_period = max(periods.items(), key=lambda x: abs(x[1].get("ic", 0)),
                         default=(None, {}))
        if best_period[0]:
            analysis_package["individual_feature_signals"][feat_name] = {
                "best_period": best_period[0],
                "ic": best_period[1].get("ic"),
                "accuracy": best_period[1].get("accuracy"),
                "p_value": best_period[1].get("ic_pvalue"),
                "n": best_period[1].get("n_observations"),
            }

    # Condense regression results
    for period, period_data in regression.items():
        period_summary = {
            "n_observations": period_data.get("n_observations"),
            "model_comparison": period_data.get("model_comparison", []),
        }
        # Lasso selected features
        if "lasso" in period_data:
            period_summary["lasso_selected"] = period_data["lasso"].get("selected_features", {})
            period_summary["lasso_eliminated"] = period_data["lasso"].get("n_features_eliminated", 0)
        # Stepwise order
        if "stepwise" in period_data:
            period_summary["stepwise_order"] = period_data["stepwise"].get("selection_order", [])
            period_summary["stepwise_r2"] = period_data["stepwise"].get("r2_cv_mean")
        # Non-linear findings
        if "polynomial" in period_data:
            period_summary["interactions"] = period_data["polynomial"].get("interaction_terms", {})
            period_summary["poly_vs_linear_r2"] = (
                period_data["polynomial"].get("r2_cv_mean", 0),
                period_data.get("ridge", {}).get("r2_cv_mean", 0)
            )
        # RF / GB importances
        if "random_forest" in period_data:
            period_summary["rf_top5"] = period_data["random_forest"].get("top_features", [])[:5]
        if "gradient_boosting" in period_data:
            period_summary["gb_top5"] = period_data["gradient_boosting"].get("top_features", [])[:5]
        # Mutual info
        if "mutual_information" in period_data:
            period_summary["mi_top5"] = period_data["mutual_information"].get("top_features", [])[:5]
        # Recommended weights
        if "recommended_weights" in period_data:
            period_summary["final_weights"] = period_data["recommended_weights"].get("weights", {})

        analysis_package["regression_by_period"][period] = period_summary

    # Notable correlations (|r| > 0.4)
    for f1, f1_corrs in correlations.items():
        for f2, corr_val in f1_corrs.items():
            if f1 < f2 and abs(corr_val) > 0.4:
                analysis_package["feature_correlations_notable"][f"{f1} ↔ {f2}"] = corr_val

    # Build the prompt
    summary_prompt = f"""You are a quantitative researcher summarizing the results of a study that tested whether 
NLP features extracted from earnings call transcripts predict forward stock returns.

Here are the complete results:

{json.dumps(analysis_package, indent=2)}

Write a comprehensive but direct summary of learnings. Structure it as follows:

## Executive Summary
2-3 sentences: Is there signal in earnings transcripts? How strong? Which horizon?

## What Works
- Which features consistently predict returns? At which horizons?
- Do linear or non-linear models perform better? What does that imply?
- Which feature combinations are strongest and why?

## What Doesn't Work
- Which features were eliminated by Lasso (no signal)?
- Any features that seem intuitive but don't actually predict returns?
- Where does the signal break down?

## Non-Linear Findings
- Any important interaction effects from polynomial regression?
- Do Random Forest/Gradient Boosting significantly outperform linear models?
- What does mutual information reveal that linear IC misses?

## Feature Redundancy
- Which features are highly correlated and therefore redundant?
- What is the optimal minimal feature set?

## Recommended Prediction Model
- Which model to use in production and why
- The specific feature weights/formula for the best horizon
- Expected accuracy and realistic Sharpe expectations

## Caveats & Risks
- Sample size concerns
- Overfitting risks
- Look-ahead bias considerations
- What would make this analysis more robust

## Next Steps
- Specific improvements to try
- Additional data sources that could help
- How to validate before deploying capital

Be specific with numbers. Reference actual R² values, feature names, and coefficients.
Do NOT pad with generic advice. Every sentence should be grounded in the actual results above.

Respond in markdown format only."""

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        markdown_summary = response.content[0].text.strip()
        print("\n" + markdown_summary)

        return {
            "markdown": markdown_summary,
            "analysis_package": analysis_package,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"\n⚠️  Could not generate Claude summary: {e}")
        print("  Generating basic summary from data...")

        # Fallback: generate a basic summary without Claude
        return generate_basic_summary(analysis_package)


def generate_basic_summary(pkg: dict) -> dict:
    """Generate a basic summary without Claude API, purely from the numbers."""
    lines = []
    lines.append("# Earnings Signal Lab — Results Summary\n")
    lines.append(f"Dataset: {pkg['dataset']['n_events']} earnings events, "
                 f"{pkg['dataset']['n_companies']} companies, "
                 f"{pkg['dataset']['date_range'][0]} to {pkg['dataset']['date_range'][1]}\n")

    # Best individual features
    lines.append("## Strongest Individual Features\n")
    feat_list = [(name, data) for name, data in pkg["individual_feature_signals"].items()]
    feat_list.sort(key=lambda x: abs(x[1].get("ic", 0)), reverse=True)
    for name, data in feat_list[:6]:
        sig = "***" if data.get("p_value", 1) < 0.01 else "**" if data.get("p_value", 1) < 0.05 else "*" if data.get("p_value", 1) < 0.1 else ""
        lines.append(f"- **{name}**: IC={data['ic']:.3f} at {data['best_period']}, "
                     f"accuracy={data['accuracy']:.1%}, p={data['p_value']:.3f} {sig}")
    lines.append("")

    # Model comparison by period
    lines.append("## Best Model by Horizon\n")
    for period, pdata in pkg["regression_by_period"].items():
        comparison = pdata.get("model_comparison", [])
        if comparison:
            best = comparison[0]
            lines.append(f"- **{period}**: {best['model']} (CV R²={best['cv_r2']:.4f})")

            # How many features does Lasso keep?
            n_elim = pdata.get("lasso_eliminated", 0)
            if n_elim > 0:
                lines.append(f"  - Lasso eliminated {n_elim} features as noise")

            # Non-linear vs linear
            poly_r2 = pdata.get("poly_vs_linear_r2", (0, 0))
            if poly_r2[0] > poly_r2[1] + 0.02:
                lines.append(f"  - Non-linear model (R²={poly_r2[0]:.4f}) beats linear "
                             f"(R²={poly_r2[1]:.4f}) — interaction effects matter")
    lines.append("")

    # Recommended weights for best period
    lines.append("## Recommended Feature Weights\n")
    best_period = None
    best_r2 = -999
    for period, pdata in pkg["regression_by_period"].items():
        comparison = pdata.get("model_comparison", [])
        if comparison and comparison[0]["cv_r2"] > best_r2:
            best_r2 = comparison[0]["cv_r2"]
            best_period = period

    if best_period:
        weights = pkg["regression_by_period"][best_period].get("final_weights", {})
        lines.append(f"Best horizon: **{best_period}** (CV R²={best_r2:.4f})\n")
        sorted_w = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, w in sorted_w:
            direction = "↑ bullish" if w > 0 else "↓ bearish"
            lines.append(f"- {name}: **{w:+.4f}** ({direction})")
    lines.append("")

    # Correlated features
    notable_corr = pkg.get("feature_correlations_notable", {})
    if notable_corr:
        lines.append("## Highly Correlated Features (potential redundancy)\n")
        for pair, corr in sorted(notable_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"- {pair}: r={corr:.2f}")
    lines.append("")

    # Caveats
    n = pkg["dataset"]["n_events"]
    lines.append("## Caveats\n")
    if n < 100:
        lines.append(f"- **Small sample size ({n} events)** — results may not generalize. "
                     f"Target 200+ events for robust conclusions.")
    if n < 200:
        lines.append(f"- Cross-validation with {n} events uses very small test folds. "
                     f"R² estimates have high variance.")
    lines.append("- NLP feature extraction via LLM introduces subjectivity — "
                 "scores may vary across model versions.")
    lines.append("- No transaction costs, slippage, or market impact modeled.")
    lines.append("- Survivorship bias: only analyzing companies that still exist and report.\n")

    markdown = "\n".join(lines)
    print("\n" + markdown)

    return {
        "markdown": markdown,
        "analysis_package": pkg,
        "generated_at": datetime.now().isoformat(),
    }


# ============================================================
# BATCH ANALYSIS (Message Batches API — 50% cheaper)
# ============================================================

BATCH_STATE_FILE = DATA_DIR / "batch_state.json"


def _truncate_content(content: str) -> str:
    """Truncate very long transcripts to stay within context."""
    if len(content) > 80000:
        content = content[:30000] + "\n\n[... middle section truncated ...]\n\n" + content[-50000:]
    return content


def submit_batch():
    """Submit all pending transcripts as a Message Batch (50% cheaper, no rate limits)."""
    print("\n" + "=" * 60)
    print("STEP 2: Submitting Batch Analysis (Message Batches API)")
    print("=" * 60)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build work queue
    work_items = []
    total_cached = 0

    for cache_file in sorted(TRANSCRIPTS_DIR.glob("*_transcripts.json")):
        symbol = cache_file.stem.replace("_transcripts", "")
        transcripts = json.loads(cache_file.read_text())

        for t in transcripts:
            content = t.get("content", "")
            if not content or len(content) < 500:
                continue

            q = t.get("quarter")
            y = t.get("year")

            if YEAR_RANGE and y is not None and not (YEAR_RANGE[0] <= y <= YEAR_RANGE[1]):
                continue

            analysis_file = ANALYSIS_DIR / f"{symbol}_Q{q}_{y}_analysis.json"
            if analysis_file.exists():
                total_cached += 1
                continue

            work_items.append((symbol, q, y, content))

    print(f"\n  {len(work_items)} transcripts to submit, {total_cached} already cached")

    if not work_items:
        print("  Nothing to submit.")
        return

    # Batches API supports up to 10,000 requests per batch
    # Split into chunks if needed
    BATCH_SIZE = 10000
    batch_ids = []

    for chunk_start in range(0, len(work_items), BATCH_SIZE):
        chunk = work_items[chunk_start:chunk_start + BATCH_SIZE]
        chunk_num = chunk_start // BATCH_SIZE + 1
        total_chunks = (len(work_items) + BATCH_SIZE - 1) // BATCH_SIZE

        if total_chunks > 1:
            print(f"\n  Submitting batch {chunk_num}/{total_chunks} ({len(chunk)} requests)...")
        else:
            print(f"\n  Submitting batch ({len(chunk)} requests)...")

        requests = []
        for symbol, q, y, content in chunk:
            content = _truncate_content(content)
            requests.append({
                "custom_id": f"{symbol}_Q{q}_{y}",
                "params": {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Here is the earnings call transcript for {symbol} Q{q} {y}:\n\n{content}\n\n{EXTRACTION_PROMPT}"
                        }
                    ],
                }
            })

        batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.processing_status}")
        print(f"  Expires: {batch.expires_at}")

    # Save batch state for later retrieval
    state = {
        "batch_ids": batch_ids,
        "submitted_at": datetime.now().isoformat(),
        "total_requests": len(work_items),
    }
    BATCH_STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"\n  Batch state saved to {BATCH_STATE_FILE}")
    print(f"\n  Run '--step batch-results' to check status and retrieve results.")
    print(f"  Results typically arrive within a few hours (max 24h).")


def retrieve_batch_results():
    """Check batch status and retrieve results when ready."""
    print("\n" + "=" * 60)
    print("STEP 2: Retrieving Batch Results")
    print("=" * 60)

    if not BATCH_STATE_FILE.exists():
        print("\n  No batch state found. Run '--step batch' first.")
        return

    state = json.loads(BATCH_STATE_FILE.read_text())
    batch_ids = state.get("batch_ids", [])
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    total_succeeded = 0
    total_errored = 0
    total_processing = 0
    all_done = True

    for batch_id in batch_ids:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"\n  Batch {batch_id}:")
        print(f"    Status:     {batch.processing_status}")
        print(f"    Succeeded:  {counts.succeeded}")
        print(f"    Errored:    {counts.errored}")
        print(f"    Expired:    {counts.expired}")
        print(f"    Canceled:   {counts.canceled}")
        print(f"    Processing: {counts.processing}")

        if batch.processing_status != "ended":
            all_done = False
            total_processing += counts.processing
            continue

        # Retrieve results
        print(f"    Retrieving results...")
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id  # e.g. "AAPL_Q3_2023"
            parts = custom_id.rsplit("_Q", 1)
            if len(parts) != 2:
                print(f"    ✗ Unexpected custom_id format: {custom_id}")
                total_errored += 1
                continue

            symbol = parts[0]
            q_y = parts[1].split("_")
            if len(q_y) != 2:
                print(f"    ✗ Unexpected custom_id format: {custom_id}")
                total_errored += 1
                continue
            q, y = int(q_y[0]), int(q_y[1])

            analysis_file = ANALYSIS_DIR / f"{symbol}_Q{q}_{y}_analysis.json"
            if analysis_file.exists():
                total_succeeded += 1
                continue

            if result.result.type == "succeeded":
                message = result.result.message
                text = message.content[0].text.strip()

                # Clean potential markdown formatting
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                text = text.strip()

                try:
                    analysis = json.loads(text)
                    analysis["symbol"] = symbol
                    analysis["quarter"] = q
                    analysis["year"] = y
                    analysis_file.write_text(json.dumps(analysis, indent=2))
                    total_succeeded += 1
                except json.JSONDecodeError as e:
                    debug_file = ANALYSIS_DIR / f"{symbol}_Q{q}_{y}_raw.txt"
                    debug_file.write_text(text)
                    total_errored += 1
            else:
                print(f"    ✗ {custom_id}: {result.result.type}")
                total_errored += 1

    print(f"\n  Results: {total_succeeded} succeeded, {total_errored} errored")
    if total_processing > 0:
        print(f"  Still processing: {total_processing} requests")
        print(f"  Run '--step batch-results' again later.")
    if all_done:
        print(f"\n  All batches complete!")
        # Clean up state file
        total_analyses = len(list(ANALYSIS_DIR.glob("*_analysis.json")))
        print(f"  Total analyses on disk: {total_analyses}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Earnings Transcript Signal Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run modes:
  python earnings_signal_pipeline.py                  Full pipeline (pull, analyze, prices, backtest)
  python earnings_signal_pipeline.py --refresh        Refresh incomplete price data & re-run backtest
  python earnings_signal_pipeline.py --refresh-all    Re-fetch ALL price data from scratch & backtest
  python earnings_signal_pipeline.py --step pull      Run only transcript pulling
  python earnings_signal_pipeline.py --step analyze   Run only Claude analysis (concurrent, real-time)
  python earnings_signal_pipeline.py --step batch     Submit analysis as batch (50% cheaper, async)
  python earnings_signal_pipeline.py --step batch-results  Retrieve batch results
  python earnings_signal_pipeline.py --step prices    Run only price fetching
  python earnings_signal_pipeline.py --step backtest  Run only backtest (on existing data)
  python earnings_signal_pipeline.py --status         Show cache status without running anything
        """
    )
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh price data for events with incomplete returns or recent earnings, then re-run backtest")
    parser.add_argument("--refresh-all", action="store_true",
                        help="Re-fetch ALL price data from Yahoo Finance and re-run backtest")
    parser.add_argument("--step", choices=["pull", "analyze", "batch", "batch-results", "prices", "backtest"],
                        help="Run only a specific pipeline step (batch/batch-results use the 50%% cheaper Batches API)")
    parser.add_argument("--status", action="store_true",
                        help="Show cache status: how many transcripts, analyses, price records exist")
    parser.add_argument("--years", type=str, default=None,
                        help="Year range filter, e.g. 2023-2025. Only process transcripts within this range.")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip confirmation prompts (for automation/cron)")
    return parser.parse_args()


def show_status():
    """Print a summary of what's cached and what needs work."""
    print("=" * 60)
    print("  CACHE STATUS")
    print("=" * 60)

    # Transcripts
    transcript_count = 0
    companies_with_transcripts = set()
    for f in TRANSCRIPTS_DIR.glob("*_transcripts.json"):
        data = json.loads(f.read_text())
        transcript_count += len(data)
        symbol = f.stem.replace("_transcripts", "")
        companies_with_transcripts.add(symbol)
    print(f"\nTranscripts: {transcript_count} across {len(companies_with_transcripts)} companies")

    # Analyses
    analysis_count = len(list(ANALYSIS_DIR.glob("*_analysis.json")))
    print(f"Analyses:    {analysis_count} completed")
    if transcript_count > analysis_count:
        print(f"             {transcript_count - analysis_count} transcripts pending analysis")

    # Price data
    price_file = DATA_DIR / "price_data.json"
    if price_file.exists():
        price_data = json.loads(price_file.read_text())
        total_prices = len(price_data)
        incomplete = 0
        recent = 0
        today = datetime.now()
        for key, pdata in price_data.items():
            rets = pdata.get("returns", {})
            if any(v is None for k, v in rets.items() if k != "earnings_day"):
                incomplete += 1
            try:
                ed = pd.Timestamp(pdata.get("earnings_date", "2000-01-01"))
                if (today - ed).days < MAX_HOLDING_CALENDAR_DAYS:
                    recent += 1
            except:
                pass
        print(f"Price data:  {total_prices} events")
        if incomplete > 0:
            print(f"             {incomplete} with incomplete returns (need --refresh)")
        if recent > 0:
            print(f"             {recent} recent events (would update with --refresh)")
        missing = analysis_count - total_prices
        if missing > 0:
            print(f"             {missing} analyses without price data")
    else:
        print("Price data:  None (run pipeline first)")

    # Results
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        meta = results.get("metadata", {})
        print(f"\nBacktest:    {meta.get('total_events', '?')} events")
        print(f"             Generated: {meta.get('generated_at', 'unknown')}")
        n_features = len(results.get("features", {}))
        n_combos = len(results.get("combinations", []))
        print(f"             {n_features} features, {n_combos} combinations tested")
    else:
        print("\nBacktest:    Not yet run")

    print()


def main():
    global YEAR_RANGE
    args = parse_args()

    # Parse --years flag
    if args.years:
        parts = args.years.split("-")
        if len(parts) == 2:
            YEAR_RANGE = (int(parts[0]), int(parts[1]))
        elif len(parts) == 1:
            YEAR_RANGE = (int(parts[0]), int(parts[0]))
        else:
            print(f"Invalid --years format: {args.years}. Use e.g. 2023-2025")
            return

    print("=" * 60)
    print("  EARNINGS TRANSCRIPT SIGNAL ANALYZER")
    print("  Real Data Pipeline")
    print("=" * 60)
    print(f"\nData source: HuggingFace ({HF_DATASET})")
    print(f"Companies: S&P 500 (~496 tickers)")
    print(f"Quarters back: {'all available' if QUARTERS_BACK == 0 else QUARTERS_BACK}")
    if YEAR_RANGE:
        print(f"Year filter: {YEAR_RANGE[0]}-{YEAR_RANGE[1]}")
    print(f"Holding periods: {list(HOLDING_PERIODS.keys())}")
    print(f"Features: 16 granular NLP features")
    print(f"Data directory: {DATA_DIR}")

    # Status mode
    if args.status:
        show_status()
        return

    # Check API keys (not needed for backtest-only, status, or pull)
    needs_claude = args.step in [None, "analyze", "batch", "batch-results"]

    if needs_claude and ANTHROPIC_API_KEY == "YOUR_CLAUDE_KEY_HERE":
        print("\n⚠️  Set ANTHROPIC_API_KEY environment variable or edit this script.")
        print("   Get a key at https://console.anthropic.com")
        return

    # ---- Refresh modes ----
    if args.refresh or args.refresh_all:
        if args.refresh_all:
            print("\n🔄 REFRESH ALL: Re-fetching all price data from Yahoo Finance...")
            # Delete existing price cache to force full re-fetch
            price_file = DATA_DIR / "price_data.json"
            if price_file.exists():
                backup = DATA_DIR / f"price_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                price_file.rename(backup)
                print(f"   Backed up old price data to {backup.name}")
            gather_all_price_data(refresh=False)  # Fresh fetch since cache is gone
        else:
            print("\n🔄 REFRESH: Updating incomplete and recent price data...")
            gather_all_price_data(refresh=True)

        # Always re-run backtest after refresh
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            os.system("pip install scipy")
        run_backtest()
        return

    # ---- Single step mode ----
    if args.step:
        if args.step == "pull":
            pull_all_transcripts()
        elif args.step == "analyze":
            if not args.no_confirm:
                n_existing = len(list(ANALYSIS_DIR.glob("*_analysis.json")))
                n_transcripts = 0
                for f in TRANSCRIPTS_DIR.glob("*_transcripts.json"):
                    for t in json.loads(f.read_text()):
                        y = t.get("year")
                        if YEAR_RANGE and y is not None and not (YEAR_RANGE[0] <= y <= YEAR_RANGE[1]):
                            continue
                        n_transcripts += 1
                pending = max(0, n_transcripts - n_existing)
                if pending > 0:
                    print(f"\n{pending} transcripts to analyze (~${pending * 0.02:.2f})")
                    proceed = input("Proceed? (y/n): ").strip().lower()
                    if proceed != "y":
                        return
            analyze_all_transcripts()
        elif args.step == "batch":
            submit_batch()
        elif args.step == "batch-results":
            retrieve_batch_results()
        elif args.step == "prices":
            gather_all_price_data(refresh=False)
        elif args.step == "backtest":
            try:
                from scipy import stats as scipy_stats
            except ImportError:
                os.system("pip install scipy")
            run_backtest()
        return

    # ---- Full pipeline ----

    # Step 1: Pull transcripts
    pull_all_transcripts()

    # Step 2: Analyze with Claude
    if not args.no_confirm:
        # Count transcripts from cache files to estimate cost
        n_transcripts = 0
        for f in TRANSCRIPTS_DIR.glob("*_transcripts.json"):
            for t in json.loads(f.read_text()):
                y = t.get("year")
                if YEAR_RANGE and y is not None and not (YEAR_RANGE[0] <= y <= YEAR_RANGE[1]):
                    continue
                n_transcripts += 1
        n_existing = len(list(ANALYSIS_DIR.glob("*_analysis.json")))
        n_pending = max(0, n_transcripts - n_existing)
        print(f"\nNote: Claude API calls cost ~$0.01-0.03 per transcript.")
        print(f"Transcripts: {n_transcripts} total, {n_existing} already analyzed, {n_pending} pending")
        if n_pending > 0:
            print(f"Estimated cost for {n_pending} pending: ~${n_pending * 0.02:.2f}")
        proceed = input("Proceed with analysis? (y/n): ").strip().lower()
        if proceed != "y":
            print("Skipping analysis. Run with --step analyze later.")
            return

    analyze_all_transcripts()

    # Step 3: Get price data
    gather_all_price_data(refresh=False)

    # Step 4: Backtest
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\nInstalling scipy for statistical tests...")
        os.system("pip install scipy")

    run_backtest()


if __name__ == "__main__":
    main()
