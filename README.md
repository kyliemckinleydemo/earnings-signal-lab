# Earnings Signal Lab

**Test whether granular NLP features from earnings call transcripts predict forward stock returns.**

This is not a sentiment analysis tool. Instead of computing a single "positive/negative" score, it extracts **16 specific behavioral and linguistic features** from earnings transcripts and Q&A sessions, then backtests each one against real price data.

## The 16 Features

| Category | Feature | Signal Direction |
|----------|---------|-----------------|
| **Management Behavior** | Hedging Language | Bearish when high |
| | Q&A Deflection Rate | Bearish when high |
| | Guidance Specificity | Bullish when high |
| | Confidence Shift (Prepared→Q&A) | Bearish when high |
| **Analyst Behavior** | Analyst Skepticism | Bearish when high |
| | Surprise Indicators | Context-dependent |
| | Question Clustering | Context-dependent |
| **Forward Guidance** | Guidance Revision Direction | Bullish when raised |
| | Qualifier Density | Bearish when high |
| **Risk Signals** | New Risk Factor Mentions | Bearish when high |
| | External Blame Attribution | Bearish when high |
| **Strategic Signals** | CapEx/Investment Tone | Bullish when aggressive |
| | Hiring & Headcount | Bullish when growing |
| | Competitive Positioning | Bearish when concerned |
| **Demand Signals** | Customer/Demand Descriptors | Bullish when strong |
| | Pricing Power Indicators | Bullish when strong |

## How It Works

```
FMP Transcripts → Claude NLP Extraction → Yahoo Finance Prices → Statistical Backtest
```

1. **Pull transcripts** from Financial Modeling Prep (free tier, 250 calls/day)
2. **Extract features** using Claude API — each transcript is analyzed for all 16 features with scores (0-1), evidence quotes, and section attribution (prepared remarks vs Q&A)
3. **Get price data** from Yahoo Finance at 1D, 5D, 10D, 21D post-earnings
4. **Backtest** — Information Coefficient, directional accuracy, Sharpe ratios, p-values, feature correlations, and multi-feature combination tests

## Quick Start

### Prerequisites

```bash
pip install requests yfinance pandas numpy anthropic scipy
```

### API Keys

Get free keys from:
- **FMP**: https://financialmodelingprep.com/developer (free tier: 250 calls/day)
- **Anthropic**: https://console.anthropic.com

```bash
export FMP_API_KEY="your_fmp_key"
export ANTHROPIC_API_KEY="your_claude_key"
```

### Run the Pipeline

```bash
# Full pipeline (first run)
python earnings_signal_pipeline.py

# Check what's cached
python earnings_signal_pipeline.py --status
```

The pipeline will:
- Pull transcripts for 30 companies × 8 quarters (~240 API calls)
- Analyze each with Claude (~$0.02/transcript, ~$5 total)
- Fetch price data from Yahoo Finance
- Output `earnings_signal_data/backtest_results.json`

All data is cached, so re-runs skip already-processed transcripts.

### Refresh Mode

After the initial run, use refresh mode to update price data as holding periods mature:

```bash
# Update only events with incomplete returns (e.g., 21D hadn't elapsed yet)
# and recent earnings (within last 35 days), then re-run backtest
python earnings_signal_pipeline.py --refresh

# Re-fetch ALL price data from scratch (backs up old data first)
python earnings_signal_pipeline.py --refresh-all
```

Refresh mode is smart about what it updates — it shows you exactly what changed:
```
↻ NVDA_Q4_2024 [incomplete returns]: 21D: None→+8.3%
↻ AAPL_Q1_2025 [recent (12d ago)]: 10D: +2.1%→+2.4%, 21D: None→-1.2%
```

### Run Individual Steps

```bash
python earnings_signal_pipeline.py --step pull       # Only pull new transcripts
python earnings_signal_pipeline.py --step analyze    # Only run Claude on unanalyzed transcripts
python earnings_signal_pipeline.py --step prices     # Only fetch missing price data
python earnings_signal_pipeline.py --step backtest   # Re-run backtest on existing data
python earnings_signal_pipeline.py --no-confirm      # Skip prompts (for cron/automation)
```

### View Results

Upload `backtest_results.json` to the React dashboard (`dashboard/earnings-signal-analyzer.jsx`), or load it in Claude.ai as an artifact.

## Key Metrics

- **Information Coefficient (IC)**: Spearman rank correlation between feature score and forward returns. IC > 0.10 is considered meaningful in quant finance.
- **Directional Accuracy**: How often the signal correctly predicts up/down.
- **Sharpe Ratio**: Risk-adjusted return when the signal is triggered.
- **p-value**: Statistical significance of the IC.

## Customization

### Change the company universe

Edit the `COMPANIES` list in `earnings_signal_pipeline.py`:

```python
COMPANIES = ["AAPL", "MSFT", "GOOGL", ...]  # Add your 700 tickers
```

For 700 companies, you'll need ~3 days to pull all transcripts within the free tier limit. The pipeline tracks progress and caches everything.

### Adjust holding periods

```python
HOLDING_PERIODS = {
    "1D": 1,
    "5D": 5,
    "10D": 10,
    "21D": 21,
}
```

### Modify the feature extraction prompt

The `EXTRACTION_PROMPT` variable contains the full Claude prompt for feature extraction. You can add, remove, or modify features here.

## Project Structure

```
earnings-signal-lab/
├── earnings_signal_pipeline.py    # Main pipeline (pull, analyze, backtest)
├── server/
│   └── app.py                     # FastAPI web server + public API
├── static/
│   └── index.html                 # Public dashboard (no raw transcripts)
├── dashboard/
│   └── earnings-signal-analyzer.jsx  # React dashboard (for Claude.ai artifact)
├── cron_runner.py                 # Automated daily updates
├── first_run.py                   # One-time initial data population
├── Dockerfile                     # Railway/Docker deployment
├── railway.toml                   # Railway configuration
├── requirements.txt
├── .gitignore
└── README.md
```

## Output Structure

```
earnings_signal_data/
├── transcripts/          # Cached raw transcripts per company
├── analysis/             # Claude feature extractions per earnings call
├── price_data.json       # Yahoo Finance price data
├── backtest_results.json # Final results (served via API)
├── backtest_dataset.csv  # Raw dataset for custom analysis
└── SUMMARY.md            # Claude-generated analysis summary
```

## Deploy to Railway

The app runs as a full-stack Python service on Railway: FastAPI serves the public dashboard and API, a cron job runs daily updates.

### Step 1: Initial data population (local)

```bash
export FMP_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
python first_run.py
```

This takes ~30 min and costs ~$5 in Claude API calls. All data is saved to `earnings_signal_data/`.

### Step 2: Push to GitHub

```bash
git add -A
git commit -m "Initial commit with data"
git push origin main
```

Note: The `.gitignore` excludes `earnings_signal_data/` by default. For Railway, you have two options:
- **Option A**: Remove `earnings_signal_data/` from `.gitignore` and commit the data (simpler, ~50MB)
- **Option B**: Use a Railway persistent volume (better for ongoing updates)

### Step 3: Create Railway project

1. Go to [railway.app](https://railway.app) and create a new project
2. Connect your GitHub repo
3. Railway will auto-detect the Dockerfile and deploy

### Step 4: Set environment variables

In Railway dashboard → your service → Variables:

```
FMP_API_KEY=your_fmp_key
ANTHROPIC_API_KEY=your_anthropic_key
PORT=8000
```

### Step 5: Add persistent volume (Option B)

If using a persistent volume for data:

1. In Railway dashboard → your service → Settings → Volumes
2. Add a volume mounted at `/app/earnings_signal_data`
3. Upload your local `earnings_signal_data/` contents to the volume

### Step 6: Set up cron job

1. In your Railway project, add a second service
2. Set the source to the same GitHub repo
3. Set the start command to: `python cron_runner.py`
4. In service settings, set schedule: `0 18 * * 1-5` (weekdays at 6pm ET)
5. Add the same environment variables

### Public API Endpoints

Once deployed, your Railway URL exposes:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Public dashboard |
| `GET /api/results` | Full results (features, regression, combos) |
| `GET /api/predictions` | Just prediction weights and expected performance |
| `GET /api/features` | Individual feature signal strength |
| `GET /api/summary` | Markdown analysis summary |
| `GET /api/status` | Pipeline health and data freshness |
| `GET /docs` | Interactive API documentation |

No raw transcripts or Claude analysis text is exposed through any endpoint.

## Legal Notes

- **Transcripts**: Pulled from FMP for personal analysis. Not redistributed.
- **Price Data**: Yahoo Finance data is for personal use per their terms.
- **Predictions**: Your derived analysis/predictions are your own work.

## License

MIT
