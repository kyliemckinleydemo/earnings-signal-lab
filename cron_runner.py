"""
Cron runner for Railway.
Deploy this as a separate Railway cron service that runs on a schedule.

Railway cron setup:
  1. Create a second service in your Railway project
  2. Set it as a "Cron" type
  3. Schedule: 0 18 * * 1-5 (weekdays at 6pm ET / market close)
  4. Command: python cron_runner.py
  5. Same env vars as the main service

What it does:
  - Pulls any new transcripts from FMP
  - Analyzes new transcripts with Claude
  - Refreshes incomplete price data
  - Re-runs the backtest
  - Generates updated summary
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    print(f"[{datetime.now().isoformat()}] Earnings Signal Lab — Cron Run Starting")
    print("=" * 60)

    # Check env vars
    if not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Step 1: Pull new transcripts
    print("\n[1/4] Pulling new transcripts...")
    result = subprocess.run(
        ["python", "earnings_signal_pipeline.py", "--step", "pull"],
        capture_output=False
    )

    # Step 2: Analyze new transcripts
    print("\n[2/4] Analyzing new transcripts with Claude...")
    result = subprocess.run(
        ["python", "earnings_signal_pipeline.py", "--step", "analyze", "--no-confirm"],
        capture_output=False
    )

    # Step 3: Refresh price data + re-backtest
    print("\n[3/4] Refreshing price data and running backtest...")
    result = subprocess.run(
        ["python", "earnings_signal_pipeline.py", "--refresh", "--no-confirm"],
        capture_output=False
    )

    # Step 4: Notify (optional — you could add a webhook here)
    print(f"\n[{datetime.now().isoformat()}] Cron run complete")

    # If the web server is running, it will pick up the new results
    # automatically on next /api/results request (file mtime check)


if __name__ == "__main__":
    main()
