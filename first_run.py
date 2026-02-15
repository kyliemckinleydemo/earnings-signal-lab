"""
First-run script: Populates initial data locally before deploying to Railway.

Run this ONCE on your local machine to build the initial dataset,
then push the data directory to Railway's persistent volume.

Usage:
    export EARNINGSCALL_API_KEY="your_key"
    export ANTHROPIC_API_KEY="your_key"
    python first_run.py
"""

import subprocess
import sys

def main():
    print("=" * 60)
    print("  EARNINGS SIGNAL LAB — First Run Setup")
    print("=" * 60)
    print()
    print("This will:")
    print("  1. Pull transcripts via earningscall (~2 min)")
    print("  2. Analyze with Claude (~$5, ~20 min)")
    print("  3. Get price data from Yahoo Finance (~2 min)")
    print("  4. Run regression analysis (~5 min)")
    print("  5. Generate summary")
    print()
    print("After this completes, deploy to Railway.")
    print()

    proceed = input("Ready? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborted.")
        return

    result = subprocess.run(
        [sys.executable, "earnings_signal_pipeline.py"],
        capture_output=False
    )

    if result.returncode == 0:
        print()
        print("=" * 60)
        print("  ✅ SETUP COMPLETE")
        print("=" * 60)
        print()
        print("Your data is in earnings_signal_data/")
        print()
        print("Next steps:")
        print("  1. Push to GitHub")
        print("  2. Connect repo to Railway")
        print("  3. Add a persistent volume mounted at /app/earnings_signal_data")
        print("  4. Upload the earnings_signal_data/ folder to the volume")
        print("  5. Set env vars: EARNINGSCALL_API_KEY, ANTHROPIC_API_KEY")
        print("  6. Deploy!")
    else:
        print(f"\nPipeline exited with code {result.returncode}")
        print("Check the output above for errors.")


if __name__ == "__main__":
    main()
