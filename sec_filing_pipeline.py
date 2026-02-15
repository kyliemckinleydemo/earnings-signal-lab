"""
SEC Filing Signal Analyzer — Model Zoo Pipeline
=================================================
Downloads pre-exported SEC filing data (352 rows) from GitHub,
runs the full model zoo (OLS, Ridge, Lasso, ElasticNet, Stepwise,
Polynomial, Random Forest, Gradient Boosting, Mutual Information),
and optionally combines with earnings transcript features.

Data source:
    GitHub: kyliemckinleydemo/sec-filing-analyzer → data/ml_dataset_with_concern.csv

Usage:
    python sec_filing_pipeline.py                    # Full pipeline (pull + backtest)
    python sec_filing_pipeline.py --step pull        # Download CSV from GitHub
    python sec_filing_pipeline.py --step backtest    # Run model zoo
    python sec_filing_pipeline.py --step train-model # Train scoring model
    python sec_filing_pipeline.py --step score       # Score all filings
    python sec_filing_pipeline.py --step combined    # Merge with earnings data + run model zoo

Requirements:
    pip install pandas numpy scipy scikit-learn
"""

import os
import json
import math
import argparse
import urllib.request
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("sec_filing_data")
RAW_DIR = DATA_DIR / "raw"
RESULTS_FILE = DATA_DIR / "backtest_results.json"
MODEL_FILE = DATA_DIR / "scoring_model.json"
SCORES_FILE = DATA_DIR / "scores.json"

COMBINED_DIR = Path("combined_data")
COMBINED_RESULTS = COMBINED_DIR / "backtest_results.json"
COMBINED_MODEL = COMBINED_DIR / "scoring_model.json"
COMBINED_SCORES = COMBINED_DIR / "scores.json"

CSV_FILE = RAW_DIR / "ml_dataset_with_concern.csv"

GITHUB_API_URL = "https://api.github.com/repos/kyliemckinleydemo/sec-filing-analyzer/contents/data/ml_dataset_with_concern.csv"

# Holding periods to test
HOLDING_PERIODS = {
    "7D": "actual7dReturn",
    "30D": "actual30dReturn",
}

# Also test alpha (market-relative) if columns exist
ALPHA_PERIODS = {
    "7D_alpha": "actual7dAlpha",
    "30D_alpha": "actual30dAlpha",
}

# ============================================================
# FEATURE METADATA
# ============================================================

# Map CSV column names → clean snake_case names + metadata
FEATURE_COLUMNS = {
    "riskScore": "risk_score",
    "sentimentScore": "sentiment_score",
    "concernLevel": "concern_level",
    "peRatio": "pe_ratio",
    "forwardPE": "forward_pe",
    "priceToHigh": "price_to_high",
    "priceToLow": "price_to_low",
    "priceToTarget": "price_to_target",
    "priceToMA30": "price_to_ma30",
    "priceToMA50": "price_to_ma50",
    "rsi14": "rsi14",
    "volatility30": "volatility30",
    "return30d": "return_30d",
    "spxReturn7d": "spx_return_7d",
    "spxReturn30d": "spx_return_30d",
    "vixClose": "vix_close",
    "analystUpsidePotential": "analyst_upside_potential",
    "analystConsensusScore": "analyst_consensus_score",
    "analystCoverage": "analyst_coverage",
    "netUpgrades": "net_upgrades",
}

SEC_FEATURES = {
    "risk_score": {"name": "Risk Score", "cat": "AI Analysis", "color": "#e74c3c", "bear": True, "desc": "Claude-assessed risk severity (0-10)"},
    "sentiment_score": {"name": "Sentiment", "cat": "AI Analysis", "color": "#27ae60", "bear": False, "desc": "MD&A sentiment (-1 to +1)"},
    "concern_level": {"name": "Concern Level", "cat": "AI Analysis", "color": "#c0392b", "bear": True, "desc": "Multi-factor concern synthesis (0-10)"},
    "pe_ratio": {"name": "P/E Ratio", "cat": "Valuation", "color": "#8e44ad", "bear": True, "desc": "Price-to-earnings multiple"},
    "forward_pe": {"name": "Forward P/E", "cat": "Valuation", "color": "#9b59b6", "bear": True, "desc": "Forward price-to-earnings"},
    "price_to_high": {"name": "Price to High", "cat": "Valuation", "color": "#2980b9", "bear": True, "desc": "Price relative to 52-week high"},
    "price_to_low": {"name": "Price to Low", "cat": "Valuation", "color": "#3498db", "bear": False, "desc": "Price relative to 52-week low"},
    "price_to_target": {"name": "Price to Target", "cat": "Valuation", "color": "#1abc9c", "bear": False, "desc": "Price relative to analyst target"},
    "price_to_ma30": {"name": "Price/MA30", "cat": "Technical", "color": "#e67e22", "bear": False, "desc": "Price relative to 30-day moving average"},
    "price_to_ma50": {"name": "Price/MA50", "cat": "Technical", "color": "#d35400", "bear": False, "desc": "Price relative to 50-day moving average"},
    "rsi14": {"name": "RSI (14)", "cat": "Technical", "color": "#f39c12", "bear": True, "desc": "Relative Strength Index (overbought > 70)"},
    "volatility30": {"name": "Volatility 30D", "cat": "Technical", "color": "#e74c3c", "bear": True, "desc": "30-day price volatility"},
    "return_30d": {"name": "Prior 30D Return", "cat": "Technical", "color": "#2ecc71", "bear": False, "desc": "Trailing 30-day return (momentum)"},
    "spx_return_7d": {"name": "SPX 7D Return", "cat": "Market Context", "color": "#3498db", "bear": False, "desc": "S&P 500 7-day return"},
    "spx_return_30d": {"name": "SPX 30D Return", "cat": "Market Context", "color": "#2980b9", "bear": False, "desc": "S&P 500 30-day return"},
    "vix_close": {"name": "VIX Close", "cat": "Market Context", "color": "#e74c3c", "bear": True, "desc": "CBOE VIX volatility index"},
    "analyst_upside_potential": {"name": "Analyst Upside", "cat": "Analyst Activity", "color": "#27ae60", "bear": False, "desc": "Analyst-implied upside potential"},
    "analyst_consensus_score": {"name": "Analyst Consensus", "cat": "Analyst Activity", "color": "#2ecc71", "bear": False, "desc": "Analyst consensus rating score"},
    "analyst_coverage": {"name": "Analyst Coverage", "cat": "Analyst Activity", "color": "#16a085", "bear": False, "desc": "Number of covering analysts"},
    "net_upgrades": {"name": "Net Upgrades", "cat": "Analyst Activity", "color": "#1abc9c", "bear": False, "desc": "Recent upgrades minus downgrades"},
}

# Conservative ML settings for small dataset (n=352)
RF_PARAMS = {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 10, "random_state": 42, "n_jobs": -1}
GB_PARAMS = {"n_estimators": 100, "max_depth": 2, "min_samples_leaf": 10, "subsample": 0.8, "random_state": 42}
POLY_TOP_N = 4  # Only use top 4 features for polynomial to avoid explosion
N_SPLITS = 5


# ============================================================
# STEP: PULL — Download CSV from GitHub
# ============================================================

def pull_data():
    """Download the SEC filing CSV from GitHub."""
    print("\n" + "=" * 60)
    print("STEP: Pull SEC Filing Data")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        print(f"  CSV already exists: {CSV_FILE} ({len(df)} rows)")
        return df

    print(f"  Downloading from GitHub...")
    print(f"  URL: {GITHUB_API_URL}")

    try:
        req = urllib.request.Request(GITHUB_API_URL)
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "sec-filing-pipeline")

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

        content_b64 = data["content"]
        csv_bytes = base64.b64decode(content_b64)
        CSV_FILE.write_bytes(csv_bytes)
        print(f"  Saved to {CSV_FILE}")

    except Exception as e:
        print(f"  GitHub API failed: {e}")
        print(f"  Trying direct raw download...")

        raw_url = "https://raw.githubusercontent.com/kyliemckinleydemo/sec-filing-analyzer/main/data/ml_dataset_with_concern.csv"
        try:
            req = urllib.request.Request(raw_url)
            req.add_header("User-Agent", "sec-filing-pipeline")
            with urllib.request.urlopen(req) as resp:
                CSV_FILE.write_bytes(resp.read())
            print(f"  Saved to {CSV_FILE}")
        except Exception as e2:
            print(f"  Direct download also failed: {e2}")
            print(f"  Please manually download the CSV and place it at: {CSV_FILE}")
            return None

    df = pd.read_csv(CSV_FILE)
    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Show basic stats
    if "ticker" in df.columns:
        print(f"  Tickers: {df['ticker'].nunique()}")
    if "filingDate" in df.columns:
        print(f"  Date range: {df['filingDate'].min()} to {df['filingDate'].max()}")

    return df


def load_csv():
    """Load the CSV file, downloading if necessary."""
    if not CSV_FILE.exists():
        return pull_data()
    return pd.read_csv(CSV_FILE)


# ============================================================
# STEP: BACKTEST — Run Model Zoo
# ============================================================

def run_backtest(df=None):
    """Run the full model zoo on SEC filing features."""
    print("\n" + "=" * 60)
    print("STEP: Backtest — SEC Filing Model Zoo")
    print("=" * 60)

    if df is None:
        df = load_csv()
    if df is None:
        print("No data available.")
        return None

    from scipy import stats as scipy_stats
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import r2_score
    from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_regression

    # Identify available features
    available_features = []
    for csv_col, clean_name in FEATURE_COLUMNS.items():
        if csv_col in df.columns:
            available_features.append((csv_col, clean_name))

    if not available_features:
        print("  No feature columns found in CSV!")
        print(f"  Available columns: {list(df.columns)}")
        return None

    print(f"\n  Dataset: {len(df)} filings × {len(available_features)} features")

    # Rename columns for consistency
    rename_map = {csv_col: clean_name for csv_col, clean_name in available_features}
    df_work = df.rename(columns=rename_map).copy()
    feature_names = [clean_name for _, clean_name in available_features]

    # Sort by filing date for time-series CV
    date_col = None
    for col in ["filingDate", "filing_date", "date"]:
        if col in df.columns:
            date_col = col
            break
    if date_col:
        df_work["_sort_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df_work = df_work.sort_values("_sort_date").reset_index(drop=True)

    # Identify target columns
    target_cols = {}
    for label, col in {**HOLDING_PERIODS, **ALPHA_PERIODS}.items():
        if col in df.columns:
            target_cols[label] = col
        # Also check renamed
        clean = FEATURE_COLUMNS.get(col, col)
        if clean in df_work.columns and label not in target_cols:
            target_cols[label] = clean

    if not target_cols:
        print("  No target return columns found!")
        print(f"  Looking for: {list(HOLDING_PERIODS.values()) + list(ALPHA_PERIODS.values())}")
        return None

    print(f"  Target columns: {list(target_cols.keys())}")
    print(f"  Features: {feature_names}")

    # Determine ticker and date columns for metadata
    ticker_col = None
    for col in ["ticker", "symbol", "Ticker"]:
        if col in df.columns:
            ticker_col = col
            break

    tickers = sorted(df[ticker_col].unique().tolist()) if ticker_col else []

    date_range = []
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            date_range = [str(dates.min().date()), str(dates.max().date())]

    # Build results structure
    results = {
        "metadata": {
            "total_events": len(df),
            "companies": tickers,
            "date_range": date_range,
            "generated_at": datetime.now().isoformat(),
            "source": "sec_filings",
            "feature_metadata": SEC_FEATURES,
        },
        "features": {},
        "sample_extractions": {},
        "correlation_matrix": {},
        "combinations": [],
        "regression": {},
    }

    # ============================================================
    # PER-FEATURE SIGNAL ANALYSIS
    # ============================================================

    print("\n--- Feature Signal Analysis ---\n")
    print(f"{'Feature':<30} {'Period':<10} {'IC':<10} {'Accuracy':<10} {'Sharpe':<10} {'p-value':<10} {'n':<6}")
    print("-" * 90)

    for feat_name in feature_names:
        if feat_name not in df_work.columns:
            continue
        feat_results = {}
        meta = SEC_FEATURES.get(feat_name, {})
        is_bearish = meta.get("bear", False)

        for period_label, target_col in target_cols.items():
            if target_col not in df_work.columns:
                continue

            valid = df_work[[feat_name, target_col]].dropna()
            if len(valid) < 20:
                continue

            scores = valid[feat_name].values
            returns = valid[target_col].values

            # Information Coefficient
            ic, ic_pvalue = scipy_stats.spearmanr(scores, returns)

            # Directional accuracy using median split (features aren't 0-1)
            median_score = np.median(scores)
            if is_bearish:
                predictions = scores > median_score
                actuals = returns < 0
            else:
                predictions = scores > median_score
                actuals = returns > 0
            accuracy = np.mean(predictions == actuals) if len(predictions) > 0 else 0.5

            # Win rate: top quartile signal
            q75 = np.percentile(scores, 75)
            q25 = np.percentile(scores, 25)

            if is_bearish:
                signal_returns = -returns[scores > q75]
            else:
                signal_returns = returns[scores > q75]

            if len(signal_returns) > 1:
                avg_ret = np.mean(signal_returns)
                std_ret = np.std(signal_returns)
                sharpe = avg_ret / std_ret * np.sqrt(4) if std_ret > 0 else 0
                win_rate = np.mean(signal_returns > 0)
            else:
                avg_ret = 0
                sharpe = 0
                win_rate = 0.5

            feat_results[period_label] = {
                "ic": round(float(ic), 4),
                "ic_pvalue": round(float(ic_pvalue), 4),
                "accuracy": round(float(accuracy), 4),
                "sharpe": round(float(sharpe), 4),
                "avg_return_pct": round(float(avg_ret), 4),
                "win_rate": round(float(win_rate), 4),
                "n_observations": len(valid),
                "n_signal_triggered": len(signal_returns),
            }

            print(f"{feat_name:<30} {period_label:<10} {ic:>8.4f}  {accuracy:>8.1%}  {sharpe:>8.2f}  {ic_pvalue:>8.4f}  {len(valid):<6}")

        if feat_results:
            results["features"][feat_name] = feat_results

        # Sample extractions
        valid_all = df_work[df_work[feat_name].notna()].copy()
        if len(valid_all) > 0:
            valid_all = valid_all.sort_values(feat_name, ascending=False)
            samples = []
            for _, row in valid_all.head(3).iterrows():
                s = {
                    "symbol": row.get(ticker_col, row.get("ticker", "?")),
                    "quarter": str(row.get("quarter", "?")),
                    "score": round(float(row[feat_name]), 3),
                }
                for tl, tc in target_cols.items():
                    if tc in row and pd.notna(row[tc]):
                        s[f"return_{tl}"] = round(float(row[tc]), 2)
                samples.append(s)
            for _, row in valid_all.tail(3).iterrows():
                s = {
                    "symbol": row.get(ticker_col, row.get("ticker", "?")),
                    "quarter": str(row.get("quarter", "?")),
                    "score": round(float(row[feat_name]), 3),
                }
                for tl, tc in target_cols.items():
                    if tc in row and pd.notna(row[tc]):
                        s[f"return_{tl}"] = round(float(row[tc]), 2)
                samples.append(s)
            results["sample_extractions"][feat_name] = samples

    # ============================================================
    # CORRELATION MATRIX
    # ============================================================

    feat_cols_present = [f for f in feature_names if f in df_work.columns]
    feat_df = df_work[feat_cols_present].dropna(axis=1, how="all")
    if len(feat_df.columns) > 1:
        corr = feat_df.corr(method="spearman")
        results["correlation_matrix"] = {
            c1: {c2: round(float(corr.loc[c1, c2]), 3) for c2 in corr.columns}
            for c1 in corr.index
        }

    # ============================================================
    # REGRESSION MODEL ZOO
    # ============================================================

    print("\n" + "=" * 60)
    print("Regression Modeling & Optimal Weightings")
    print("=" * 60)

    for period_label, target_col in target_cols.items():
        if target_col not in df_work.columns:
            continue

        print(f"\n{'='*50}")
        print(f"  Modeling: {period_label} forward returns")
        print(f"{'='*50}")

        # Drop rows where target is NaN, but impute feature NaN with median
        has_target = df_work[target_col].notna()
        valid = df_work.loc[has_target, feat_cols_present + [target_col]].copy()
        if len(valid) < 30:
            print(f"  Skipping — only {len(valid)} observations (need 30+)")
            continue

        # Drop features with >50% missing, then impute remaining NaN with median
        good_feats = [f for f in feat_cols_present if valid[f].notna().mean() > 0.5]
        for f in good_feats:
            valid[f] = valid[f].fillna(valid[f].median())
        valid = valid[good_feats + [target_col]].dropna()
        if len(valid) < 30:
            print(f"  Skipping — only {len(valid)} after imputation (need 30+)")
            continue

        X = valid[good_feats].values
        y = valid[target_col].values
        fn = list(good_feats)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Sort by date for time-series CV
        if "_sort_date" in df_work.columns:
            valid_sorted = valid.copy()
            valid_sorted["_sd"] = df_work.loc[valid.index, "_sort_date"]
            valid_sorted = valid_sorted.sort_values("_sd")
            X_sorted = scaler.transform(valid_sorted[good_feats].values)
            y_sorted = valid_sorted[target_col].values
        else:
            X_sorted = X_scaled
            y_sorted = y

        n_splits = min(N_SPLITS, max(2, len(valid) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        period_results = {
            "n_observations": len(valid),
            "y_mean": round(float(y.mean()), 4),
            "y_std": round(float(y.std()), 4),
        }

        # --------------------------------------------------------
        # OLS
        # --------------------------------------------------------
        print("\n  --- OLS Linear Regression ---")
        ols = LinearRegression()
        ols.fit(X_scaled, y)
        y_pred = ols.predict(X_scaled)
        r2_in = r2_score(y, y_pred)
        cv_r2_ols = cross_val_score(ols, X_sorted, y_sorted, cv=tscv, scoring="r2")
        cv_rmse = cross_val_score(ols, X_sorted, y_sorted, cv=tscv, scoring="neg_mean_squared_error")

        ols_weights = dict(zip(fn, [round(float(c), 4) for c in ols.coef_]))
        ols_sorted = sorted(ols_weights.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"  In-sample R²: {r2_in:.4f}")
        print(f"  CV R² (mean): {cv_r2_ols.mean():.4f} ± {cv_r2_ols.std():.4f}")
        for name, w in ols_sorted[:5]:
            print(f"    {name:<35} {w:>8.4f}")

        period_results["ols"] = {
            "r2_insample": round(float(r2_in), 4),
            "r2_cv_mean": round(float(cv_r2_ols.mean()), 4),
            "r2_cv_std": round(float(cv_r2_ols.std()), 4),
            "rmse_cv": round(float(np.sqrt(-cv_rmse.mean())), 4),
            "intercept": round(float(ols.intercept_), 4),
            "coefficients": ols_weights,
            "top_features": [{"feature": n, "weight": w} for n, w in ols_sorted[:8]],
        }

        # --------------------------------------------------------
        # Ridge
        # --------------------------------------------------------
        print("\n  --- Ridge Regression (L2) ---")
        best_ridge_alpha, best_ridge_score = 1.0, -np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(ridge, X_sorted, y_sorted, cv=tscv, scoring="r2")
            if scores.mean() > best_ridge_score:
                best_ridge_score = scores.mean()
                best_ridge_alpha = alpha

        ridge = Ridge(alpha=best_ridge_alpha)
        ridge.fit(X_scaled, y)
        cv_r2_ridge = cross_val_score(ridge, X_sorted, y_sorted, cv=tscv, scoring="r2")
        ridge_weights = dict(zip(fn, [round(float(c), 4) for c in ridge.coef_]))
        ridge_sorted = sorted(ridge_weights.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"  Best alpha: {best_ridge_alpha}")
        print(f"  CV R² (mean): {cv_r2_ridge.mean():.4f} ± {cv_r2_ridge.std():.4f}")

        period_results["ridge"] = {
            "best_alpha": best_ridge_alpha,
            "r2_cv_mean": round(float(cv_r2_ridge.mean()), 4),
            "r2_cv_std": round(float(cv_r2_ridge.std()), 4),
            "coefficients": ridge_weights,
            "top_features": [{"feature": n, "weight": w} for n, w in ridge_sorted[:8]],
        }

        # --------------------------------------------------------
        # Lasso
        # --------------------------------------------------------
        print("\n  --- Lasso Regression (L1 — sparse) ---")
        best_lasso_alpha, best_lasso_score = 0.1, -np.inf
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            scores = cross_val_score(lasso, X_sorted, y_sorted, cv=tscv, scoring="r2")
            if scores.mean() > best_lasso_score:
                best_lasso_score = scores.mean()
                best_lasso_alpha = alpha

        lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        cv_r2_lasso = cross_val_score(lasso, X_sorted, y_sorted, cv=tscv, scoring="r2")
        lasso_weights = {name: round(float(c), 4) for name, c in zip(fn, lasso.coef_) if abs(c) > 1e-6}
        lasso_sorted = sorted(lasso_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        n_selected = sum(1 for c in lasso.coef_ if abs(c) > 1e-6)
        n_eliminated = len(fn) - n_selected

        print(f"  Best alpha: {best_lasso_alpha}")
        print(f"  CV R² (mean): {cv_r2_lasso.mean():.4f} ± {cv_r2_lasso.std():.4f}")
        print(f"  Features kept: {n_selected}/{len(fn)} ({n_eliminated} eliminated)")
        for name, w in lasso_sorted:
            print(f"    {name:<35} {w:>8.4f}")

        period_results["lasso"] = {
            "best_alpha": best_lasso_alpha,
            "r2_cv_mean": round(float(cv_r2_lasso.mean()), 4),
            "r2_cv_std": round(float(cv_r2_lasso.std()), 4),
            "n_features_selected": n_selected,
            "n_features_eliminated": n_eliminated,
            "selected_features": lasso_weights,
            "top_features": [{"feature": n, "weight": w} for n, w in lasso_sorted],
        }

        # --------------------------------------------------------
        # ElasticNet
        # --------------------------------------------------------
        print("\n  --- ElasticNet (L1+L2 blend) ---")
        best_en, best_en_score = (0.1, 0.5), -np.inf
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
        en_weights = {name: round(float(c), 4) for name, c in zip(fn, en.coef_) if abs(c) > 1e-6}

        print(f"  Best alpha={best_en[0]}, l1_ratio={best_en[1]}")
        print(f"  CV R² (mean): {cv_r2_en.mean():.4f}")

        period_results["elasticnet"] = {
            "best_alpha": best_en[0],
            "best_l1_ratio": best_en[1],
            "r2_cv_mean": round(float(cv_r2_en.mean()), 4),
            "selected_features": en_weights,
        }

        # --------------------------------------------------------
        # Stepwise Forward Selection
        # --------------------------------------------------------
        print("\n  --- Forward Stepwise Selection ---")
        if len(valid) >= 50:
            base_model = Ridge(alpha=best_ridge_alpha)
            max_feats = min(8, len(fn))
            try:
                sfs = SequentialFeatureSelector(
                    base_model, n_features_to_select=max_feats,
                    direction="forward", cv=tscv, scoring="r2"
                )
                sfs.fit(X_sorted, y_sorted)
                selected_mask = sfs.get_support()
                selected_names = [fn[i] for i, s in enumerate(selected_mask) if s]

                X_sel = X_scaled[:, selected_mask]
                X_sel_sorted = X_sorted[:, selected_mask]
                final_m = Ridge(alpha=best_ridge_alpha)
                final_m.fit(X_sel, y)
                cv_r2_sfs = cross_val_score(final_m, X_sel_sorted, y_sorted, cv=tscv, scoring="r2")

                stepwise_weights = dict(zip(selected_names, [round(float(c), 4) for c in final_m.coef_]))
                stepwise_sorted = sorted(stepwise_weights.items(), key=lambda x: abs(x[1]), reverse=True)

                print(f"  Selected {len(selected_names)} features")
                print(f"  CV R² (mean): {cv_r2_sfs.mean():.4f} ± {cv_r2_sfs.std():.4f}")
                for name, w in stepwise_sorted:
                    print(f"    {name:<35} {w:>8.4f}")

                period_results["stepwise"] = {
                    "n_selected": len(selected_names),
                    "r2_cv_mean": round(float(cv_r2_sfs.mean()), 4),
                    "r2_cv_std": round(float(cv_r2_sfs.std()), 4),
                    "selected_features": stepwise_weights,
                    "selection_order": selected_names,
                }
            except Exception as e:
                print(f"  Stepwise failed: {e}")
        else:
            print(f"  Skipping — need 50+ observations")

        # --------------------------------------------------------
        # Polynomial (degree 2)
        # --------------------------------------------------------
        print("\n  --- Polynomial Regression (degree 2) ---")
        top_feat_names = [n for n, _ in lasso_sorted[:POLY_TOP_N]] if lasso_sorted else [n for n, _ in ols_sorted[:POLY_TOP_N]]
        top_feat_indices = [fn.index(n) for n in top_feat_names if n in fn]

        if len(top_feat_indices) >= 2 and len(valid) >= 50:
            X_top = X_scaled[:, top_feat_indices]
            X_top_sorted = X_sorted[:, top_feat_indices]

            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            X_poly = poly.fit_transform(X_top)
            X_poly_sorted = poly.transform(X_top_sorted)
            poly_names = poly.get_feature_names_out([top_feat_names[i] for i in range(len(top_feat_indices))])

            poly_model = Ridge(alpha=10.0)
            poly_model.fit(X_poly, y)
            cv_r2_poly = cross_val_score(poly_model, X_poly_sorted, y_sorted, cv=tscv, scoring="r2")
            r2_in_poly = r2_score(y, poly_model.predict(X_poly))

            poly_weights = dict(zip(poly_names, poly_model.coef_))
            poly_sorted = sorted(poly_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            interaction_terms = {n: w for n, w in poly_sorted if " " in n and abs(w) > 0.01}

            print(f"  Base features: {len(top_feat_names)}")
            print(f"  Polynomial features: {X_poly.shape[1]}")
            print(f"  In-sample R²: {r2_in_poly:.4f}")
            print(f"  CV R² (mean): {cv_r2_poly.mean():.4f} ± {cv_r2_poly.std():.4f}")

            period_results["polynomial"] = {
                "base_features": top_feat_names,
                "n_poly_features": X_poly.shape[1],
                "r2_insample": round(float(r2_in_poly), 4),
                "r2_cv_mean": round(float(cv_r2_poly.mean()), 4),
                "r2_cv_std": round(float(cv_r2_poly.std()), 4),
                "top_terms": [{"term": n, "weight": round(float(w), 4)} for n, w in poly_sorted[:12]],
                "interaction_terms": {n: round(float(w), 4) for n, w in interaction_terms.items()},
            }
        else:
            print(f"  Skipping — insufficient features or data")

        # --------------------------------------------------------
        # Random Forest (conservative)
        # --------------------------------------------------------
        print("\n  --- Random Forest ---")
        if len(valid) >= 40:
            rf = RandomForestRegressor(**RF_PARAMS)
            cv_r2_rf = cross_val_score(rf, X_sorted, y_sorted, cv=tscv, scoring="r2")
            rf.fit(X_scaled, y)

            rf_importances = dict(zip(fn, [round(float(i), 4) for i in rf.feature_importances_]))
            rf_sorted = sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)

            print(f"  CV R² (mean): {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}")
            for name, imp in rf_sorted[:8]:
                bar = "█" * int(imp * 100)
                print(f"    {name:<35} {imp:.4f}  {bar}")

            period_results["random_forest"] = {
                "r2_cv_mean": round(float(cv_r2_rf.mean()), 4),
                "r2_cv_std": round(float(cv_r2_rf.std()), 4),
                "feature_importances": rf_importances,
                "top_features": [{"feature": n, "importance": i} for n, i in rf_sorted[:10]],
            }

        # --------------------------------------------------------
        # Gradient Boosting (conservative)
        # --------------------------------------------------------
        print("\n  --- Gradient Boosting ---")
        if len(valid) >= 40:
            best_gb, best_gb_score = (0.05, 2), -np.inf
            for lr in [0.01, 0.05, 0.1]:
                for depth in [2, 3]:
                    gb = GradientBoostingRegressor(
                        n_estimators=GB_PARAMS["n_estimators"], learning_rate=lr,
                        max_depth=depth, min_samples_leaf=GB_PARAMS["min_samples_leaf"],
                        subsample=GB_PARAMS["subsample"], random_state=42
                    )
                    scores = cross_val_score(gb, X_sorted, y_sorted, cv=tscv, scoring="r2")
                    if scores.mean() > best_gb_score:
                        best_gb_score = scores.mean()
                        best_gb = (lr, depth)

            gb = GradientBoostingRegressor(
                n_estimators=GB_PARAMS["n_estimators"], learning_rate=best_gb[0],
                max_depth=best_gb[1], min_samples_leaf=GB_PARAMS["min_samples_leaf"],
                subsample=GB_PARAMS["subsample"], random_state=42
            )
            cv_r2_gb = cross_val_score(gb, X_sorted, y_sorted, cv=tscv, scoring="r2")
            gb.fit(X_scaled, y)

            gb_importances = dict(zip(fn, [round(float(i), 4) for i in gb.feature_importances_]))
            gb_sorted = sorted(gb_importances.items(), key=lambda x: x[1], reverse=True)

            print(f"  Best lr={best_gb[0]}, depth={best_gb[1]}")
            print(f"  CV R² (mean): {cv_r2_gb.mean():.4f} ± {cv_r2_gb.std():.4f}")

            period_results["gradient_boosting"] = {
                "best_learning_rate": best_gb[0],
                "best_max_depth": best_gb[1],
                "r2_cv_mean": round(float(cv_r2_gb.mean()), 4),
                "r2_cv_std": round(float(cv_r2_gb.std()), 4),
                "feature_importances": gb_importances,
                "top_features": [{"feature": n, "importance": i} for n, i in gb_sorted[:10]],
            }

        # --------------------------------------------------------
        # Mutual Information
        # --------------------------------------------------------
        print("\n  --- Mutual Information ---")
        mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
        mi_dict = dict(zip(fn, [round(float(s), 4) for s in mi_scores]))
        mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

        period_results["mutual_information"] = {
            "scores": mi_dict,
            "top_features": [{"feature": n, "mi_score": s} for n, s in mi_sorted[:10]],
        }

        # --------------------------------------------------------
        # Model Comparison
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
                print(f"    {model_name:<25} CV R²: {r2:>8.4f}")

        period_results["model_comparison"] = sorted(model_summary, key=lambda x: x["cv_r2"], reverse=True)

        # --------------------------------------------------------
        # Recommended Weights
        # --------------------------------------------------------
        print("\n  --- Recommended Weights (Ridge-Lasso blend) ---")
        final_weights = {}
        for name in fn:
            ridge_w = ridge_weights.get(name, 0)
            lasso_w = lasso_weights.get(name, 0)
            if lasso_w == 0:
                blended = ridge_w * 0.2
            else:
                blended = (ridge_w + lasso_w) / 2
            if abs(blended) > 0.001:
                final_weights[name] = round(float(blended), 4)

        final_sorted = sorted(final_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, w in final_sorted:
            print(f"    {'↑' if w > 0 else '↓'} {name:<35} {w:>8.4f}")

        period_results["recommended_weights"] = {
            "method": "Ridge-Lasso blend (avg, Lasso-zero features downweighted 80%)",
            "weights": dict(final_sorted),
            "intercept": round(float(ols.intercept_), 4),
        }

        results["regression"][period_label] = period_results

    # Generate basic summary
    results["summary"] = _generate_summary(results)

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {RESULTS_FILE}")

    return results


def _sanitize_nan(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _generate_summary(results):
    """Generate a basic markdown summary from results."""
    lines = ["# SEC Filing Signal Lab — Results Summary\n"]
    meta = results.get("metadata", {})
    lines.append(f"Dataset: {meta.get('total_events', '?')} SEC filings, "
                 f"{len(meta.get('companies', []))} companies")
    if meta.get("date_range"):
        lines.append(f", {meta['date_range'][0]} to {meta['date_range'][1]}")
    lines.append("\n")

    # Best features by IC
    lines.append("## Strongest Individual Features\n")
    feat_data = results.get("features", {})
    feat_list = []
    for fname, periods in feat_data.items():
        best = max(periods.values(), key=lambda x: abs(x.get("ic", 0)), default={})
        if best:
            feat_list.append((fname, best))
    feat_list.sort(key=lambda x: abs(x[1].get("ic", 0)), reverse=True)
    for fname, best in feat_list[:8]:
        ic = best.get("ic", 0)
        acc = best.get("accuracy", 0)
        p = best.get("ic_pvalue", 1)
        lines.append(f"- **{fname}**: IC={ic:.3f}, accuracy={acc:.1%}, p={p:.3f}")
    lines.append("")

    # Model comparison
    lines.append("## Best Model by Horizon\n")
    for period, pdata in results.get("regression", {}).items():
        comparison = pdata.get("model_comparison", [])
        if comparison:
            best = comparison[0]
            lines.append(f"- **{period}**: {best['model']} (CV R²={best['cv_r2']:.4f})")
    lines.append("")

    markdown = "\n".join(lines)
    return {
        "markdown": markdown,
        "generated_at": datetime.now().isoformat(),
    }


# ============================================================
# STEP: TRAIN-MODEL
# ============================================================

def train_scoring_model(data_dir=None, model_file=None, results_file=None):
    """Train a Lasso scoring model on the best horizon."""
    if data_dir is None:
        data_dir = DATA_DIR
    if model_file is None:
        model_file = MODEL_FILE
    if results_file is None:
        results_file = RESULTS_FILE

    print("\n" + "=" * 60)
    print("TRAINING SCORING MODEL (SEC Filings)")
    print("=" * 60)

    df = load_csv()
    if df is None:
        print("  No data available.")
        return None

    # Determine best horizon from backtest results
    best_horizon = "7D"
    best_r2 = -np.inf
    if results_file.exists():
        res = json.loads(results_file.read_text())
        for period, pdata in res.get("regression", {}).items():
            comparison = pdata.get("model_comparison", [])
            if comparison and comparison[0]["cv_r2"] > best_r2:
                best_r2 = comparison[0]["cv_r2"]
                best_horizon = period
    print(f"  Best horizon: {best_horizon} (CV R²={best_r2:.4f})")

    # Get target column
    target_col = HOLDING_PERIODS.get(best_horizon) or ALPHA_PERIODS.get(best_horizon)
    if target_col is None or target_col not in df.columns:
        # Fallback to 7D
        target_col = HOLDING_PERIODS.get("7D", "actual7dReturn")
        best_horizon = "7D"
    if target_col not in df.columns:
        print(f"  Target column {target_col} not in CSV. Available: {list(df.columns)}")
        return None

    # Rename features
    rename_map = {csv_col: clean for csv_col, clean in FEATURE_COLUMNS.items() if csv_col in df.columns}
    df_work = df.rename(columns=rename_map)
    feature_names = [clean for clean in rename_map.values()]
    feat_cols = [f for f in feature_names if f in df_work.columns]

    # Sort by filing date
    date_col = None
    for col in ["filingDate", "filing_date", "date"]:
        if col in df.columns:
            date_col = col
            break
    if date_col:
        df_work["_sort_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df_work = df_work.sort_values("_sort_date")

    # Drop rows where target is NaN, impute feature NaN with median
    has_target = df_work[target_col].notna()
    valid = df_work.loc[has_target, feat_cols + [target_col]].copy()
    good_feats = [f for f in feat_cols if valid[f].notna().mean() > 0.5]
    for f in good_feats:
        valid[f] = valid[f].fillna(valid[f].median())
    valid = valid[good_feats + [target_col]].dropna()
    feat_cols = good_feats
    print(f"  Training samples: {len(valid)}")

    if len(valid) < 30:
        print("  Not enough data for training.")
        return None

    X = valid[feat_cols].values
    y = valid[target_col].values

    # Standardise
    means = X.mean(axis=0).tolist()
    stds = X.std(axis=0).tolist()
    stds = [s if s > 1e-9 else 1.0 for s in stds]
    X_scaled = (X - np.array(means)) / np.array(stds)

    # Feature percentiles
    percentiles = {}
    for i, fname in enumerate(feat_cols):
        col = X[:, i]
        percentiles[fname] = {
            "p25": float(np.percentile(col, 25)),
            "p50": float(np.percentile(col, 50)),
            "p75": float(np.percentile(col, 75)),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }

    # Fit Lasso with CV
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    best_alpha, best_score = 0.1, -np.inf
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        scores = cross_val_score(lasso, X_scaled, y, cv=tscv, scoring="r2")
        if scores.mean() > best_score:
            best_alpha, best_score = alpha, scores.mean()

    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X_scaled, y)

    weights = {fname: round(float(w), 6) for fname, w in zip(feat_cols, lasso.coef_)}
    intercept = float(lasso.intercept_)
    selected = {k: v for k, v in weights.items() if abs(v) > 1e-6}

    print(f"  Best alpha: {best_alpha}")
    print(f"  CV R²: {best_score:.4f}")
    print(f"  Features selected: {len(selected)}/{len(feat_cols)}")
    for fname, w in sorted(selected.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {fname:<35} {'+' if w > 0 else ''}{w:.4f}")

    # Score distribution
    train_scores = X_scaled @ lasso.coef_ + intercept
    score_percentiles = {
        "p10": float(np.percentile(train_scores, 10)),
        "p25": float(np.percentile(train_scores, 25)),
        "p50": float(np.percentile(train_scores, 50)),
        "p75": float(np.percentile(train_scores, 75)),
        "p90": float(np.percentile(train_scores, 90)),
        "mean": float(train_scores.mean()),
        "std": float(train_scores.std()),
    }

    model = {
        "version": 1,
        "trained_at": datetime.now().isoformat(),
        "horizon": best_horizon,
        "n_training_samples": len(valid),
        "cv_r2": round(best_score, 4),
        "lasso_alpha": best_alpha,
        "features": feat_cols,
        "weights": weights,
        "intercept": round(intercept, 6),
        "scaler_means": {fname: round(m, 6) for fname, m in zip(feat_cols, means)},
        "scaler_stds": {fname: round(s, 6) for fname, s in zip(feat_cols, stds)},
        "feature_percentiles": percentiles,
        "score_percentiles": score_percentiles,
    }

    data_dir.mkdir(exist_ok=True)
    model_file.write_text(json.dumps(model, indent=2))
    print(f"\n  Model saved to {model_file}")
    return model


# ============================================================
# STEP: SCORE
# ============================================================

def score_filing(model, feature_values):
    """Score a single filing using the trained model.

    feature_values: dict of {feature_name: value}
    """
    weights = model["weights"]
    means = model["scaler_means"]
    stds = model["scaler_stds"]
    intercept = model["intercept"]
    sp = model["score_percentiles"]

    contributions = {}
    raw_score = intercept

    for fname in model["features"]:
        val = feature_values.get(fname)
        w = weights.get(fname, 0.0)
        if val is not None and abs(w) > 1e-9 and not np.isnan(val):
            z = (val - means[fname]) / stds[fname]
            contrib = z * w
            raw_score += contrib
            contributions[fname] = round(contrib, 4)

    # Percentile
    if raw_score >= sp["p90"]:
        pct_label = ">90th"
    elif raw_score >= sp["p75"]:
        pct_label = "75-90th"
    elif raw_score >= sp["p50"]:
        pct_label = "50-75th"
    elif raw_score >= sp["p25"]:
        pct_label = "25-50th"
    elif raw_score >= sp["p10"]:
        pct_label = "10-25th"
    else:
        pct_label = "<10th"

    # Signal
    if raw_score >= sp["p75"]:
        signal = "LONG"
        confidence = "high" if raw_score >= sp["p90"] else "medium"
    elif raw_score <= sp["p25"]:
        signal = "SHORT"
        confidence = "high" if raw_score <= sp["p10"] else "medium"
    else:
        signal = "NEUTRAL"
        confidence = "low"

    sorted_contribs = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "raw_score": round(raw_score, 4),
        "expected_return_pct": round(raw_score, 2),
        "signal": signal,
        "confidence": confidence,
        "percentile": pct_label,
        "feature_contributions": sorted_contribs,
    }


def score_all(data_dir=None, model_file=None, scores_file=None):
    """Score all SEC filings."""
    if data_dir is None:
        data_dir = DATA_DIR
    if model_file is None:
        model_file = MODEL_FILE
    if scores_file is None:
        scores_file = SCORES_FILE

    if not model_file.exists():
        print("No scoring model found. Run --step train-model first.")
        return None

    model = json.loads(model_file.read_text())

    print("\n" + "=" * 60)
    print("SCORING ALL SEC FILINGS")
    print(f"  Model: Lasso ({model['horizon']} horizon, CV R²={model['cv_r2']:.4f})")
    print("=" * 60)

    df = load_csv()
    if df is None:
        return None

    # Rename features
    rename_map = {csv_col: clean for csv_col, clean in FEATURE_COLUMNS.items() if csv_col in df.columns}
    df_work = df.rename(columns=rename_map)

    # Determine ticker column
    ticker_col = None
    for col in ["ticker", "symbol", "Ticker"]:
        if col in df.columns:
            ticker_col = col
            break

    scored = []
    for idx, row in df_work.iterrows():
        feature_values = {fname: row.get(fname) for fname in model["features"]
                         if fname in row and pd.notna(row.get(fname))}
        if not feature_values:
            continue

        result = score_filing(model, feature_values)

        # Add metadata
        if ticker_col and ticker_col in df.columns:
            result["symbol"] = df.loc[idx, ticker_col]
        elif "ticker" in df_work.columns:
            result["symbol"] = row.get("ticker", "?")
        else:
            result["symbol"] = "?"

        # Extract quarter/year from filing date
        date_col = None
        for col in ["filingDate", "filing_date", "date"]:
            if col in df.columns:
                date_col = col
                break
        if date_col:
            try:
                d = pd.to_datetime(df.loc[idx, date_col])
                result["quarter"] = (d.month - 1) // 3 + 1
                result["year"] = d.year
            except Exception:
                result["quarter"] = "?"
                result["year"] = "?"
        else:
            result["quarter"] = "?"
            result["year"] = "?"

        scored.append(result)

    scored.sort(key=lambda x: x["raw_score"], reverse=True)

    n_long = sum(1 for s in scored if s["signal"] == "LONG")
    n_short = sum(1 for s in scored if s["signal"] == "SHORT")
    n_neutral = sum(1 for s in scored if s["signal"] == "NEUTRAL")

    print(f"\n  Scored: {len(scored)} filings")
    print(f"  Signals: {n_long} LONG | {n_neutral} NEUTRAL | {n_short} SHORT")

    print(f"\n  --- TOP 10 LONG ---")
    for i, s in enumerate(scored[:10], 1):
        top = next(iter(s["feature_contributions"]), "")
        print(f"  {i:<3} {s['symbol']:<8} Q{s['quarter']} {s['year']}  {s['raw_score']:>+7.3f}  {s['signal']:<8}  {top}")

    print(f"\n  --- TOP 10 SHORT ---")
    for i, s in enumerate(reversed(scored[-10:]), 1):
        top = next(iter(s["feature_contributions"]), "")
        print(f"  {i:<3} {s['symbol']:<8} Q{s['quarter']} {s['year']}  {s['raw_score']:>+7.3f}  {s['signal']:<8}  {top}")

    data_dir.mkdir(exist_ok=True)
    scores_file.write_text(json.dumps(scored, indent=2))
    print(f"\n  Scores saved to {scores_file}")
    return scored


# ============================================================
# STEP: COMBINED — Merge SEC + Earnings features
# ============================================================

def run_combined():
    """Merge SEC filing features with earnings transcript features and run model zoo."""
    print("\n" + "=" * 60)
    print("STEP: Combined Model — SEC Filings + Earnings Transcripts")
    print("=" * 60)

    from scipy import stats as scipy_stats
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import r2_score
    from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_regression

    # Load SEC filing data
    sec_df = load_csv()
    if sec_df is None:
        print("  No SEC filing data. Run --step pull first.")
        return None

    # Load earnings analysis files
    earnings_dir = Path("earnings_signal_data/analysis")
    if not earnings_dir.exists():
        print(f"  No earnings data at {earnings_dir}")
        print("  Run the earnings pipeline first.")
        return None

    # Load earnings price data for returns
    earnings_price_file = Path("earnings_signal_data/price_data.json")

    # Rename SEC features
    rename_map = {csv_col: f"sec_{clean}" for csv_col, clean in FEATURE_COLUMNS.items() if csv_col in sec_df.columns}
    sec_work = sec_df.rename(columns=rename_map)

    # Determine ticker and date columns
    ticker_col = None
    for col in ["ticker", "symbol", "Ticker"]:
        if col in sec_df.columns:
            ticker_col = col
            break

    date_col = None
    for col in ["filingDate", "filing_date", "date"]:
        if col in sec_df.columns:
            date_col = col
            break

    if not ticker_col:
        print("  No ticker column found in SEC data!")
        return None

    # Derive quarter + year from SEC filing date for matching
    if date_col:
        sec_dates = pd.to_datetime(sec_df[date_col], errors="coerce")
        sec_work["_quarter"] = ((sec_dates.dt.month - 1) // 3 + 1).astype("Int64")
        sec_work["_year"] = sec_dates.dt.year.astype("Int64")
        sec_work["_ticker"] = sec_df[ticker_col].str.upper()
    else:
        print("  No date column in SEC data for matching!")
        return None

    # Load earnings features
    EARNINGS_FEATURES = [
        "mgmt_hedging", "mgmt_deflection", "mgmt_specificity", "mgmt_confidence_shift",
        "analyst_skepticism", "analyst_surprise", "analyst_focus_cluster",
        "guidance_revision_dir", "guidance_qualifiers",
        "new_risk_mention", "macro_blame", "capex_language", "hiring_language",
        "competitive_mentions", "customer_language", "pricing_power",
    ]

    earnings_rows = []
    for analysis_file in sorted(earnings_dir.glob("*_analysis.json")):
        try:
            analysis = json.loads(analysis_file.read_text())
        except Exception:
            continue

        symbol = analysis.get("symbol", analysis_file.stem.split("_")[0]).upper()
        quarter = analysis.get("quarter")
        year = analysis.get("year")

        if not quarter or not year:
            continue

        row = {"_ticker": symbol, "_quarter": int(quarter), "_year": int(year)}
        features = analysis.get("features", {})
        for fname in EARNINGS_FEATURES:
            feat_data = features.get(fname, {})
            if isinstance(feat_data, dict):
                row[f"nlp_{fname}"] = feat_data.get("score")
        earnings_rows.append(row)

    if not earnings_rows:
        print("  No earnings analysis files found!")
        return None

    earnings_df = pd.DataFrame(earnings_rows)
    print(f"  Loaded {len(earnings_df)} earnings analyses")
    print(f"  SEC filings: {len(sec_work)} rows")

    # Match on ticker + quarter + year
    merged = pd.merge(
        sec_work, earnings_df,
        on=["_ticker", "_quarter", "_year"],
        how="inner"
    )

    print(f"  Matched: {len(merged)} events (ticker + quarter + year)")

    if len(merged) < 20:
        print("  Too few matched events for meaningful modeling.")
        print("  This is expected if the SEC and earnings datasets cover different time periods.")
        return None

    # Identify all feature columns
    sec_feat_cols = [c for c in merged.columns if c.startswith("sec_")]
    nlp_feat_cols = [c for c in merged.columns if c.startswith("nlp_")]
    all_feat_cols = sec_feat_cols + nlp_feat_cols

    # Identify target columns (from SEC data)
    target_cols = {}
    for label, orig_col in {**HOLDING_PERIODS, **ALPHA_PERIODS}.items():
        if orig_col in merged.columns:
            target_cols[label] = orig_col

    if not target_cols:
        print("  No target return columns in merged data!")
        return None

    print(f"  Combined features: {len(sec_feat_cols)} SEC + {len(nlp_feat_cols)} NLP = {len(all_feat_cols)}")
    print(f"  Target columns: {list(target_cols.keys())}")

    # Build combined feature metadata
    combined_features_meta = {}
    for col in sec_feat_cols:
        clean = col.replace("sec_", "")
        if clean in SEC_FEATURES:
            meta = SEC_FEATURES[clean].copy()
            meta["name"] = "SEC: " + meta["name"]
            combined_features_meta[col] = meta
    for col in nlp_feat_cols:
        clean = col.replace("nlp_", "")
        combined_features_meta[col] = {
            "name": "NLP: " + clean.replace("_", " ").title(),
            "cat": "NLP (Earnings)",
            "color": "#9b59b6",
            "bear": clean in ["mgmt_hedging", "mgmt_deflection", "mgmt_confidence_shift",
                              "analyst_skepticism", "guidance_qualifiers", "new_risk_mention",
                              "macro_blame", "competitive_mentions"],
            "desc": f"Earnings transcript feature: {clean}",
        }

    # Sort by date
    if date_col and date_col in sec_df.columns:
        merged["_sort_date"] = pd.to_datetime(sec_df.loc[merged.index.intersection(sec_df.index), date_col] if len(merged) == len(sec_df) else merged.get("_sort_date"), errors="coerce")
        if "_sort_date" not in merged.columns or merged["_sort_date"].isna().all():
            # Try reconstructing from _year/_quarter
            merged["_sort_date"] = pd.to_datetime(
                merged["_year"].astype(str) + "-" + (merged["_quarter"] * 3).astype(str) + "-15",
                errors="coerce"
            )
        merged = merged.sort_values("_sort_date").reset_index(drop=True)
    else:
        merged["_sort_date"] = pd.to_datetime(
            merged["_year"].astype(str) + "-" + (merged["_quarter"] * 3).astype(str) + "-15",
            errors="coerce"
        )
        merged = merged.sort_values("_sort_date").reset_index(drop=True)

    # Now run the same model zoo as run_backtest
    tickers = sorted(merged["_ticker"].unique().tolist())

    results = {
        "metadata": {
            "total_events": len(merged),
            "companies": tickers,
            "date_range": [str(merged["_sort_date"].min().date()), str(merged["_sort_date"].max().date())] if merged["_sort_date"].notna().any() else [],
            "generated_at": datetime.now().isoformat(),
            "source": "combined",
            "n_sec_features": len(sec_feat_cols),
            "n_nlp_features": len(nlp_feat_cols),
            "feature_metadata": combined_features_meta,
        },
        "features": {},
        "sample_extractions": {},
        "correlation_matrix": {},
        "combinations": [],
        "regression": {},
    }

    # Per-feature signal analysis
    print("\n--- Combined Feature Signal Analysis ---\n")
    print(f"{'Feature':<40} {'Period':<10} {'IC':<10} {'Accuracy':<10} {'n':<6}")
    print("-" * 80)

    for feat_name in all_feat_cols:
        if feat_name not in merged.columns:
            continue
        feat_results = {}
        clean = feat_name.replace("sec_", "").replace("nlp_", "")
        meta = combined_features_meta.get(feat_name, SEC_FEATURES.get(clean, {}))
        is_bearish = meta.get("bear", False)

        for period_label, target_col in target_cols.items():
            valid = merged[[feat_name, target_col]].dropna()
            if len(valid) < 15:
                continue

            scores_arr = valid[feat_name].values
            returns_arr = valid[target_col].values

            ic, ic_pvalue = scipy_stats.spearmanr(scores_arr, returns_arr)

            median_score = np.median(scores_arr)
            if is_bearish:
                predictions = scores_arr > median_score
                actuals = returns_arr < 0
            else:
                predictions = scores_arr > median_score
                actuals = returns_arr > 0
            accuracy = np.mean(predictions == actuals)

            q75 = np.percentile(scores_arr, 75)
            if is_bearish:
                sig_rets = -returns_arr[scores_arr > q75]
            else:
                sig_rets = returns_arr[scores_arr > q75]

            avg_ret = np.mean(sig_rets) if len(sig_rets) > 1 else 0
            std_ret = np.std(sig_rets) if len(sig_rets) > 1 else 1
            sharpe = avg_ret / std_ret * np.sqrt(4) if std_ret > 0 else 0
            win_rate = np.mean(sig_rets > 0) if len(sig_rets) > 0 else 0.5

            feat_results[period_label] = {
                "ic": round(float(ic), 4),
                "ic_pvalue": round(float(ic_pvalue), 4),
                "accuracy": round(float(accuracy), 4),
                "sharpe": round(float(sharpe), 4),
                "avg_return_pct": round(float(avg_ret), 4),
                "win_rate": round(float(win_rate), 4),
                "n_observations": len(valid),
                "n_signal_triggered": len(sig_rets),
            }

            print(f"{feat_name:<40} {period_label:<10} {ic:>8.4f}  {accuracy:>8.1%}  {len(valid):<6}")

        if feat_results:
            results["features"][feat_name] = feat_results

    # Correlation matrix
    feat_df = merged[all_feat_cols].dropna(axis=1, how="all")
    if len(feat_df.columns) > 1:
        corr = feat_df.corr(method="spearman")
        results["correlation_matrix"] = {
            c1: {c2: round(float(corr.loc[c1, c2]), 3) for c2 in corr.columns}
            for c1 in corr.index
        }

    # Regression model zoo on combined features
    print("\n" + "=" * 60)
    print("Combined Regression Modeling")
    print("=" * 60)

    for period_label, target_col in target_cols.items():
        if target_col not in merged.columns:
            continue

        print(f"\n{'='*50}")
        print(f"  Modeling: {period_label} forward returns (combined)")
        print(f"{'='*50}")

        feat_present = [f for f in all_feat_cols if f in merged.columns]

        # Handle NaN: drop rows missing target, impute feature NaN with median
        has_target = merged[target_col].notna()
        valid = merged.loc[has_target, feat_present + [target_col]].copy()

        # Drop features with >50% missing, impute rest with median
        good_feats = [f for f in feat_present if valid[f].notna().mean() > 0.5]
        for f in good_feats:
            valid[f] = valid[f].fillna(valid[f].median())
        valid = valid[good_feats + [target_col]].dropna()

        if len(valid) < 30:
            print(f"  Skipping — only {len(valid)} observations")
            continue

        X = valid[good_feats].values
        y = valid[target_col].values
        fn = list(good_feats)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if "_sort_date" in merged.columns:
            valid_sorted = valid.copy()
            valid_sorted["_sd"] = merged.loc[valid.index, "_sort_date"]
            valid_sorted = valid_sorted.sort_values("_sd")
            X_sorted = scaler.transform(valid_sorted[good_feats].values)
            y_sorted = valid_sorted[target_col].values
        else:
            X_sorted, y_sorted = X_scaled, y

        n_splits = min(N_SPLITS, max(2, len(valid) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        period_results = {
            "n_observations": len(valid),
            "y_mean": round(float(y.mean()), 4),
            "y_std": round(float(y.std()), 4),
        }

        # Run all models (same as run_backtest)
        # OLS
        ols = LinearRegression()
        ols.fit(X_scaled, y)
        cv_r2_ols = cross_val_score(ols, X_sorted, y_sorted, cv=tscv, scoring="r2")
        ols_weights = dict(zip(fn, [round(float(c), 4) for c in ols.coef_]))
        ols_sorted = sorted(ols_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        period_results["ols"] = {
            "r2_insample": round(float(r2_score(y, ols.predict(X_scaled))), 4),
            "r2_cv_mean": round(float(cv_r2_ols.mean()), 4),
            "r2_cv_std": round(float(cv_r2_ols.std()), 4),
            "coefficients": ols_weights,
            "top_features": [{"feature": n, "weight": w} for n, w in ols_sorted[:8]],
        }

        # Ridge
        best_ra, best_rs = 1.0, -np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            scores = cross_val_score(Ridge(alpha=alpha), X_sorted, y_sorted, cv=tscv, scoring="r2")
            if scores.mean() > best_rs:
                best_ra, best_rs = alpha, scores.mean()
        ridge = Ridge(alpha=best_ra)
        ridge.fit(X_scaled, y)
        cv_r2_ridge = cross_val_score(ridge, X_sorted, y_sorted, cv=tscv, scoring="r2")
        ridge_weights = dict(zip(fn, [round(float(c), 4) for c in ridge.coef_]))
        ridge_sorted = sorted(ridge_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        period_results["ridge"] = {
            "best_alpha": best_ra,
            "r2_cv_mean": round(float(cv_r2_ridge.mean()), 4),
            "r2_cv_std": round(float(cv_r2_ridge.std()), 4),
            "coefficients": ridge_weights,
            "top_features": [{"feature": n, "weight": w} for n, w in ridge_sorted[:8]],
        }

        # Lasso
        best_la, best_ls = 0.1, -np.inf
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            scores = cross_val_score(Lasso(alpha=alpha, max_iter=10000), X_sorted, y_sorted, cv=tscv, scoring="r2")
            if scores.mean() > best_ls:
                best_la, best_ls = alpha, scores.mean()
        lasso = Lasso(alpha=best_la, max_iter=10000)
        lasso.fit(X_scaled, y)
        cv_r2_lasso = cross_val_score(lasso, X_sorted, y_sorted, cv=tscv, scoring="r2")
        lasso_weights = {n: round(float(c), 4) for n, c in zip(fn, lasso.coef_) if abs(c) > 1e-6}
        lasso_sorted = sorted(lasso_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        period_results["lasso"] = {
            "best_alpha": best_la,
            "r2_cv_mean": round(float(cv_r2_lasso.mean()), 4),
            "r2_cv_std": round(float(cv_r2_lasso.std()), 4),
            "n_features_selected": sum(1 for c in lasso.coef_ if abs(c) > 1e-6),
            "n_features_eliminated": sum(1 for c in lasso.coef_ if abs(c) <= 1e-6),
            "selected_features": lasso_weights,
            "top_features": [{"feature": n, "weight": w} for n, w in lasso_sorted],
        }

        # ElasticNet
        best_en, best_es = (0.1, 0.5), -np.inf
        for alpha in [0.01, 0.1, 0.5, 1.0]:
            for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                scores = cross_val_score(ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000),
                                         X_sorted, y_sorted, cv=tscv, scoring="r2")
                if scores.mean() > best_es:
                    best_en, best_es = (alpha, l1), scores.mean()
        en = ElasticNet(alpha=best_en[0], l1_ratio=best_en[1], max_iter=10000)
        en.fit(X_scaled, y)
        en_weights = {n: round(float(c), 4) for n, c in zip(fn, en.coef_) if abs(c) > 1e-6}
        period_results["elasticnet"] = {
            "best_alpha": best_en[0],
            "best_l1_ratio": best_en[1],
            "r2_cv_mean": round(float(cross_val_score(en, X_sorted, y_sorted, cv=tscv, scoring="r2").mean()), 4),
            "selected_features": en_weights,
        }

        # Stepwise
        if len(valid) >= 50:
            try:
                sfs = SequentialFeatureSelector(
                    Ridge(alpha=best_ra), n_features_to_select=min(8, len(fn)),
                    direction="forward", cv=tscv, scoring="r2"
                )
                sfs.fit(X_sorted, y_sorted)
                sel_mask = sfs.get_support()
                sel_names = [fn[i] for i, s in enumerate(sel_mask) if s]
                final_m = Ridge(alpha=best_ra)
                final_m.fit(X_scaled[:, sel_mask], y)
                cv_sfs = cross_val_score(final_m, X_sorted[:, sel_mask], y_sorted, cv=tscv, scoring="r2")
                sw = dict(zip(sel_names, [round(float(c), 4) for c in final_m.coef_]))
                period_results["stepwise"] = {
                    "n_selected": len(sel_names),
                    "r2_cv_mean": round(float(cv_sfs.mean()), 4),
                    "r2_cv_std": round(float(cv_sfs.std()), 4),
                    "selected_features": sw,
                    "selection_order": sel_names,
                }
            except Exception as e:
                print(f"  Stepwise failed: {e}")

        # Polynomial
        top_names = [n for n, _ in lasso_sorted[:POLY_TOP_N]] if lasso_sorted else [n for n, _ in ols_sorted[:POLY_TOP_N]]
        top_idx = [fn.index(n) for n in top_names if n in fn]
        if len(top_idx) >= 2 and len(valid) >= 50:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_scaled[:, top_idx])
            X_poly_s = poly.transform(X_sorted[:, top_idx])
            poly_model = Ridge(alpha=10.0)
            poly_model.fit(X_poly, y)
            cv_poly = cross_val_score(poly_model, X_poly_s, y_sorted, cv=tscv, scoring="r2")
            pn = poly.get_feature_names_out([top_names[i] for i in range(len(top_idx))])
            pw = dict(zip(pn, poly_model.coef_))
            ps = sorted(pw.items(), key=lambda x: abs(x[1]), reverse=True)
            period_results["polynomial"] = {
                "base_features": top_names,
                "n_poly_features": X_poly.shape[1],
                "r2_insample": round(float(r2_score(y, poly_model.predict(X_poly))), 4),
                "r2_cv_mean": round(float(cv_poly.mean()), 4),
                "r2_cv_std": round(float(cv_poly.std()), 4),
                "top_terms": [{"term": n, "weight": round(float(w), 4)} for n, w in ps[:12]],
                "interaction_terms": {n: round(float(w), 4) for n, w in ps if " " in n and abs(w) > 0.01},
            }

        # Random Forest
        if len(valid) >= 40:
            rf = RandomForestRegressor(**RF_PARAMS)
            cv_rf = cross_val_score(rf, X_sorted, y_sorted, cv=tscv, scoring="r2")
            rf.fit(X_scaled, y)
            rf_imp = dict(zip(fn, [round(float(i), 4) for i in rf.feature_importances_]))
            rf_s = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
            period_results["random_forest"] = {
                "r2_cv_mean": round(float(cv_rf.mean()), 4),
                "r2_cv_std": round(float(cv_rf.std()), 4),
                "feature_importances": rf_imp,
                "top_features": [{"feature": n, "importance": i} for n, i in rf_s[:10]],
            }

        # Gradient Boosting
        if len(valid) >= 40:
            best_gb, best_gbs = (0.05, 2), -np.inf
            for lr in [0.01, 0.05, 0.1]:
                for d in [2, 3]:
                    scores = cross_val_score(
                        GradientBoostingRegressor(n_estimators=100, learning_rate=lr, max_depth=d,
                                                  min_samples_leaf=10, subsample=0.8, random_state=42),
                        X_sorted, y_sorted, cv=tscv, scoring="r2")
                    if scores.mean() > best_gbs:
                        best_gb, best_gbs = (lr, d), scores.mean()
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=best_gb[0], max_depth=best_gb[1],
                                           min_samples_leaf=10, subsample=0.8, random_state=42)
            cv_gb = cross_val_score(gb, X_sorted, y_sorted, cv=tscv, scoring="r2")
            gb.fit(X_scaled, y)
            gb_imp = dict(zip(fn, [round(float(i), 4) for i in gb.feature_importances_]))
            gb_s = sorted(gb_imp.items(), key=lambda x: x[1], reverse=True)
            period_results["gradient_boosting"] = {
                "best_learning_rate": best_gb[0],
                "best_max_depth": best_gb[1],
                "r2_cv_mean": round(float(cv_gb.mean()), 4),
                "r2_cv_std": round(float(cv_gb.std()), 4),
                "feature_importances": gb_imp,
                "top_features": [{"feature": n, "importance": i} for n, i in gb_s[:10]],
            }

        # Mutual Information
        mi = mutual_info_regression(X_scaled, y, random_state=42)
        mi_dict = dict(zip(fn, [round(float(s), 4) for s in mi]))
        mi_s = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        period_results["mutual_information"] = {
            "scores": mi_dict,
            "top_features": [{"feature": n, "mi_score": s} for n, s in mi_s[:10]],
        }

        # Model comparison
        model_summary = []
        for mn, mk in [("OLS", "ols"), ("Ridge", "ridge"), ("Lasso", "lasso"),
                        ("ElasticNet", "elasticnet"), ("Stepwise+Ridge", "stepwise"),
                        ("Polynomial", "polynomial"), ("Random Forest", "random_forest"),
                        ("Gradient Boost", "gradient_boosting")]:
            if mk in period_results:
                r2 = period_results[mk].get("r2_cv_mean", 0)
                model_summary.append({"model": mn, "cv_r2": r2})
                print(f"    {mn:<25} CV R²: {r2:>8.4f}")
        period_results["model_comparison"] = sorted(model_summary, key=lambda x: x["cv_r2"], reverse=True)

        # Recommended weights
        final_w = {}
        for name in fn:
            rw = ridge_weights.get(name, 0)
            lw = lasso_weights.get(name, 0)
            blended = (rw + lw) / 2 if lw != 0 else rw * 0.2
            if abs(blended) > 0.001:
                final_w[name] = round(float(blended), 4)
        final_ws = sorted(final_w.items(), key=lambda x: abs(x[1]), reverse=True)
        period_results["recommended_weights"] = {
            "method": "Ridge-Lasso blend (combined SEC+NLP features)",
            "weights": dict(final_ws),
            "intercept": round(float(ols.intercept_), 4),
        }

        results["regression"][period_label] = period_results

    # Summary
    results["summary"] = _generate_summary(results)

    # Save (sanitize NaN for JSON compatibility)
    results = _sanitize_nan(results)
    COMBINED_DIR.mkdir(exist_ok=True)
    COMBINED_RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\n  Combined results saved to {COMBINED_RESULTS}")

    # Train combined scoring model
    print("\n  Training combined scoring model...")
    # Find best horizon
    best_h, best_r2 = "7D", -np.inf
    for p, pd_ in results.get("regression", {}).items():
        comp = pd_.get("model_comparison", [])
        if comp and comp[0]["cv_r2"] > best_r2:
            best_h, best_r2 = p, comp[0]["cv_r2"]

    target_col = HOLDING_PERIODS.get(best_h) or ALPHA_PERIODS.get(best_h)
    if target_col and target_col in merged.columns:
        feat_present = [f for f in all_feat_cols if f in merged.columns]
        # Handle NaN: drop rows missing target, impute feature NaN with median
        has_target = merged[target_col].notna()
        valid = merged.loc[has_target, feat_present + [target_col]].copy()
        good_feats = [f for f in feat_present if valid[f].notna().mean() > 0.5]
        for f in good_feats:
            valid[f] = valid[f].fillna(valid[f].median())
        valid = valid[good_feats + [target_col]].dropna()
        feat_present = good_feats  # Update reference for downstream code

        if len(valid) >= 30:
            X = valid[feat_present].values
            y_vals = valid[target_col].values
            means = X.mean(axis=0).tolist()
            stds_vals = X.std(axis=0).tolist()
            stds_vals = [s if s > 1e-9 else 1.0 for s in stds_vals]
            X_s = (X - np.array(means)) / np.array(stds_vals)

            tscv = TimeSeriesSplit(n_splits=N_SPLITS)
            b_a, b_s = 0.1, -np.inf
            for a in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
                sc = cross_val_score(Lasso(alpha=a, max_iter=10000), X_s, y_vals, cv=tscv, scoring="r2")
                if sc.mean() > b_s:
                    b_a, b_s = a, sc.mean()

            final_lasso = Lasso(alpha=b_a, max_iter=10000)
            final_lasso.fit(X_s, y_vals)

            train_sc = X_s @ final_lasso.coef_ + final_lasso.intercept_
            combined_model = {
                "version": 1,
                "trained_at": datetime.now().isoformat(),
                "horizon": best_h,
                "n_training_samples": len(valid),
                "cv_r2": round(b_s, 4),
                "lasso_alpha": b_a,
                "features": feat_present,
                "weights": {n: round(float(w), 6) for n, w in zip(feat_present, final_lasso.coef_)},
                "intercept": round(float(final_lasso.intercept_), 6),
                "scaler_means": {n: round(m, 6) for n, m in zip(feat_present, means)},
                "scaler_stds": {n: round(s, 6) for n, s in zip(feat_present, stds_vals)},
                "feature_percentiles": {
                    n: {"p25": float(np.percentile(X[:, i], 25)),
                        "p50": float(np.percentile(X[:, i], 50)),
                        "p75": float(np.percentile(X[:, i], 75)),
                        "mean": float(X[:, i].mean()),
                        "std": float(X[:, i].std())}
                    for i, n in enumerate(feat_present)
                },
                "score_percentiles": {
                    "p10": float(np.percentile(train_sc, 10)),
                    "p25": float(np.percentile(train_sc, 25)),
                    "p50": float(np.percentile(train_sc, 50)),
                    "p75": float(np.percentile(train_sc, 75)),
                    "p90": float(np.percentile(train_sc, 90)),
                    "mean": float(train_sc.mean()),
                    "std": float(train_sc.std()),
                },
            }
            COMBINED_MODEL.write_text(json.dumps(combined_model, indent=2))
            print(f"  Combined model saved to {COMBINED_MODEL}")

            # Score all matched events
            scored = []
            for idx, row in merged.iterrows():
                fv = {n: row.get(n) for n in feat_present if n in row and pd.notna(row.get(n))}
                if not fv:
                    continue
                result = score_filing(combined_model, fv)
                result["symbol"] = row.get("_ticker", "?")
                result["quarter"] = int(row["_quarter"]) if pd.notna(row.get("_quarter")) else "?"
                result["year"] = int(row["_year"]) if pd.notna(row.get("_year")) else "?"
                scored.append(result)
            scored.sort(key=lambda x: x["raw_score"], reverse=True)
            COMBINED_SCORES.write_text(json.dumps(scored, indent=2))
            print(f"  Combined scores saved to {COMBINED_SCORES} ({len(scored)} events)")

    return results


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SEC Filing Signal Analyzer — Model Zoo Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run modes:
  python sec_filing_pipeline.py                    Full pipeline (pull + backtest)
  python sec_filing_pipeline.py --step pull        Download CSV from GitHub
  python sec_filing_pipeline.py --step backtest    Run model zoo on SEC data
  python sec_filing_pipeline.py --step train-model Train scoring model
  python sec_filing_pipeline.py --step score       Score all filings
  python sec_filing_pipeline.py --step combined    Merge with earnings data + model zoo
        """
    )
    parser.add_argument("--step", choices=["pull", "backtest", "train-model", "score", "combined"],
                        help="Run only a specific pipeline step")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  SEC FILING SIGNAL ANALYZER — Model Zoo")
    print("=" * 60)
    print(f"\nData source: GitHub (sec-filing-analyzer/ml_dataset_with_concern.csv)")
    print(f"Data directory: {DATA_DIR}")

    if args.step:
        if args.step == "pull":
            pull_data()
        elif args.step == "backtest":
            run_backtest()
        elif args.step == "train-model":
            train_scoring_model()
        elif args.step == "score":
            score_all()
        elif args.step == "combined":
            run_combined()
        return

    # Full pipeline: pull → backtest → train-model → score
    pull_data()
    run_backtest()
    train_scoring_model()
    score_all()


if __name__ == "__main__":
    main()
