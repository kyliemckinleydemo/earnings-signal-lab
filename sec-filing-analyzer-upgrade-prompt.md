# Claude Code Session: Upgrade sec-filing-analyzer Prediction Model

## Context

The sec-filing-analyzer at `kyliemckinleydemo/sec-filing-analyzer` is a Next.js 14 + Prisma + PostgreSQL app that analyzes SEC filings and predicts stock returns. It has **three prediction models** across two API endpoints, all broken:

**Path 1 — Analyze endpoint** (user-triggered, primary path):
- `GET /api/analyze/[accession]` → `generateMLPrediction()` in `lib/ml-prediction.ts` → `scripts/predict_single_filing.py`
- Model: **RandomForestRegressor** (200 trees, max_depth=10) that **retrains from `data/ml_dataset.csv` on every single API call**
- `extractMLFeatures()` in `lib/ml-prediction.ts` hardcodes `riskScore = 5` and `sentimentScore = 0` — only `concernLevel` from Claude AI is actually variable
- Confidence is fake (starts at 0.65, bumps for upgrades/coverage/prediction magnitude), not from the model
- Predicts raw 7-day return

**Path 2a — Predict endpoint, baseline branch** (when EPS data available):
- `GET /api/predict/[accession]` → `predictionEngine.predictBaseline(actualEPS, estimatedEPS)` (TypeScript method in `lib/predictions.ts`, NOT the Python `scripts/predict_baseline.py` which is a separate standalone CLI tool)
- Model: **Logistic Regression** on 6 EPS-surprise features
- Predicts every stock as positive. Confusion matrix: TN=0, FP=344, FN=0, TP=713. The "67.5% accuracy" is just the base rate. Zero discriminative power.

**Path 2b — Predict endpoint, rules fallback** (when no EPS data):
- Same endpoint falls back to `predictionEngine.predict(features)` in `lib/predictions.ts`
- Model: Hand-tuned **~15-factor rule engine** with hardcoded weights and sophisticated interaction effects (concern-adjusted sentiment inversion, P/E multipliers, market cap multipliers, market regime adjustments, macro indicators from `marketMomentumClient` and `macroIndicatorsClient`)
- Unlike Path 1, this path reads ACTUAL `sentimentScore` and `concernLevel` from the DB — the hardcoding problem is only in `ml-prediction.ts`
- The weights are arbitrary (never validated against out-of-sample data)

**Critical: The predict endpoint caches predictions.** If `filing.predicted7dReturn` is already set, it skips prediction entirely and returns the cached value + accuracy check. This means migrating to the new model requires clearing or ignoring existing `predicted7dReturn` values.

**Both should be replaced.** We ran a systematic model zoo (OLS, Ridge, Lasso, ElasticNet, Stepwise Forward Selection, Polynomial, Random Forest, Gradient Boosting, Mutual Information) on the `data/ml_dataset_with_concern.csv` dataset (352 filings, 29 features) using 5-fold TimeSeriesSplit cross-validation. Key findings:

- **RandomForest (their current live model) performed poorly**: CV R² = -0.067 on 30D alpha. Negative R² means worse than predicting the mean. With max_depth=10 and n=352, it overfits massively.
- **Stepwise+Ridge was the best**: CV R² = 0.043 on 30-day market-relative alpha, selecting 8 features from 29.
- **Backtested directional accuracy**: 56.3% overall, **62.5% on high-confidence signals**, with a **+7.64pp LONG-SHORT spread**
- **SHORT signals are the strongest**: 62.7% accuracy identifying relative losers
- **Targeting alpha (stock return minus S&P 500) instead of raw returns** is critical — it removes market noise

The goal: replace all three prediction models with a single TypeScript-native model (no Python bridge), target 30-day alpha instead of 7-day raw returns, fix the database pipeline gaps, and update the documentation.

**WARNING — Feature Scale Mismatch:** The existing `ml-prediction.ts` computes `priceToHigh` and `priceToLow` as **percentages** (`(price/high - 1) * 100`, e.g., -12.0). But the model was trained on **ratios** from the CSV (`price/high`, e.g., 0.88). The `extractAlphaFeatures()` function in Task 1 uses the correct ratio scale. Do NOT copy the computation from `ml-prediction.ts` — it will completely break the model.

---

## Task 1: Create the New Model — `lib/alpha-model.ts`

This replaces BOTH `lib/ml-prediction.ts` (RandomForest) and `lib/baseline-features.ts` (Logistic Regression). It's pure TypeScript — no Python bridge, no scikit-learn, no retraining on every call.

The model uses 8 features selected by forward stepwise selection from 29 candidates, with Ridge regression weights.

```typescript
/**
 * Alpha Prediction Model — Stepwise+Ridge (8 features)
 *
 * Predicts 30-day market-relative alpha (stock return minus S&P 500 return).
 * Trained on 340 SEC filings with 5-fold TimeSeriesSplit cross-validation.
 * CV R² = 0.043 ± 0.056
 *
 * Replaces:
 *   - lib/ml-prediction.ts (RandomForest via predict_single_filing.py)
 *   - lib/baseline-features.ts (Logistic Regression via predict_baseline.py)
 *   - The rule-based engine in lib/predictions.ts
 *
 * Why this model wins over the existing RandomForest:
 *   - RF had CV R² = -0.067 (worse than predicting the mean)
 *   - RF overfits with max_depth=10 on n=352 samples
 *   - RF retrained from CSV on every call (no stability)
 *   - This model is a fixed formula — deterministic, fast, no Python needed
 *
 * Backtested directional accuracy:
 *   All signals: 56.3% (80/142)
 *   High confidence: 62.5% (30/48)
 *   LONG-SHORT spread: +3.73pp (high-conf: +7.64pp)
 */

// Training set statistics (DO NOT MODIFY — these are the exact values
// from the 340-sample training set used to fit the model)
const FEATURE_STATS = {
  priceToLow:             { mean: 1.3978, std: 0.4174 },
  majorDowngrades:        { mean: 0.1029, std: 0.3812 },
  analystUpsidePotential: { mean: 13.518, std: 10.887 },
  priceToHigh:            { mean: 0.8588, std: 0.0912 },
  concernLevel:           { mean: 5.345,  std: 1.578  },
  marketCap:              { mean: 682_892_847_207, std: 1_043_282_559_376 },
  sentimentScore:         { mean: 0.0236, std: 0.1206 },
  upgradesLast30d:        { mean: 0.1941, std: 0.4576 },
} as const;

// Stepwise+Ridge model weights (standardized feature space)
// These are the exact coefficients from Ridge regression (alpha=100)
// after forward stepwise selection chose these 8 features from 29 candidates
const WEIGHTS = {
  priceToLow:             +1.3191,  // Momentum: far above 52W low → continues outperforming
  majorDowngrades:        +0.7783,  // Contrarian: major bank downgrades → market overreaction → recovery
  analystUpsidePotential: -0.4069,  // Value trap: high upside target → stock has been underperforming, continues to lag
  priceToHigh:            +0.3872,  // Momentum: near 52W high → strength continues
  concernLevel:           -0.1165,  // AI signal: higher Claude-assessed concern → lower alpha
  marketCap:              +0.0822,  // Size effect: larger companies → more predictable positive alpha
  sentimentScore:         +0.0413,  // AI signal: positive filing sentiment → positive alpha (weak)
  upgradesLast30d:        -0.0112,  // Negligible after other analyst features captured
} as const;

// Score distribution percentiles from training data
// Used to classify signals as LONG/SHORT/NEUTRAL and assign confidence
const SCORE_PERCENTILES = {
  p10: -1.0345,
  p25: -0.8114,
  p50: -0.4263,
  p75: +0.0438,
  p90: +1.6600,
  mean: -0.1343,
  std: 1.1164,
} as const;

export interface AlphaFeatures {
  priceToLow: number;             // currentPrice / fiftyTwoWeekLow
  majorDowngrades: number;        // count of downgrades from top-tier banks in last 30 days
  analystUpsidePotential: number; // ((analystTargetPrice / currentPrice) - 1) * 100
  priceToHigh: number;            // currentPrice / fiftyTwoWeekHigh
  concernLevel: number;           // Claude AI concern level (0-10)
  marketCap: number;              // market capitalization in dollars
  sentimentScore: number;         // Claude AI sentiment (-1 to +1)
  upgradesLast30d: number;        // count of analyst upgrades in last 30 days
}

export interface AlphaPrediction {
  rawScore: number;               // continuous score (higher = more bullish)
  expectedAlpha: number;          // expected 30-day alpha in percentage points
  signal: 'LONG' | 'SHORT' | 'NEUTRAL';
  confidence: 'high' | 'medium' | 'low';
  percentile: string;             // where this score falls in training distribution
  featureContributions: Record<string, number>;  // per-feature contribution to score
  predicted30dReturn: number;     // alpha + market baseline (~0.8%/mo)
}

/**
 * Score a filing using the Stepwise+Ridge alpha model.
 */
export function predictAlpha(features: AlphaFeatures): AlphaPrediction {
  // Standardize features: z = (x - mean) / std
  const contributions: Record<string, number> = {};
  let score = 0;

  for (const [name, weight] of Object.entries(WEIGHTS)) {
    const stat = FEATURE_STATS[name as keyof typeof FEATURE_STATS];
    const raw = features[name as keyof AlphaFeatures] as number;
    const zVal = (raw - stat.mean) / stat.std;
    const contribution = weight * zVal;
    contributions[name] = Math.round(contribution * 10000) / 10000;
    score += contribution;
  }

  // Classify signal based on training score percentiles
  let signal: 'LONG' | 'SHORT' | 'NEUTRAL';
  let confidence: 'high' | 'medium' | 'low';
  let percentile: string;

  if (score > SCORE_PERCENTILES.p90) {
    signal = 'LONG'; confidence = 'high'; percentile = '>90th';
  } else if (score > SCORE_PERCENTILES.p75) {
    signal = 'LONG'; confidence = 'medium'; percentile = '75th-90th';
  } else if (score < SCORE_PERCENTILES.p10) {
    signal = 'SHORT'; confidence = 'high'; percentile = '<10th';
  } else if (score < SCORE_PERCENTILES.p25) {
    signal = 'SHORT'; confidence = 'medium'; percentile = '10th-25th';
  } else {
    signal = 'NEUTRAL'; confidence = 'low'; percentile = '25th-75th';
  }

  const expectedAlpha = score;
  const marketBaseline30d = 0.8;  // long-run monthly S&P 500 return
  const predicted30dReturn = expectedAlpha + marketBaseline30d;

  return {
    rawScore: Math.round(score * 10000) / 10000,
    expectedAlpha: Math.round(expectedAlpha * 100) / 100,
    signal,
    confidence,
    percentile,
    featureContributions: contributions,
    predicted30dReturn: Math.round(predicted30dReturn * 100) / 100,
  };
}

/**
 * Extract AlphaFeatures from database records.
 *
 * This replaces the feature extraction in lib/ml-prediction.ts (extractMLFeatures)
 * which built ~33 features including hardcoded riskScore=5 and sentimentScore=0.
 * We only need 8 features, and we use the ACTUAL sentimentScore from Claude.
 */
/**
 * CRITICAL: priceToHigh and priceToLow must be RATIOS, not percentages.
 *
 * The existing ml-prediction.ts computes these as PERCENTAGES:
 *   priceToHigh = (currentPrice / fiftyTwoWeekHigh - 1) * 100  // e.g., -12.0
 *   priceToLow  = (currentPrice / fiftyTwoWeekLow - 1) * 100   // e.g., +27.0
 *
 * But the model was trained on RATIOS from the CSV export:
 *   priceToHigh = currentPrice / fiftyTwoWeekHigh               // e.g., 0.88
 *   priceToLow  = currentPrice / fiftyTwoWeekLow                // e.g., 1.27
 *
 * Using the wrong scale will completely break the model. The FEATURE_STATS
 * confirm the ratio scale: priceToHigh mean=0.8588, priceToLow mean=1.3978.
 */
export function extractAlphaFeatures(
  company: {
    currentPrice: number;
    fiftyTwoWeekHigh: number;
    fiftyTwoWeekLow: number;
    marketCap: number;
    analystTargetPrice?: number | null;
  },
  filing: {
    concernLevel?: number | null;
    sentimentScore?: number | null;
  },
  analystActivity: {
    upgradesLast30d: number;
    majorDowngradesLast30d: number;
  },
): AlphaFeatures {
  // RATIO (not percentage!) — see warning above
  const priceToLow = company.fiftyTwoWeekLow > 0
    ? company.currentPrice / company.fiftyTwoWeekLow
    : FEATURE_STATS.priceToLow.mean;

  // RATIO (not percentage!) — see warning above
  const priceToHigh = company.fiftyTwoWeekHigh > 0
    ? company.currentPrice / company.fiftyTwoWeekHigh
    : FEATURE_STATS.priceToHigh.mean;

  const analystUpsidePotential = company.analystTargetPrice && company.currentPrice > 0
    ? ((company.analystTargetPrice / company.currentPrice) - 1) * 100
    : FEATURE_STATS.analystUpsidePotential.mean;

  return {
    priceToLow,
    priceToHigh,
    majorDowngrades: analystActivity.majorDowngradesLast30d,
    analystUpsidePotential,
    concernLevel: filing.concernLevel ?? FEATURE_STATS.concernLevel.mean,
    marketCap: company.marketCap,
    sentimentScore: filing.sentimentScore ?? FEATURE_STATS.sentimentScore.mean,
    upgradesLast30d: analystActivity.upgradesLast30d,
  };
}
```

### Why this replaces the RandomForest

The current live model (`predict_single_filing.py`) does this on every API call:
```python
df = pd.read_csv('data/ml_dataset.csv')  # Load 352 rows
# ... build X, y ...
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)  # Retrain every time
prediction = model.predict(X_new)[0]
```

Problems:
1. **Retraining on every call** — non-deterministic if the CSV changes, wasteful, ~2-3 seconds per prediction
2. **max_depth=10 on n=352** — massive overfitting. Our CV R² for RF was -0.067 (worse than mean)
3. **Predicts raw 7-day return** — dominated by market direction, not filing-specific signal
4. **33 features, many hardcoded** — riskScore always 5, sentimentScore always 0
5. **No confidence calibration** — confidence is rule-based (starts at 0.65, bumps up), not derived from model

The replacement:
1. **Pure TypeScript formula** — instant, deterministic, no Python subprocess
2. **8 features, all meaningful** — selected by forward stepwise from 29 candidates
3. **Predicts 30-day alpha** — removes market noise, isolates filing-specific signal
4. **Confidence from training distribution** — percentile-based, calibrated to backtest accuracy

---

## Task 2: Update the Analyze Endpoint (User-Triggered Predictions)

This is the **primary change**. The analyze endpoint at `app/api/analyze/[accession]/route.ts` is the path users actually trigger.

### Current flow (lines ~935-951 of the route):

1. Claude AI analyzes the filing (risk, sentiment, financials, concern assessment) — 4 parallel calls
2. XBRL + Yahoo Finance data enriches the analysis
3. Sentiment adjusted based on earnings surprises
4. `generateMLPrediction()` called → spawns Python → RandomForest prediction
5. Results stored in DB

### New flow:

Replace step 4. After Claude analysis and Yahoo Finance enrichment:

1. Import `predictAlpha`, `extractAlphaFeatures` from `lib/alpha-model.ts`
2. The Company record is already available (Yahoo Finance enrichment provides `currentPrice`, `fiftyTwoWeekHigh`, `fiftyTwoWeekLow`, `marketCap`)
3. The Filing record now has `concernLevel` and `sentimentScore` from Claude's analysis (step 1)
4. Query `AnalystActivity` for the 30 days before `filing.filingDate`.

   **Note:** The predict endpoint (`app/api/predict/[accession]/route.ts`) already has this exact query pattern — look for the `analystActivities` block around line 170+. It queries `AnalystActivity`, filters by companyId and date range, and classifies by major firm. Adapt that existing code rather than writing from scratch.

   Use their existing major firms list for consistency:
   ```typescript
   const MAJOR_FIRMS = ['Goldman Sachs', 'Morgan Stanley', 'JP Morgan', 'Bank of America',
                        'Citi', 'Wells Fargo', 'Barclays', 'UBS'];
   ```

   Query for upgrade count and major-firm downgrade count:

```typescript
const thirtyDaysAgo = new Date(filing.filingDate);
thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

const analystActivities = await prisma.analystActivity.findMany({
  where: {
    companyId: company.id,
    activityDate: { gte: thirtyDaysAgo, lt: filing.filingDate },
  },
  select: { actionType: true, firm: true },
});

const upgradeCount = analystActivities.filter(a => a.actionType === 'upgrade').length;
const majorDowngradeCount = analystActivities.filter(a =>
  a.actionType === 'downgrade' && MAJOR_FIRMS.some(f => a.firm.includes(f))
).length;
```

5. Build features and predict:

```typescript
const alphaFeatures = extractAlphaFeatures(
  company,
  { concernLevel: analysis.concernLevel, sentimentScore: analysis.sentimentScore },
  { upgradesLast30d: upgradeCount, majorDowngradesLast30d: majorDowngradeCount },
);

const prediction = predictAlpha(alphaFeatures);
```

6. Store results:

```typescript
await prisma.filing.update({
  where: { id: filing.id },
  data: {
    predicted30dReturn: prediction.predicted30dReturn,
    predicted30dAlpha: prediction.expectedAlpha,
    predictionConfidence: prediction.confidence === 'high' ? 0.85
      : prediction.confidence === 'medium' ? 0.65 : 0.5,
    // Keep predicted7dReturn for backward compat — rough estimate
    predicted7dReturn: prediction.predicted30dReturn * (7 / 30),
  },
});

await prisma.prediction.create({
  data: {
    filingId: filing.id,
    predictedReturn: prediction.predicted30dReturn,
    confidence: prediction.confidence === 'high' ? 0.85
      : prediction.confidence === 'medium' ? 0.65 : 0.5,
    features: prediction.featureContributions as any,
    modelVersion: 'alpha-v1.0',
  },
});
```

7. Return to frontend — include the new fields in the response:

```typescript
return {
  // ... existing fields ...
  prediction: {
    signal: prediction.signal,           // 'LONG' | 'SHORT' | 'NEUTRAL'
    confidence: prediction.confidence,   // 'high' | 'medium' | 'low'
    expectedAlpha: prediction.expectedAlpha,
    predicted30dReturn: prediction.predicted30dReturn,
    featureContributions: prediction.featureContributions,
    percentile: prediction.percentile,
    modelVersion: 'alpha-v1.0',
  },
};
```

### Remove the Python bridge

- Delete the call to `generateMLPrediction()` in the analyze route
- The function `generateMLPrediction()` in `lib/ml-prediction.ts` can be deprecated (keep the file but add a deprecation notice at the top)
- `scripts/predict_single_filing.py` is no longer called — archive it

### Important: Keep the Claude AI analysis

The Claude analysis steps (risk assessment, sentiment analysis, financial analysis, concern assessment) must stay — they produce `concernLevel` and `sentimentScore` which are inputs to our model. The change is only in what happens AFTER Claude analyzes the filing.

---

## Task 3: Update the Predict Endpoint (Batch/Cron Path)

The predict endpoint at `app/api/predict/[accession]/route.ts` currently has three layers:
- **Cache check**: If `filing.predicted7dReturn` is already set → returns cached prediction immediately (never re-predicts)
- **Path A**: `predictionEngine.predictBaseline(actualEPS, estimatedEPS)` — a TypeScript method in `lib/predictions.ts` (NOT the Python `scripts/predict_baseline.py`)
- **Path B**: `predictionEngine.predict(features)` — the ~15-factor hand-tuned rules engine in `lib/predictions.ts`

The endpoint also already queries AnalystActivity, fetches market momentum, and has paper trading integration — all of which can be reused or adapted.

Replace the prediction logic with the alpha model:

1. Import `predictAlpha`, `extractAlphaFeatures` from `lib/alpha-model.ts`
2. **Update the cache check**: The route currently skips prediction if `filing.predicted7dReturn !== null` (around line 60). Change this to check `filing.predicted30dAlpha !== null` instead, so existing 7-day predictions don't prevent the new model from running.
3. **Reuse the existing AnalystActivity query**: The route already queries this table (around line 170). Adapt it to extract `upgradeCount` and `majorDowngradeCount` as needed by `extractAlphaFeatures()`.
4. Same `extractAlphaFeatures()` + `predictAlpha()` pattern as Task 2
5. Remove the calls to `predictBaseline()` and `predictionEngine.predict()`
6. The existing market momentum (`marketMomentumClient`) and macro indicator (`macroIndicatorsClient`) fetches can be kept for display purposes but are NOT used by the alpha model
7. Keep the rule-based engine in `lib/predictions.ts` as a last-resort fallback ONLY if both `fiftyTwoWeekLow` AND `concernLevel` are unavailable (this should be extremely rare since Yahoo Finance provides price data for all S&P 500 stocks)

### Migration: Re-predict existing filings

After deploying the new model, existing filings will have `predicted7dReturn` set but `predicted30dAlpha` null. To generate alpha predictions for all historical filings:

```sql
-- Find filings that need new predictions
SELECT COUNT(*) FROM "Filing" WHERE "predicted7dReturn" IS NOT NULL AND "predicted30dAlpha" IS NULL;
```

Write a one-time migration script that iterates over these filings and runs `predictAlpha()` on each. This will populate the new column for the dashboard and accuracy tracking.

---

## Task 4: Ensure the Database Pipeline Populates Required Features

The model needs 8 features. Most are already in the database. Verify each:

| Feature | Source | DB Location | Status | Action |
|---------|--------|-------------|--------|--------|
| `priceToLow` | Computed | `Company.currentPrice / Company.fiftyTwoWeekLow` | Available | None needed |
| `priceToHigh` | Computed | `Company.currentPrice / Company.fiftyTwoWeekHigh` | Available | None needed |
| `concernLevel` | Claude AI | `Filing.concernLevel` | Available (from analyze step) | None needed |
| `sentimentScore` | Claude AI | `Filing.sentimentScore` | **Hardcoded to 0 in `ml-prediction.ts` (analyze path only)**. The predict endpoint correctly reads `filing.sentimentScore \|\| 0` from DB. | The new `extractAlphaFeatures()` reads the actual value. Just ensure it's populated after Claude analysis. |
| `marketCap` | Yahoo Finance | `Company.marketCap` | Available | None needed |
| `upgradesLast30d` | Yahoo Finance | `AnalystActivity` table | **VERIFY COVERAGE** | See below |
| `majorDowngrades` | Yahoo Finance | `AnalystActivity` table (filtered by firm) | **VERIFY COVERAGE** | See below |
| `analystUpsidePotential` | Computed | `((Company.analystTargetPrice / Company.currentPrice) - 1) * 100` | **VERIFY** `analystTargetPrice` population | See below |

### Critical Check 1: `AnalystActivity` table coverage

The `AnalystActivity` table exists in the Prisma schema with columns: `companyId, activityDate, firm, actionType, previousRating, newRating, previousTarget, newTarget`. But it may not be consistently populated.

Run these queries:
```sql
SELECT COUNT(*) FROM "AnalystActivity";
SELECT DISTINCT "firm" FROM "AnalystActivity" LIMIT 20;
SELECT "actionType", COUNT(*) FROM "AnalystActivity" GROUP BY "actionType";
```

If the table is sparse or empty:
- Check which cron job or script populates it (look in `app/api/cron/` and `scripts/`)
- The Yahoo Finance client (`lib/yahoo-finance-client.ts`) likely has a function to fetch analyst upgrades/downgrades — ensure it's being called by a daily cron
- If no cron exists for this, create one at `app/api/cron/analyst-activity/route.ts` that:
  1. Iterates over all tracked companies
  2. Fetches recent analyst activity from `yahoo-finance2` (`recommendationTrend` or `upgradeDowngradeHistory` modules)
  3. Upserts into the `AnalystActivity` table
  4. Runs daily

If the table IS empty but you can't immediately populate it, the model gracefully handles missing data — `majorDowngrades=0` and `upgradesLast30d=0` are valid values (they'll standardize to near their training means and contribute minimally to the score). The model still works with just the other 6 features.

### Critical Check 2: `analystTargetPrice` in Company table

```sql
SELECT COUNT(*) FROM "Company" WHERE "analystTargetPrice" IS NOT NULL AND "analystTargetPrice" > 0;
```

If this returns 0 or very low:
- The Yahoo Finance client needs to fetch `targetMeanPrice` from the `financialData` module of `yahoo-finance2`'s `quoteSummary`
- If not available, the `extractAlphaFeatures` function falls back to the training mean (13.5% upside), which is acceptable

---

## Task 5: Add `predicted30dAlpha` to the Filing Table

The model's primary output is 30-day alpha, not raw return. Add a column:

### Prisma schema change (`prisma/schema.prisma`):

Add to the `Filing` model:
```prisma
predicted30dAlpha     Float?
```

Run: `npx prisma migrate dev --name add-predicted-30d-alpha`

---

## Task 6: Update Accuracy Tracking

The current `lib/accuracy-tracker.ts` compares `predicted7dReturn` vs `actual7dReturn` after 10+ calendar days. Update it for 30-day alpha:

1. **Timing**: Check accuracy after **35+ calendar days** instead of 10 (need 30 trading days + buffer)
2. **Alpha calculation**: When fetching actual returns, also fetch the S&P 500 return for the same period:
   ```typescript
   const actual30dAlpha = actual30dReturn - spxReturn30d;
   ```
   The `MacroIndicators` table has `spxReturn30d` — use it, or compute from `StockPrice` table using SPY
3. **Directional accuracy**: Track whether the signal (LONG/SHORT) matched the actual alpha direction:
   ```typescript
   const signalCorrect =
     (signal === 'LONG' && actual30dAlpha > 0) ||
     (signal === 'SHORT' && actual30dAlpha < 0);
   ```
4. **New stats method**:
   ```typescript
   interface AlphaModelStats {
     totalPredictions: number;
     predictionsWithActuals: number;
     directionalAccuracy: number;        // % of LONG/SHORT signals correct
     highConfDirectionalAccuracy: number; // % of high-conf signals correct
     longShortSpread: number;            // avg LONG alpha - avg SHORT alpha (pp)
     avgLongAlpha: number;
     avgShortAlpha: number;
   }
   ```

---

## Task 7: Update Paper Trading

The paper trading system in `lib/paper-trading.ts` currently uses:
```
BUY:  confidence >= 0.60 AND predicted return >= +2.0%
SELL: confidence >= 0.60 AND predicted return <= -2.0%
HOLD: everything else
```

Replace with signal-based logic:

1. **Use signal directly**: `LONG` → BUY, `SHORT` → SELL, `NEUTRAL` → HOLD
2. **Position sizing by confidence**:
   - High confidence → full position size
   - Medium confidence → half position size
   - Low confidence (NEUTRAL) → no trade
3. **Hold period**: Change from 7 days to **30 days** (the model targets 30-day alpha)
4. **Stop-loss / take-profit** (adjust for 30-day horizon):
   - High-confidence LONG: target +8%, stop -5%
   - Medium-confidence LONG: target +5%, stop -3%
   - High-confidence SHORT: target +8% gain (stock drops 8%), stop -5%
   - Medium-confidence SHORT: target +5% gain, stop -3%

---

## Task 8: Fix Documentation and Accuracy Claims

### README.md

The current claims are invalid:
- "60.26% directional accuracy" — came from `backtest-v3-optimized.py` which **uses the actual return to simulate input features** (data leakage). The sentiment and risk scores are generated FROM the target variable.
- "67.5% accuracy" — the Logistic Regression baseline predicts every single stock as positive. The 67.5% is the base rate of positive returns in a bull market.

Replace with:

```markdown
## Model Performance

**Alpha Model v1.0** — Stepwise+Ridge regression predicting 30-day market-relative alpha

| Metric | Value |
|--------|-------|
| Cross-validated R² | 0.043 (5-fold TimeSeriesSplit) |
| Directional accuracy (all signals) | 56.3% |
| **Directional accuracy (high confidence)** | **62.5%** |
| LONG-SHORT spread (all) | +3.73 percentage points |
| **LONG-SHORT spread (high confidence)** | **+7.64 percentage points** |
| Training set | 340 SEC filings (107 S&P 500 companies, Oct 2023 – Oct 2025) |
| Features | 8 (selected from 29 candidates via forward stepwise selection) |

The model is strongest at **identifying relative losers** — SHORT signals have 62.7% directional accuracy.

### Key Features (by importance)

1. **Price to 52W Low** (+1.32): Stocks far above their 52-week low continue outperforming (momentum)
2. **Major Downgrades** (+0.78): Recent downgrades from top-tier banks are a contrarian buy signal — the market overreacts
3. **Analyst Upside Potential** (-0.41): High analyst upside targets signal value traps that continue underperforming
4. **Price to 52W High** (+0.39): Stocks near all-time highs continue showing strength
5. **Concern Level** (-0.12): Claude AI concern assessment — higher concern correlates with lower alpha
6. **Market Cap** (+0.08): Larger companies have more predictable positive alpha post-filing
7. **Sentiment Score** (+0.04): Claude AI sentiment — weak but positive contributor
8. **Upgrades Last 30D** (-0.01): Negligible after other analyst features are included
```

### MODEL-DEVELOPMENT-JOURNEY.md

Add a section at the end:

```markdown
## v4.0: Systematic Model Zoo (Earnings Signal Lab collaboration)

### Approach
Instead of hand-tuning weights or training a single model, we ran a systematic search
across 8 model types (OLS, Ridge, Lasso, ElasticNet, Stepwise Forward Selection,
Polynomial, Random Forest, Gradient Boosting) plus Mutual Information analysis.
All models used 5-fold TimeSeriesSplit cross-validation to prevent lookahead bias.

### Key Discovery: Target Alpha, Not Returns
Predicting raw 7-day returns is dominated by market direction (bull vs bear).
Switching to **30-day market-relative alpha** (stock return minus S&P 500 return)
dramatically improved signal quality. The market noise is removed, leaving only
the stock-specific signal from the filing.

### Key Discovery: RandomForest Was Overfitting
The production RandomForest (max_depth=10, n=352 samples) had CV R² = -0.067.
Negative R² means it predicted worse than just guessing the average return.
The model memorized the training data but couldn't generalize. Simpler linear
models (Ridge, Lasso) significantly outperformed.

### Key Discovery: The Baseline Model Had No Signal
The v3.0 Logistic Regression on EPS surprise features predicted EVERY stock as
positive. Its confusion matrix: TN=0, FP=344, FN=0, TP=713. The "67.5% accuracy"
equaled the base rate of positive returns. The model had zero discriminative power.

### Key Discovery: Contrarian Analyst Signals
Major bank downgrades are the **second strongest bullish signal** (+0.78 weight).
This directly contradicts the v2.0 rule-based model which penalized downgrades.
The market systematically overreacts to downgrades from Goldman Sachs, JPMorgan,
and similar firms, creating a 30-day recovery opportunity.

### Key Discovery: AI Features Are Weak but Real
The Claude-generated features (riskScore, sentimentScore, concernLevel) collectively
contribute ~0.16 weight out of ~3.3 total. Market structure features (price ratios,
analyst activity, market cap) dominate. However, concernLevel (-0.12) does add signal
that survives stepwise selection — it's the 5th most important feature. The previous
model hardcoded riskScore=5 and sentimentScore=0, throwing away the AI signal entirely.

### Key Discovery: riskScore and sentimentScore Were Wasted (Analyze Path)
The ml-prediction.ts feature extraction (used by the analyze endpoint) hardcoded
riskScore=5 and sentimentScore=0 for every filing. These are the training set means,
so they contributed zero signal. Note: the predict endpoint correctly read
sentimentScore from the DB, but the primary user-facing analyze path did not.
The new model uses the ACTUAL sentimentScore from Claude's analysis and replaces
riskScore with concernLevel (which was the only Claude feature that varied per filing).

### Key Discovery: The Predict Endpoint's Rules Engine Was Sophisticated But Unvalidated
The v2.0 rules engine in lib/predictions.ts had ~15 interacting factors including
concern-adjusted sentiment (inverts positive sentiment when concern > 7), P/E
multipliers, market cap multipliers, market regime adjustments, and macro indicators
(dollar strength, GDP proxy). These are reasonable heuristics, but the weights were
hand-tuned and never validated against out-of-sample data. The systematic model zoo
found that most of these factors don't survive cross-validation — only 8 features
out of 29 candidates contribute real signal.
```

### `lib/confidence-scores.ts`

The hardcoded per-ticker confidence scores (NVDA: 46.7%, HD: 80%, etc.) are based on ~15 samples each and are not statistically meaningful. Deprecate this system — the new model has its own confidence framework based on score percentiles calibrated to backtest accuracy.

---

## Task 9: Update UI Components

### Prediction display on filing analysis page

The UI currently shows predictions as simple percentage returns. Update to show:
- **Signal badge**: LONG (green), SHORT (red), NEUTRAL (gray)
- **Confidence level**: high/medium/low with visual indicator
- **Expected 30-day alpha**: "Expected to outperform S&P 500 by X.X%" or "Expected to underperform..."
- **Top feature drivers**: Show the top 2-3 feature contributions from `featureContributions`
- **Model version**: Small text showing "Alpha Model v1.0"

### Latest filings page

If there's a table/list view of recent filings, add a signal column with color-coded badges. High-confidence LONGs and SHORTs should be visually prominent — those are the most reliable predictions.

### Trading signal display

Replace the current BUY/SELL/HOLD with the new signal format:
- `LONG (high confidence)` → "Strong Buy — model expects significant outperformance"
- `LONG (medium confidence)` → "Buy — model expects moderate outperformance"
- `SHORT (high confidence)` → "Strong Sell — model expects significant underperformance"
- `SHORT (medium confidence)` → "Sell — model expects moderate underperformance"
- `NEUTRAL` → "Hold — no clear signal"

---

## Task 10: Clean Up Deprecated Code

After the new model is working:

| File | Action |
|------|--------|
| `scripts/predict_single_filing.py` | Archive — no longer called |
| `scripts/predict_baseline.py` | Archive — no longer called |
| `models/baseline_model.pkl` | Archive — no longer loaded |
| `models/baseline_scaler.pkl` | Archive — no longer loaded |
| `models/baseline_features.json` | Archive |
| `models/baseline_results.json` | Archive (but save for historical reference) |
| `lib/ml-prediction.ts` | Add deprecation notice; keep temporarily for reference |
| `lib/baseline-features.ts` | Add deprecation notice |
| `lib/predictions.ts` | Keep as last-resort fallback, but add deprecation notice and note that it's rarely used |
| `lib/confidence-scores.ts` | Deprecate the hardcoded per-ticker scores |

Do NOT delete `data/ml_dataset_with_concern.csv` — this is the training data the model was built on.

---

## Summary of Changes (Priority Order)

| # | File | Action | Priority |
|---|------|--------|----------|
| 1 | `lib/alpha-model.ts` | **Create** — new model with exact weights | P0 |
| 2 | `app/api/analyze/[accession]/route.ts` | **Modify** — replace `generateMLPrediction()` with `predictAlpha()` | P0 |
| 3 | `app/api/predict/[accession]/route.ts` | **Modify** — replace baseline + rule-based with `predictAlpha()` | P0 |
| 4 | `prisma/schema.prisma` | **Modify** — add `predicted30dAlpha` column | P0 |
| 5 | `lib/accuracy-tracker.ts` | **Modify** — track 30-day alpha accuracy + directional accuracy | P1 |
| 6 | `lib/paper-trading.ts` | **Modify** — use signal/confidence, 30-day hold period | P1 |
| 7 | `README.md` | **Modify** — fix accuracy claims with real numbers | P1 |
| 8 | `MODEL-DEVELOPMENT-JOURNEY.md` | **Modify** — add v4.0 section | P2 |
| 9 | UI components | **Modify** — show signal/confidence/alpha/drivers | P2 |
| 10 | Deprecated scripts/models | **Archive** — move or add deprecation notices | P2 |

## Verification Steps

After making the changes:
1. `npx prisma migrate dev` — apply schema changes
2. Pick 3-5 recent filings that already have Claude analysis (`concernLevel` and `sentimentScore` populated). Run the new model against them and verify the scores match the formula manually.
3. Check that `AnalystActivity` queries return reasonable data. If the table is empty, verify the model still produces valid predictions (it will — analyst features just won't contribute).
4. Trigger a full analysis on a new filing through the UI — verify the new signal/confidence displays correctly.
5. Verify paper trading doesn't break with the new signal format.
6. Run the accuracy tracker on historical data where `actual30dReturn` exists to get real-world performance numbers.
