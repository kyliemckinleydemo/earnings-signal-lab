# Claude Code Session: Upgrade sec-filing-analyzer Prediction Model

## Context

The sec-filing-analyzer at `kyliemckinleydemo/sec-filing-analyzer` is a Next.js 14 + Prisma + PostgreSQL app that analyzes SEC filings and predicts 7-day and 30-day stock returns. Its current production model (Logistic Regression on 6 EPS-surprise features) has zero discriminative power — it predicts every stock as positive and achieves "67.5% accuracy" which is just the base rate of positive returns in a bull market. The confusion matrix shows 0 true negatives and 0 false negatives.

We ran a systematic model zoo (OLS, Ridge, Lasso, ElasticNet, Stepwise Forward Selection, Polynomial, Random Forest, Gradient Boosting, Mutual Information) on the `data/ml_dataset_with_concern.csv` dataset (352 filings, 29 features) using TimeSeriesSplit cross-validation. The best model is **Stepwise+Ridge on 30-day market-relative alpha** (CV R² = 0.043). When backtested, it achieves:

- **56.3% directional accuracy** on 30-day alpha (vs 50% for the current model's alpha predictions)
- **62.5% accuracy on high-confidence signals** with a **+7.64 percentage point LONG-SHORT spread**
- SHORT signals are especially strong: 62.7% accuracy identifying losers

The goal is to replace the broken production model with this proven one, update the database pipeline to populate the required features, and fix the inaccurate accuracy claims in the documentation.

---

## Task 1: Replace the Production Prediction Model

### Current flow (to be replaced)

The predict endpoint at `app/api/predict/[accession]/route.ts` currently:
1. Tries Path A (baseline): extracts 6 EPS features via `lib/baseline-features.ts`, calls `python3 scripts/predict_baseline.py` which loads `models/baseline_model.pkl` (Logistic Regression)
2. Falls back to Path B (legacy): rule-based 12-factor model in `lib/predictions.ts`

### New flow

Replace both paths with a single TypeScript-native scoring function. No Python bridge needed — the model is a simple linear formula.

### Create `lib/alpha-model.ts`

This is the new production model. It uses 8 features, all of which already exist in the database schema (Company, TechnicalIndicators, AnalystActivity tables).

```typescript
/**
 * Alpha Prediction Model — Stepwise+Ridge (8 features)
 *
 * Predicts 30-day market-relative alpha (stock return minus S&P 500 return).
 * Trained on 340 SEC filings with 5-fold TimeSeriesSplit cross-validation.
 * CV R² = 0.043 ± 0.056
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
  predicted7dReturn?: number;     // rough 7D estimate (alpha * 7/30 + market baseline)
  predicted30dReturn?: number;    // rough 30D estimate (alpha + market baseline)
}

/**
 * Score a filing using the Stepwise+Ridge alpha model.
 *
 * @param features - The 8 input features (all required)
 * @returns AlphaPrediction with signal, confidence, and expected alpha
 */
export function predictAlpha(features: AlphaFeatures): AlphaPrediction {
  // Standardize features: z = (x - mean) / std
  const z: Record<string, number> = {};
  const contributions: Record<string, number> = {};
  let score = 0;

  for (const [name, weight] of Object.entries(WEIGHTS)) {
    const stat = FEATURE_STATS[name as keyof typeof FEATURE_STATS];
    const raw = features[name as keyof AlphaFeatures] as number;
    const zVal = (raw - stat.mean) / stat.std;
    z[name] = zVal;
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

  // Expected alpha ≈ raw score (the model is trained to predict alpha in pct points)
  const expectedAlpha = score;

  // Rough return estimates (alpha + assumed market baseline of ~0.8%/month)
  const marketBaseline30d = 0.8;  // long-run monthly S&P 500 return
  const predicted30dReturn = expectedAlpha + marketBaseline30d;
  const predicted7dReturn = predicted30dReturn * (7 / 30);

  return {
    rawScore: Math.round(score * 10000) / 10000,
    expectedAlpha: Math.round(expectedAlpha * 100) / 100,
    signal,
    confidence,
    percentile,
    featureContributions: contributions,
    predicted7dReturn: Math.round(predicted7dReturn * 100) / 100,
    predicted30dReturn: Math.round(predicted30dReturn * 100) / 100,
  };
}

/**
 * Extract AlphaFeatures from database records.
 * Call this when assembling features for a prediction.
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
  const priceToLow = company.fiftyTwoWeekLow > 0
    ? company.currentPrice / company.fiftyTwoWeekLow
    : FEATURE_STATS.priceToLow.mean;

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

---

## Task 2: Update the Prediction Endpoint

Modify `app/api/predict/[accession]/route.ts` to use the new model:

1. Import `predictAlpha`, `extractAlphaFeatures` from `lib/alpha-model.ts`
2. Fetch the required data:
   - `Company` record (already fetched) — provides `currentPrice`, `fiftyTwoWeekHigh`, `fiftyTwoWeekLow`, `marketCap`, `analystTargetPrice`
   - `Filing` record (already fetched) — provides `concernLevel`, `sentimentScore`
   - Query `AnalystActivity` for upgrades/downgrades in the 30 days before `filing.filingDate`:
     ```typescript
     const thirtyDaysAgo = new Date(filing.filingDate);
     thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

     const analystActivity = await prisma.analystActivity.groupBy({
       by: ['actionType'],
       where: {
         companyId: company.id,
         activityDate: { gte: thirtyDaysAgo, lte: filing.filingDate },
       },
       _count: true,
     });

     const MAJOR_FIRMS = ['Goldman Sachs', 'JPMorgan', 'Morgan Stanley', 'Bank of America',
                          'Citigroup', 'Wells Fargo', 'Barclays', 'Deutsche Bank', 'UBS',
                          'Credit Suisse', 'HSBC', 'BNP Paribas'];

     const majorDowngrades = await prisma.analystActivity.count({
       where: {
         companyId: company.id,
         activityDate: { gte: thirtyDaysAgo, lte: filing.filingDate },
         actionType: 'downgrade',
         firm: { in: MAJOR_FIRMS },
       },
     });

     const upgradesLast30d = analystActivity
       .filter(a => a.actionType === 'upgrade')
       .reduce((sum, a) => sum + a._count, 0);
     ```
3. Call `extractAlphaFeatures(company, filing, { upgradesLast30d, majorDowngradesLast30d: majorDowngrades })`
4. Call `predictAlpha(features)` to get the prediction
5. Store: update `filing.predicted7dReturn`, `filing.predicted30dReturn`, `filing.predictionConfidence`
6. Create `Prediction` record with `modelVersion: 'alpha-v1.0'`

**Remove or deprecate:**
- The Python bridge call to `predict_baseline.py`
- The `predictBaseline()` method in `lib/predictions.ts`
- The fallback to the rule-based engine (keep it as a last-resort fallback if `concernLevel` is null AND `fiftyTwoWeekLow` is null, but this should be rare)

---

## Task 3: Ensure the Database Pipeline Populates Required Features

The model needs 8 features. Check that each is reliably populated:

| Feature | Source Table | Column | Currently Populated? | Action Needed |
|---------|-------------|--------|---------------------|---------------|
| `priceToLow` | Computed | `currentPrice / fiftyTwoWeekLow` | Yes (Company table has both) | None |
| `priceToHigh` | Computed | `currentPrice / fiftyTwoWeekHigh` | Yes (Company table has both) | None |
| `majorDowngrades` | `AnalystActivity` | count where `actionType='downgrade'` and `firm` in major list | **Check coverage** | Ensure `scripts/fetch-analyst-activity.ts` or the cron that populates `AnalystActivity` is running and includes the `firm` column |
| `analystUpsidePotential` | Computed | `((analystTargetPrice / currentPrice) - 1) * 100` | Depends on `analystTargetPrice` in Company | **Check if `analystTargetPrice` is populated** — it was 0/352 in the CSV. Ensure `yahoo-finance-client.ts` fetches and stores this field. If unavailable, the model falls back to the training mean (13.5%) |
| `concernLevel` | `Filing` | `concernLevel` | Yes (Claude AI analysis) | None |
| `marketCap` | `Company` | `marketCap` | Yes | None |
| `sentimentScore` | `Filing` | `sentimentScore` | Yes (Claude AI analysis) | None |
| `upgradesLast30d` | `AnalystActivity` | count where `actionType='upgrade'` | Same as majorDowngrades | Same as above |

### Critical: Verify `AnalystActivity` table is populated

The `AnalystActivity` table exists in the schema but may not be consistently populated. Check:
1. `SELECT COUNT(*) FROM "AnalystActivity";` — how many rows?
2. `SELECT DISTINCT "firm" FROM "AnalystActivity" LIMIT 20;` — are major firms present?
3. `SELECT "actionType", COUNT(*) FROM "AnalystActivity" GROUP BY "actionType";` — are upgrade/downgrade entries present?

If the table is sparse, look at the cron jobs or scripts that populate it. The Yahoo Finance client (`lib/yahoo-finance-client.ts`) likely fetches analyst data — ensure it writes to `AnalystActivity`. If there's no cron for this, add one to `app/api/cron/` that runs daily and fetches recent analyst activity for all tracked companies.

### Critical: Verify `analystTargetPrice` in Company table

Run `SELECT COUNT(*) FROM "Company" WHERE "analystTargetPrice" IS NOT NULL AND "analystTargetPrice" > 0;`. If this is 0 or very low, the Yahoo Finance client needs to be updated to fetch the `targetMeanPrice` field from `yahoo-finance2`'s `quoteSummary` module (under `financialData.targetMeanPrice`).

---

## Task 4: Add `predicted30dAlpha` to the Filing Table

The model predicts **30-day alpha** (market-relative return), not raw return. Add a column to track this:

### Prisma schema change (`prisma/schema.prisma`):

Add to the `Filing` model:
```prisma
predicted30dAlpha     Float?
```

Then run `npx prisma migrate dev --name add-predicted-alpha`.

Update the prediction endpoint to also store:
```typescript
await prisma.filing.update({
  where: { id: filing.id },
  data: {
    predicted7dReturn: prediction.predicted7dReturn,
    predicted30dReturn: prediction.predicted30dReturn,
    predicted30dAlpha: prediction.expectedAlpha,
    predictionConfidence: prediction.confidence === 'high' ? 0.85 : prediction.confidence === 'medium' ? 0.65 : 0.5,
  },
});
```

---

## Task 5: Update Accuracy Tracking

The current `lib/accuracy-tracker.ts` compares `predicted7dReturn` vs `actual7dReturn`. Update it to also track alpha accuracy:

1. When checking accuracy (10+ days after filing), also compute `actual30dAlpha = actual30dReturn - spxReturn30d`
2. Store the alpha accuracy alongside return accuracy
3. Track directional accuracy of the signal (LONG/SHORT) vs actual alpha direction
4. The key metric to optimize is **high-confidence directional accuracy** — our model achieves 62.5% on this

Add a method to compute model stats that includes:
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

## Task 6: Update Paper Trading

The paper trading system in `lib/paper-trading.ts` currently uses `predicted7dReturn` and `predictionConfidence` to decide trades. Update it to:

1. Use the `signal` field ('LONG'/'SHORT'/'NEUTRAL') instead of return sign
2. Use the `confidence` field ('high'/'medium'/'low') instead of the numeric confidence
3. Only trade on LONG or SHORT signals (skip NEUTRAL)
4. For high-confidence signals, use full position size
5. For medium-confidence, use half position size
6. Set stop-loss and take-profit based on the score magnitude:
   - High-confidence LONG: target +5%, stop -3%
   - Medium-confidence LONG: target +3%, stop -2%
   - High-confidence SHORT: target -5% (or +5% for short position), stop +3%

---

## Task 7: Fix Documentation and Accuracy Claims

### README.md

Replace the accuracy claims. The current claims are:

> "60.26% directional accuracy"
> "60% accuracy"

These are not valid:
- The 60.26% came from `backtest-v3-optimized.py` which simulates features **using the actual return as input** (data leakage)
- The 67.5% in `baseline_results.json` is the base rate (model predicts all positive)

Replace with honest metrics from the new model:

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
| HIGH-conf LONG avg 30D alpha | +4.72% |
| HIGH-conf SHORT avg 30D alpha | -2.92% |
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

Add a section at the end documenting this model iteration:

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

### Key Discovery: The Production Model Had No Signal
The v3.0 Logistic Regression on EPS surprise features predicted EVERY stock as
positive. Its confusion matrix: TN=0, FP=344, FN=0, TP=713. The "67.5% accuracy"
equaled the base rate of positive returns. The model had zero discriminative power.

### Key Discovery: Contrarian Analyst Signals
Major bank downgrades are the **second strongest bullish signal** (+0.78 weight).
This directly contradicts the v2.0 rule-based model which penalized downgrades.
The market systematically overreacts to downgrades from Goldman Sachs, JPMorgan,
and similar firms, creating a 30-day recovery opportunity.

### Key Discovery: AI Features Are Weak
The Claude-generated features (riskScore, sentimentScore, concernLevel) collectively
contribute ~0.16 weight out of ~3.3 total. Market structure features (price ratios,
analyst activity, market cap) dominate. The AI analysis adds marginal value but is
not the primary signal source.
```

### `lib/confidence-scores.ts`

The default confidence of 56.8% and the per-ticker hardcoded scores should be deprecated. The new model has its own confidence framework based on score percentiles. Either:
- Remove the hardcoded per-ticker scores entirely, or
- Keep them as informational metadata but don't use them for trade sizing

### Remove from `scripts/`:
- `predict_baseline.py` — no longer needed (model is pure TypeScript now)
- The Python model files `models/baseline_model.pkl` and `models/baseline_scaler.pkl` can be archived

---

## Task 8: Update UI Components

### Prediction display

The UI currently shows predictions as simple percentage returns. Update to also show:
- The **signal** (LONG/SHORT/NEUTRAL) with color coding (green/red/gray)
- The **confidence** level (high/medium/low)
- The **expected alpha** (separate from raw return)
- The **top feature contribution** — which feature drove the prediction most

This information is in the `AlphaPrediction` response from `predictAlpha()`.

### Latest filings page

If there's a dashboard or latest-filings view, add a column or badge showing the signal. High-confidence LONGs and SHORTs should be visually prominent since those are the most reliable predictions.

---

## Summary of Changes

| File | Action | Priority |
|------|--------|----------|
| `lib/alpha-model.ts` | **Create** — new model with exact weights | P0 |
| `app/api/predict/[accession]/route.ts` | **Modify** — use new model | P0 |
| `prisma/schema.prisma` | **Modify** — add `predicted30dAlpha` column | P0 |
| `lib/accuracy-tracker.ts` | **Modify** — track alpha accuracy + directional accuracy | P1 |
| `lib/paper-trading.ts` | **Modify** — use signal/confidence instead of raw returns | P1 |
| `README.md` | **Modify** — fix accuracy claims | P1 |
| `MODEL-DEVELOPMENT-JOURNEY.md` | **Modify** — add v4.0 section | P2 |
| `lib/confidence-scores.ts` | **Deprecate** — remove hardcoded per-ticker scores | P2 |
| `scripts/predict_baseline.py` | **Archive** — no longer used | P2 |
| UI components | **Modify** — show signal/confidence/alpha | P2 |

## Verification Steps

After making the changes:
1. Run `npx prisma migrate dev` to apply schema changes
2. Pick 5 recent filings with known actual returns and run the new model against them manually — verify the scores match the formula
3. Compare the new predictions against `actual30dAlpha` for any filings where actuals exist
4. Verify the paper trading system doesn't break with the new signal format
5. Run the accuracy tracker on historical data to get the real-world performance numbers
