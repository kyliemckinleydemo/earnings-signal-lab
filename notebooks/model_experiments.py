# %% [markdown]
# # Earnings Signal Lab — Model Experimentation
#
# Test gradient boosting variants, neural nets, tune with Optuna,
# explain with SHAP, and run walk-forward validation.
#
# **Usage**: Open in JupyterLab and run cells, or run as a script:
# `python model_experiments.py`

# %% Imports
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Railway
import matplotlib.pyplot as plt
import seaborn as sns

print('All imports OK')

# %% Config
DATA_DIR = Path('../earnings_signal_data')
if not DATA_DIR.exists():
    DATA_DIR = Path('earnings_signal_data')  # Running from repo root

# TARGET: which return horizon to model
TARGET = 'return_5D'  # Options: return_1D, return_5D, return_10D, return_21D

# Optuna trials (more = better but slower)
N_TRIALS = 50

# Walk-forward settings
WF_MIN_TRAIN = 50   # Minimum training observations before first prediction
WF_STEP = 10        # Predict this many at a time, then retrain

# %% Load Data
df = pd.read_csv(DATA_DIR / 'backtest_dataset.csv')
print(f'Dataset: {len(df)} rows × {len(df.columns)} columns')
print(f'Companies: {df["symbol"].nunique()}')
print(f'Date range: {df["earnings_date"].min()} to {df["earnings_date"].max()}')

FEAT_COLS = [c for c in df.columns if c.startswith('feat_')]
RET_COLS = [c for c in df.columns if c.startswith('return_') and c != 'return_earnings_day']
FEATURE_NAMES = [c.replace('feat_', '') for c in FEAT_COLS]

print(f'\nFeatures ({len(FEAT_COLS)}): {FEATURE_NAMES}')
print(f'Return periods: {[c.replace("return_", "") for c in RET_COLS]}')

# Sort by date for time-series splits
df = df.sort_values('earnings_date').reset_index(drop=True)

# %% Prepare modeling data
model_df = df[FEAT_COLS + [TARGET, 'earnings_date', 'symbol']].dropna()
print(f'\nModeling {TARGET}: {len(model_df)} usable observations')

X = model_df[FEAT_COLS].values
y = model_df[TARGET].values
dates = model_df['earnings_date'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

N_SPLITS = min(5, max(2, len(model_df) // 20))
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
print(f'Using {N_SPLITS}-fold time-series CV')
print(f'Target stats: mean={y.mean():.2f}%, std={y.std():.2f}%, median={np.median(y):.2f}%')


# %% Helper
def evaluate_model(model, X, y, tscv, name='Model'):
    """Cross-validate and print results."""
    r2 = cross_val_score(model, X, y, cv=tscv, scoring='r2')
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error'))
    print(f'{name}')
    print(f'  CV R²:  {r2.mean():.4f} ± {r2.std():.4f}  (folds: {[f"{v:.4f}" for v in r2]})')
    print(f'  CV RMSE: {rmse.mean():.4f}')
    return r2.mean(), r2.std()


# %% [markdown]
# ## 1. Baselines

# %% Baselines
print('=' * 60)
print('BASELINES')
print('=' * 60)

baseline_scores = {}
for name, model in [('Ridge', Ridge(alpha=1.0)),
                     ('Lasso', Lasso(alpha=0.1, max_iter=10000))]:
    mean, std = evaluate_model(model, X_scaled, y, tscv, name)
    baseline_scores[name] = mean

best_baseline = max(baseline_scores.values())
print(f'\nBaseline to beat: {best_baseline:.4f} R²')


# %% [markdown]
# ## 2. XGBoost

# %% XGBoost default
print('\n' + '=' * 60)
print('XGBOOST')
print('=' * 60)

xgb_default = xgb.XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
)
xgb_score, _ = evaluate_model(xgb_default, X_scaled, y, tscv, 'XGBoost (default)')


# %% [markdown]
# ## 3. LightGBM

# %% LightGBM default
print('\n' + '=' * 60)
print('LIGHTGBM')
print('=' * 60)

lgb_default = lgb.LGBMRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1,
)
lgb_score, _ = evaluate_model(lgb_default, X_scaled, y, tscv, 'LightGBM (default)')


# %% [markdown]
# ## 4. CatBoost

# %% CatBoost default
print('\n' + '=' * 60)
print('CATBOOST')
print('=' * 60)

cat_default = CatBoostRegressor(
    iterations=300, depth=3, learning_rate=0.05,
    subsample=0.8, l2_leaf_reg=3.0, random_seed=42, verbose=0,
)
cat_score, _ = evaluate_model(cat_default, X_scaled, y, tscv, 'CatBoost (default)')


# %% [markdown]
# ## 5. Neural Network (PyTorch)

# %% Neural net
print('\n' + '=' * 60)
print('NEURAL NETWORK')
print('=' * 60)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EarningsNet(nn.Module):
    def __init__(self, n_features, hidden_sizes=[64, 32, 16], dropout=0.3):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h), nn.BatchNorm1d(h),
                nn.ReLU(), nn.Dropout(dropout),
            ])
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_neural_net(X_train, y_train, X_val, y_val,
                     hidden_sizes=[64, 32, 16], dropout=0.3,
                     lr=0.001, epochs=200, batch_size=32, patience=20):
    """Train with early stopping."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = EarningsNet(X_train.shape[1], hidden_sizes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_v).numpy()
    return model, preds


nn_r2_folds = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    _, preds = train_neural_net(X_tr, y_tr, X_val, y_val)
    fold_r2 = r2_score(y_val, preds)
    nn_r2_folds.append(fold_r2)
    print(f'  Fold {fold+1}: R² = {fold_r2:.4f}')

nn_score = np.mean(nn_r2_folds)
print(f'\nNeural Net CV R²: {nn_score:.4f} ± {np.std(nn_r2_folds):.4f}')


# %% [markdown]
# ## 6. Optuna Hyperparameter Tuning

# %% Optuna
print('\n' + '=' * 60)
print(f'OPTUNA TUNING ({N_TRIALS} trials each)')
print('=' * 60)


def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42, 'verbosity': 0,
    }
    model = xgb.XGBRegressor(**params)
    return cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2').mean()


def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42, 'verbosity': -1,
    }
    model = lgb.LGBMRegressor(**params)
    return cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2').mean()


def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'random_seed': 42, 'verbose': 0,
    }
    model = CatBoostRegressor(**params)
    return cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2').mean()


def nn_objective(trial):
    h1 = trial.suggest_int('h1', 16, 128)
    h2 = trial.suggest_int('h2', 8, 64)
    h3 = trial.suggest_int('h3', 4, 32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    folds_r2 = []
    for train_idx, val_idx in tscv.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        _, preds = train_neural_net(X_tr, y_tr, X_val, y_val,
                                     hidden_sizes=[h1, h2, h3],
                                     dropout=dropout, lr=lr, epochs=150, patience=15)
        folds_r2.append(r2_score(y_val, preds))
    return np.mean(folds_r2)


studies = {}
for name, objective in [('XGBoost', xgb_objective),
                         ('LightGBM', lgb_objective),
                         ('CatBoost', cat_objective),
                         ('NeuralNet', nn_objective)]:
    print(f'\nTuning {name}...')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    studies[name] = study
    print(f'  Best R²: {study.best_value:.4f}')
    print(f'  Best params: {study.best_params}')


# %% [markdown]
# ## 7. Full Model Comparison

# %% Comparison
print('\n' + '=' * 60)
print('FULL MODEL COMPARISON')
print('=' * 60)

tuned_xgb = xgb.XGBRegressor(**studies['XGBoost'].best_params, random_state=42, verbosity=0)
tuned_lgb = lgb.LGBMRegressor(**studies['LightGBM'].best_params, random_state=42, verbosity=-1)
tuned_cat = CatBoostRegressor(**studies['CatBoost'].best_params, random_seed=42, verbose=0)

all_models = {
    'Ridge (baseline)': Ridge(alpha=1.0),
    'Lasso (baseline)': Lasso(alpha=0.1, max_iter=10000),
    'XGBoost (default)': xgb_default,
    'XGBoost (tuned)': tuned_xgb,
    'LightGBM (default)': lgb_default,
    'LightGBM (tuned)': tuned_lgb,
    'CatBoost (default)': cat_default,
    'CatBoost (tuned)': tuned_cat,
}

results = []
for name, model in all_models.items():
    r2 = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
    results.append({'Model': name, 'CV_R2_Mean': r2.mean(), 'CV_R2_Std': r2.std()})

# Add neural net results
results.append({'Model': 'NeuralNet (default)', 'CV_R2_Mean': nn_score, 'CV_R2_Std': np.std(nn_r2_folds)})
results.append({'Model': 'NeuralNet (tuned)', 'CV_R2_Mean': studies['NeuralNet'].best_value, 'CV_R2_Std': 0.0})

results_df = pd.DataFrame(results).sort_values('CV_R2_Mean', ascending=False)
results_df.index = range(1, len(results_df) + 1)
results_df.index.name = 'Rank'
print(results_df.to_string())

# Save comparison chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#3de8a0' if v > 0 else '#f06878' for v in results_df['CV_R2_Mean']]
ax.barh(results_df['Model'], results_df['CV_R2_Mean'],
        xerr=results_df['CV_R2_Std'], color=colors, alpha=0.85, capsize=3)
ax.set_xlabel('CV R²')
ax.set_title(f'Model Comparison — {TARGET}')
ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)
ax.set_facecolor('#0c0c1a')
fig.patch.set_facecolor('#06060f')
ax.tick_params(colors='#d4d4e8')
ax.xaxis.label.set_color('#d4d4e8')
ax.title.set_color('#d4d4e8')
for spine in ax.spines.values():
    spine.set_color('#1a1a35')
plt.tight_layout()
plt.savefig(DATA_DIR / 'model_comparison.png', dpi=150, facecolor='#06060f')
print(f'\nSaved: {DATA_DIR / "model_comparison.png"}')


# %% [markdown]
# ## 8. SHAP Analysis

# %% SHAP
print('\n' + '=' * 60)
print('SHAP FEATURE IMPORTANCE')
print('=' * 60)

try:
    import shap

    # Use best tree model for SHAP
    tree_results = {n: r for n, r in zip(
        [r['Model'] for r in results],
        [r['CV_R2_Mean'] for r in results]
    ) if 'XGB' in n or 'Light' in n or 'Cat' in n}
    best_tree_name = max(tree_results, key=tree_results.get)
    best_tree = all_models.get(best_tree_name, tuned_xgb)
    print(f'SHAP on: {best_tree_name}')

    best_tree.fit(X_scaled, y)
    explainer = shap.TreeExplainer(best_tree)
    shap_values = explainer.shap_values(X_scaled)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=FEATURE_NAMES,
                      show=False, plot_size=(10, 8))
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {DATA_DIR / "shap_summary.png"}')
    plt.close()

    # Feature importance bar chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=FEATURE_NAMES,
                      plot_type='bar', show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'shap_importance.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {DATA_DIR / "shap_importance.png"}')
    plt.close()

    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': FEATURE_NAMES, 'mean_abs_shap': mean_shap})
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    print('\nSHAP Feature Rankings:')
    for _, row in shap_df.iterrows():
        bar = '█' * int(row['mean_abs_shap'] / shap_df['mean_abs_shap'].max() * 30)
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:.4f}  {bar}")

    # SHAP interaction effects (if dataset is small enough)
    if len(X_scaled) < 500:
        print('\nComputing SHAP interaction values...')
        shap_interaction = explainer.shap_interaction_values(X_scaled)
        mean_int = np.abs(shap_interaction).mean(axis=0)
        np.fill_diagonal(mean_int, 0)

        pairs = []
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], mean_int[i, j]))

        top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
        print('\nTop 10 Feature Interactions:')
        for f1, f2, val in top_pairs:
            print(f'  {f1} × {f2}: {val:.4f}')
    else:
        print('\nSkipping interaction analysis (dataset > 500 rows)')

except Exception as e:
    print(f'SHAP analysis failed: {e}')


# %% [markdown]
# ## 9. Walk-Forward Validation

# %% Walk-forward
print('\n' + '=' * 60)
print('WALK-FORWARD VALIDATION')
print('=' * 60)


def walk_forward_test(model_fn, X, y, dates, min_train=WF_MIN_TRAIN, step=WF_STEP):
    """True out-of-sample: train on past, predict next batch, slide forward."""
    all_preds, all_actuals, all_dates = [], [], []

    start = min_train
    while start < len(X):
        end = min(start + step, len(X))
        X_train, y_train = X[:start], y[:start]
        X_test, y_test = X[start:end], y[start:end]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates.extend(dates[start:end])
        start = end

    return np.array(all_preds), np.array(all_actuals), all_dates


# Neural net walk-forward wrapper
def nn_walk_forward(X, y, dates, best_params, min_train=WF_MIN_TRAIN, step=WF_STEP):
    """Walk-forward for neural net (needs special training loop)."""
    all_preds, all_actuals, all_dates = [], [], []

    start = min_train
    while start < len(X):
        end = min(start + step, len(X))
        X_tr, y_tr = X[:start], y[:start]
        X_te, y_te = X[start:end], y[start:end]

        _, preds = train_neural_net(
            X_tr, y_tr, X_te, y_te,
            hidden_sizes=[best_params.get('h1', 64),
                         best_params.get('h2', 32),
                         best_params.get('h3', 16)],
            dropout=best_params.get('dropout', 0.3),
            lr=best_params.get('lr', 0.001),
            epochs=150, patience=15,
        )
        all_preds.extend(preds)
        all_actuals.extend(y_te)
        all_dates.extend(dates[start:end])
        start = end

    return np.array(all_preds), np.array(all_actuals), all_dates


wf_models = {
    'Ridge': lambda: Ridge(alpha=1.0),
    'XGBoost (tuned)': lambda: xgb.XGBRegressor(
        **studies['XGBoost'].best_params, random_state=42, verbosity=0),
    'LightGBM (tuned)': lambda: lgb.LGBMRegressor(
        **studies['LightGBM'].best_params, random_state=42, verbosity=-1),
    'CatBoost (tuned)': lambda: CatBoostRegressor(
        **studies['CatBoost'].best_params, random_seed=42, verbose=0),
}

print(f'Walk-Forward (min_train={WF_MIN_TRAIN}, step={WF_STEP})\n')
print(f'{"Model":<25} {"R²":>8} {"RMSE":>8} {"MAE":>8} {"Dir Acc":>8} {"Sharpe":>8} {"n":>6}')
print('-' * 75)

wf_results = {}
for name, model_fn in wf_models.items():
    preds, actuals, pred_dates = walk_forward_test(model_fn, X_scaled, y, dates)

    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    dir_acc = np.mean((preds > 0) == (actuals > 0))

    # Strategy: long when predicted positive, short when negative
    signals = np.where(preds > 0, 1, -1)
    strategy_rets = signals * actuals
    sharpe = strategy_rets.mean() / strategy_rets.std() * np.sqrt(252 / 5) if strategy_rets.std() > 0 else 0

    wf_results[name] = {
        'preds': preds, 'actuals': actuals, 'dates': pred_dates,
        'r2': r2, 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc, 'sharpe': sharpe,
    }
    print(f'{name:<25} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f} {dir_acc:>8.1%} {sharpe:>8.2f} {len(preds):>6}')

# Neural net walk-forward
print('\nRunning Neural Net walk-forward (slower)...')
nn_preds, nn_actuals, nn_dates = nn_walk_forward(
    X_scaled, y, dates, studies['NeuralNet'].best_params)
nn_r2_wf = r2_score(nn_actuals, nn_preds)
nn_dir = np.mean((nn_preds > 0) == (nn_actuals > 0))
nn_strat = np.where(nn_preds > 0, 1, -1) * nn_actuals
nn_sharpe = nn_strat.mean() / nn_strat.std() * np.sqrt(252 / 5) if nn_strat.std() > 0 else 0
wf_results['NeuralNet (tuned)'] = {
    'preds': nn_preds, 'actuals': nn_actuals, 'dates': nn_dates,
    'r2': nn_r2_wf, 'dir_acc': nn_dir, 'sharpe': nn_sharpe,
}
print(f'{"NeuralNet (tuned)":<25} {nn_r2_wf:>8.4f} {"":>8} {"":>8} {nn_dir:>8.1%} {nn_sharpe:>8.2f} {len(nn_preds):>6}')


# Walk-forward cumulative return chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#06060f')

for name, data in wf_results.items():
    signals = np.where(data['preds'] > 0, 1, -1)
    cum_returns = np.cumsum(signals * data['actuals'])
    axes[0].plot(range(len(cum_returns)), cum_returns, label=name, linewidth=1.5)
    axes[1].scatter(data['preds'], data['actuals'], alpha=0.3, s=15, label=name)

for ax in axes:
    ax.set_facecolor('#0c0c1a')
    ax.tick_params(colors='#d4d4e8')
    for spine in ax.spines.values():
        spine.set_color('#1a1a35')

axes[0].set_title('Walk-Forward Cumulative Returns', color='#d4d4e8')
axes[0].set_xlabel('Trade #', color='#d4d4e8')
axes[0].set_ylabel('Cumulative Return (%)', color='#d4d4e8')
axes[0].legend(fontsize=7)
axes[0].axhline(y=0, color='white', linewidth=0.5, alpha=0.3)

axes[1].set_title('Predicted vs Actual Returns', color='#d4d4e8')
axes[1].set_xlabel('Predicted (%)', color='#d4d4e8')
axes[1].set_ylabel('Actual (%)', color='#d4d4e8')
axes[1].axhline(y=0, color='white', linewidth=0.3, alpha=0.3)
axes[1].axvline(x=0, color='white', linewidth=0.3, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'walk_forward.png', dpi=150, facecolor='#06060f')
print(f'\nSaved: {DATA_DIR / "walk_forward.png"}')


# %% [markdown]
# ## 10. Promote Best Model to Production

# %% Promote
print('\n' + '=' * 60)
print('PROMOTE TO PRODUCTION')
print('=' * 60)

# Rank by walk-forward Sharpe (most realistic metric for trading)
wf_ranked = sorted(wf_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
best_name, best_data = wf_ranked[0]

print(f'\nBest walk-forward model: {best_name}')
print(f'  R²:        {best_data["r2"]:.4f}')
print(f'  Dir Acc:   {best_data["dir_acc"]:.1%}')
print(f'  Sharpe:    {best_data["sharpe"]:.2f}')

# Load existing results and update with experiment findings
results_file = DATA_DIR / 'backtest_results.json'
if results_file.exists():
    production_results = json.loads(results_file.read_text())
else:
    production_results = {}

# Save experiment results
experiment_record = {
    'target': TARGET,
    'run_date': datetime.now().isoformat(),
    'n_observations': len(model_df),
    'n_optuna_trials': N_TRIALS,
    'cv_comparison': results_df.to_dict('records'),
    'walk_forward': {
        name: {
            'r2': float(data['r2']),
            'dir_acc': float(data['dir_acc']),
            'sharpe': float(data['sharpe']),
        }
        for name, data in wf_results.items()
    },
    'best_model': {
        'name': best_name,
        'sharpe': float(best_data['sharpe']),
        'r2': float(best_data['r2']),
        'dir_acc': float(best_data['dir_acc']),
    },
    'optuna_best_params': {
        'XGBoost': studies['XGBoost'].best_params,
        'LightGBM': studies['LightGBM'].best_params,
        'CatBoost': studies['CatBoost'].best_params,
        'NeuralNet': studies['NeuralNet'].best_params,
    },
}

# Feature importance from SHAP (if available)
if 'shap_df' in dir() and shap_df is not None:
    experiment_record['shap_importance'] = shap_df.set_index('feature')['mean_abs_shap'].to_dict()

production_results['experiments'] = production_results.get('experiments', [])
production_results['experiments'].append(experiment_record)

# Save
results_file.write_text(json.dumps(production_results, indent=2, default=str))
print(f'\nExperiment saved to {results_file}')
print(f'Total experiments recorded: {len(production_results["experiments"])}')

# Also save a standalone experiment report
report_file = DATA_DIR / f'experiment_{TARGET}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
report_file.write_text(json.dumps(experiment_record, indent=2, default=str))
print(f'Standalone report: {report_file}')

print('\n✅ Done. Refresh the web dashboard to see updated results.')
