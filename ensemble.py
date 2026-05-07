"""
Ensemble Learning - Chennai House Price Prediction
Compares: Random Forest, XGBoost, LightGBM, Gradient Boosting + Stacking Ensemble
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────
print("=" * 60)
print("  ENSEMBLE LEARNING — CHENNAI HOUSE PRICE PREDICTION")
print("=" * 60)

df = pd.read_csv('C:/Users/eshwa/FOML PROJECT/Housing-Prices-in-Chennai.csv')
print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Encode Location
le = LabelEncoder()
df['Location_enc'] = le.fit_transform(df['Location'])

# Features & Target
drop_cols = ['Price', 'Location']
X = df.drop(columns=drop_cols)
y = df['Price']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 2. DEFINE MODELS
# ─────────────────────────────────────────
models = {
    'Random Forest':      RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
    'XGBoost':            xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42, verbosity=0),
    'LightGBM':           lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=63, random_state=42, verbose=-1),
}

# Stacking Ensemble (base = all 4, meta = Ridge)
stacking = StackingRegressor(
    estimators=[(name, mdl) for name, mdl in models.items()],
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1
)

# ─────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n📊 Training models...\n")
for name, model in {**models, 'Stacking Ensemble': stacking}.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mape  = np.mean(np.abs((y_test - preds) / y_test)) * 100

    cv_r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2', n_jobs=-1).mean()

    results[name] = {
        'model': model, 'predictions': preds,
        'RMSE': rmse, 'MAE': mae, 'R²': r2,
        'MAPE': mape, 'CV R²': cv_r2
    }
    print(f"  {name:<22} | R²={r2:.4f} | RMSE=₹{rmse:,.0f} | MAE=₹{mae:,.0f} | MAPE={mape:.2f}%")

# ─────────────────────────────────────────
# 4. SAVE BEST MODEL
# ─────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['R²'])
best_model = results[best_name]['model']
with open('C:/Users/eshwa/FOML PROJECT/best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'label_encoder': le, 'features': X.columns.tolist()}, f)
print(f"\n✓ Best model saved: {best_name} (R²={results[best_name]['R²']:.4f})")

# ─────────────────────────────────────────
# 5. FULL VISUALISATION  (12 panels)
# ─────────────────────────────────────────
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
PALETTE = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
model_names = list(results.keys())
colors = dict(zip(model_names, PALETTE))

fig = plt.figure(figsize=(24, 28))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel titles helper
def panel_title(ax, text):
    ax.set_title(text, fontsize=12, fontweight='bold', color='#212121', pad=8)

# ── 5-1  R² Comparison (bar)
ax1 = fig.add_subplot(gs[0, 0])
r2s = [results[n]['R²'] for n in model_names]
bars = ax1.barh(model_names, r2s, color=[colors[n] for n in model_names], edgecolor='white', height=0.6)
ax1.set_xlim(min(r2s) - 0.02, 1.0)
for bar, val in zip(bars, r2s):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
panel_title(ax1, 'R² Score Comparison')
ax1.set_xlabel('R² Score')
ax1.axvline(x=max(r2s), color='red', linestyle='--', alpha=0.4, linewidth=1)

# ── 5-2  RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
rmses = [results[n]['RMSE']/1e6 for n in model_names]
bars2 = ax2.barh(model_names, rmses, color=[colors[n] for n in model_names], edgecolor='white', height=0.6)
for bar, val in zip(bars2, rmses):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'₹{val:.2f}M', va='center', fontsize=9)
panel_title(ax2, 'RMSE Comparison (₹ Millions)')
ax2.set_xlabel('RMSE (₹M)')

# ── 5-3  Cross-Validation R²
ax3 = fig.add_subplot(gs[0, 2])
cv_r2s = [results[n]['CV R²'] for n in model_names]
bars3 = ax3.barh(model_names, cv_r2s, color=[colors[n] for n in model_names], edgecolor='white', height=0.6)
for bar, val in zip(bars3, cv_r2s):
    ax3.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
panel_title(ax3, '5-Fold Cross-Validation R²')
ax3.set_xlabel('CV R²')

# ── 5-4 to 5-8  Actual vs Predicted (one per model)
positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
for idx, (name, (row, col)) in enumerate(zip(model_names, positions)):
    ax = fig.add_subplot(gs[row, col])
    preds = results[name]['predictions']
    y_mln = y_test.values / 1e6
    p_mln = preds / 1e6
    ax.scatter(y_mln, p_mln, alpha=0.35, s=12, color=colors[name], edgecolors='none')
    lims = [min(y_mln.min(), p_mln.min()), max(y_mln.max(), p_mln.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Price (₹M)')
    ax.set_ylabel('Predicted Price (₹M)')
    panel_title(ax, f'{name}\nActual vs Predicted')
    ax.text(0.05, 0.92, f'R²={results[name]["R²"]:.4f}', transform=ax.transAxes,
            fontsize=9, color='#333', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# ── 5-9  Residuals Distribution
ax9 = fig.add_subplot(gs[2, 2])
for name in model_names:
    res = (y_test.values - results[name]['predictions']) / 1e6
    ax9.hist(res, bins=40, alpha=0.45, label=name, color=colors[name], edgecolor='none')
ax9.axvline(0, color='black', linewidth=1.2, linestyle='--')
ax9.set_xlabel('Residual (₹M)')
ax9.set_ylabel('Frequency')
panel_title(ax9, 'Residual Distribution')
ax9.legend(fontsize=8, loc='upper right')

# ── 5-10  Feature Importance (best model)
ax10 = fig.add_subplot(gs[3, 0:2])
feat_model = results['Random Forest']['model']
importances = feat_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True).tail(15)
ax10.barh(feat_df['Feature'], feat_df['Importance'], color='#2196F3', edgecolor='white')
panel_title(ax10, 'Top 15 Feature Importances (Random Forest)')
ax10.set_xlabel('Importance Score')

# ── 5-11  Metrics Table
ax11 = fig.add_subplot(gs[3, 2])
ax11.axis('off')
table_data = []
for name in model_names:
    r = results[name]
    table_data.append([
        name[:18],
        f"{r['R²']:.4f}",
        f"₹{r['RMSE']/1e6:.2f}M",
        f"{r['MAPE']:.1f}%",
        f"{r['CV R²']:.4f}"
    ])
col_labels = ['Model', 'R²', 'RMSE', 'MAPE', 'CV R²']
tbl = ax11.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#E3F2FD')
    cell.set_edgecolor('white')
panel_title(ax11, 'Summary Metrics Table')

# Main title
fig.suptitle(
    'Ensemble Learning — Chennai House Price Prediction\n'
    'Random Forest | Gradient Boosting | XGBoost | LightGBM | Stacking Ensemble',
    fontsize=15, fontweight='bold', color='#0D47A1', y=0.98
)

plt.savefig('C:/Users/eshwa/FOML PROJECT/outputs/ensemble_report.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("✓ Visualisation saved → ensemble_report.png")

# ─────────────────────────────────────────
# 6. PRINT FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<25} {'R²':>8} {'RMSE (₹M)':>12} {'MAE (₹M)':>11} {'MAPE%':>8} {'CV R²':>8}")
print("-" * 72)
for name in model_names:
    r = results[name]
    marker = " ← BEST" if name == best_name else ""
    print(f"{name:<25} {r['R²']:>8.4f} {r['RMSE']/1e6:>12.3f} {r['MAE']/1e6:>11.3f} {r['MAPE']:>7.2f}% {r['CV R²']:>8.4f}{marker}")
print("=" * 60)
