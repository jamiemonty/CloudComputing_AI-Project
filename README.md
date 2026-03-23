# Stock Day Trading Classifier — LightGBM Pipeline

CPU-efficient, production-quality day trading classifier with strict anti-leakage
guarantees, triple-barrier labeling, walk-forward CV, meta-labeling, and
proportional position sizing.

---

## Project Structure

```
trading_classifier/
├── rawSP500data/               <- Place TICKER_1min.csv files here
├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
├── logs/
├── utils/
│   ├── config.py               <- All constants, paths, hyperparams
│   ├── feature_utils.py        <- Vectorised features (shift(1) enforced)
│   └── label_utils.py          <- Vectorised triple-barrier labeling
├── script1_data_preparation.py
├── script2_feature_engineering.py
├── script3_labeling.py
├── script4_dataset_builder.py
├── script5_split_data.py
├── script6_train_lightgbm.py   <- Walk-forward CV + final training
├── script7_train_meta_model.py <- Meta-labeling + position sizing schedule
├── script8_evaluate_model.py
├── script9_backtest.py         <- Proportional position sizing backtest
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Place data files in rawSP500data/
```

---

## Running the Pipeline

```bash
python script1_data_preparation.py
python script2_feature_engineering.py
python script3_labeling.py
python script4_dataset_builder.py
python script5_split_data.py
python script6_train_lightgbm.py   # Runs WF-CV then trains final model
python script7_train_meta_model.py
python script8_evaluate_model.py
python script9_backtest.py
```

---

## Key Parameters (utils/config.py)

### Label Distribution (calibrated from real data)
```python
BARRIER_MULTIPLIER = 3.0   # was 1.75 -- raised to achieve ~25% HOLD
FORWARD_WINDOW     = 24    # was 30  -- tuned alongside multiplier
# Result: BUY ~37.8%  SELL ~37.4%  HOLD ~24.8%
```

### Walk-Forward CV Folds
```python
WF_FOLDS = [
    {"train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-06-30"},
    {"train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"train_end": "2023-06-30", "val_start": "2023-07-01", "val_end": "2023-12-31"},
]
```

### Position Sizing
```python
META_CONFIDENCE_THRESHOLD = 0.55   # minimum confidence to trade
MIN_POSITION_SIZE         = 0.25   # 25% capital at threshold
MAX_POSITION_SIZE         = 1.00   # 100% capital at confidence=1.0
# Linear: size = 0.25 + (confidence - 0.55) / 0.45 * 0.75
```

---

## Anti-Leakage Guarantees

| Rule | Where |
|---|---|
| All rolling stats use `.shift(1)` | `feature_utils._rolling_then_shift()` |
| Cross-day nulling via `_bar_idx` | Every rolling feature function |
| Scaler fit on train only | `script2` |
| Vol regime thresholds from train only | `script2` |
| Triple-barrier: no cross-day look-ahead | `label_utils._label_group_vectorized()` |
| Walk-forward CV: no val/test data seen | `script6.run_walk_forward_cv()` |
| Test evaluated exactly once | `script8`, `script9` only |

---

## What's New vs Standard Pipelines

### 1. Calibrated Label Distribution
Most implementations use arbitrary multipliers. Here BARRIER_MULTIPLIER=3.0 and
FORWARD_WINDOW=24 were derived empirically from actual SP500 1-min data to
achieve BUY~38%, SELL~37%, HOLD~25%.

### 2. Walk-Forward Cross-Validation
Expanding-window CV on training data only. 3 folds, each adding a year of data.
Validates that the model generalises across market regimes before final training.
Results saved to models/walk_forward_results.joblib.

### 3. Meta-Labeling
Secondary binary classifier on trade signals. Separates direction prediction
from trade quality prediction. Only signals above META_CONFIDENCE_THRESHOLD
are executed.

### 4. Proportional Position Sizing
Position size scales linearly with meta confidence:
    size = 0.25 at confidence = 0.55 (threshold)
    size = 1.00 at confidence = 1.00
High-conviction signals get full capital. Marginal signals get 25%.
Equity curve and position size histogram saved in logs/.

---

## Outputs

| File | Description |
|---|---|
| `models/scaler.joblib` | StandardScaler (train only) |
| `models/feature_list.joblib` | Feature column ordering |
| `models/lgbm_model.joblib` | Final LightGBM classifier |
| `models/walk_forward_results.joblib` | Per-fold CV metrics |
| `models/meta_model.joblib` | Meta-labeling binary classifier |
| `logs/eval_*.txt` | Classification reports |
| `logs/confusion_matrix_*.png` | Confusion matrices |
| `logs/equity_curve_*.png` | Primary and meta equity curves |
| `logs/position_size_distribution_meta.png` | Confidence -> size histogram |
| `logs/backtest_results.txt` | All backtest metrics |
