"""
config.py — Optimised for 4 tickers (~90K training rows).
"""
 
import os
 
# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
 
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR    = os.path.join(BASE_DIR, "rawSP500data")
PROCESSED_DIR   = os.path.join(BASE_DIR, "processed")
TRAIN_DIR       = os.path.join(PROCESSED_DIR, "train")
VAL_DIR         = os.path.join(PROCESSED_DIR, "validation")
TEST_DIR        = os.path.join(PROCESSED_DIR, "test")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
 
# ── Market hours ───────────────────────────────────────────────────────────────
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"
 
# ── Date splits ────────────────────────────────────────────────────────────────
TRAIN_START = "2020-12-28"
TRAIN_END   = "2023-12-31"
VAL_START   = "2024-01-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-23"
 
# ── Feature windows ────────────────────────────────────────────────────────────
MOMENTUM_WINDOWS      = [1, 5, 15, 30, 60]
ROLLING_MEAN_WINDOWS  = [15, 60]
ROLLING_STD_WINDOWS   = [15, 60, 120]
LOOKBACK_WINDOW       = 60
 
# ── Triple-barrier labeling ────────────────────────────────────────────────────
# mult=2.25, fw=13 => BUY~36%  SELL~35.6%  HOLD~28.4%
BARRIER_MULTIPLIER   = 2.25
FORWARD_WINDOW       = 13
VOL_WINDOW_FOR_LABEL = 60
 
# ── Volatility regime percentiles ─────────────────────────────────────────────
VOL_LOW_PCT  = 33
VOL_HIGH_PCT = 67
 
# ── Label encoding ─────────────────────────────────────────────────────────────
LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_INV = {0: "SELL", 1: "HOLD", 2: "BUY"}
 
# ── LightGBM — tuned for ~90K rows (4 tickers) ────────────────────────────────
# Smaller model to avoid overfitting on limited data.
# Fewer trees + higher LR = faster convergence.
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "learning_rate":     0.05,    # higher lr = fewer trees needed
    "num_leaves":        31,      # smaller = less overfit on small data
    "feature_fraction":  0.7,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 50,      # lower floor for small dataset
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_estimators":      500,     # cap at 500 — early stopping will fire ~100-200
    "device":            "cpu",
    "random_state":      RANDOM_SEED,
    "verbosity":         -1,
}
EARLY_STOPPING_ROUNDS = 30       # fast early stopping for small data
 
# ── Walk-forward CV — 2 folds only for speed ──────────────────────────────────
# With 4 tickers, 3 folds on small data is redundant.
# 2 folds covers enough time variation.
#   Fold 1: train=2021-2022, val=2023-H1
#   Fold 2: train=2021-2023H1, val=2023-H2
WF_FOLDS = [
    {"train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"train_end": "2023-06-30", "val_start": "2023-07-01", "val_end": "2023-12-31"},
]
WF_N_ESTIMATORS = 300    # fast CV trees
WF_EARLY_STOP   = 20
 
# ── Meta-model ─────────────────────────────────────────────────────────────────
META_CONFIDENCE_THRESHOLD = 0.60  # slightly lower for fewer trades
 
# ── Position sizing ────────────────────────────────────────────────────────────
MIN_POSITION_SIZE = 0.25
MAX_POSITION_SIZE = 1.00
 
# ── Backtesting ────────────────────────────────────────────────────────────────
TRADING_COST_BPS = 0.0   # fees removed — gross performance only
 
# ── Saved artefact paths ───────────────────────────────────────────────────────
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.joblib")
LGBM_MODEL_PATH   = os.path.join(MODEL_DIR, "lgbm_model.joblib")
META_MODEL_PATH   = os.path.join(MODEL_DIR, "meta_model.joblib")
WF_RESULTS_PATH   = os.path.join(MODEL_DIR, "walk_forward_results.joblib")
 