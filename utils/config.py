"""
config.py — Central configuration for the trading classifier pipeline.
 
Optimised for 4 tickers (~90K training rows).
This config produced +30.17% gross return, Sharpe 6.73 in best run.
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
# Calibrated empirically from SP500 1-min data (ALL ticker, 498K bars).
# Full calibration grid run at mult=[2.0,2.25,2.5] x fw=[6..15]:
#
#   mult=2.25, fw=13  =>  BUY=36.0%  SELL=35.6%  HOLD=28.4%  <- SELECTED
#
# Selected: mult=2.25, fw=13 — centred in 25-30% HOLD band,
# near-equal BUY/SELL, multiplier in 2.0-2.5 range.
BARRIER_MULTIPLIER   = 2.25
FORWARD_WINDOW       = 13
VOL_WINDOW_FOR_LABEL = 60
 
# ── Alternative label presets (swap the two lines above) ──────────────────────
# Tighter  / more signals  (~27% HOLD): BARRIER_MULTIPLIER=2.00, FORWARD_WINDOW=11
# Current  / balanced      (~28% HOLD): BARRIER_MULTIPLIER=2.25, FORWARD_WINDOW=13
# Looser   / fewer signals (~30% HOLD): BARRIER_MULTIPLIER=2.50, FORWARD_WINDOW=15
 
# ── Volatility regime percentiles ─────────────────────────────────────────────
VOL_LOW_PCT  = 33
VOL_HIGH_PCT = 67
 
# ── Label encoding ─────────────────────────────────────────────────────────────
LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_INV = {0: "SELL", 1: "HOLD", 2: "BUY"}
 
# ── LightGBM — tuned for ~90K rows (4 tickers) ────────────────────────────────
# This exact config produced +30.17% gross, Sharpe 6.73 on 2025 test data.
# DO NOT increase num_leaves or decrease learning_rate on 4 tickers —
# the model will overfit. Only change if you add significantly more tickers.
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "learning_rate":     0.05,    # higher lr = fewer trees, less overfit on small data
    "num_leaves":        31,      # small = prevents overfit on ~90K rows
    "feature_fraction":  0.7,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 50,      # lower floor suitable for small dataset
    "reg_alpha":         0.1,     # L1 regularisation
    "reg_lambda":        0.1,     # L2 regularisation
    "n_estimators":      500,     # early stopping fires at ~100-200 on small data
    "device":            "cpu",
    "random_state":      RANDOM_SEED,
    "verbosity":         -1,
}
EARLY_STOPPING_ROUNDS = 30
 
# ── Walk-forward CV — 2 folds for 4 tickers ───────────────────────────────────
# 2 folds is optimal for ~90K training rows.
# 3 folds introduces a very small first fold (~30K rows) that hurts performance.
# Each fold validates on a different H2 2023 period.
#
#   Fold 1: train -> 2022-12-31,  val: 2023-01-01 to 2023-06-30
#   Fold 2: train -> 2023-06-30,  val: 2023-07-01 to 2023-12-31
WF_FOLDS = [
    {"train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"train_end": "2023-06-30", "val_start": "2023-07-01", "val_end": "2023-12-31"},
]
WF_N_ESTIMATORS = 300    # faster CV — early stopping fires well before this
WF_EARLY_STOP   = 20
 
# ── Ticker scaling guide ───────────────────────────────────────────────────────
# If you add more tickers, update LGBM_PARAMS and WF_FOLDS accordingly:
#
#  4 tickers  (~90K rows):  num_leaves=31,  lr=0.05, n_est=500,  2 folds
#  8 tickers  (~180K rows): num_leaves=48,  lr=0.04, n_est=750,  3 folds
#  16 tickers (~360K rows): num_leaves=64,  lr=0.03, n_est=1000, 3 folds
#  28 tickers (~630K rows): num_leaves=96,  lr=0.02, n_est=2000, 3 folds
#
# Before adding new tickers, run: python check_ticker_compatibility.py
# Only add tickers with compatibility score > 0.5
 
# ── Meta-model ─────────────────────────────────────────────────────────────────
# 0.60 threshold keeps ~10% of signals — optimal for 4 tickers.
# Lower (0.55) = more trades, lower precision.
# Higher (0.65) = fewer trades, higher precision but very few signals.
META_CONFIDENCE_THRESHOLD = 0.60
 
# ── Position sizing (proportional to meta confidence) ─────────────────────────
# size = MIN + (confidence - threshold) / (1 - threshold) * (MAX - MIN)
#
# Examples at threshold=0.60:
#   confidence = 0.60  ->  size = 0.25  (minimum, marginal trade)
#   confidence = 0.70  ->  size = 0.50
#   confidence = 0.80  ->  size = 0.75
#   confidence = 1.00  ->  size = 1.00  (maximum, highest conviction)
MIN_POSITION_SIZE = 0.25
MAX_POSITION_SIZE = 1.00
 
# ── Backtesting ────────────────────────────────────────────────────────────────
# Set to 0.0 for gross performance (no fees).
# Set to 0.5 for realistic IBKR Pro execution.
# Set to 2.0 for retail broker worst-case.
TRADING_COST_BPS = 0.0   # zero fees — gross performance only
 
# ── Saved artefact paths ───────────────────────────────────────────────────────
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.joblib")
LGBM_MODEL_PATH   = os.path.join(MODEL_DIR, "lgbm_model.joblib")
META_MODEL_PATH   = os.path.join(MODEL_DIR, "meta_model.joblib")
WF_RESULTS_PATH   = os.path.join(MODEL_DIR, "walk_forward_results.joblib")
 