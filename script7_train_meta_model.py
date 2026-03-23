"""
script7_train_meta_model.py — Train meta-labeling model on ACTUAL trade P&L.
 
KEY FIX (v2):
    Previous version trained meta-model on whether primary model's class
    prediction matched the true label. This does NOT correlate with actual
    trade profitability because:
      - A correct BUY label doesn't mean the trade made money (costs, timing)
      - An incorrect label can still produce a profitable trade
 
    This version simulates actual barrier exits for every training signal and
    uses the resulting P&L sign as the binary target:
        y_meta = 1  if simulated_net_return > 0  (profitable trade)
        y_meta = 0  otherwise
 
    This directly teaches the meta-model to filter for profitable trades.
 
Inputs:
    processed/train/train.parquet
    processed/validation/validation.parquet
    processed/all_tickers_labeled.parquet  (for close prices + vol)
    models/lgbm_model.joblib
    models/feature_list.joblib
 
Outputs:
    models/meta_model.joblib
"""
 
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    TRAIN_DIR, VAL_DIR, PROCESSED_DIR, MODEL_DIR,
    LGBM_MODEL_PATH, FEATURE_LIST_PATH, META_MODEL_PATH,
    RANDOM_SEED, LABEL_MAP,
    BARRIER_MULTIPLIER, FORWARD_WINDOW, TRADING_COST_BPS,
    META_CONFIDENCE_THRESHOLD, MIN_POSITION_SIZE, MAX_POSITION_SIZE,
)
 
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script7.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
 
HOLD_LABEL = LABEL_MAP["HOLD"]
BUY_LABEL  = LABEL_MAP["BUY"]
SELL_LABEL = LABEL_MAP["SELL"]
COST       = TRADING_COST_BPS * 1e-4
np.random.seed(RANDOM_SEED)
 
 
def load_split(split_dir: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(split_dir, f"{name}.parquet"))
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info(f"Loaded {name}: {len(df):,} rows")
    return df
 
 
def simulate_trade_exit(entry_price: float,
                         future_closes: np.ndarray,
                         vol: float,
                         direction: int) -> float:
    """
    Same barrier exit logic as script9.
    Returns gross log-return (before cost).
    """
    if len(future_closes) == 0 or vol <= 0 or np.isnan(vol):
        return 0.0
 
    tp = BARRIER_MULTIPLIER * vol
    sl = -BARRIER_MULTIPLIER * vol
    exit_price = future_closes[-1]
 
    for price in future_closes:
        if np.isnan(price):
            break
        ret = np.log(price / entry_price)
        if direction == 1:
            if ret >= tp or ret <= sl:
                exit_price = price
                break
        else:
            if ret <= sl or ret >= tp:
                exit_price = price
                break
 
    raw_ret = np.log(exit_price / entry_price)
    return direction * raw_ret
 
 
def build_session_index(price_df: pd.DataFrame) -> dict:
    """Pre-index close prices by (Ticker, date) for fast lookup."""
    price_df = price_df.sort_values(["Ticker", "Date"]).copy()
    price_df["_date"] = price_df["Date"].dt.date
    idx = {}
    for (ticker, date), grp in price_df.groupby(["Ticker", "_date"], sort=False):
        grp = grp.sort_values("Date")
        idx[(ticker, date)] = {
            "closes": grp["Close"].values,
            "dates":  grp["Date"].values,
            "vols":   grp["rolling_std_60"].values,
        }
    return idx
 
 
def compute_actual_pnl(signal_df: pd.DataFrame,
                        session_idx: dict) -> np.ndarray:
    """
    For each signal row, simulate the actual trade and return net P&L.
    signal_df must have: Date, Ticker, signal (BUY/SELL label), entry_close, entry_vol
    """
    returns = []
    for _, row in signal_df.iterrows():
        ticker    = row["Ticker"]
        sig_date  = pd.Timestamp(row["Date"])
        date_only = sig_date.date()
        direction = 1 if row["signal"] == BUY_LABEL else -1
        entry     = row["entry_close"]
        vol       = row["entry_vol"]
 
        key = (ticker, date_only)
        if key not in session_idx:
            returns.append(np.nan)
            continue
 
        sess      = session_idx[key]
        bar_pos   = np.searchsorted(sess["dates"], np.datetime64(sig_date))
        fut_close = sess["closes"][bar_pos + 1: bar_pos + 1 + FORWARD_WINDOW]
 
        if len(fut_close) == 0:
            returns.append(np.nan)
            continue
 
        raw_ret = simulate_trade_exit(entry, fut_close, vol, direction)
        net_ret = raw_ret - COST
        returns.append(net_ret)
 
    return np.array(returns)
 
 
def build_meta_dataset(df: pd.DataFrame,
                        primary_model,
                        feature_cols: list,
                        price_df: pd.DataFrame,
                        session_idx: dict,
                        split_name: str):
    """
    Build meta-model input matrix and binary target based on actual P&L.
 
    Steps:
        1. Get primary model predictions for all bars
        2. Keep only BUY/SELL predictions
        3. Simulate actual barrier-exit trade for each signal
        4. y_meta = 1 if net_return > 0 else 0
        5. X_meta = [original features, primary proba, predicted class]
    """
    X      = df[feature_cols].values
    proba  = primary_model.predict_proba(X)
    pred   = primary_model.predict(X).astype(int)
 
    trade_mask = pred != HOLD_LABEL
    logger.info(
        f"[{split_name}] Trade signals: {trade_mask.sum():,}/{len(df):,} "
        f"({100*trade_mask.mean():.1f}%)"
    )
 
    if trade_mask.sum() == 0:
        raise ValueError("No trade signals from primary model.")
 
    # Build signal dataframe for P&L simulation
    signal_df = df[trade_mask][["Date", "Ticker"]].copy()
    signal_df["signal"] = pred[trade_mask]
 
    # Attach entry price and vol
    signal_df = signal_df.merge(
        price_df[["Date", "Ticker", "Close", "rolling_std_60"]]
                .rename(columns={"Close": "entry_close",
                                  "rolling_std_60": "entry_vol"}),
        on=["Date", "Ticker"],
        how="left",
    )
    signal_df.dropna(subset=["entry_close", "entry_vol"], inplace=True)
 
    logger.info(f"[{split_name}] Simulating {len(signal_df):,} trades for meta target ...")
    actual_returns = compute_actual_pnl(signal_df, session_idx)
 
    # Align with trade_mask (some rows may have been dropped in merge)
    valid = ~np.isnan(actual_returns)
    actual_returns = actual_returns[valid]
    signal_df      = signal_df[valid].reset_index(drop=True)
 
    # Meta target: 1 = profitable trade, 0 = losing trade
    y_meta = (actual_returns > 0).astype(int)
    prof_rate = y_meta.mean()
    logger.info(
        f"[{split_name}] Meta target profitable rate: {prof_rate:.3f} "
        f"({y_meta.sum():,} profitable / {len(y_meta):,} total)"
    )
 
    # Rebuild feature indices aligned to signal_df after merge+dropna
    merged_idx = signal_df.index if hasattr(signal_df, 'index') else range(len(signal_df))
 
    # Get trade rows from original df aligned to signal_df dates/tickers
    # Re-extract from primary model for the filtered rows
    trade_df = df[trade_mask].reset_index(drop=True)
    # Drop same rows as signal_df dropna did
    valid_mask_in_trade = np.zeros(trade_mask.sum(), dtype=bool)
    # Recompute valid based on merge — use position match
    trade_df_merged = trade_df.merge(
        price_df[["Date", "Ticker", "Close", "rolling_std_60"]],
        on=["Date", "Ticker"], how="left"
    )
    has_price = trade_df_merged["Close"].notna().values
 
    X_trade    = X[trade_mask][has_price]
    proba_tr   = proba[trade_mask][has_price]
    pred_tr    = pred[trade_mask][has_price].reshape(-1, 1)
 
    # Final valid mask after nan return removal
    X_trade    = X_trade[valid]
    proba_tr   = proba_tr[valid]
    pred_tr    = pred_tr[valid]
 
    X_meta = np.concatenate([X_trade, proba_tr, pred_tr], axis=1)
 
    return X_meta, y_meta
 
 
def position_size_from_confidence(confidence: float) -> float:
    t = META_CONFIDENCE_THRESHOLD
    return float(np.clip(
        MIN_POSITION_SIZE + (confidence - t) / (1.0 - t + 1e-9) *
        (MAX_POSITION_SIZE - MIN_POSITION_SIZE),
        MIN_POSITION_SIZE, MAX_POSITION_SIZE
    ))
 
 
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
 
    logger.info("Loading primary model ...")
    primary_model = joblib.load(LGBM_MODEL_PATH)
    feature_cols  = joblib.load(FEATURE_LIST_PATH)
 
    train_df = load_split(TRAIN_DIR, "train")
    val_df   = load_split(VAL_DIR,   "validation")
 
    logger.info("Loading price data for P&L simulation ...")
    price_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "all_tickers_labeled.parquet"),
        columns=["Date", "Ticker", "Close", "rolling_std_60"],
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df.sort_values(["Ticker", "Date"], inplace=True)
 
    logger.info("Building session index ...")
    session_idx = build_session_index(price_df)
 
    # ── Build meta datasets ────────────────────────────────────────────────────
    logger.info("\nBuilding meta-TRAIN dataset (actual P&L targets) ...")
    X_tr, y_tr = build_meta_dataset(
        train_df, primary_model, feature_cols,
        price_df, session_idx, "train"
    )
 
    logger.info("\nBuilding meta-VAL dataset (actual P&L targets) ...")
    X_va, y_va = build_meta_dataset(
        val_df, primary_model, feature_cols,
        price_df, session_idx, "validation"
    )
 
    # ── Train LightGBM meta-model ──────────────────────────────────────────────
    logger.info("\nTraining meta-model (LightGBM binary) ...")
    meta_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.01,       # slow learning = better generalisation
        num_leaves=16,            # small = less overfitting
        feature_fraction=0.6,     # aggressive dropout
        bagging_fraction=0.7,
        bagging_freq=5,
        min_child_samples=500,    # need 500 samples per leaf — prevents noise fitting
        reg_alpha=1.0,            # strong L1
        reg_lambda=1.0,           # strong L2
        device="cpu",
        random_state=RANDOM_SEED,
        verbosity=-1,
        is_unbalance=True,
    )
    meta_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(75, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
 
    # ── Validation report ──────────────────────────────────────────────────────
    val_preds = meta_model.predict(X_va)
    val_proba = meta_model.predict_proba(X_va)[:, 1]
    logger.info("\nMeta-model validation report:")
    logger.info(classification_report(
        y_va, val_preds,
        target_names=["Unprofitable", "Profitable"],
        digits=4,
    ))
 
    # Distribution of confidence scores
    conf_above = (val_proba >= META_CONFIDENCE_THRESHOLD).mean()
    logger.info(
        f"Signals above threshold ({META_CONFIDENCE_THRESHOLD}): "
        f"{conf_above*100:.1f}% of trade signals"
    )
 
    # Position sizing schedule
    logger.info("Position sizing schedule:")
    for conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]:
        if conf >= META_CONFIDENCE_THRESHOLD:
            logger.info(
                f"  confidence={conf:.2f}  ->  "
                f"position_size={position_size_from_confidence(conf):.3f}"
            )
 
    joblib.dump(meta_model, META_MODEL_PATH)
    logger.info(f"\nMeta-model saved -> {META_MODEL_PATH}")
 
 
if __name__ == "__main__":
    main()