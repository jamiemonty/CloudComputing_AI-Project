"""
script2_feature_engineering.py — Compute all features and normalise.

LEAKAGE PREVENTION:
  * Scaler fitted ONLY on training rows (Date <= TRAIN_END).
  * Vol regime thresholds derived from training data only.
  * All rolling stats use .shift(1) inside feature_utils.

Inputs:  processed/all_tickers_clean.parquet
Outputs: processed/all_tickers_featured.parquet
         models/scaler.joblib
         models/vol_regime_thresholds.joblib
         models/feature_list.joblib
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    PROCESSED_DIR, MODEL_DIR, SCALER_PATH, FEATURE_LIST_PATH,
    TRAIN_END, RANDOM_SEED, VOL_LOW_PCT, VOL_HIGH_PCT,
)
from utils.feature_utils import build_all_features, FEATURE_COLUMNS

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script2.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

INPUT_PATH      = os.path.join(PROCESSED_DIR, "all_tickers_clean.parquet")
OUTPUT_PATH     = os.path.join(PROCESSED_DIR, "all_tickers_featured.parquet")
VOL_THRESH_PATH = os.path.join(MODEL_DIR, "vol_regime_thresholds.joblib")

np.random.seed(RANDOM_SEED)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info(f"Loading clean data from {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded {len(df):,} rows.")

    logger.info("Running feature engineering pipeline ...")
    df, _, _ = build_all_features(df, vol_low_pct=None, vol_high_pct=None)

    # Re-compute vol regime thresholds from training data only
    train_mask  = df["Date"] <= TRAIN_END
    train_ratio = df.loc[train_mask, "volatility_ratio"].dropna()
    vol_low     = float(np.nanpercentile(train_ratio, VOL_LOW_PCT))
    vol_high    = float(np.nanpercentile(train_ratio, VOL_HIGH_PCT))
    logger.info(f"Vol regime thresholds (train only) -- low: {vol_low:.6f}, high: {vol_high:.6f}")

    r = df["volatility_ratio"].values
    df["vol_regime"] = np.where(np.isnan(r), np.nan,
                        np.where(r <= vol_low, 0.0,
                         np.where(r <= vol_high, 1.0, 2.0)))

    joblib.dump({"low": vol_low, "high": vol_high}, VOL_THRESH_PATH)

    # Drop NaN feature rows (rolling warm-up)
    before = len(df)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    logger.info(f"Dropped {before - len(df):,} NaN rows from rolling warm-up.")

    # Fit scaler on training rows only
    logger.info("Fitting StandardScaler on training data ...")
    train_feats = df.loc[df["Date"] <= TRAIN_END, FEATURE_COLUMNS]
    scaler      = StandardScaler()
    scaler.fit(train_feats)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved -> {SCALER_PATH}")

    # Apply to all rows
    scaled     = scaler.transform(df[FEATURE_COLUMNS].values)
    scaled_cols = [f"{c}_scaled" for c in FEATURE_COLUMNS]
    df[scaled_cols] = scaled

    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    logger.info(f"Feature list saved ({len(FEATURE_COLUMNS)} features) -> {FEATURE_LIST_PATH}")

    df.reset_index(drop=True, inplace=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved -> {OUTPUT_PATH}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
