"""
script3_labeling.py — Apply triple-barrier labeling.

Calibrated parameters (from data analysis):
    BARRIER_MULTIPLIER = 3.0
    FORWARD_WINDOW     = 24
Expected distribution: BUY~37.8%  SELL~37.4%  HOLD~24.8%

Inputs:  processed/all_tickers_featured.parquet
Outputs: processed/all_tickers_labeled.parquet
"""

import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import PROCESSED_DIR, BARRIER_MULTIPLIER, FORWARD_WINDOW
from utils.label_utils import label_triple_barrier, log_label_distribution

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script3.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

INPUT_PATH  = os.path.join(PROCESSED_DIR, "all_tickers_featured.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "all_tickers_labeled.parquet")


def main():
    logger.info(f"Loading featured data from {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers.")
    logger.info(f"Barrier: mult={BARRIER_MULTIPLIER}, forward_window={FORWARD_WINDOW} bars")

    missing = {"Date", "Close", "Ticker", "rolling_std_60"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    logger.info("Running triple-barrier labeling ...")
    df = label_triple_barrier(df)
    log_label_distribution(df, split_name="all")

    df.reset_index(drop=True, inplace=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved -> {OUTPUT_PATH}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
