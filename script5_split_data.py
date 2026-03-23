"""
script5_split_data.py — Time-based train / validation / test split.

Train:      2020-12-28 -> 2023-12-31
Validation: 2024-01-01 -> 2024-12-31
Test:       2025-01-01 -> 2025-12-23

Inputs:  processed/all_tickers_windowed.parquet
Outputs: processed/train/train.parquet
         processed/validation/validation.parquet
         processed/test/test.parquet
"""

import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)
from utils.label_utils import log_label_distribution

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script5.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
INPUT_PATH = os.path.join(PROCESSED_DIR, "all_tickers_windowed.parquet")


def main():
    logger.info(f"Loading windowed dataset from {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info(f"Total samples: {len(df):,}")

    splits = {
        "train":      (TRAIN_START, TRAIN_END,  TRAIN_DIR),
        "validation": (VAL_START,   VAL_END,    VAL_DIR),
        "test":       (TEST_START,  TEST_END,   TEST_DIR),
    }

    for name, (start, end, out_dir) in splits.items():
        os.makedirs(out_dir, exist_ok=True)
        mask   = (df["Date"] >= start) & (df["Date"] <= end)
        subset = df[mask].copy().reset_index(drop=True)
        path   = os.path.join(out_dir, f"{name}.parquet")
        subset.to_parquet(path, index=False)
        logger.info(
            f"[{name.upper():>10}] {len(subset):>9,} samples | "
            f"{subset['Date'].min().date()} to {subset['Date'].max().date()} | "
            f"Tickers: {subset['Ticker'].nunique()}"
        )
        log_label_distribution(subset, split_name=name)

    logger.info("Data split complete.")


if __name__ == "__main__":
    main()
