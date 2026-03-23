"""
script4_dataset_builder.py — Build flattened 60-bar lookback windows for LightGBM.

Window for bar t = scaled features of bars [t-60, t-1] (flattened).
Uses numpy.lib.stride_tricks.sliding_window_view — zero-copy, no bar loop.

Inputs:  processed/all_tickers_labeled.parquet
Outputs: processed/all_tickers_windowed.parquet
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import PROCESSED_DIR, LOOKBACK_WINDOW, RANDOM_SEED
from utils.feature_utils import FEATURE_COLUMNS

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script4.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

INPUT_PATH   = os.path.join(PROCESSED_DIR, "all_tickers_labeled.parquet")
OUTPUT_PATH  = os.path.join(PROCESSED_DIR, "all_tickers_windowed.parquet")
SCALED_COLS  = [f"{c}_scaled" for c in FEATURE_COLUMNS]
N_BASE_FEATS = len(SCALED_COLS)

np.random.seed(RANDOM_SEED)


def build_windows_for_group(feat_matrix, labels, dates, tickers):
    """
    Zero-copy sliding windows via stride tricks.
    Window i predicts bar i + LOOKBACK_WINDOW.
    """
    n = len(feat_matrix)
    if n <= LOOKBACK_WINDOW:
        return None

    # Shape: (n - LB + 1, N_BASE_FEATS, LB) -- zero copy
    views = sliding_window_view(feat_matrix, LOOKBACK_WINDOW, axis=0)
    valid = views[:n - LOOKBACK_WINDOW]                         # (n-LB, NF, LB)
    flat  = valid.transpose(0, 2, 1).reshape(len(valid), -1)   # (n-LB, LB*NF)

    return (
        flat,
        labels[LOOKBACK_WINDOW:],
        dates[LOOKBACK_WINDOW:],
        tickers[LOOKBACK_WINDOW:],
    )


def main():
    logger.info(f"Loading labeled data from {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded {len(df):,} rows.")

    missing = set(SCALED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Scaled columns missing -- did script2 run?")

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["_date"] = df["Date"].dt.date

    feat_arr   = df[SCALED_COLS].values.astype(np.float32)
    label_arr  = df["label"].values
    date_arr   = df["Date"].values
    ticker_arr = df["Ticker"].values

    groups = df.groupby(["Ticker", "_date"], sort=False).indices
    total  = len(groups)
    logger.info(f"Building windows for {total:,} (ticker, day) groups ...")

    flat_parts = []; label_parts = []; date_parts = []; ticker_parts = []
    skipped    = 0

    for idx, (key, row_idx) in enumerate(groups.items()):
        row_idx = np.sort(row_idx)
        result  = build_windows_for_group(
            feat_arr[row_idx], label_arr[row_idx],
            date_arr[row_idx], ticker_arr[row_idx],
        )
        if result is None:
            skipped += 1
            continue
        flat, lbl, dates, tickers = result
        flat_parts.append(flat); label_parts.append(lbl)
        date_parts.append(dates); ticker_parts.append(tickers)

        if (idx + 1) % 2000 == 0:
            logger.info(f"  Processed {idx+1:,}/{total:,} groups ...")

    logger.info(f"Skipped {skipped:,} groups (< {LOOKBACK_WINDOW} bars).")

    logger.info("Stacking all windows ...")
    X = np.concatenate(flat_parts,   axis=0)
    y = np.concatenate(label_parts,  axis=0)
    d = np.concatenate(date_parts,   axis=0)
    t = np.concatenate(ticker_parts, axis=0)

    feat_names = [
        f"{col}_lag{LOOKBACK_WINDOW - 1 - lag}"
        for lag in range(LOOKBACK_WINDOW)
        for col in SCALED_COLS
    ]

    windowed = pd.DataFrame(X, columns=feat_names)
    windowed["label"]  = y.astype(np.int8)
    windowed["Date"]   = d
    windowed["Ticker"] = t
    windowed.sort_values(["Ticker", "Date"], inplace=True)
    windowed.reset_index(drop=True, inplace=True)

    logger.info(
        f"Windowed: {len(windowed):,} samples | "
        f"{windowed.shape[1]} cols | "
        f"feature dim: {LOOKBACK_WINDOW * N_BASE_FEATS}"
    )
    windowed.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
