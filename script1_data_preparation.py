"""
script1_data_preparation.py — Load, validate, and clean raw 1-minute CSV data.

Outputs: processed/all_tickers_clean.parquet
"""

import os
import sys
import glob
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    RAW_DATA_DIR, PROCESSED_DIR,
    MARKET_OPEN, MARKET_CLOSE,
    TRAIN_START, TEST_END,
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script1.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "all_tickers_clean.parquet")


def discover_files(raw_dir: str) -> list:
    pattern = os.path.join(raw_dir, "*_1min.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No *_1min.csv files found in {raw_dir}.")
    logger.info(f"Found {len(files)} ticker file(s).")
    return files


def load_single_file(path: str):
    ticker = os.path.basename(path).replace("_1min.csv", "").upper()
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception as e:
        logger.warning(f"[{ticker}] Failed to read: {e}")
        return None

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.warning(f"[{ticker}] Missing columns {missing} -- skipping.")
        return None

    df["Ticker"] = ticker
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]].copy()

    before = len(df)
    df.dropna(subset=["Date", "Close"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"[{ticker}] Dropped {dropped} rows with null Date/Close.")

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"[{ticker}] Loaded {len(df):,} rows.")
    return df


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    time    = df["Date"].dt.time
    open_t  = pd.Timestamp(f"1970-01-01 {MARKET_OPEN}").time()
    close_t = pd.Timestamp(f"1970-01-01 {MARKET_CLOSE}").time()
    mask_hours = (time >= open_t) & (time <= close_t)
    mask_dates = (df["Date"] >= TRAIN_START) & (df["Date"] <= TEST_END)
    return df[mask_hours & mask_dates].copy()


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    dfs = []
    for path in discover_files(RAW_DATA_DIR):
        df = load_single_file(path)
        if df is not None:
            df = filter_market_hours(df)
            if len(df):
                dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid data loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["Ticker", "Date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        f"Combined: {len(combined):,} rows | "
        f"{combined['Ticker'].nunique()} tickers | "
        f"{combined['Date'].min()} to {combined['Date'].max()}"
    )
    combined.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
