"""
label_utils.py — Triple-barrier labeling utilities (vectorised).

Parameters (from config):
    BARRIER_MULTIPLIER = 3.0   (calibrated to give ~25% HOLD on SP500 1-min data)
    FORWARD_WINDOW     = 24    (24 bars forward, same day only)

Label encoding:
    BUY  = 2  (take-profit hit first)
    SELL = 0  (stop-loss hit first)
    HOLD = 1  (neither barrier hit within FORWARD_WINDOW bars)

Expected distribution: BUY ~37.8%  SELL ~37.4%  HOLD ~24.8%
"""

import numpy as np
import pandas as pd
import logging

from utils.config import BARRIER_MULTIPLIER, FORWARD_WINDOW, LABEL_MAP

logger = logging.getLogger(__name__)

NO_HIT   = FORWARD_WINDOW + 1
BUY_LBL  = LABEL_MAP["BUY"]
SELL_LBL = LABEL_MAP["SELL"]
HOLD_LBL = LABEL_MAP["HOLD"]


def _label_group_vectorized(closes: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Vectorised triple-barrier labeling for one (Ticker, day) group.

    Builds a (n, FORWARD_WINDOW) matrix of forward log-returns using numpy
    broadcasting, then uses argmax to find first barrier crossing in C.
    No Python loop over individual bars.
    """
    n      = len(closes)
    fw     = FORWARD_WINDOW
    labels = np.full(n, HOLD_LBL, dtype=np.int8)

    valid = ~np.isnan(stds) & (stds > 0) & ~np.isnan(closes)
    if not valid.any():
        return labels

    i_idx = np.arange(n, dtype=np.int32)[:, None]   # (n, 1)
    j_idx = np.arange(fw, dtype=np.int32)[None, :]   # (1, fw)
    dst   = i_idx + j_idx + 1                         # (n, fw) future bar indices

    in_bounds = dst < n
    dst_safe  = np.clip(dst, 0, n - 1)

    future_ret = np.where(
        in_bounds,
        np.log(closes[dst_safe] / closes[i_idx]),
        np.nan,
    ).astype(np.float32)

    tp_thresh = ( BARRIER_MULTIPLIER * stds)[:, None]  # (n, 1)
    sl_thresh = (-BARRIER_MULTIPLIER * stds)[:, None]

    not_nan = ~np.isnan(future_ret)
    tp_hit  = not_nan & (future_ret >= tp_thresh)
    sl_hit  = not_nan & (future_ret <= sl_thresh)

    tp_any   = tp_hit.any(axis=1)
    sl_any   = sl_hit.any(axis=1)
    tp_first = np.where(tp_any, np.argmax(tp_hit, axis=1), NO_HIT)
    sl_first = np.where(sl_any, np.argmax(sl_hit, axis=1), NO_HIT)

    buy_mask  = valid & tp_any & (tp_first < sl_first)
    sell_mask = valid & sl_any & (sl_first < tp_first)

    labels[buy_mask]  = BUY_LBL
    labels[sell_mask] = SELL_LBL
    return labels


def label_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign triple-barrier labels to every bar.
    Groups by (Ticker, date) to prevent cross-day look-ahead.
    Vectorised numpy inside each group — no Python bar loop.
    """
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["label"] = HOLD_LBL
    date_arr   = df["Date"].dt.date
    total      = df.groupby(["Ticker", date_arr]).ngroups
    processed  = 0

    for (ticker, _date), grp in df.groupby(["Ticker", date_arr], sort=False):
        idx    = grp.index
        closes = grp["Close"].values.astype(np.float64)
        stds   = grp["rolling_std_60"].values.astype(np.float64)
        df.loc[idx, "label"] = _label_group_vectorized(closes, stds)
        processed += 1
        if processed % 1000 == 0:
            logger.info(f"  Labeled {processed:,}/{total:,} (ticker-day) groups ...")

    df["label"] = df["label"].astype(np.int8)
    return df


def log_label_distribution(df: pd.DataFrame, split_name: str = "") -> None:
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    tag    = f"[{split_name}] " if split_name else ""
    logger.info(f"{tag}Label distribution (n={total:,}):")
    for lbl, cnt in counts.items():
        name = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(int(lbl), "?")
        logger.info(f"  {name} ({lbl}): {cnt:>8,}  ({100*cnt/total:.1f}%)")
