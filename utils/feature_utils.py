"""
feature_utils.py — Feature engineering utilities (optimised, vectorised).
 
LEAKAGE RULES (enforced throughout):
  * All rolling stats use .shift(1) so bar t only sees history up to t-1.
  * Cross-day contamination is removed by nulling any bar where the within-day
    position (_bar_idx) is less than the window size.
  * Log returns null out the first bar of every trading day.
 
PERFORMANCE:
  * No groupby(Ticker, date).apply() — uses single-level groupby(Ticker).rolling()
    which pandas executes in optimised C code (28 passes vs ~35,000).
  * Time features use fully vectorised dt accessors.
  * Market features use a single groupby("Date") aggregate + vectorised shift.
"""
 
import numpy as np
import pandas as pd
import logging
 
from utils.config import (
    MOMENTUM_WINDOWS, ROLLING_MEAN_WINDOWS, ROLLING_STD_WINDOWS,
    VOL_LOW_PCT, VOL_HIGH_PCT,
)
 
logger = logging.getLogger(__name__)
 
 
def _bar_idx(df: pd.DataFrame) -> pd.Series:
    """Within-day bar position (0-indexed). df must be sorted by [Ticker, Date]."""
    return df.groupby(["Ticker", df["Date"].dt.date]).cumcount()
 
 
def _rolling_then_shift(df: pd.DataFrame, col: str, window: int, agg: str) -> pd.Series:
    """
    Single-level groupby(Ticker) rolling + shift(1) within each ticker.
    Cross-day masking is handled by the caller via _bar_idx, not by date grouping.
    """
    grouped = df.groupby("Ticker", sort=False)[col]
 
    if agg == "sum":
        rolled = grouped.rolling(window, min_periods=window).sum()
    elif agg == "mean":
        rolled = grouped.rolling(window, min_periods=window).mean()
    elif agg == "std":
        rolled = grouped.rolling(window, min_periods=window).std()
    else:
        raise ValueError(f"Unknown agg: {agg}")
 
    rolled = rolled.reset_index(level=0, drop=True)
    return rolled.groupby(df["Ticker"]).shift(1)
 
 
def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    log_return_t = log(close_t / close_{t-1}), nulled at day boundaries.
    """
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    prev_close = df.groupby("Ticker", sort=False)["Close"].shift(1)
    df["log_return"] = np.log(df["Close"] / prev_close)
    first_bar = _bar_idx(df) == 0
    df.loc[first_bar, "log_return"] = np.nan
    return df
 
 
def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """return_Nm = rolling_sum(log_return, N).shift(1), nulled at cross-day positions."""
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    bidx = _bar_idx(df)
    for w in MOMENTUM_WINDOWS:
        col = f"return_{w}m"
        df[col] = _rolling_then_shift(df, "log_return", w, "sum")
        df.loc[bidx < w, col] = np.nan
    return df
 
 
def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std — ALL shift(1), cross-day positions nulled."""
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    bidx = _bar_idx(df)
 
    for w in ROLLING_MEAN_WINDOWS:
        col = f"rolling_mean_{w}"
        df[col] = _rolling_then_shift(df, "log_return", w, "mean")
        df.loc[bidx < w, col] = np.nan
 
    for w in ROLLING_STD_WINDOWS:
        col = f"rolling_std_{w}"
        df[col] = _rolling_then_shift(df, "log_return", w, "std")
        df.loc[bidx < w, col] = np.nan
 
    return df
 
 
def compute_volatility_regime(df: pd.DataFrame, low_pct=None, high_pct=None):
    """volatility_ratio = std_15 / std_120, bucketed into 0/1/2 (low/normal/high)."""
    df = df.copy()
    df["volatility_ratio"] = (
        df["rolling_std_15"] / df["rolling_std_120"].replace(0, np.nan)
    )
    ratio = df["volatility_ratio"].dropna().values
    if low_pct is None:
        low_pct = float(np.nanpercentile(ratio, VOL_LOW_PCT))
    if high_pct is None:
        high_pct = float(np.nanpercentile(ratio, VOL_HIGH_PCT))
 
    r = df["volatility_ratio"].values
    regime = np.where(np.isnan(r), np.nan,
               np.where(r <= low_pct, 0.0,
                np.where(r <= high_pct, 1.0, 2.0)))
    df["vol_regime"] = regime
    return df, low_pct, high_pct
 
 
def compute_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """distance_from_60m_mean and zscore_60 — uses already-shifted rolling stats."""
    df = df.copy()
    shifted_lr = df.groupby("Ticker", sort=False)["log_return"].shift(1)
    df["distance_from_60m_mean"] = shifted_lr - df["rolling_mean_60"]
    df["zscore_60"] = (
        df["distance_from_60m_mean"] / df["rolling_std_60"].replace(0, np.nan)
    )
    return df
 
 
def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time-of-day encoding — fully vectorised."""
    df = df.copy()
    df["minute_of_day"] = (
        df["Date"].dt.hour * 60 + df["Date"].dt.minute - (9 * 60 + 30)
    )
    phase = 2 * np.pi * df["minute_of_day"] / 390.0
    df["sin_time"] = np.sin(phase)
    df["cos_time"] = np.cos(phase)
    return df
 
 
def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-ticker market return and volatility, shifted by 1 timestamp."""
    df = df.copy()
    mkt = (
        df.groupby("Date", sort=True)["log_return"]
          .agg(market_return_raw="mean", market_volatility_raw="std")
          .reset_index()
          .sort_values("Date")
    )
    mkt["market_return"]     = mkt["market_return_raw"].shift(1)
    mkt["market_volatility"] = mkt["market_volatility_raw"].shift(1)
    mkt.drop(columns=["market_return_raw", "market_volatility_raw"], inplace=True)
    df = df.merge(mkt, on="Date", how="left")
    return df
 
 
def build_all_features(df: pd.DataFrame, vol_low_pct=None, vol_high_pct=None):
    """Run all feature steps in order. Returns (featured_df, vol_low_pct, vol_high_pct)."""
    logger.info("Computing log returns ...")
    df = compute_log_returns(df)
 
    logger.info("Computing momentum features ...")
    df = compute_momentum_features(df)
 
    logger.info("Computing rolling stats (with shift(1)) ...")
    df = compute_rolling_stats(df)
 
    logger.info("Computing volatility regime ...")
    df, vol_low_pct, vol_high_pct = compute_volatility_regime(
        df, low_pct=vol_low_pct, high_pct=vol_high_pct
    )
 
    logger.info("Computing mean-reversion features ...")
    df = compute_mean_reversion_features(df)
 
    logger.info("Computing time features ...")
    df = compute_time_features(df)
 
    logger.info("Computing cross-ticker market features ...")
    df = compute_market_features(df)
 
    logger.info("Computing volume and High/Low features ...")
    df = compute_volume_and_hl_features(df)
 
    logger.info("Feature engineering complete.")
    return df, vol_low_pct, vol_high_pct
 
 
FEATURE_COLUMNS = (
    [f"return_{w}m"        for w in MOMENTUM_WINDOWS]
    + [f"rolling_mean_{w}" for w in ROLLING_MEAN_WINDOWS]
    + [f"rolling_std_{w}"  for w in ROLLING_STD_WINDOWS]
    + ["volatility_ratio", "vol_regime",
       "distance_from_60m_mean", "zscore_60",
       "minute_of_day", "sin_time", "cos_time",
       "market_return", "market_volatility",
       # Volume and High/Low features
       "volume_ratio", "volume_trend",
       "price_position", "hl_range", "hl_range_ratio",
       "vwap_dist", "rolling_vwap_dist"]
)
 
 
# ── Volume and High/Low features (high-signal additions) ──────────────────────
 
def compute_volume_and_hl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume and High/Low derived features.
 
    These require the raw OHLV columns to still be present in df.
    All rolling stats use shift(1) to prevent leakage.
 
    Features added:
        volume_ratio      : volume / rolling_mean_volume_20 — is volume spiking?
        volume_trend      : rolling_mean_volume_5 / rolling_mean_volume_20 — momentum
        price_position    : (Close - Low) / (High - Low) — where in bar range?
        hl_range          : (High - Low) / Close — normalised bar width
        hl_range_ratio    : hl_range / rolling_mean_hl_range_20 — range vs recent avg
        vwap_dist         : (Close - vwap) / Close — distance from typical price
        rolling_vwap_dist : 15-bar rolling mean of vwap_dist (shifted)
    """
    required = {"High", "Low", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        logger.warning(f"Cannot compute volume/HL features — missing columns: {missing}")
        return df
 
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    bidx = _bar_idx(df)
 
    # ── Per-bar features (no rolling, no leakage risk) ────────────────────────
    hl        = df["High"] - df["Low"]
    hl_safe   = hl.replace(0, np.nan)
    df["price_position"] = (df["Close"] - df["Low"]) / hl_safe   # 0=bottom, 1=top
    df["hl_range"]       = hl_safe / df["Close"]                  # normalised width
    df["vwap"]           = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["vwap_dist"]      = (df["Close"] - df["vwap"]) / df["Close"]
 
    # ── Rolling volume features — shift(1) enforced ───────────────────────────
    log_vol = np.log1p(df["Volume"].clip(lower=0))
    df["_log_vol"] = log_vol
 
    for w, col in [(5, "_vol_mean_5"), (20, "_vol_mean_20")]:
        rolled = (
            df.groupby("Ticker", sort=False)["_log_vol"]
              .rolling(w, min_periods=w).mean()
              .reset_index(level=0, drop=True)
        )
        df[col] = rolled.groupby(df["Ticker"]).shift(1)
        df.loc[bidx < w, col] = np.nan
 
    df["volume_ratio"] = df["_vol_mean_5"] / df["_vol_mean_20"].replace(0, np.nan)
    df["volume_trend"] = df["_vol_mean_5"] - df["_vol_mean_20"]  # log-space difference
 
    # ── Rolling HL range — shift(1) enforced ──────────────────────────────────
    rolled_hl = (
        df.groupby("Ticker", sort=False)["hl_range"]
          .rolling(20, min_periods=20).mean()
          .reset_index(level=0, drop=True)
    )
    df["_hl_mean_20"] = rolled_hl.groupby(df["Ticker"]).shift(1)
    df.loc[bidx < 20, "_hl_mean_20"] = np.nan
    df["hl_range_ratio"] = df["hl_range"] / df["_hl_mean_20"].replace(0, np.nan)
 
    # ── Rolling VWAP distance — shift(1) enforced ─────────────────────────────
    rolled_vd = (
        df.groupby("Ticker", sort=False)["vwap_dist"]
          .rolling(15, min_periods=15).mean()
          .reset_index(level=0, drop=True)
    )
    df["rolling_vwap_dist"] = rolled_vd.groupby(df["Ticker"]).shift(1)
    df.loc[bidx < 15, "rolling_vwap_dist"] = np.nan
 
    # Drop helper columns
    df.drop(columns=["_log_vol", "_vol_mean_5", "_vol_mean_20",
                      "_hl_mean_20", "vwap"], inplace=True)
 
    logger.info("Volume and High/Low features computed.")
    return df