"""
script9_backtest.py — Backtest + full HTML dashboard.
 
Zero transaction costs — gross performance only.
HTML includes:
  - KPI strips (incl. avg gross PnL per trade)
  - Equity curves + drawdown
  - Return distributions + monthly returns
  - Model comparison bar charts
  - Label & signal analysis
  - Walk-forward accuracy per fold
  - Full metrics tables
"""
 
import os, sys, json, logging
import numpy as np
import pandas as pd
import joblib
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    TEST_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR,
    LGBM_MODEL_PATH, META_MODEL_PATH, FEATURE_LIST_PATH,
    META_CONFIDENCE_THRESHOLD, TRADING_COST_BPS,
    MIN_POSITION_SIZE, MAX_POSITION_SIZE,
    BARRIER_MULTIPLIER, FORWARD_WINDOW,
    LABEL_MAP, WF_RESULTS_PATH,
)
 
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script9.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
 
HOLD_LABEL = LABEL_MAP["HOLD"]
BUY_LABEL  = LABEL_MAP["BUY"]
SELL_LABEL = LABEL_MAP["SELL"]
COST       = TRADING_COST_BPS * 1e-4
 
 
# ── Position sizing ────────────────────────────────────────────────────────────
 
def confidence_to_position_size(confidence):
    t = META_CONFIDENCE_THRESHOLD
    return np.clip(
        MIN_POSITION_SIZE + (confidence - t) / (1.0 - t + 1e-9) *
        (MAX_POSITION_SIZE - MIN_POSITION_SIZE),
        MIN_POSITION_SIZE, MAX_POSITION_SIZE
    )
 
 
# ── Barrier exit simulation ────────────────────────────────────────────────────
 
def simulate_trade_exit(entry_price, future_closes, vol, direction):
    if len(future_closes) == 0 or vol <= 0 or np.isnan(vol):
        return 0.0
    tp = BARRIER_MULTIPLIER * vol
    sl = -BARRIER_MULTIPLIER * vol
    exit_price = future_closes[-1]
    for price in future_closes:
        if np.isnan(price): break
        ret = np.log(price / entry_price)
        if direction == 1:
            if ret >= tp or ret <= sl: exit_price = price; break
        else:
            if ret <= sl or ret >= tp: exit_price = price; break
    return direction * np.log(exit_price / entry_price)
 
 
# ── Signal generation ──────────────────────────────────────────────────────────
 
def generate_signals_primary(test_df, primary_model, feature_cols):
    X    = test_df[feature_cols].values
    pred = primary_model.predict(X).astype(int)
    return pd.Series(pred, index=test_df.index), np.ones(len(pred), dtype=np.float32), pred
 
 
def generate_signals_meta(test_df, primary_model, meta_model, feature_cols):
    X             = test_df[feature_cols].values
    proba_primary = primary_model.predict_proba(X)
    pred_primary  = primary_model.predict(X).astype(int)
    signals  = pd.Series(HOLD_LABEL, index=test_df.index)
    sizes    = pd.Series(0.0,        index=test_df.index)
    trade_mask = pred_primary != HOLD_LABEL
    if trade_mask.sum() == 0:
        return signals, sizes.values, pred_primary
    X_meta    = np.concatenate([X[trade_mask], proba_primary[trade_mask],
                                  pred_primary[trade_mask].reshape(-1,1)], axis=1)
    meta_proba = meta_model.predict_proba(X_meta)[:, 1]
    confident  = meta_proba >= META_CONFIDENCE_THRESHOLD
    pos_sizes  = confidence_to_position_size(meta_proba[confident])
    trade_idx  = test_df.index[trade_mask]
    conf_idx   = trade_idx[confident]
    signals.loc[conf_idx] = pred_primary[trade_mask][confident]
    sizes.loc[conf_idx]   = pos_sizes
    logger.info(f"Meta-filtered: {confident.sum():,}/{trade_mask.sum():,} trades kept "
                f"({100*confident.mean():.1f}%) | avg size={pos_sizes.mean():.3f}")
    return signals, sizes.values, pred_primary
 
 
# ── Trade simulation ───────────────────────────────────────────────────────────
 
def simulate_trades(test_df, signals, sizes, price_df):
    trade_mask = signals != HOLD_LABEL
    if trade_mask.sum() == 0:
        return np.array([]), np.array([]), pd.DataFrame()
 
    signal_df = test_df[trade_mask][["Date","Ticker"]].copy()
    signal_df["signal"]   = signals[trade_mask].values
    signal_df["pos_size"] = sizes[trade_mask]
    signal_df = signal_df.merge(
        price_df[["Date","Ticker","Close","rolling_std_60"]]
                .rename(columns={"Close":"entry_close","rolling_std_60":"entry_vol"}),
        on=["Date","Ticker"], how="left"
    )
    signal_df.dropna(subset=["entry_close","entry_vol"], inplace=True)
 
    price_df2 = price_df.sort_values(["Ticker","Date"]).copy()
    price_df2["_date"] = price_df2["Date"].dt.date
    sess_closes = {}; sess_dates = {}
    for (ticker, date), grp in price_df2.groupby(["Ticker","_date"], sort=False):
        grp = grp.sort_values("Date")
        sess_closes[(ticker,date)] = grp["Close"].values
        sess_dates[(ticker,date)]  = grp["Date"].values
 
    logger.info(f"Simulating {len(signal_df):,} trades ...")
    returns=[]; pos_sizes=[]; directions=[]; dates_out=[]; gross_returns=[]
 
    for _, row in signal_df.iterrows():
        ticker=row["Ticker"]; sig_date=pd.Timestamp(row["Date"])
        date_only=sig_date.date(); direction=1 if row["signal"]==BUY_LABEL else -1
        entry=row["entry_close"]; vol=row["entry_vol"]; pos_size=row["pos_size"]
        key=(ticker,date_only)
        if key not in sess_closes: continue
        bar_pos  = np.searchsorted(sess_dates[key], np.datetime64(sig_date))
        fut_close= sess_closes[key][bar_pos+1: bar_pos+1+FORWARD_WINDOW]
        if len(fut_close)==0: continue
        raw_ret = simulate_trade_exit(entry, fut_close, vol, direction)
        gross_returns.append(raw_ret * pos_size)
        net_ret = raw_ret * pos_size - COST
        returns.append(net_ret); pos_sizes.append(pos_size)
        directions.append(direction); dates_out.append(sig_date)
 
    td = pd.DataFrame({
        "date": dates_out, "direction": directions,
        "return": returns, "gross_return": gross_returns, "pos_size": pos_sizes,
    })
    return np.array(returns), np.array(pos_sizes), td
 
 
# ── Metrics ────────────────────────────────────────────────────────────────────
 
def compute_metrics(returns, pos_sizes=None, gross_returns=None):
    n = len(returns)
    if n == 0: return {}
    wins         = returns[returns > 0]
    losses       = returns[returns < 0]
    win_rate     = len(wins) / n
    avg_ret      = float(np.mean(returns))
    gross_profit = float(wins.sum())    if len(wins)   > 0 else 0.0
    gross_loss   = float(-losses.sum()) if len(losses) > 0 else 0.0
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    cum          = np.exp(np.cumsum(returns))
    peak         = np.maximum.accumulate(cum)
    drawdown     = (cum - peak) / peak
    max_dd       = float(drawdown.min())
    bars_per_yr  = 390 * 252
    sharpe       = float((avg_ret/returns.std())*np.sqrt(bars_per_yr)) if returns.std()>0 else 0.0
    calmar       = abs(float(cum[-1]-1)/max_dd) if max_dd!=0 else 0.0
    avg_win      = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss     = float(losses.mean()) if len(losses) > 0 else 0.0
    expectancy   = win_rate*avg_win + (1-win_rate)*avg_loss
    avg_size     = float(np.mean(pos_sizes)) if pos_sizes is not None else 1.0
    avg_gross    = float(np.mean(gross_returns)) if gross_returns is not None and len(gross_returns)>0 else avg_ret
    return {
        "number_of_trades":   n,
        "win_rate":           win_rate,
        "loss_rate":          1-win_rate,
        "average_return":     avg_ret,
        "avg_gross_pnl":      avg_gross,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "expectancy":         expectancy,
        "profit_factor":      pf,
        "gross_profit":       gross_profit,
        "gross_loss":         gross_loss,
        "max_drawdown":       max_dd,
        "sharpe_ratio":       sharpe,
        "calmar_ratio":       calmar,
        "total_return":       float(cum[-1]-1),
        "avg_position_size":  avg_size,
        "_cum_equity":        cum.tolist(),
        "_drawdown":          drawdown.tolist(),
        "_returns":           returns.tolist(),
    }
 
 
def print_metrics(metrics, label=""):
    tag = f"[{label}] " if label else ""
    logger.info(f"\n{tag}Backtest Metrics:")
    for k in ["number_of_trades","win_rate","avg_gross_pnl","average_return",
              "profit_factor","max_drawdown","sharpe_ratio","total_return"]:
        v = metrics.get(k, float("nan"))
        logger.info(f"  {k:<22} : {v:.6f}" if isinstance(v,float) else f"  {k:<22} : {v:,}")
 
 
# ── Label & signal analysis ────────────────────────────────────────────────────
 
def compute_signal_analysis(test_df, pred_primary, pred_meta_signals,
                              feature_cols, primary_model):
    """
    Returns dicts describing:
      - True label distribution in test set
      - Primary model prediction distribution
      - Meta model signal distribution
      - Per-class precision proxy (predicted vs true)
    """
    y_true = test_df["label"].values.astype(int) if "label" in test_df.columns else None
 
    # True label counts
    label_names = {0:"SELL", 1:"HOLD", 2:"BUY"}
    true_counts = {}
    if y_true is not None:
        for k,v in label_names.items():
            true_counts[v] = int((y_true==k).sum())
 
    # Primary prediction counts
    primary_counts = {}
    for k,v in label_names.items():
        primary_counts[v] = int((pred_primary==k).sum())
 
    # Meta signal counts (only BUY/SELL, HOLD = filtered out)
    meta_arr = pred_meta_signals.values if hasattr(pred_meta_signals,'values') else pred_meta_signals
    meta_counts = {
        "SELL": int((meta_arr==SELL_LABEL).sum()),
        "HOLD": int((meta_arr==HOLD_LABEL).sum()),
        "BUY":  int((meta_arr==BUY_LABEL).sum()),
    }
 
    # Per-class accuracy if labels available
    per_class_acc = {}
    if y_true is not None:
        for k,v in label_names.items():
            mask = y_true == k
            if mask.sum() > 0:
                per_class_acc[v] = float((pred_primary[mask]==k).sum() / mask.sum())
 
    return {
        "true_counts":    true_counts,
        "primary_counts": primary_counts,
        "meta_counts":    meta_counts,
        "per_class_acc":  per_class_acc,
    }
 
 
# ── HTML Report ────────────────────────────────────────────────────────────────
 
def generate_html(results, trade_details, signal_analysis, wf_results):
    p = results.get("primary", {})
    m = results.get("meta",    {})
 
    def subsample(arr, n=500):
        if len(arr) <= n: return arr
        idx = np.linspace(0, len(arr)-1, n, dtype=int)
        return [arr[i] for i in idx]
 
    p_equity = subsample(p.get("_cum_equity",[1]))
    m_equity = subsample(m.get("_cum_equity",[1]))
    p_dd     = subsample(p.get("_drawdown",  [0]))
    m_dd     = subsample(m.get("_drawdown",  [0]))
 
    def hist_data(rets, bins=40):
        if len(rets)==0: return [],[]
        arr=np.array(rets); counts,edges=np.histogram(arr,bins=bins)
        return ((edges[:-1]+edges[1:])/2*10000).round(4).tolist(), counts.tolist()
 
    p_hx,p_hy = hist_data(p.get("_returns",[]))
    m_hx,m_hy = hist_data(m.get("_returns",[]))
 
    # Monthly returns
    monthly_returns={}
    if trade_details.get("meta") is not None and len(trade_details["meta"])>0:
        td=trade_details["meta"].copy()
        td["month"]=pd.to_datetime(td["date"]).dt.to_period("M").astype(str)
        monthly_returns=td.groupby("month")["return"].sum().to_dict()
    months   = list(monthly_returns.keys())
    mret_val = [round(v*100,4) for v in monthly_returns.values()]
    mret_col = ["rgba(0,255,136,0.75)" if v>=0 else "rgba(255,75,75,0.75)" for v in mret_val]
 
    # Walk-forward data
    wf_folds=[]; wf_f1=[]; wf_iters=[]
    if wf_results:
        for f in wf_results.get("folds",[]):
            wf_folds.append(f"Fold {f['fold']}<br><small>{f['val_start'][:7]} to {f['val_end'][:7]}</small>")
            wf_f1.append(round(f["macro_f1"],4))
            wf_iters.append(f.get("best_iter",0))
    wf_avg  = round(wf_results.get("avg_f1",0),4) if wf_results else 0
    wf_std  = round(wf_results.get("std_f1",0),4) if wf_results else 0
 
    # Signal analysis data
    sa = signal_analysis or {}
    tc = sa.get("true_counts",    {"SELL":0,"HOLD":0,"BUY":0})
    pc = sa.get("primary_counts", {"SELL":0,"HOLD":0,"BUY":0})
    mc = sa.get("meta_counts",    {"SELL":0,"HOLD":0,"BUY":0})
    pca= sa.get("per_class_acc",  {"SELL":0,"HOLD":0,"BUY":0})
 
    true_vals    = [tc.get("SELL",0), tc.get("HOLD",0), tc.get("BUY",0)]
    primary_vals = [pc.get("SELL",0), pc.get("HOLD",0), pc.get("BUY",0)]
    meta_vals    = [mc.get("SELL",0), mc.get("HOLD",0), mc.get("BUY",0)]
    pca_vals     = [round(pca.get("SELL",0)*100,2),
                    round(pca.get("HOLD",0)*100,2),
                    round(pca.get("BUY", 0)*100,2)]
 
    def fmt(v, pct=False, bps=False):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "N/A"
        if pct:  return f"{v*100:.2f}%"
        if bps:  return f"{v*10000:.4f} bps"
        if isinstance(v,int): return f"{v:,}"
        return f"{v:.4f}"
 
    win_data   = [round(p.get("win_rate",0)*100,2), round(m.get("win_rate",0)*100,2)]
    pf_data    = [round(min(p.get("profit_factor",0),5),4), round(min(m.get("profit_factor",0),5),4)]
    sharpe_data= [round(p.get("sharpe_ratio",0),4), round(m.get("sharpe_ratio",0),4)]
 
    # Colour helpers
    def vc(v, good=lambda x:x>=0):
        return "val-up" if good(v) else "val-down"
 
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Trading System — Backtest Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#080c10;--surface:#0d1219;--border:#1a2535;
  --accent:#00ff88;--accent2:#0099ff;--danger:#ff4b4b;--warn:#ffaa00;
  --text:#e2e8f0;--muted:#64748b;--up:#00ff88;--down:#ff4b4b;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;font-size:13px;line-height:1.6;min-height:100vh;}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,0.02) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}}
.container{{max-width:1400px;margin:0 auto;padding:40px 24px;position:relative;z-index:1;}}
.header{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:48px;border-bottom:1px solid var(--border);padding-bottom:24px;}}
.header-left h1{{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;letter-spacing:-1px;color:var(--accent);text-shadow:0 0 40px rgba(0,255,136,0.3);}}
.header-left p{{color:var(--muted);font-size:12px;margin-top:4px;}}
.badge{{background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);color:var(--accent);padding:6px 14px;border-radius:4px;font-size:11px;letter-spacing:1px;text-transform:uppercase;}}
.badge.warn{{background:rgba(255,170,0,0.1);border-color:rgba(255,170,0,0.3);color:var(--warn);}}
.kpi-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin-bottom:2px;}}
.kpi-strip+.kpi-strip{{margin-bottom:32px;}}
.kpi{{background:var(--surface);border:1px solid var(--border);padding:20px 24px;position:relative;overflow:hidden;transition:border-color 0.2s;}}
.kpi:hover{{border-color:var(--accent);}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent-bar,var(--accent));}}
.kpi.danger::before{{--accent-bar:var(--danger);}}
.kpi.warn::before{{--accent-bar:var(--warn);}}
.kpi.blue::before{{--accent-bar:var(--accent2);}}
.kpi-label{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:8px;}}
.kpi-value{{font-family:'Syne',sans-serif;font-size:28px;font-weight:700;}}
.kpi-value.up{{color:var(--up);}} .kpi-value.down{{color:var(--down);}} .kpi-value.neutral{{color:var(--text);}}
.kpi-sub{{font-size:11px;color:var(--muted);margin-top:4px;}}
.section-label{{font-size:10px;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin:40px 0 16px;display:flex;align-items:center;gap:12px;}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border);}}
.charts-2{{display:grid;grid-template-columns:1fr 1fr;gap:2px;margin-bottom:2px;}}
.charts-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;margin-bottom:2px;}}
.chart-box{{background:var(--surface);border:1px solid var(--border);padding:24px;}}
.chart-box.full{{grid-column:1/-1;}}
.chart-title{{font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;}}
.chart-sub{{font-size:11px;color:var(--muted);margin-bottom:20px;}}
.chart-wrap{{position:relative;height:220px;}}
.chart-wrap.tall{{height:300px;}}
.chart-wrap.short{{height:160px;}}
.metrics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:2px;margin-bottom:32px;}}
.metrics-table{{background:var(--surface);border:1px solid var(--border);overflow:hidden;}}
.metrics-table h3{{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;padding:16px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;}}
.dot{{width:8px;height:8px;border-radius:50%;display:inline-block;}}
.dot.green{{background:var(--accent);box-shadow:0 0 6px var(--accent);}}
.dot.blue{{background:var(--accent2);box-shadow:0 0 6px var(--accent2);}}
.dot.yellow{{background:var(--warn);box-shadow:0 0 6px var(--warn);}}
.metrics-table table{{width:100%;border-collapse:collapse;}}
.metrics-table tr{{border-bottom:1px solid rgba(26,37,53,0.5);}}
.metrics-table tr:last-child{{border-bottom:none;}}
.metrics-table tr:hover{{background:rgba(255,255,255,0.02);}}
.metrics-table td{{padding:10px 20px;}}
.metrics-table td:first-child{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:0.5px;}}
.metrics-table td:last-child{{text-align:right;font-weight:500;}}
.val-up{{color:var(--up);}} .val-down{{color:var(--down);}} .val-warn{{color:var(--warn);}}
.wf-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:2px;margin-bottom:32px;}}
.wf-fold{{background:var(--surface);border:1px solid var(--border);padding:20px 24px;position:relative;}}
.wf-fold::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent2);}}
.wf-fold-label{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:6px;}}
.wf-fold-f1{{font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:var(--accent2);}}
.wf-fold-meta{{font-size:11px;color:var(--muted);margin-top:8px;display:flex;gap:16px;}}
.wf-summary{{background:var(--surface);border:1px solid rgba(0,153,255,0.3);padding:16px 24px;margin-bottom:32px;display:flex;gap:40px;align-items:center;}}
.wf-summary-val{{font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:var(--accent2);}}
.wf-summary-label{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);}}
.sa-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;margin-bottom:2px;}}
.sa-card{{background:var(--surface);border:1px solid var(--border);padding:20px;text-align:center;}}
.sa-card-label{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:8px;}}
.sa-card-val{{font-family:'Syne',sans-serif;font-size:24px;font-weight:700;}}
.sa-card-sub{{font-size:11px;color:var(--muted);margin-top:4px;}}
.pill{{display:inline-block;padding:2px 8px;border-radius:2px;font-size:10px;font-weight:500;}}
.pill.buy{{background:rgba(0,255,136,0.15);color:var(--accent);border:1px solid rgba(0,255,136,0.3);}}
.pill.sell{{background:rgba(255,75,75,0.15);color:var(--danger);border:1px solid rgba(255,75,75,0.3);}}
.pill.hold{{background:rgba(255,170,0,0.15);color:var(--warn);border:1px solid rgba(255,170,0,0.3);}}
.footer{{margin-top:60px;padding-top:24px;border-top:1px solid var(--border);color:var(--muted);font-size:11px;display:flex;justify-content:space-between;}}
@keyframes scanline{{0%{{transform:translateY(-100%);}}100%{{transform:translateY(100vh);}}}}
.scanline{{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(transparent,rgba(0,255,136,0.06),transparent);animation:scanline 8s linear infinite;pointer-events:none;z-index:999;}}
@media(max-width:900px){{.kpi-strip,.charts-2,.charts-3,.metrics-grid,.sa-row{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="container">
 
<div class="header">
  <div class="header-left">
    <h1>BACKTEST RESULTS</h1>
    <p>LightGBM Multiclass · Triple-Barrier mult={BARRIER_MULTIPLIER} fw={FORWARD_WINDOW} · Meta-Filtered · 2025 Test Period</p>
  </div>
  <span class="badge warn">ZERO FEES — GROSS ONLY</span>
</div>
 
<!-- ── PRIMARY KPIs ── -->
<div class="section-label">Primary Model — All Signals</div>
<div class="kpi-strip">
  <div class="kpi {'blue' if p.get('total_return',0)>=0 else 'danger'}">
    <div class="kpi-label">Total Return</div>
    <div class="kpi-value {'up' if p.get('total_return',0)>=0 else 'down'}">{fmt(p.get('total_return'),pct=True)}</div>
    <div class="kpi-sub">{p.get('number_of_trades',0):,} trades</div>
  </div>
  <div class="kpi {'blue' if p.get('win_rate',0)>=0.5 else 'warn'}">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value {'up' if p.get('win_rate',0)>=0.5 else 'warn'}">{fmt(p.get('win_rate'),pct=True)}</div>
    <div class="kpi-sub">breakeven = 50.00%</div>
  </div>
  <div class="kpi {'blue' if p.get('profit_factor',0)>=1 else 'danger'}">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-value {'up' if p.get('profit_factor',0)>=1 else 'down'}">{fmt(p.get('profit_factor'))}</div>
    <div class="kpi-sub">target > 1.00</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Avg Gross PnL / Trade</div>
    <div class="kpi-value {'up' if p.get('avg_gross_pnl',0)>=0 else 'down'}">{fmt(p.get('avg_gross_pnl'),bps=True)}</div>
    <div class="kpi-sub">before any costs</div>
  </div>
</div>
<div class="kpi-strip">
  <div class="kpi {'blue' if p.get('sharpe_ratio',0)>=0 else 'danger'}">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value {'up' if p.get('sharpe_ratio',0)>=0 else 'down'}">{fmt(p.get('sharpe_ratio'))}</div>
    <div class="kpi-sub">annualised</div>
  </div>
  <div class="kpi danger">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value down">{fmt(p.get('max_drawdown'),pct=True)}</div>
    <div class="kpi-sub">peak-to-trough</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Avg Win</div>
    <div class="kpi-value up">{fmt(p.get('avg_win'),bps=True)}</div>
    <div class="kpi-sub">per winning trade</div>
  </div>
  <div class="kpi danger">
    <div class="kpi-label">Avg Loss</div>
    <div class="kpi-value down">{fmt(p.get('avg_loss'),bps=True)}</div>
    <div class="kpi-sub">per losing trade</div>
  </div>
</div>
 
<!-- ── META KPIs ── -->
<div class="section-label">Meta Model — High-Confidence Filtered</div>
<div class="kpi-strip">
  <div class="kpi {'blue' if m.get('total_return',0)>=0 else 'danger'}">
    <div class="kpi-label">Total Return</div>
    <div class="kpi-value {'up' if m.get('total_return',0)>=0 else 'down'}">{fmt(m.get('total_return'),pct=True)}</div>
    <div class="kpi-sub">{m.get('number_of_trades',0):,} trades · threshold={META_CONFIDENCE_THRESHOLD}</div>
  </div>
  <div class="kpi {'blue' if m.get('win_rate',0)>=0.5 else 'warn'}">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value {'up' if m.get('win_rate',0)>=0.5 else 'warn'}">{fmt(m.get('win_rate'),pct=True)}</div>
    <div class="kpi-sub">vs primary {fmt(p.get('win_rate'),pct=True)}</div>
  </div>
  <div class="kpi {'blue' if m.get('profit_factor',0)>=1 else 'danger'}">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-value {'up' if m.get('profit_factor',0)>=1 else 'down'}">{fmt(m.get('profit_factor'))}</div>
    <div class="kpi-sub">target > 1.00</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Avg Gross PnL / Trade</div>
    <div class="kpi-value {'up' if m.get('avg_gross_pnl',0)>=0 else 'down'}">{fmt(m.get('avg_gross_pnl'),bps=True)}</div>
    <div class="kpi-sub">before any costs</div>
  </div>
</div>
<div class="kpi-strip">
  <div class="kpi {'blue' if m.get('sharpe_ratio',0)>=0 else 'danger'}">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value {'up' if m.get('sharpe_ratio',0)>=0 else 'down'}">{fmt(m.get('sharpe_ratio'))}</div>
    <div class="kpi-sub">annualised</div>
  </div>
  <div class="kpi danger">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value down">{fmt(m.get('max_drawdown'),pct=True)}</div>
    <div class="kpi-sub">peak-to-trough</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Avg Position Size</div>
    <div class="kpi-value neutral">{fmt(m.get('avg_position_size'))}</div>
    <div class="kpi-sub">proportional sizing</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Expectancy</div>
    <div class="kpi-value {'up' if m.get('expectancy',0)>=0 else 'down'}">{fmt(m.get('expectancy'),bps=True)}</div>
    <div class="kpi-sub">avg return per trade</div>
  </div>
</div>
 
<!-- ── EQUITY CURVES ── -->
<div class="section-label">Equity Curves</div>
<div class="charts-2">
  <div class="chart-box">
    <div class="chart-title">Primary Model — Equity Curve</div>
    <div class="chart-sub">Cumulative return, uniform position sizing</div>
    <div class="chart-wrap tall"><canvas id="equityPrimary"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Meta Model — Equity Curve</div>
    <div class="chart-sub">Cumulative return, proportional position sizing</div>
    <div class="chart-wrap tall"><canvas id="equityMeta"></canvas></div>
  </div>
</div>
 
<!-- ── DRAWDOWN ── -->
<div class="section-label">Drawdown Analysis</div>
<div class="charts-2">
  <div class="chart-box">
    <div class="chart-title">Primary — Drawdown</div>
    <div class="chart-sub">Rolling peak-to-trough decline</div>
    <div class="chart-wrap"><canvas id="ddPrimary"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Meta — Drawdown</div>
    <div class="chart-sub">Rolling peak-to-trough decline</div>
    <div class="chart-wrap"><canvas id="ddMeta"></canvas></div>
  </div>
</div>
 
<!-- ── RETURN ANALYSIS ── -->
<div class="section-label">Return Analysis</div>
<div class="charts-3">
  <div class="chart-box">
    <div class="chart-title">Primary — Return Distribution</div>
    <div class="chart-sub">Per-trade log-returns (basis points)</div>
    <div class="chart-wrap"><canvas id="distPrimary"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Meta — Return Distribution</div>
    <div class="chart-sub">Per-trade log-returns (basis points)</div>
    <div class="chart-wrap"><canvas id="distMeta"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Meta — Monthly Returns</div>
    <div class="chart-sub">Aggregated monthly P&L (%)</div>
    <div class="chart-wrap"><canvas id="monthly"></canvas></div>
  </div>
</div>
 
<!-- ── LABEL & SIGNAL ANALYSIS ── -->
<div class="section-label">Label & Signal Analysis</div>
<div class="sa-row">
  <div class="sa-card">
    <div class="sa-card-label">True Label Distribution</div>
    <div style="display:flex;justify-content:center;gap:24px;margin:12px 0;">
      <div><div class="sa-card-val" style="color:var(--danger)">{true_vals[0]:,}</div><div class="sa-card-sub"><span class="pill sell">SELL</span></div></div>
      <div><div class="sa-card-val" style="color:var(--warn)">{true_vals[1]:,}</div><div class="sa-card-sub"><span class="pill hold">HOLD</span></div></div>
      <div><div class="sa-card-val" style="color:var(--accent)">{true_vals[2]:,}</div><div class="sa-card-sub"><span class="pill buy">BUY</span></div></div>
    </div>
    <div class="sa-card-sub">Actual labels in 2025 test set</div>
  </div>
  <div class="sa-card">
    <div class="sa-card-label">Primary Model Predictions</div>
    <div style="display:flex;justify-content:center;gap:24px;margin:12px 0;">
      <div><div class="sa-card-val" style="color:var(--danger)">{primary_vals[0]:,}</div><div class="sa-card-sub"><span class="pill sell">SELL</span></div></div>
      <div><div class="sa-card-val" style="color:var(--warn)">{primary_vals[1]:,}</div><div class="sa-card-sub"><span class="pill hold">HOLD</span></div></div>
      <div><div class="sa-card-val" style="color:var(--accent)">{primary_vals[2]:,}</div><div class="sa-card-sub"><span class="pill buy">BUY</span></div></div>
    </div>
    <div class="sa-card-sub">What the primary model predicted</div>
  </div>
  <div class="sa-card">
    <div class="sa-card-label">Meta Model Signals</div>
    <div style="display:flex;justify-content:center;gap:24px;margin:12px 0;">
      <div><div class="sa-card-val" style="color:var(--danger)">{meta_vals[0]:,}</div><div class="sa-card-sub"><span class="pill sell">SELL</span></div></div>
      <div><div class="sa-card-val" style="color:var(--warn)">{meta_vals[1]:,}</div><div class="sa-card-sub"><span class="pill hold">FILTERED</span></div></div>
      <div><div class="sa-card-val" style="color:var(--accent)">{meta_vals[2]:,}</div><div class="sa-card-sub"><span class="pill buy">BUY</span></div></div>
    </div>
    <div class="sa-card-sub">After meta confidence filter</div>
  </div>
</div>
<div class="charts-2" style="margin-top:2px;">
  <div class="chart-box">
    <div class="chart-title">Prediction vs True Label Distribution</div>
    <div class="chart-sub">How model predictions compare to actual label counts</div>
    <div class="chart-wrap"><canvas id="labelDist"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Per-Class Recall</div>
    <div class="chart-sub">How often the model correctly identifies each class (%)</div>
    <div class="chart-wrap"><canvas id="classAcc"></canvas></div>
  </div>
</div>
 
<!-- ── WALK-FORWARD CV ── -->
<div class="section-label">Walk-Forward Cross-Validation</div>
<div class="wf-summary">
  <div><div class="wf-summary-label">Average Macro F1</div><div class="wf-summary-val">{wf_avg:.4f}</div></div>
  <div><div class="wf-summary-label">Std Dev (Stability)</div><div class="wf-summary-val">±{wf_std:.4f}</div></div>
  <div><div class="wf-summary-label">Folds Completed</div><div class="wf-summary-val">{len(wf_folds)}</div></div>
  <div style="margin-left:auto;font-size:11px;color:var(--muted);max-width:300px;">
    Lower std dev = more consistent across market regimes.<br>
    All folds use only training data — val/test never touched.
  </div>
</div>
<div class="wf-grid">
{"".join([f'''
  <div class="wf-fold">
    <div class="wf-fold-label">Fold {i+1} · {wf_results["folds"][i]["val_start"][:7]} to {wf_results["folds"][i]["val_end"][:7]}</div>
    <div class="wf-fold-f1">{wf_f1[i]:.4f}</div>
    <div class="wf-fold-meta">
      <span>Macro F1</span>
      <span>Train rows: {wf_results["folds"][i]["n_train"]:,}</span>
      <span>Best iter: {wf_results["folds"][i]["best_iter"]}</span>
    </div>
  </div>''' for i in range(len(wf_folds))]) if wf_results else '<div class="wf-fold"><div class="wf-fold-label">No walk-forward data found</div></div>'}
</div>
<div class="chart-box" style="margin-bottom:2px;">
  <div class="chart-title">Walk-Forward Macro F1 Per Fold</div>
  <div class="chart-sub">Expanding window — each fold sees more training data. Flat line = stable generalisation.</div>
  <div class="chart-wrap short"><canvas id="wfChart"></canvas></div>
</div>
 
<!-- ── MODEL COMPARISON ── -->
<div class="section-label">Model Comparison</div>
<div class="charts-3">
  <div class="chart-box">
    <div class="chart-title">Win Rate</div>
    <div class="chart-sub">Primary vs Meta (%)</div>
    <div class="chart-wrap"><canvas id="cmpWin"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Profit Factor</div>
    <div class="chart-sub">Primary vs Meta</div>
    <div class="chart-wrap"><canvas id="cmpPF"></canvas></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Sharpe Ratio</div>
    <div class="chart-sub">Primary vs Meta (annualised)</div>
    <div class="chart-wrap"><canvas id="cmpSharpe"></canvas></div>
  </div>
</div>
 
<!-- ── FULL METRICS TABLES ── -->
<div class="section-label">Full Metrics</div>
<div class="metrics-grid">
  <div class="metrics-table">
    <h3><span class="dot green"></span> Primary Model</h3>
    <table>
      <tr><td>Number of Trades</td><td>{p.get('number_of_trades',0):,}</td></tr>
      <tr><td>Win Rate</td><td class="{vc(p.get('win_rate',0)-0.5)}">{fmt(p.get('win_rate'),pct=True)}</td></tr>
      <tr><td>Loss Rate</td><td class="val-down">{fmt(p.get('loss_rate'),pct=True)}</td></tr>
      <tr><td>Avg Gross PnL / Trade</td><td class="{vc(p.get('avg_gross_pnl',0))}">{fmt(p.get('avg_gross_pnl'),bps=True)}</td></tr>
      <tr><td>Total Return</td><td class="{vc(p.get('total_return',0))}">{fmt(p.get('total_return'),pct=True)}</td></tr>
      <tr><td>Profit Factor</td><td class="{vc(p.get('profit_factor',0)-1)}">{fmt(p.get('profit_factor'))}</td></tr>
      <tr><td>Sharpe Ratio</td><td class="{vc(p.get('sharpe_ratio',0))}">{fmt(p.get('sharpe_ratio'))}</td></tr>
      <tr><td>Calmar Ratio</td><td>{fmt(p.get('calmar_ratio'))}</td></tr>
      <tr><td>Max Drawdown</td><td class="val-down">{fmt(p.get('max_drawdown'),pct=True)}</td></tr>
      <tr><td>Avg Win (bps)</td><td class="val-up">{fmt(p.get('avg_win'),bps=True)}</td></tr>
      <tr><td>Avg Loss (bps)</td><td class="val-down">{fmt(p.get('avg_loss'),bps=True)}</td></tr>
      <tr><td>Expectancy (bps)</td><td class="{vc(p.get('expectancy',0))}">{fmt(p.get('expectancy'),bps=True)}</td></tr>
      <tr><td>Gross Profit (bps)</td><td class="val-up">{fmt(p.get('gross_profit'),bps=True)}</td></tr>
      <tr><td>Gross Loss (bps)</td><td class="val-down">{fmt(p.get('gross_loss'),bps=True)}</td></tr>
    </table>
  </div>
  <div class="metrics-table">
    <h3><span class="dot blue"></span> Meta Model (Filtered)</h3>
    <table>
      <tr><td>Number of Trades</td><td>{m.get('number_of_trades',0):,}</td></tr>
      <tr><td>Win Rate</td><td class="{vc(m.get('win_rate',0)-0.5)}">{fmt(m.get('win_rate'),pct=True)}</td></tr>
      <tr><td>Loss Rate</td><td class="val-down">{fmt(m.get('loss_rate'),pct=True)}</td></tr>
      <tr><td>Avg Gross PnL / Trade</td><td class="{vc(m.get('avg_gross_pnl',0))}">{fmt(m.get('avg_gross_pnl'),bps=True)}</td></tr>
      <tr><td>Total Return</td><td class="{vc(m.get('total_return',0))}">{fmt(m.get('total_return'),pct=True)}</td></tr>
      <tr><td>Profit Factor</td><td class="{vc(m.get('profit_factor',0)-1)}">{fmt(m.get('profit_factor'))}</td></tr>
      <tr><td>Sharpe Ratio</td><td class="{vc(m.get('sharpe_ratio',0))}">{fmt(m.get('sharpe_ratio'))}</td></tr>
      <tr><td>Calmar Ratio</td><td>{fmt(m.get('calmar_ratio'))}</td></tr>
      <tr><td>Max Drawdown</td><td class="val-down">{fmt(m.get('max_drawdown'),pct=True)}</td></tr>
      <tr><td>Avg Win (bps)</td><td class="val-up">{fmt(m.get('avg_win'),bps=True)}</td></tr>
      <tr><td>Avg Loss (bps)</td><td class="val-down">{fmt(m.get('avg_loss'),bps=True)}</td></tr>
      <tr><td>Expectancy (bps)</td><td class="{vc(m.get('expectancy',0))}">{fmt(m.get('expectancy'),bps=True)}</td></tr>
      <tr><td>Gross Profit (bps)</td><td class="val-up">{fmt(m.get('gross_profit'),bps=True)}</td></tr>
      <tr><td>Gross Loss (bps)</td><td class="val-down">{fmt(m.get('gross_loss'),bps=True)}</td></tr>
      <tr><td>Avg Position Size</td><td>{fmt(m.get('avg_position_size'))}</td></tr>
    </table>
  </div>
</div>
 
<div class="footer">
  <span>LightGBM Trading Classifier · Test Period 2025 · Barrier mult={BARRIER_MULTIPLIER} · FW={FORWARD_WINDOW} bars</span>
  <span>Transaction Costs: NONE (gross performance)</span>
</div>
</div>
 
<script>
const C=Chart;
C.defaults.color='#64748b';
C.defaults.borderColor='#1a2535';
C.defaults.font.family="'DM Mono',monospace";
C.defaults.font.size=11;
 
const pEq={json.dumps(p_equity)};
const mEq={json.dumps(m_equity)};
const pDd={json.dumps(p_dd)};
const mDd={json.dumps(m_dd)};
const pHx={json.dumps(p_hx)};const pHy={json.dumps(p_hy)};
const mHx={json.dumps(m_hx)};const mHy={json.dumps(m_hy)};
const mnths={json.dumps(months)};
const mRet={json.dumps(mret_val)};
const mCol={json.dumps(mret_col)};
const wData={json.dumps(win_data)};
const pfData={json.dumps(pf_data)};
const shData={json.dumps(sharpe_data)};
const trueVals={json.dumps(true_vals)};
const primVals={json.dumps(primary_vals)};
const metaVals={json.dumps(meta_vals)};
const pcaVals={json.dumps(pca_vals)};
const wfF1={json.dumps(wf_f1)};
const wfAvg={wf_avg};
 
function lineChart(id,labels,datasets,yFmt){{
  new C(document.getElementById(id),{{
    type:'line',data:{{labels,datasets}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:datasets.length>1}},tooltip:{{callbacks:{{label:ctx=>yFmt?yFmt(ctx.parsed.y):ctx.parsed.y.toFixed(4)}}}}}},
      scales:{{x:{{display:false}},y:{{grid:{{color:'rgba(26,37,53,0.8)'}},ticks:{{callback:v=>yFmt?yFmt(v):v.toFixed(3)}}}}}},
      elements:{{point:{{radius:0}},line:{{tension:0.2}}}}}}
  }});
}}
 
function barChart(id,labels,data,colors,yFmt){{
  new C(document.getElementById(id),{{
    type:'bar',
    data:{{labels,datasets:[{{data,backgroundColor:colors,borderRadius:2}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>yFmt?yFmt(ctx.parsed.y):ctx.parsed.y.toFixed(4)}}}}}},
      scales:{{x:{{grid:{{display:false}},ticks:{{font:{{size:10}}}}}},y:{{grid:{{color:'rgba(26,37,53,0.8)'}}}}}}}}
  }});
}}
 
function groupedBar(id,labels,datasets){{
  new C(document.getElementById(id),{{
    type:'bar',data:{{labels,datasets}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:true,labels:{{boxWidth:10,font:{{size:10}}}}}}}},
      scales:{{x:{{grid:{{display:false}}}},y:{{grid:{{color:'rgba(26,37,53,0.8)'}}}}}},
      barPercentage:0.7}}
  }});
}}
 
const idxP=Array.from({{length:pEq.length}},(_,i)=>i);
const idxM=Array.from({{length:mEq.length}},(_,i)=>i);
 
lineChart('equityPrimary',idxP,[{{label:'Equity',data:pEq,borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,0.05)',fill:true,borderWidth:1.5}}],v=>(v*100-100).toFixed(2)+'%');
lineChart('equityMeta',   idxM,[{{label:'Equity',data:mEq,borderColor:'#0099ff',backgroundColor:'rgba(0,153,255,0.05)',fill:true,borderWidth:1.5}}],v=>(v*100-100).toFixed(2)+'%');
lineChart('ddPrimary',    idxP,[{{label:'DD',data:pDd,borderColor:'#ff4b4b',backgroundColor:'rgba(255,75,75,0.1)',fill:true,borderWidth:1}}],v=>(v*100).toFixed(2)+'%');
lineChart('ddMeta',       idxM,[{{label:'DD',data:mDd,borderColor:'#ff4b4b',backgroundColor:'rgba(255,75,75,0.1)',fill:true,borderWidth:1}}],v=>(v*100).toFixed(2)+'%');
barChart('distPrimary',pHx,pHy,pHx.map(v=>v>=0?'rgba(0,255,136,0.6)':'rgba(255,75,75,0.6)'));
barChart('distMeta',   mHx,mHy,mHx.map(v=>v>=0?'rgba(0,153,255,0.6)':'rgba(255,75,75,0.6)'));
barChart('monthly',mnths,mRet,mCol,v=>v.toFixed(2)+'%');
 
// Label & signal
groupedBar('labelDist',['SELL','HOLD','BUY'],[
  {{label:'True Labels',  data:trueVals, backgroundColor:'rgba(100,116,139,0.5)'}},
  {{label:'Primary Pred', data:primVals, backgroundColor:'rgba(0,255,136,0.5)'}},
  {{label:'Meta Signals', data:metaVals, backgroundColor:'rgba(0,153,255,0.5)'}}
]);
barChart('classAcc',['SELL Recall','HOLD Recall','BUY Recall'],pcaVals,
  ['rgba(255,75,75,0.7)','rgba(255,170,0,0.7)','rgba(0,255,136,0.7)'],
  v=>v.toFixed(1)+'%'
);
 
// Walk-forward
const wfLabels=wfF1.map((_,i)=>'Fold '+(i+1));
new C(document.getElementById('wfChart'),{{
  type:'line',
  data:{{labels:wfLabels,datasets:[
    {{label:'Macro F1',data:wfF1,borderColor:'#0099ff',backgroundColor:'rgba(0,153,255,0.1)',fill:true,borderWidth:2,pointRadius:6,pointBackgroundColor:'#0099ff'}},
    {{label:'Average', data:wfF1.map(()=>wfAvg),borderColor:'rgba(255,170,0,0.6)',borderDash:[5,5],borderWidth:1,pointRadius:0}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:true,labels:{{boxWidth:10,font:{{size:10}}}}}},tooltip:{{callbacks:{{label:ctx=>ctx.parsed.y.toFixed(4)}}}}}},
    scales:{{
      x:{{grid:{{color:'rgba(26,37,53,0.8)'}}}},
      y:{{grid:{{color:'rgba(26,37,53,0.8)'}},min:0.3,max:0.5,ticks:{{callback:v=>v.toFixed(3)}}}}
    }}
  }}
}});
 
// Comparison
barChart('cmpWin',   ['Primary','Meta'],wData, ['rgba(0,255,136,0.7)','rgba(0,153,255,0.7)'],v=>v.toFixed(2)+'%');
barChart('cmpPF',    ['Primary','Meta'],pfData,['rgba(0,255,136,0.7)','rgba(0,153,255,0.7)']);
barChart('cmpSharpe',['Primary','Meta'],shData,['rgba(0,255,136,0.7)','rgba(0,153,255,0.7)']);
</script>
</body>
</html>"""
    return html
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.info(f"Transaction costs: {TRADING_COST_BPS} bps {'(ZERO)' if TRADING_COST_BPS==0 else ''}")
 
    primary_model = joblib.load(LGBM_MODEL_PATH)
    feature_cols  = joblib.load(FEATURE_LIST_PATH)
    meta_model    = joblib.load(META_MODEL_PATH) if os.path.exists(META_MODEL_PATH) else None
    wf_results    = joblib.load(WF_RESULTS_PATH) if os.path.exists(WF_RESULTS_PATH) else None
 
    test_df = pd.read_parquet(os.path.join(TEST_DIR, "test.parquet"))
    test_df["Date"] = pd.to_datetime(test_df["Date"])
 
    labeled_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "all_tickers_labeled.parquet"),
        columns=["Date","Ticker","Close","rolling_std_60"],
    )
    labeled_df["Date"] = pd.to_datetime(labeled_df["Date"])
    labeled_df = labeled_df[labeled_df["Date"] >= test_df["Date"].min()].copy()
 
    results={}; trade_details={}
 
    # Backtest 1 — Primary
    logger.info("--- BACKTEST 1: Primary ---")
    sig_p, sz_p, pred_p = generate_signals_primary(test_df, primary_model, feature_cols)
    ret_p, ps_p, td_p   = simulate_trades(test_df, sig_p, sz_p, labeled_df)
    gr_p = td_p["gross_return"].values if len(td_p)>0 else np.array([])
    m_p  = compute_metrics(ret_p, ps_p, gr_p)
    print_metrics(m_p, "Primary")
    results["primary"]=m_p; trade_details["primary"]=td_p
 
    # Backtest 2 — Meta
    pred_meta_signals = pd.Series(HOLD_LABEL, index=test_df.index)
    if meta_model is not None:
        logger.info("--- BACKTEST 2: Meta ---")
        sig_m, sz_m, _  = generate_signals_meta(test_df, primary_model, meta_model, feature_cols)
        pred_meta_signals = sig_m
        ret_m, ps_m, td_m = simulate_trades(test_df, sig_m, sz_m, labeled_df)
        gr_m = td_m["gross_return"].values if len(td_m)>0 else np.array([])
        m_m  = compute_metrics(ret_m, ps_m, gr_m)
        print_metrics(m_m, "Meta")
        results["meta"]=m_m; trade_details["meta"]=td_m
 
    # Signal analysis
    signal_analysis = compute_signal_analysis(
        test_df, pred_p, pred_meta_signals, feature_cols, primary_model
    )
 
    # HTML
    logger.info("Generating HTML report ...")
    html      = generate_html(results, trade_details, signal_analysis, wf_results)
    html_path = os.path.join(LOG_DIR, "results.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report -> {html_path}")
 
    # txt summary
    keys = ["number_of_trades","win_rate","avg_gross_pnl","average_return",
            "profit_factor","max_drawdown","sharpe_ratio","total_return"]
    logger.info(f"\n{'Metric':<22} {'Primary':>14} {'Meta':>14}")
    logger.info("-"*52)
    for k in keys:
        vp=results.get("primary",{}).get(k,float("nan"))
        vm=results.get("meta",   {}).get(k,float("nan"))
        logger.info(f"{k:<22} {vp:>14.4f} {vm:>14.4f}")
 
 
if __name__ == "__main__":
    main()