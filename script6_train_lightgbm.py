"""
script6_train_lightgbm.py — Walk-forward cross-validation + final LightGBM training.

WALK-FORWARD CV (runs first, on training data only):
    Uses an expanding window — each fold adds one year of training data.
    Fold 1: train=2021,      val=2022-H1
    Fold 2: train=2021-2022, val=2023-H1
    Fold 3: train=2021-2023H1, val=2023-H2
    Reports per-fold and average macro-F1 across folds.
    Fold results are saved to models/walk_forward_results.joblib.

FINAL TRAINING:
    Trains on full training split (2020-2023) with early stopping on the
    2024 validation set using the same hyperparameters.

Inputs:
    processed/train/train.parquet
    processed/validation/validation.parquet
Outputs:
    models/lgbm_model.joblib
    models/lgbm_model.txt
    models/walk_forward_results.joblib
    models/feature_list.joblib
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    TRAIN_DIR, VAL_DIR, MODEL_DIR,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS,
    RANDOM_SEED, FEATURE_LIST_PATH, LGBM_MODEL_PATH,
    WF_FOLDS, WF_N_ESTIMATORS, WF_EARLY_STOP,
    WF_RESULTS_PATH, TRAIN_START,
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script6.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

TRAIN_LSTM = False
np.random.seed(RANDOM_SEED)


def load_split(split_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(split_dir, f"{name}.parquet")
    df   = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info(f"Loaded {name}: {len(df):,} rows")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in {"Date", "Ticker", "label"}]


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    wdict   = dict(zip(classes.tolist(), weights.tolist()))
    return np.array([wdict[int(yi)] for yi in y])


def fit_lgbm(X_tr, y_tr, X_va, y_va,
              feature_cols, n_estimators, early_stop, tag="") -> lgb.LGBMClassifier:
    """Fit one LightGBM model with early stopping."""
    params = LGBM_PARAMS.copy()
    params.pop("n_estimators", None)

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
    sw    = make_sample_weights(y_tr)

    logger.info(f"{tag} Fitting LightGBM: {len(X_tr):,} train / {len(X_va):,} val ...")
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    logger.info(f"{tag} Best iteration: {model.best_iteration_}")
    return model


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def run_walk_forward_cv(train_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Expanding-window walk-forward CV on the training split.

    Each fold trains from TRAIN_START to fold['train_end'], then evaluates
    on fold['val_start']..fold['val_end'].  All windows are strictly within
    the training period — the 2024 validation set is never touched here.

    Returns a dict with per-fold metrics and averages.
    """
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD CROSS-VALIDATION")
    logger.info("="*60)

    fold_results = []

    for fold_num, fold in enumerate(WF_FOLDS, start=1):
        tr_mask = (
            (train_df["Date"] >= TRAIN_START) &
            (train_df["Date"] <= fold["train_end"])
        )
        va_mask = (
            (train_df["Date"] >= fold["val_start"]) &
            (train_df["Date"] <= fold["val_end"])
        )

        fold_tr = train_df[tr_mask]
        fold_va = train_df[va_mask]

        if len(fold_tr) == 0 or len(fold_va) == 0:
            logger.warning(f"Fold {fold_num}: empty split -- skipping.")
            continue

        logger.info(
            f"\nFold {fold_num}/{len(WF_FOLDS)} | "
            f"Train: {fold['train_end']}  ({len(fold_tr):,} samples) | "
            f"Val:   {fold['val_start']} to {fold['val_end']}  ({len(fold_va):,} samples)"
        )

        X_tr = fold_tr[feature_cols].values
        y_tr = fold_tr["label"].values.astype(int)
        X_va = fold_va[feature_cols].values
        y_va = fold_va["label"].values.astype(int)

        model = fit_lgbm(
            X_tr, y_tr, X_va, y_va,
            feature_cols,
            n_estimators=WF_N_ESTIMATORS,
            early_stop=WF_EARLY_STOP,
            tag=f"[Fold {fold_num}]",
        )

        preds    = model.predict(X_va).astype(int)
        macro_f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        report   = classification_report(
            y_va, preds,
            target_names=["SELL", "HOLD", "BUY"],
            digits=4, zero_division=0,
        )

        logger.info(f"[Fold {fold_num}] Macro F1: {macro_f1:.4f}")
        logger.info(f"\n{report}")

        fold_results.append({
            "fold":       fold_num,
            "train_end":  fold["train_end"],
            "val_start":  fold["val_start"],
            "val_end":    fold["val_end"],
            "macro_f1":   macro_f1,
            "n_train":    len(fold_tr),
            "n_val":      len(fold_va),
            "best_iter":  model.best_iteration_,
        })

    # Summary
    f1_scores = [r["macro_f1"] for r in fold_results]
    avg_f1    = np.mean(f1_scores)
    std_f1    = np.std(f1_scores)
    logger.info("\n" + "-"*60)
    logger.info(f"Walk-forward CV complete: {len(fold_results)} folds")
    logger.info(f"  Macro F1 per fold: {[f'{v:.4f}' for v in f1_scores]}")
    logger.info(f"  Average macro F1 : {avg_f1:.4f} (+/- {std_f1:.4f})")
    logger.info("-"*60)

    return {
        "folds":    fold_results,
        "avg_f1":   avg_f1,
        "std_f1":   std_f1,
        "f1_scores": f1_scores,
    }


# ── Final training ─────────────────────────────────────────────────────────────

def train_final_model(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       feature_cols: list) -> lgb.LGBMClassifier:
    logger.info("\n" + "="*60)
    logger.info("FINAL MODEL TRAINING (full train split + 2024 val for early stop)")
    logger.info("="*60)

    X_tr = train_df[feature_cols].values
    y_tr = train_df["label"].values.astype(int)
    X_va = val_df[feature_cols].values
    y_va = val_df["label"].values.astype(int)

    params = LGBM_PARAMS.copy()
    n_est  = params.pop("n_estimators")

    model  = fit_lgbm(
        X_tr, y_tr, X_va, y_va,
        feature_cols,
        n_estimators=n_est,
        early_stop=EARLY_STOPPING_ROUNDS,
        tag="[Final]",
    )
    return model


# ── Optional lightweight LSTM ──────────────────────────────────────────────────

def train_lightweight_lstm(train_df, val_df, feature_cols, lookback=60):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not installed -- skipping LSTM.")
        return

    torch.manual_seed(RANDOM_SEED)

    def reshape(df):
        X = df[feature_cols].values.astype(np.float32)
        n_steps    = lookback
        n_per_step = X.shape[1] // n_steps
        return torch.tensor(X.reshape(-1, n_steps, n_per_step)), \
               torch.tensor(df["label"].values.astype(np.int64))

    X_tr, y_tr = reshape(train_df)
    X_va, y_va = reshape(val_df)
    n_per_step = X_tr.shape[2]

    class TinyLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_per_step, 32, batch_first=True)
            self.fc   = nn.Linear(32, 3)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    model_lstm = TinyLSTM()
    opt        = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
    criterion  = nn.CrossEntropyLoss()
    dl_tr      = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=True)
    dl_va      = DataLoader(TensorDataset(X_va, y_va), batch_size=512)

    best_loss = float("inf")
    patience  = 2
    for epoch in range(5):
        model_lstm.train()
        for xb, yb in dl_tr:
            opt.zero_grad(); criterion(model_lstm(xb), yb).backward(); opt.step()
        model_lstm.eval()
        val_loss = np.mean([criterion(model_lstm(xb), yb).item()
                            for xb, yb in dl_va])
        logger.info(f"  LSTM epoch {epoch+1}/5 -- val_loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model_lstm.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))
        else:
            patience -= 1
            if patience == 0:
                break
    logger.info(f"LSTM saved -> {os.path.join(MODEL_DIR, 'lstm_model.pt')}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = load_split(TRAIN_DIR, "train")
    val_df   = load_split(VAL_DIR,   "validation")

    feature_cols = get_feature_cols(train_df)
    logger.info(f"Feature dimension: {len(feature_cols)}")

    joblib.dump(feature_cols, FEATURE_LIST_PATH)
    logger.info(f"Feature list saved -> {FEATURE_LIST_PATH}")

    # Step 1: Walk-forward CV on training data
    wf_results = run_walk_forward_cv(train_df, feature_cols)
    joblib.dump(wf_results, WF_RESULTS_PATH)
    logger.info(f"Walk-forward results saved -> {WF_RESULTS_PATH}")

    # Step 2: Final model on full training data
    final_model = train_final_model(train_df, val_df, feature_cols)

    joblib.dump(final_model, LGBM_MODEL_PATH)
    logger.info(f"Model saved -> {LGBM_MODEL_PATH}")

    txt_path = os.path.join(MODEL_DIR, "lgbm_model.txt")
    final_model.booster_.save_model(txt_path)
    logger.info(f"Native LightGBM model saved -> {txt_path}")

    if TRAIN_LSTM:
        from utils.config import LOOKBACK_WINDOW
        train_lightweight_lstm(train_df, val_df, feature_cols, LOOKBACK_WINDOW)


if __name__ == "__main__":
    main()
