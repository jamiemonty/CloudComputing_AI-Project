"""
script8_evaluate_model.py — Evaluate on validation and test sets.

Prints:
    Accuracy, Precision, Recall, F1 (per class)
    Macro Precision, Macro Recall, Macro F1
    Walk-forward CV summary (from script6)
    Confusion matrices saved as PNG

Inputs:  processed/validation/validation.parquet
         processed/test/test.parquet
         models/lgbm_model.joblib
         models/meta_model.joblib
         models/walk_forward_results.joblib
Outputs: logs/eval_validation.txt
         logs/eval_test.txt
         logs/confusion_matrix_*.png
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    VAL_DIR, TEST_DIR, MODEL_DIR, LOG_DIR,
    LGBM_MODEL_PATH, META_MODEL_PATH, FEATURE_LIST_PATH,
    WF_RESULTS_PATH, META_CONFIDENCE_THRESHOLD, LABEL_MAP,
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "script8.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["SELL", "HOLD", "BUY"]
HOLD_LABEL  = LABEL_MAP["HOLD"]


def load_split(split_dir, name):
    return pd.read_parquet(os.path.join(split_dir, f"{name}.parquet"))


def _save_confusion_matrix(y_true, y_pred, split_name, tag):
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=True, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix -- {split_name} ({tag})")
    plt.tight_layout()
    path = os.path.join(LOG_DIR, f"confusion_matrix_{split_name}_{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {path}")


def evaluate_split(split_name, df, primary_model, meta_model, feature_cols):
    logger.info(f"\n{'='*60}\nEVALUATING: {split_name.upper()}\n{'='*60}")

    X      = df[feature_cols].values
    y_true = df["label"].values.astype(int)

    proba_primary = primary_model.predict_proba(X)
    pred_primary  = primary_model.predict(X).astype(int)

    acc        = accuracy_score(y_true, pred_primary)
    macro_prec = precision_score(y_true, pred_primary, average="macro",  zero_division=0)
    macro_rec  = recall_score(   y_true, pred_primary, average="macro",  zero_division=0)
    macro_f1   = f1_score(       y_true, pred_primary, average="macro",  zero_division=0)
    report     = classification_report(
        y_true, pred_primary, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    logger.info(f"[Primary LightGBM -- {split_name}]")
    logger.info(f"  Accuracy        : {acc:.4f}")
    logger.info(f"  Macro Precision : {macro_prec:.4f}")
    logger.info(f"  Macro Recall    : {macro_rec:.4f}")
    logger.info(f"  Macro F1        : {macro_f1:.4f}")
    logger.info(f"\n{report}")

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"eval_{split_name}.txt"), "w") as f:
        f.write(f"Primary LightGBM -- {split_name}\n")
        f.write(f"Accuracy        : {acc:.4f}\n")
        f.write(f"Macro Precision : {macro_prec:.4f}\n")
        f.write(f"Macro Recall    : {macro_rec:.4f}\n")
        f.write(f"Macro F1        : {macro_f1:.4f}\n\n")
        f.write(report)

    _save_confusion_matrix(y_true, pred_primary, split_name, "primary")

    # Meta-filtered evaluation
    if meta_model is not None:
        _evaluate_meta(X, y_true, pred_primary, proba_primary,
                       meta_model, split_name)

    return dict(accuracy=acc, macro_precision=macro_prec,
                macro_recall=macro_rec, macro_f1=macro_f1)


def _evaluate_meta(X, y_true, pred_primary, proba_primary,
                    meta_model, split_name):
    trade_mask = pred_primary != HOLD_LABEL
    if trade_mask.sum() == 0:
        return

    X_meta    = np.concatenate([
        X[trade_mask],
        proba_primary[trade_mask],
        pred_primary[trade_mask].reshape(-1, 1)
    ], axis=1)

    meta_proba = meta_model.predict_proba(X_meta)[:, 1]
    confident  = meta_proba >= META_CONFIDENCE_THRESHOLD

    logger.info(
        f"\n[Meta-filtered -- {split_name}] "
        f"Threshold={META_CONFIDENCE_THRESHOLD} | "
        f"Trades kept: {confident.sum():,}/{trade_mask.sum():,} "
        f"({100*confident.mean():.1f}%)"
    )
    if confident.sum() == 0:
        return

    y_conf   = y_true[trade_mask][confident]
    pred_conf = pred_primary[trade_mask][confident]
    report    = classification_report(
        y_conf, pred_conf, labels=[0, 2],
        target_names=["SELL", "BUY"], digits=4, zero_division=0
    )
    logger.info(f"Meta-filtered report:\n{report}")
    _save_confusion_matrix(y_conf, pred_conf, split_name, "meta")


def print_wf_summary():
    if not os.path.exists(WF_RESULTS_PATH):
        return
    wf = joblib.load(WF_RESULTS_PATH)
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD CV SUMMARY")
    logger.info("="*60)
    for r in wf["folds"]:
        logger.info(
            f"  Fold {r['fold']}: train->  {r['train_end']} | "
            f"val: {r['val_start']} to {r['val_end']} | "
            f"Macro F1: {r['macro_f1']:.4f} | "
            f"Best iter: {r['best_iter']}"
        )
    logger.info(f"  Average Macro F1: {wf['avg_f1']:.4f} (+/- {wf['std_f1']:.4f})")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info("Loading models ...")
    primary_model = joblib.load(LGBM_MODEL_PATH)
    feature_cols  = joblib.load(FEATURE_LIST_PATH)

    meta_model = None
    if os.path.exists(META_MODEL_PATH):
        meta_model = joblib.load(META_MODEL_PATH)
        logger.info("Meta-model loaded.")

    print_wf_summary()

    val_df  = load_split(VAL_DIR,  "validation")
    val_m   = evaluate_split("validation", val_df,  primary_model, meta_model, feature_cols)

    # Test used exactly once
    test_df = load_split(TEST_DIR, "test")
    test_m  = evaluate_split("test",       test_df, primary_model, meta_model, feature_cols)

    logger.info("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    header = f"{'Metric':<22} {'Validation':>12} {'Test':>12}"
    logger.info(header)
    logger.info("-" * len(header))
    for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        logger.info(f"{k:<22} {val_m[k]:>12.4f} {test_m[k]:>12.4f}")


if __name__ == "__main__":
    main()
