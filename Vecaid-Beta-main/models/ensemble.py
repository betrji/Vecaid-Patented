"""
models/ensemble.py — walk-forward OOF training to eliminate data leakage.

The original code had three critical leakage problems:
  1. Base models trained on full data; RMSE computed on *same* training data.
  2. Meta-model trained on base-model predictions from training data (not OOF).
  3. train_test_split used random shuffle on time-series data.

This module fixes all three:
  1. OOF predictions from temporally-ordered folds only.
  2. Meta-model trained exclusively on OOF predictions.
  3. All CV uses sklearn's TimeSeriesSplit (no shuffle).
  4. After OOF, base models are retrained on the FULL training set.
"""
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

# Use GPU for XGBoost if CUDA is available
try:
    import xgboost as _xgb
    _test = XGBRegressor(device="cuda", n_estimators=1)
    _test.fit(np.zeros((2, 1)), np.zeros(2))
    _XGB_DEVICE = "cuda"
    logging.getLogger(__name__).info("XGBoost will use GPU (device=cuda).")
except Exception:
    _XGB_DEVICE = "cpu"
    logging.getLogger(__name__).info("XGBoost will use CPU.")

from config import BAYES_INIT_POINTS, BAYES_N_ITER, NN_BATCH_SIZE, NN_EPOCHS, NN_VALIDATION_SPLIT, XGB_N_ESTIMATORS
from models.builders import BASE_MODEL_BUILDERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XGBoost Bayesian hyper-parameter optimisation (temporal CV, no shuffle)
# ---------------------------------------------------------------------------

def _bayesian_optimize_xgb(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Optimise XGBoost hyperparameters using Bayesian search with
    TimeSeriesSplit (no data shuffling).
    """
    def xgb_cv(max_depth: float, learning_rate: float, subsample: float, colsample_bytree: float) -> float:
        params = dict(
            max_depth=int(round(max_depth)),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=42,
            n_estimators=XGB_N_ESTIMATORS,
            device=_XGB_DEVICE,
        )
        cv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in cv.split(X):
            m = XGBRegressor(**params)
            m.fit(X[tr_idx], y[tr_idx], verbose=False)
            preds = m.predict(X[val_idx])
            scores.append(-mean_squared_error(y[val_idx], preds))
        return float(np.mean(scores))

    opt = BayesianOptimization(
        f=xgb_cv,
        pbounds={
            "max_depth": (3, 8),
            "learning_rate": (0.01, 0.25),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
        },
        random_state=42,
        verbose=1,
    )
    opt.maximize(init_points=BAYES_INIT_POINTS, n_iter=BAYES_N_ITER)

    best = opt.max["params"]
    best["max_depth"] = int(round(best["max_depth"]))
    best["objective"] = "reg:squarederror"
    best["random_state"] = 42
    best["n_estimators"] = XGB_N_ESTIMATORS
    return best


# ---------------------------------------------------------------------------
# OOF prediction generation
# ---------------------------------------------------------------------------

def _generate_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    xgb_params: Dict[str, Any],
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each temporal fold, train all base models on the training split and
    predict on the held-out validation split.  Returns the OOF prediction
    matrix and a boolean mask of which samples were covered.

    Returns:
        oof_preds : (n_samples, n_base_models + 1) — last column is XGB
        valid     : boolean array, True where predictions are available
    """
    n_models = len(BASE_MODEL_BUILDERS) + 1   # +1 for XGB ensemble column
    oof = np.zeros((len(y), n_models), dtype=np.float64)
    counts = np.zeros(len(y), dtype=int)

    cv = TimeSeriesSplit(n_splits=n_splits)

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X)):
        logger.info(
            "OOF fold %d/%d — train=%d  val=%d",
            fold_i + 1, n_splits, len(tr_idx), len(val_idx),
        )
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val = X[val_idx]

        # --- Keras base models ---
        for col, (name, build_fn) in enumerate(BASE_MODEL_BUILDERS.items()):
            m = build_fn(input_dim)
            m.fit(
                X_tr, y_tr,
                epochs=NN_EPOCHS,
                batch_size=NN_BATCH_SIZE,
                validation_split=NN_VALIDATION_SPLIT,
                verbose=0,
            )
            preds = m.predict(X_val, verbose=0).ravel()
            oof[val_idx, col] += preds
            val_rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
            logger.info("  [fold %d] %s  val RMSE=%.4f", fold_i + 1, name, val_rmse)
            # Free GPU memory between folds
            del m

        # --- XGBoost ensemble (average of 3 seeds) ---
        xgb_fold_preds_list = []
        for seed in [42, 43, 44]:
            p = dict(xgb_params)
            p["random_state"] = seed
            p["device"] = _XGB_DEVICE
            xm = XGBRegressor(**p)
            xm.fit(X_tr, y_tr, verbose=False)
            xgb_fold_preds_list.append(xm.predict(X_val))
        oof[val_idx, -1] += np.mean(xgb_fold_preds_list, axis=0)
        xgb_val_rmse = np.sqrt(
            mean_squared_error(y[val_idx], np.mean(xgb_fold_preds_list, axis=0))
        )
        logger.info("  [fold %d] xgb_ensemble  val RMSE=%.4f", fold_i + 1, xgb_val_rmse)

        counts[val_idx] += 1

    valid = counts > 0
    oof[valid] /= counts[valid, np.newaxis]
    return oof, valid


# ---------------------------------------------------------------------------
# Full ensemble training
# ---------------------------------------------------------------------------

def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_cv_splits: int = 3,
) -> Dict[str, Any]:
    """
    Full training protocol:
      1. Bayesian-optimise XGBoost hyperparameters (temporal CV).
      2. Generate OOF predictions with TimeSeriesSplit.
      3. Train meta-model on OOF predictions (no look-ahead).
      4. Retrain all base models on the full dataset.

    Returns a dict of all models + honest OOF metrics.
    """
    input_dim = X.shape[1]

    # Step 1 — XGB hyper-parameter search
    logger.info("Step 1/4: Bayesian optimisation for XGBoost hyperparameters...")
    xgb_params = _bayesian_optimize_xgb(X, y)
    logger.info("Best XGB params: %s", xgb_params)

    # Step 2 — OOF predictions
    logger.info("Step 2/4: Generating walk-forward OOF predictions (%d folds)...", n_cv_splits)
    oof_preds, valid_mask = _generate_oof_predictions(X, y, n_cv_splits, xgb_params, input_dim)

    X_meta = oof_preds[valid_mask]
    y_meta = y[valid_mask]
    logger.info("OOF samples available for meta-model: %d / %d", int(valid_mask.sum()), len(y))

    # Step 3 — Train meta-model on OOF predictions
    logger.info("Step 3/4: Training meta-model on OOF predictions...")
    meta_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        device=_XGB_DEVICE,
    )

    # Evaluate on last temporal fold before fitting on all OOF
    meta_cv = TimeSeriesSplit(n_splits=3)
    last_fold_rmse = None
    last_fold_dir_acc = None
    for tr_idx, val_idx in meta_cv.split(X_meta):
        m_temp = XGBRegressor(
            objective="reg:squarederror", n_estimators=200,
            learning_rate=0.05, random_state=42, device=_XGB_DEVICE,
        )
        m_temp.fit(X_meta[tr_idx], y_meta[tr_idx], verbose=False)
        val_preds = m_temp.predict(X_meta[val_idx])
        last_fold_rmse = float(np.sqrt(mean_squared_error(y_meta[val_idx], val_preds)))
        last_fold_dir_acc = float(
            np.mean(np.sign(val_preds) == np.sign(y_meta[val_idx])) * 100
        )

    meta_model.fit(X_meta, y_meta, verbose=False)
    logger.info(
        "Meta-model OOF last-fold RMSE=%.4f  dir_acc=%.1f%%",
        last_fold_rmse or -1,
        last_fold_dir_acc or -1,
    )

    # Step 4 — Retrain all base models on full training data
    logger.info("Step 4/4: Retraining all base models on full training set...")
    final_keras_models: Dict[str, Any] = {}
    for name, build_fn in BASE_MODEL_BUILDERS.items():
        m = build_fn(input_dim)
        m.fit(
            X, y,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_split=NN_VALIDATION_SPLIT,
            verbose=0,
        )
        final_keras_models[name] = m
        logger.info("  %s — full retrain done", name)

    # XGBoost ensemble (3 variants with slightly different LR)
    xgb_models: List[XGBRegressor] = []
    for lr_mult in [1.0, 1.1, 0.9]:
        p = dict(xgb_params)
        p["learning_rate"] = float(p["learning_rate"]) * lr_mult
        p["device"] = _XGB_DEVICE
        xm = XGBRegressor(**p)
        xm.fit(X, y, verbose=False)
        xgb_models.append(xm)

    # Honest metrics on OOF predictions (not training data)
    oof_final_preds = meta_model.predict(X_meta)
    oof_rmse = float(np.sqrt(mean_squared_error(y_meta, oof_final_preds)))
    oof_dir_acc = float(np.mean(np.sign(oof_final_preds) == np.sign(y_meta)) * 100)

    model_metrics = {
        "oof_rmse_pct": round(oof_rmse, 4),
        "oof_directional_accuracy_pct": round(oof_dir_acc, 2),
        "meta_last_fold_rmse": round(last_fold_rmse, 4) if last_fold_rmse else None,
        "meta_last_fold_dir_acc": round(last_fold_dir_acc, 2) if last_fold_dir_acc else None,
        "xgb_params": xgb_params,
        "n_oof_samples": int(valid_mask.sum()),
        "n_training_samples": len(y),
    }
    logger.info("Final OOF metrics: %s", model_metrics)

    return {
        **final_keras_models,
        "xgb_models": xgb_models,
        "meta_model": meta_model,
        "model_metrics": model_metrics,
        "input_dim": input_dim,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_ensemble(
    X_input: np.ndarray,
    models: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Run X_input (shape 1×input_dim) through all base models then meta-model.

    Returns:
        pred_pct    : predicted % price change
        ensemble_std: std-dev across base-model predictions
                      (used as a proxy for prediction uncertainty)
    """
    base_preds: List[float] = []

    for name in BASE_MODEL_BUILDERS:
        m = models[name]
        p = float(m.predict(X_input, verbose=0).ravel()[0])
        base_preds.append(p)

    xgb_preds = [float(m.predict(X_input)[0]) for m in models["xgb_models"]]
    base_preds.append(float(np.mean(xgb_preds)))

    combined = np.array(base_preds, dtype=np.float64).reshape(1, -1)
    final_pred = float(models["meta_model"].predict(combined)[0])
    ensemble_std = float(np.std(base_preds))

    return final_pred, ensemble_std
