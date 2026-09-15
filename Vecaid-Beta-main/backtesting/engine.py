"""
backtesting/engine.py — walk-forward backtesting on a held-out dataset.

Design principles:
  - Models are LOADED (pre-trained), never fitted here.  No data leakage.
  - Only the last `n_test_days` rows are used as the test window.
  - The lookback window before the first test day is taken from the same
    DataFrame (it is part of the training-period data passed in).
  - Returns serialisable stats + base64-encoded plot images.
"""
import base64
import io
import logging
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.pipeline import build_feature_vector
from models.ensemble import predict_ensemble

logger = logging.getLogger(__name__)


def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img


def run_backtest(
    data: pd.DataFrame,
    models: Dict[str, Any],
    scaler,
    pca,
    window: int = 60,
    n_test_days: int = 63,
) -> Dict[str, Any]:
    """
    Walk-forward backtest over the last `n_test_days` of `data`.

    `data` must already have technical indicators computed and intraday
    features joined.  The models, scaler, and pca are pre-trained artefacts.

    Returns a dict with:
        stats                : summary metrics dict
        plot_pred_vs_actual  : base64 PNG string
        plot_price_comparison: base64 PNG string
    """
    needed = window + n_test_days + 1
    if len(data) < needed:
        raise ValueError(
            f"Backtest needs at least {needed} rows; got {len(data)}. "
            "Fetch more history or reduce n_test_days."
        )

    # Use only the tail we need (context + test)
    segment = data.iloc[-needed:].copy()

    dates_out, pred_moves, actual_moves = [], [], []
    pred_prices, actual_prices = [], []
    cumulative = [1.0]

    for i in range(window, len(segment) - 1):
        w = segment.iloc[i - window:i]

        try:
            fv = build_feature_vector(w).reshape(1, -1)
        except Exception as e:
            logger.warning("Feature build failed at step %d: %s", i, e)
            continue

        fv_scaled = scaler.transform(fv)
        fv_pca = pca.transform(fv_scaled)

        try:
            pred_pct, _ = predict_ensemble(fv_pca, models)
        except Exception as e:
            logger.warning("Ensemble predict failed at step %d: %s", i, e)
            continue

        current_price = float(segment["Close"].iloc[i])
        next_price = float(segment["Close"].iloc[i + 1])
        actual_pct = ((next_price - current_price) / current_price) * 100.0

        pred_moves.append(pred_pct)
        actual_moves.append(actual_pct)
        pred_prices.append(current_price * (1 + pred_pct / 100.0))
        actual_prices.append(next_price)
        dates_out.append(segment.index[i + 1])

        if pred_pct != 0:
            correct = np.sign(pred_pct) == np.sign(actual_pct)
            ret = abs(actual_pct) / 100.0 * (1 if correct else -1)
        else:
            ret = 0.0
        cumulative.append(cumulative[-1] * (1 + ret))

    if len(pred_moves) == 0:
        raise ValueError("No valid predictions generated during backtest.")

    preds = np.array(pred_moves)
    actuals = np.array(actual_moves)
    pred_prices_arr = np.array(pred_prices)
    actual_prices_arr = np.array(actual_prices)

    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)) * 100)
    mae = float(np.mean(np.abs(preds - actuals)))
    rmse_pct = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    rmse_price = float(np.sqrt(np.mean((pred_prices_arr - actual_prices_arr) ** 2)))

    logger.info(
        "Backtest complete — dir_acc=%.1f%%  RMSE=%.4f%%  RMSE_price=%.4f",
        dir_acc, rmse_pct, rmse_price,
    )

    # --- Plot 1: predicted vs actual % moves ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates_out, preds, label="Predicted %", color="#2d6a4f", linewidth=1.5)
    ax.plot(dates_out, actuals, label="Actual %", color="#f4a261", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("% Move")
    ax.set_title("Walk-Forward Backtest: Predicted vs Actual % Moves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img_pct = _fig_to_b64()

    # --- Plot 2: predicted vs actual price ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates_out, pred_prices_arr, label="Predicted Price", color="#2d6a4f", linewidth=1.5)
    ax.plot(dates_out, actual_prices_arr, label="Actual Price", color="#f4a261", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title("Walk-Forward Backtest: Predicted vs Actual Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img_price = _fig_to_b64()

    stats = {
        "directional_accuracy_%": round(dir_acc, 2),
        "mean_absolute_error_%": round(mae, 4),
        "rmse_%": round(rmse_pct, 4),
        "rmse_price_$": round(rmse_price, 4),
        "n_test_days": len(preds),
        "note": "All metrics computed on held-out data only (no training data used).",
    }

    return {
        "stats": stats,
        "plot_pred_vs_actual": img_pct,
        "plot_price_comparison": img_price,
    }
