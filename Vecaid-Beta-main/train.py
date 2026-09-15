#!/usr/bin/env python3
"""
train.py — offline training script.

Usage:
    python train.py --ticker AAPL
    python train.py --ticker AAPL --years 5          # use 5 years of history
    python train.py --ticker AAPL --no-backtest      # skip held-out backtest

This script is meant to be run offline (takes 20-40 min on CPU per ticker).
It saves trained models to saved_models/<TICKER>/<version>/.
The Flask app (app.py) loads these saved models at startup — no training there.

Schedule with cron for nightly retraining:
    0 2 * * 1-5 cd /app && python train.py --ticker AAPL >> logs/train.log 2>&1
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GPU setup — must happen before any TF/Keras import
# ---------------------------------------------------------------------------
def _configure_gpu() -> bool:
    """
    Enable GPU memory growth and mixed-precision (fp16) for RTX cards.
    Returns True if a GPU was found and configured.
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Mixed precision: fp16 compute, fp32 accumulation — ~2× faster on RTX
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    return True

_gpu_available = _configure_gpu()
if _gpu_available:
    logging.getLogger(__name__).info("GPU detected — mixed_float16 precision enabled.")
else:
    logging.getLogger(__name__).warning("No GPU found — training on CPU.")
# ---------------------------------------------------------------------------
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    CV_MONTH_LEN,
    CV_YEAR_LEN,
    LOOKBACK_WINDOW,
    N_CV_SPLITS,
    TRAIN_YEARS,
)
from backtesting.engine import run_backtest
from data.intraday import get_intraday_features
from data.market import build_merged_daily_weekly
from features.pipeline import build_feature_matrix
from features.technical import get_technical_indicators
from models.ensemble import train_ensemble
from models.registry import save_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train(ticker: str, years: int = TRAIN_YEARS, run_backtest_flag: bool = True) -> str:
    """Train ensemble for `ticker` and return the saved version string."""
    today = datetime.today()
    train_start = (today - relativedelta(years=years)).strftime("%Y-%m-%d")
    # Hold out last 3 months for a post-training backtest
    train_end = (today - relativedelta(months=3)).strftime("%Y-%m-%d")
    backtest_start = train_end
    backtest_end = today.strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("Training %s | %s → %s", ticker, train_start, train_end)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Fetch training data
    # -----------------------------------------------------------------------
    logger.info("[1/5] Downloading daily + weekly price data...")
    data = build_merged_daily_weekly(ticker, train_start, train_end)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    logger.info("Raw rows (daily): %d", len(data))

    logger.info("[1/5] Downloading intraday 5-min features (last 60 days)...")
    intraday = get_intraday_features(ticker, days=60, interval="5m")

    # -----------------------------------------------------------------------
    # 2. Compute technical indicators
    # -----------------------------------------------------------------------
    logger.info("[2/5] Computing technical indicators...")
    data = get_technical_indicators(data)

    if not intraday.empty:
        data = data.join(intraday, how="left")

    data.fillna(0, inplace=True)
    logger.info("Rows after indicator calculation: %d", len(data))

    # -----------------------------------------------------------------------
    # 3. Build feature matrix
    # -----------------------------------------------------------------------
    logger.info("[3/5] Building feature matrix (window=%d)...", LOOKBACK_WINDOW)
    X, y, dates = build_feature_matrix(data, window=LOOKBACK_WINDOW)
    logger.info("Feature matrix: X=%s  y=%s", X.shape, y.shape)

    if len(X) < CV_YEAR_LEN * 2:
        logger.warning(
            "Only %d training samples — consider using more years of history. "
            "Minimum recommended: %d.",
            len(X),
            CV_YEAR_LEN * 2,
        )

    # -----------------------------------------------------------------------
    # 4. Preprocessing
    # -----------------------------------------------------------------------
    logger.info("[4/5] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("[4/5] Applying PCA (retain 95%% variance)...")
    pca = PCA(n_components=0.95, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    logger.info("PCA: %d components explain ≥95%% variance", X_pca.shape[1])

    # -----------------------------------------------------------------------
    # 5. Train ensemble (OOF protocol)
    # -----------------------------------------------------------------------
    logger.info("[5/5] Training ensemble with walk-forward OOF (%d folds)...", N_CV_SPLITS)
    models = train_ensemble(X_pca, y, n_cv_splits=N_CV_SPLITS)

    logger.info("OOF metrics:")
    for k, v in models.get("model_metrics", {}).items():
        logger.info("  %s: %s", k, v)

    # -----------------------------------------------------------------------
    # 6. Save
    # -----------------------------------------------------------------------
    version = save_models(ticker, models, scaler, pca)
    logger.info("Saved to saved_models/%s/%s", ticker, version)

    # -----------------------------------------------------------------------
    # 7. Optional held-out backtest
    # -----------------------------------------------------------------------
    if run_backtest_flag:
        logger.info("Running held-out backtest (%s → %s)...", backtest_start, backtest_end)
        try:
            bt_data = build_merged_daily_weekly(ticker, backtest_start, backtest_end)
            bt_data.index = pd.to_datetime(bt_data.index)
            bt_data = get_technical_indicators(bt_data)
            if not intraday.empty:
                bt_data = bt_data.join(intraday, how="left")
            bt_data.fillna(0, inplace=True)

            bt_results = run_backtest(
                bt_data,
                models,
                scaler,
                pca,
                window=LOOKBACK_WINDOW,
                n_test_days=min(60, len(bt_data) - LOOKBACK_WINDOW - 2),
            )
            logger.info("Held-out backtest results:")
            for k, v in bt_results["stats"].items():
                logger.info("  %s: %s", k, v)
        except Exception as e:
            logger.warning("Held-out backtest failed (non-fatal): %s", e)

    logger.info("=" * 60)
    logger.info("Training complete for %s — version %s", ticker, version)
    logger.info("=" * 60)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Vecaid ensemble models for a given ticker."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--years",
        type=int,
        default=TRAIN_YEARS,
        help=f"Years of training history to use (default: {TRAIN_YEARS})",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip held-out backtest after training",
    )
    # If no arguments were passed (e.g. running directly in VS Code),
    # prompt interactively instead of crashing with SystemExit: 2.
    if len(sys.argv) == 1:
        ticker_input = input("Enter ticker symbol (e.g. AAPL): ").strip().upper()
        if not ticker_input:
            print("No ticker provided. Exiting.")
            sys.exit(1)
        train(ticker=ticker_input, years=TRAIN_YEARS, run_backtest_flag=True)
    else:
        args = parser.parse_args()
        train(
            ticker=args.ticker.upper(),
            years=args.years,
            run_backtest_flag=not args.no_backtest,
        )
