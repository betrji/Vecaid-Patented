"""
app.py — Flask web application.

This file is intentionally thin:
  - loads pre-trained models from saved_models/ at first request per ticker
  - serves predictions in <2 seconds (no training here)
  - caches results for 10 minutes (configurable)
  - rate-limits to prevent abuse
  - exposes /health and /models/<ticker> utility endpoints

To run in development:
    python app.py

To run in production:
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
"""
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from flask import Flask, jsonify, render_template, request
from pandas.tseries.offsets import BDay

from config import (
    FUND_WEIGHT,
    LOG_LEVEL,
    LOOKBACK_WINDOW,
    OPTIONS_WEIGHT,
    PORT,
    PREDICTION_CACHE_TTL_SECONDS,
    PREDICTION_LOOKBACK_DAYS,
    RATE_LIMIT,
)
from backtesting.engine import run_backtest
from data.intraday import get_intraday_features
from data.sentiment import get_sentiment
from features.pipeline import build_feature_vector
from features.technical import get_technical_indicators
from models.ensemble import predict_ensemble
from models.registry import list_versions, load_models

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Rate limiting (optional dependency)
# ---------------------------------------------------------------------------
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=[RATE_LIMIT])
    logger.info("Rate limiting enabled: %s", RATE_LIMIT)
except ImportError:
    logger.warning("flask-limiter not installed; rate limiting disabled.")
    limiter = None

# ---------------------------------------------------------------------------
# In-memory prediction cache (thread-safe, TTL-based)
# ---------------------------------------------------------------------------
_pred_cache: Dict[str, Dict[str, Any]] = {}
_pred_cache_lock = threading.Lock()


def _cache_get(key: str) -> Optional[Any]:
    with _pred_cache_lock:
        entry = _pred_cache.get(key)
        if entry and (time.monotonic() - entry["ts"] < PREDICTION_CACHE_TTL_SECONDS):
            logger.debug("Cache HIT: %s", key)
            return entry["value"]
    return None


def _cache_set(key: str, value: Any) -> None:
    with _pred_cache_lock:
        _pred_cache[key] = {"value": value, "ts": time.monotonic()}


# ---------------------------------------------------------------------------
# Per-ticker model cache (load once, reuse across requests)
# ---------------------------------------------------------------------------
_models_cache: Dict[str, Any] = {}
_models_lock = threading.Lock()


def _get_models(ticker: str) -> Optional[Dict[str, Any]]:
    with _models_lock:
        if ticker not in _models_cache:
            m = load_models(ticker)
            if m:
                _models_cache[ticker] = m
            else:
                return None
        return _models_cache.get(ticker)


# ---------------------------------------------------------------------------
# Overlay signals (options + fundamentals)
# ---------------------------------------------------------------------------

def _options_signal(ticker: str, current_price: float) -> int:
    try:
        obj = yf.Ticker(ticker)
        expirations = obj.options
        if not expirations:
            return 0
        chain = obj.option_chain(expirations[0])
        calls, puts = chain.calls, chain.puts
        if calls.empty and puts.empty:
            return 0
        if not calls.empty and not puts.empty:
            hc = calls.loc[calls["volume"].idxmax()]
            hp = puts.loc[puts["volume"].idxmax()]
            if hc["volume"] >= hp["volume"]:
                return 1 if current_price < hc["strike"] else -1
            return -1 if current_price > hp["strike"] else 1
        if not calls.empty:
            hc = calls.loc[calls["volume"].idxmax()]
            return 1 if current_price < hc["strike"] else -1
        hp = puts.loc[puts["volume"].idxmax()]
        return -1 if current_price > hp["strike"] else 1
    except Exception as e:
        logger.debug("Options signal error for %s: %s", ticker, e)
        return 0


def _fundamental_signal(ticker: str) -> int:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return 0
    if not info:
        return 0
    sig = 0
    pe = info.get("trailingPE")
    if pe is not None:
        sig += 1 if pe < 15 else (-1 if pe > 25 else 0)
    sent = get_sentiment(ticker)
    sig += 1 if sent > 0.2 else (-1 if sent < -0.2 else 0)
    return sig


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_ticker(ticker: str) -> Optional[str]:
    """
    Returns None if valid, or an error string if invalid.
    Checks format first (fast), then yfinance (slow but authoritative).
    """
    if not ticker or len(ticker) > 10:
        return "Ticker must be 1–10 characters."
    if not ticker.replace(".", "").replace("-", "").isalpha():
        return f"'{ticker}' doesn't look like a valid ticker."
    try:
        fi = yf.Ticker(ticker).fast_info
        price = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if not price or price <= 0:
            return f"No live price data found for '{ticker}'. Check the symbol."
    except Exception:
        return f"Could not verify ticker '{ticker}'. Please try again."
    return None


# ---------------------------------------------------------------------------
# Core prediction pipeline
# ---------------------------------------------------------------------------

def _run_prediction(ticker: str) -> Dict[str, Any]:
    """Build and return a prediction dict, using cache where available."""
    cache_key = f"pred:{ticker}"
    cached = _cache_get(cache_key)
    if cached:
        return {**cached, "from_cache": True}

    models = _get_models(ticker)
    if not models:
        return {
            "error": (
                f"No trained model found for {ticker}. "
                "Run: python train.py --ticker " + ticker
            )
        }

    scaler = models["scaler"]
    pca = models["pca"]

    # Fetch recent price data
    logger.info("Fetching live data for %s...", ticker)
    raw = yf.download(
        ticker,
        period=f"{PREDICTION_LOOKBACK_DAYS}d",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return {"error": f"No price data available for {ticker}."}

    if isinstance(raw.columns, pd.MultiIndex):
        lvl1 = raw.columns.get_level_values(1)
        raw = raw.xs(ticker, level=1, axis=1) if ticker in lvl1 else raw[ticker]

    data = get_technical_indicators(raw)

    intraday = get_intraday_features(ticker, days=PREDICTION_LOOKBACK_DAYS)
    data.index = pd.to_datetime(data.index)
    if not intraday.empty:
        data = data.join(intraday, how="left")
    data.fillna(0, inplace=True)

    if len(data) < LOOKBACK_WINDOW + 1:
        return {"error": "Insufficient price history for prediction (need at least 61 trading days)."}

    # Build feature vector from the last LOOKBACK_WINDOW rows
    window_data = data.iloc[-LOOKBACK_WINDOW:]
    fv = build_feature_vector(window_data).reshape(1, -1)
    fv_scaled = scaler.transform(fv)
    fv_pca = pca.transform(fv_scaled)

    pred_pct, ensemble_std = predict_ensemble(fv_pca, models)

    # Small overlay adjustments
    current_price = float(data["Close"].iloc[-1])
    opt_s = _options_signal(ticker, current_price)
    fund_s = _fundamental_signal(ticker)
    pred_pct_adj = pred_pct + FUND_WEIGHT * fund_s + OPTIONS_WEIGHT * opt_s

    predicted_price = current_price * (1 + pred_pct_adj / 100.0)
    direction = "higher" if predicted_price > current_price else "lower"

    # Confidence: inverse of ensemble disagreement
    # ensemble_std near 0 → models agree → high confidence
    confidence = float(max(0.0, min(100.0, 100.0 - ensemble_std * 15.0)))

    predicted_date = (data.index[-1] + BDay(1)).strftime("%Y-%m-%d")

    result = {
        "prediction": direction,
        "predicted_difference": round(predicted_price - current_price, 4),
        "predicted_price": round(predicted_price, 4),
        "current_price": round(current_price, 4),
        "predicted_pct_move": round(pred_pct_adj, 4),
        "confidence_percentage": round(confidence, 2),
        "ensemble_std": round(ensemble_std, 4),
        "predicted_date": predicted_date,
        "model_version": models.get("version", "unknown"),
        "model_trained_at": models.get("trained_at", "unknown"),
        "model_metrics": models.get("model_metrics", {}),
        "from_cache": False,
    }
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info", methods=["POST"])
def info():
    ticker = (request.form.get("ticker") or "").upper().strip()

    # Basic format validation
    if not ticker:
        return render_template("index.html", error="Please enter a ticker symbol.")

    validation_error = _validate_ticker(ticker)
    if validation_error:
        return render_template("index.html", error=validation_error)

    # Prediction
    try:
        ml_pred = _run_prediction(ticker)
    except Exception as e:
        logger.exception("Prediction error for %s", ticker)
        ml_pred = {"error": str(e)}

    # Backtest (uses the same cached models; fetches its own recent data)
    backtest_results: Dict[str, Any] = {}
    if "error" not in ml_pred:
        models = _get_models(ticker)
        if models:
            try:
                raw_bt = yf.download(
                    ticker, period="200d", interval="1d",
                    progress=False, auto_adjust=True,
                )
                if isinstance(raw_bt.columns, pd.MultiIndex):
                    lvl1 = raw_bt.columns.get_level_values(1)
                    raw_bt = raw_bt.xs(ticker, level=1, axis=1) if ticker in lvl1 else raw_bt[ticker]
                bt_data = get_technical_indicators(raw_bt)
                intraday_bt = get_intraday_features(ticker, days=60)
                bt_data.index = pd.to_datetime(bt_data.index)
                if not intraday_bt.empty:
                    bt_data = bt_data.join(intraday_bt, how="left")
                bt_data.fillna(0, inplace=True)
                backtest_results = run_backtest(
                    bt_data,
                    models,
                    models["scaler"],
                    models["pca"],
                    window=LOOKBACK_WINDOW,
                    n_test_days=60,
                )
            except Exception as e:
                logger.warning("Backtest error for %s: %s", ticker, e)
                backtest_results = {"error": str(e)}

    results = {
        "ticker": ticker,
        "ml_prediction": ml_pred,
        "backtest": backtest_results,
    }
    return render_template("info.html", results=results)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"})


@app.route("/models/<ticker>")
def model_info(ticker: str):
    ticker = ticker.upper()
    versions = list_versions(ticker)
    models = _get_models(ticker)
    return jsonify({
        "ticker": ticker,
        "available_versions": versions,
        "active_version": models.get("version") if models else None,
        "trained_at": models.get("trained_at") if models else None,
        "metrics": models.get("model_metrics") if models else None,
    })


if __name__ == "__main__":
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(debug=debug, host="0.0.0.0", port=PORT)
