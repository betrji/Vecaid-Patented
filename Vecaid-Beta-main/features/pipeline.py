"""
features/pipeline.py — assembles the final feature vector from a window of
technical-indicator data, and builds training matrices with proper targets.

All features are either already normalised signals (-1/0/1) or expressed as
price-independent ratios/percentages.  StandardScaler + PCA are applied on
top in train.py.
"""
import numpy as np
import pandas as pd

from features.strategies import (
    technical_signal,
    stochastic_strategy,
    adx_strategy,
    roc_strategy,
    bb_strategy,
    rsi_strategy,
    macd_strategy,
    sma_crossover_strategy,
    obv_strategy,
    volume_strategy,
    momentum_strategy,
)

# Canonical ordered feature names (used for documentation / debugging)
FEATURE_NAMES = [
    "tech_sig",
    "cmf",
    "cci_norm",
    "intraday_range_ratio",
    "stochastic_sig",
    "adx_sig",
    "roc_sig",
    "bb_sig",
    "rsi_sig",
    "macd_sig",
    "sma_cross_sig",
    "obv_sig",
    "volume_sig",
    "momentum_sig",
    "hlc3_ratio",
    "tema_dev",
    "fwma_dev",
    "bb_width",
    "atr_ratio",
    "vwap_dev",
    "rsi_norm",
    "obv_roc_norm",
    "std_ratio",
    "williams_r_norm",
    "mfi_norm",
]


def build_feature_vector(window_data: pd.DataFrame) -> np.ndarray:
    """
    Given a window (lookback rows) of data with technical indicators,
    return a 1-D float64 feature vector.

    All values are dimensionless and bounded (ratio or -1..1 signal).
    """
    latest = window_data.iloc[-1]
    close = float(latest.get("Close", 1.0)) or 1.0

    # Aggregate signals
    tech = technical_signal(latest, window_data)
    cmf = float(latest.get("CMF", 0.0) or 0.0)
    cci_norm = float(latest.get("CCI", 0.0) or 0.0) / 200.0   # CCI roughly ±200 → ±1

    # Intraday range as fraction of close (price-independent)
    intraday_range = float(latest.get("intraday_range", 0.0) or 0.0)
    intraday_range_ratio = intraday_range / close

    # Discrete strategy signals
    stoch_s = float(stochastic_strategy(latest))
    adx_s = float(adx_strategy(latest))
    roc_s = float(roc_strategy(latest))
    bb_s = float(bb_strategy(latest))
    rsi_s = float(rsi_strategy(latest))
    macd_s = float(macd_strategy(latest))
    sma_s = float(sma_crossover_strategy(latest))
    obv_s = float(obv_strategy(window_data))
    vol_s = float(volume_strategy(latest))
    mom_s = float(momentum_strategy(latest))

    # Normalised continuous features
    hlc3_ratio = float(latest.get("hlc3_ratio", 0.0) or 0.0)
    tema_dev = float(latest.get("TEMA_dev", 0.0) or 0.0)
    fwma_dev = float(latest.get("FWMA_dev", 0.0) or 0.0)
    bb_width = float(latest.get("BB_width", 0.0) or 0.0)
    atr_ratio = float(latest.get("ATR_ratio", 0.0) or 0.0)
    vwap_dev = float(latest.get("VWAP_dev", 0.0) or 0.0)
    rsi_norm = (float(latest.get("RSI", 50.0) or 50.0) - 50.0) / 50.0   # centre+scale
    obv_roc_norm = float(latest.get("OBV_roc", 0.0) or 0.0) / 100.0      # ±1 range
    std_ratio = float(latest.get("std_ratio", 0.0) or 0.0)
    williams_r_norm = (float(latest.get("Williams_R", -50.0) or -50.0) + 50.0) / 50.0
    mfi_norm = (float(latest.get("MFI", 50.0) or 50.0) - 50.0) / 50.0

    vec = np.array([
        tech,
        cmf,
        cci_norm,
        intraday_range_ratio,
        stoch_s,
        adx_s,
        roc_s,
        bb_s,
        rsi_s,
        macd_s,
        sma_s,
        obv_s,
        vol_s,
        mom_s,
        hlc3_ratio,
        tema_dev,
        fwma_dev,
        bb_width,
        atr_ratio,
        vwap_dev,
        rsi_norm,
        obv_roc_norm,
        std_ratio,
        williams_r_norm,
        mfi_norm,
    ], dtype=np.float64)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_matrix(
    data: pd.DataFrame,
    window: int = 60,
) -> tuple:
    """
    Slide a lookback window over `data` and build (X, y, dates).

    Target y[i] = next-day close % change from current close.
    Temporal ordering is preserved; no shuffling.

    Returns:
        X     : np.ndarray of shape (n_samples, n_features)
        y     : np.ndarray of shape (n_samples,)
        dates : list of pd.Timestamp — the *target* date for each sample
    """
    features, targets, dates = [], [], []

    for i in range(window, len(data) - 1):
        w = data.iloc[i - window:i]
        fv = build_feature_vector(w)
        features.append(fv)

        current_price = float(data["Close"].iloc[i])
        next_price = float(data["Close"].iloc[i + 1])
        pct = ((next_price - current_price) / current_price) * 100.0
        targets.append(pct)
        dates.append(data.index[i + 1])

    return np.array(features, dtype=np.float64), np.array(targets, dtype=np.float64), dates
