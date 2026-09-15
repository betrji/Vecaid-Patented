"""
features/strategies.py — signal functions returning -1, 0, or 1.

Bug fixes vs. original:
  - obv_strategy: now uses OBV_roc (5-day rate of change) instead of the
    broken `current_obv * 0.99` comparison which always returned 1.
  - macd_strategy: uses MACD_norm / Signal_norm (already normalised by close).
  - All strategies that need the full window receive `window_data` explicitly.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual strategy signals
# ---------------------------------------------------------------------------

def stochastic_strategy(latest: pd.Series) -> int:
    k, d = latest.get("%K", 50), latest.get("%D", 50)
    if k < 20 and d < 20:
        return 1
    if k > 80 and d > 80:
        return -1
    return 0


def adx_strategy(latest: pd.Series, threshold: float = 25) -> int:
    adx_val = latest.get("ADX", 0)
    di_plus = latest.get("DI_plus", 0)
    di_minus = latest.get("DI_minus", 0)
    if adx_val > threshold:
        return 1 if di_plus > di_minus else -1
    return 0


def roc_strategy(latest: pd.Series, threshold: float = 5) -> int:
    roc = latest.get("ROC", 0)
    if roc > threshold:
        return 1
    if roc < -threshold:
        return -1
    return 0


def bb_strategy(latest: pd.Series) -> int:
    lower = latest.get("BB_lower", 0)
    upper = latest.get("BB_upper", 0)
    close = latest.get("Close", 0)
    if close == 0:
        return 0
    if (close - lower) / close < 0.02:
        return 1
    if (upper - close) / close < 0.02:
        return -1
    return 0


def rsi_strategy(latest: pd.Series, lower: float = 30, upper: float = 70) -> int:
    rsi = latest.get("RSI", 50)
    if rsi < lower:
        return 1
    if rsi > upper:
        return -1
    return 0


def macd_strategy(latest: pd.Series) -> int:
    """Uses normalised MACD so signal is price-independent."""
    return 1 if latest.get("MACD_norm", 0) > latest.get("Signal_norm", 0) else -1


def sma_crossover_strategy(latest: pd.Series) -> int:
    sma50 = latest.get("SMA50", 0)
    sma200 = latest.get("SMA200", 0)
    return 1 if sma50 > sma200 else -1


def obv_strategy(window_data: pd.DataFrame) -> int:
    """
    Fixed: compares OBV rate-of-change over the last 5 days.
    Original bug: `prev_obv = current_obv * 0.99` — always compared OBV to
    itself, so OBV > 0 almost always returned 1.
    """
    if len(window_data) < 2:
        return 0
    obv_roc = window_data["OBV_roc"].iloc[-1]
    if pd.isna(obv_roc):
        return 0
    if obv_roc > 5:
        return 1
    if obv_roc < -5:
        return -1
    return 0


def volume_strategy(latest: pd.Series) -> int:
    vol_sma = latest.get("vol_SMA20", 0)
    vol = latest.get("Volume", 0)
    close = latest.get("Close", 0)
    sma20 = latest.get("SMA20", 0)
    if vol_sma > 0 and vol > 1.5 * vol_sma:
        return 1 if close > sma20 else -1
    return 0


def momentum_strategy(latest: pd.Series, threshold: float = 2) -> int:
    mom = latest.get("Momentum", 0)
    if pd.isna(mom):
        return 0
    if mom > threshold:
        return 1
    if mom < -threshold:
        return -1
    return 0


def bollinger_b_percent_strategy(latest: pd.Series) -> int:
    lower = latest.get("BB_lower", 0)
    upper = latest.get("BB_upper", 0)
    close = latest.get("Close", 0)
    denom = upper - lower
    if denom == 0:
        return 0
    b = (close - lower) / denom
    if b > 0.8:
        return -1
    if b < 0.2:
        return 1
    return 0


def williams_r_strategy(latest: pd.Series) -> int:
    w = latest.get("Williams_R", -50)
    if w < -80:
        return 1
    if w > -20:
        return -1
    return 0


def atr_breakout_strategy(latest: pd.Series) -> int:
    h = latest.get("High", 0)
    lo = latest.get("Low", 0)
    atr = latest.get("ATR", 0)
    if atr == 0:
        return 0
    tr = h - lo
    if tr > 1.2 * atr:
        return 1
    if tr < 0.8 * atr:
        return -1
    return 0


def vwap_reversion_strategy(latest: pd.Series, threshold: float = 0.01) -> int:
    dev = latest.get("VWAP_dev", 0)
    if dev > threshold:
        return -1
    if dev < -threshold:
        return 1
    return 0


def squeeze_momentum_indicator(latest: pd.Series, threshold: float = 0.1) -> int:
    bb_width = latest.get("BB_width", 0)
    mom = latest.get("Momentum", 0)
    if bb_width < threshold:
        return 1 if mom > 0 else -1
    return 0


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def composite_indicator_signal(latest: pd.Series, window_data: pd.DataFrame) -> float:
    signals = [
        stochastic_strategy(latest),
        adx_strategy(latest),
        roc_strategy(latest),
        bb_strategy(latest),
        rsi_strategy(latest),
        macd_strategy(latest),
        sma_crossover_strategy(latest),
        obv_strategy(window_data),      # fixed: uses window_data
        volume_strategy(latest),
        momentum_strategy(latest),
        vwap_reversion_strategy(latest),
        bollinger_b_percent_strategy(latest),
        williams_r_strategy(latest),
        atr_breakout_strategy(latest),
        squeeze_momentum_indicator(latest),
    ]
    return float(np.clip(sum(signals), -15, 15))


def technical_signal(latest: pd.Series, window_data: pd.DataFrame) -> float:
    """Aggregate technical score from core indicators + composite."""
    sig = 0.0

    rsi = latest.get("RSI", 50)
    sig += 1 if rsi < 30 else (-1 if rsi > 70 else 0)

    sig += macd_strategy(latest)

    close = latest.get("Close", 0)
    sma20 = latest.get("SMA20", 0)
    sig += 1 if close > sma20 else -1

    lower = latest.get("BB_lower", 0)
    upper = latest.get("BB_upper", 0)
    if close < lower:
        sig += 1
    elif close > upper:
        sig -= 1

    if sma20 != 0:
        dev = (close - sma20) / sma20
        if dev > 0.05:
            sig -= 1
        elif dev < -0.05:
            sig += 1

    sig += composite_indicator_signal(latest, window_data)
    return sig
