"""
features/technical.py — technical indicator calculations.

Key normalisation fixes vs. original:
  - HLC3, TEMA, FWMA are expressed as % deviations from Close so the same
    model works across $5 stocks and $500 stocks.
  - MACD is normalised by Close (same reason).
  - VWAP is expressed as % deviation from Close.
  - ATR is expressed as a ratio to Close (ATR_ratio).
  - OBV rate-of-change (OBV_roc, 5-day) replaces the raw OBV level.
  - Force Index is normalised to avoid magnitude dominance.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = data["High"], data["Low"], data["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def compute_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    data = data.copy()
    high, low, close = data["High"], data["Low"], data["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * pd.Series(plus_dm, index=data.index).rolling(period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm, index=data.index).rolling(period).sum() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data["DI_plus"] = plus_di.values
    data["DI_minus"] = minus_di.values
    data["ADX"] = dx.rolling(period).mean().values
    return data


def compute_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    mf = tp * data["Volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp <= tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(period, min_periods=1).sum()
    neg_sum = neg_mf.rolling(period, min_periods=1).sum()
    return 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))


def stochastic_oscillator(data: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    data = data.copy()
    ll = data["Low"].rolling(k, min_periods=1).min()
    hh = data["High"].rolling(k, min_periods=1).max()
    data["%K"] = 100 * (data["Close"] - ll) / (hh - ll).replace(0, np.nan)
    data["%D"] = data["%K"].rolling(d, min_periods=1).mean()
    return data


def chaikin_money_flow(data: pd.DataFrame, period: int = 20) -> pd.Series:
    denom = (data["High"] - data["Low"]).replace(0, np.nan)
    ad = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / denom * data["Volume"]
    vol_sum = data["Volume"].rolling(period, min_periods=1).sum()
    return ad.rolling(period, min_periods=1).sum() / vol_sum.replace(0, np.nan)


def commodity_channel_index(data: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    sma = tp.rolling(period, min_periods=1).mean()
    md = tp.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


# ---------------------------------------------------------------------------
# Normalised composite indicators
# ---------------------------------------------------------------------------

def bollinger_bands(data: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.DataFrame:
    data = data.copy()
    mid = data["Close"].rolling(period, min_periods=1).mean()
    std = data["Close"].rolling(period, min_periods=1).std()
    data["BB_middle"] = mid
    data["BB_upper"] = mid + num_std * std
    data["BB_lower"] = mid - num_std * std
    data["BB_width"] = (data["BB_upper"] - data["BB_lower"]) / mid.replace(0, np.nan)
    data["SMA20"] = mid
    return data


def tema(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Triple EMA expressed as % deviation from Close (avoids price-level leakage)."""
    data = data.copy()
    ema1 = data["Close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    raw = 3 * ema1 - 3 * ema2 + ema3
    data["TEMA_dev"] = (raw - data["Close"]) / data["Close"].replace(0, np.nan)
    return data


def fwma(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Linearly-weighted MA expressed as % deviation from Close."""
    data = data.copy()
    weights = np.arange(1, period + 1, dtype=float)

    def _wm(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum()) if len(x) == period else np.nan

    raw = data["Close"].rolling(period).apply(_wm, raw=True)
    data["FWMA_dev"] = (raw - data["Close"]) / data["Close"].replace(0, np.nan)
    return data


# ---------------------------------------------------------------------------
# Master indicator builder
# ---------------------------------------------------------------------------

def get_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on a daily OHLCV DataFrame.
    Returns the DataFrame with indicator columns appended.
    All price-level quantities are normalised to avoid ticker-dependency.
    """
    data = data.copy()
    close = data["Close"]

    # --- Normalised price-level features ---
    hlc3 = (data["High"] + data["Low"] + close) / 3
    data["hlc3_ratio"] = hlc3 / close.replace(0, np.nan) - 1   # near zero, dimensionless

    data = bollinger_bands(data)
    data = tema(data)
    data = fwma(data)

    # --- Volatility ---
    data["std_ratio"] = close.rolling(20, min_periods=1).std() / close.replace(0, np.nan)

    # --- RSI ---
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = up.rolling(14, min_periods=1).mean() / down.rolling(14, min_periods=1).mean().replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (normalised by close) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    signal_raw = macd_raw.ewm(span=9, adjust=False).mean()
    data["MACD_norm"] = macd_raw / close.replace(0, np.nan)
    data["Signal_norm"] = signal_raw / close.replace(0, np.nan)
    # Keep raw MACD for legacy strategy calls
    data["MACD"] = macd_raw
    data["Signal_Line"] = signal_raw

    # --- VWAP deviation ---
    data["VWAP"] = (hlc3 * data["Volume"]).cumsum() / data["Volume"].cumsum()
    data["VWAP_dev"] = (close - data["VWAP"]) / close.replace(0, np.nan)

    # --- ROC ---
    data["ROC"] = close.pct_change() * 100
    data["ROC_short"] = close.pct_change(3) * 100

    # --- Stochastics / ADX ---
    data = stochastic_oscillator(data)
    data = compute_adx(data)

    # --- Volume-based ---
    data["CMF"] = chaikin_money_flow(data)
    data["CCI"] = commodity_channel_index(data)
    data["MFI"] = compute_mfi(data)
    force_raw = close.diff(1) * data["Volume"]
    data["Force_Index_norm"] = force_raw / (close * data["Volume"] + 1e-9)

    # --- MAs ---
    data["SMA50"] = close.rolling(50, min_periods=1).mean()
    data["SMA200"] = close.rolling(200, min_periods=1).mean()
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA12"] = ema12
    data["EMA26"] = ema26

    # --- OBV + normalised OBV change ---
    obv = [0]
    for i in range(1, len(data)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + data["Volume"].iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - data["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    data["OBV"] = obv
    # OBV rate-of-change (5-day) replaces raw level; bounded, dimensionless
    data["OBV_roc"] = data["OBV"].pct_change(5).clip(-1, 1) * 100

    # --- Other indicators ---
    data["vol_SMA20"] = data["Volume"].rolling(20, min_periods=1).mean()
    data["Momentum"] = close.pct_change(10) * 100
    data["Donchian_High"] = data["High"].rolling(20, min_periods=1).max()
    data["Donchian_Low"] = data["Low"].rolling(20, min_periods=1).min()
    data["Williams_R"] = -100 * (
        (data["High"].rolling(14, min_periods=1).max() - close) /
        (
            data["High"].rolling(14, min_periods=1).max() -
            data["Low"].rolling(14, min_periods=1).min()
        ).replace(0, np.nan)
    )
    data["ATR"] = compute_atr(data, 14)
    data["ATR_ratio"] = data["ATR"] / close.replace(0, np.nan)   # normalised

    return data
