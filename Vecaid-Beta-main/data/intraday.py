"""
data/intraday.py — intraday 5-minute bar aggregation to daily features.
"""
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_intraday_features(ticker: str, days: int = 60, interval: str = "5m") -> pd.DataFrame:
    """
    Download intraday bars and aggregate to daily statistics.
    Returns a DataFrame indexed by date with columns:
        intraday_high, intraday_low, intraday_range, intraday_vol, intraday_avg_vol
    Returns an empty DataFrame on failure (caller should left-join).
    """
    try:
        intraday = yf.download(
            ticker, period=f"{days}d", interval=interval,
            progress=False, auto_adjust=True,
        )
    except Exception as e:
        logger.warning("Intraday download failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if intraday.empty:
        logger.warning("No intraday %s data for %s.", interval, ticker)
        return pd.DataFrame()

    if isinstance(intraday.columns, pd.MultiIndex):
        lvl0 = intraday.columns.get_level_values(0)
        lvl1 = intraday.columns.get_level_values(1)
        if ticker in lvl0:
            intraday = intraday[ticker]
        elif ticker in lvl1:
            intraday = intraday.xs(ticker, level=1, axis=1)

    intraday["_date"] = intraday.index.date
    grouped = intraday.groupby("_date")

    agg = pd.DataFrame({
        "intraday_high": grouped["High"].max(),
        "intraday_low": grouped["Low"].min(),
        "intraday_range": grouped["High"].max() - grouped["Low"].min(),
        "intraday_vol": grouped["Volume"].sum(),
        "intraday_avg_vol": grouped["Volume"].mean(),
    })
    agg.index = pd.to_datetime(agg.index)
    agg.sort_index(inplace=True)
    return agg
