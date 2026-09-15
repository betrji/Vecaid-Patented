"""
data/market.py — yfinance downloads and daily+weekly merging.

Download strategy (in order):
  1. yf.download()        — fastest, multi-ticker capable
  2. Ticker.history()     — different endpoint, usually not rate-limited simultaneously
  3. Retry with back-off  — 3 attempts with increasing delays on each method
"""
import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [10, 30, 60]   # seconds between retries per method


def _via_download(ticker: str, interval: str, start: str = None,
                  end: str = None, period: str = None) -> pd.DataFrame:
    kwargs = dict(interval=interval, progress=False, auto_adjust=True)
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    return yf.download(ticker, **kwargs)


def _via_history(ticker: str, interval: str, start: str = None,
                 end: str = None, period: str = None) -> pd.DataFrame:
    """Ticker.history() uses a different API endpoint — good fallback."""
    obj = yf.Ticker(ticker)
    kwargs = dict(interval=interval, auto_adjust=True)
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    return obj.history(**kwargs)


def _download_with_retry(ticker: str, interval: str, start: str = None,
                         end: str = None, period: str = None) -> pd.DataFrame:
    """
    Try yf.download first, fall back to Ticker.history(), with retries on each.
    Raises ValueError only after all strategies are exhausted.
    """
    strategies = [
        ("yf.download",       _via_download),
        ("Ticker.history",    _via_history),
    ]

    last_exc = None
    for strategy_name, strategy_fn in strategies:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                df = strategy_fn(ticker, interval, start=start, end=end, period=period)
                if not df.empty:
                    if attempt > 1 or strategy_name != "yf.download":
                        logger.info("Succeeded via %s (attempt %d)", strategy_name, attempt)
                    return df
                logger.warning(
                    "[%s] Empty result for %s (%s) — attempt %d/%d",
                    strategy_name, ticker, interval, attempt, len(_RETRY_DELAYS),
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] Error for %s (%s) attempt %d/%d: %s",
                    strategy_name, ticker, interval, attempt, len(_RETRY_DELAYS), exc,
                )

            if attempt < len(_RETRY_DELAYS):
                logger.info("Waiting %ds before next attempt…", delay)
                time.sleep(delay)

    msg = (
        f"All download strategies failed for {ticker} ({interval}). "
        f"Last error: {last_exc}. "
        "yfinance may be rate-limiting your IP — wait a few minutes and try again."
    )
    raise ValueError(msg)


def _flatten_multiindex(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse a yfinance MultiIndex column frame to a flat one."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)
    if ticker in lvl0:
        return df[ticker]
    if ticker in lvl1:
        return df.xs(ticker, level=1, axis=1)
    df.columns = df.columns.droplevel(1)
    return df


def download_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = _download_with_retry(ticker, "1d", start=start, end=end)
    return _flatten_multiindex(df, ticker)


def download_weekly(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = _download_with_retry(ticker, "1wk", start=start, end=end)
        return _flatten_multiindex(df, ticker)
    except ValueError as e:
        logger.warning("Skipping weekly data for %s: %s", ticker, e)
        return pd.DataFrame()


def build_merged_daily_weekly(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily + weekly data, forward-fill weekly to daily granularity,
    and merge on Date index.
    """
    daily = download_daily(ticker, start, end)
    daily.index = pd.to_datetime(daily.index)

    weekly = download_weekly(ticker, start, end)
    if weekly.empty:
        return daily.sort_index()

    weekly.index = pd.to_datetime(weekly.index)
    weekly_ff = weekly.resample("1D").ffill()
    weekly_ff.columns = [f"{c}_wk" for c in weekly_ff.columns]

    merged = daily.join(weekly_ff, how="left")
    merged.sort_index(inplace=True)
    merged.ffill(inplace=True)
    return merged
