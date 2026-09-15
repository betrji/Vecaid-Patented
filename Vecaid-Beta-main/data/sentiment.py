"""
data/sentiment.py — news sentiment helper.

IMPORTANT: The original Yahoo Finance HTML scraper broke constantly and
returned 0 most of the time anyway.  This module is intentionally stubbed.

To add real sentiment in a future iteration:
  - Use Alpaca News API (free tier available)
  - Use NewsAPI.org (free tier for 30-day lookback)
  - Use FinBERT via HuggingFace for better finance-specific scores

For now, all functions return 0.0 (neutral) so the rest of the pipeline
is unaffected.  Set SENTIMENT_ENABLED=1 in your .env and implement
_fetch_headlines() once you have a provider.
"""
import logging
import os

logger = logging.getLogger(__name__)

SENTIMENT_ENABLED = os.environ.get("SENTIMENT_ENABLED", "0") == "1"


def get_sentiment(ticker: str) -> float:
    """
    Returns a sentiment score in [-1.0, 1.0].
    Currently returns 0.0 (neutral stub).
    """
    if not SENTIMENT_ENABLED:
        return 0.0

    # --- Placeholder for future implementation ---
    # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    # headlines = _fetch_headlines(ticker)
    # analyzer = SentimentIntensityAnalyzer()
    # scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    # return sum(scores) / len(scores) if scores else 0.0

    logger.debug("Sentiment stubbed for %s — returning 0.0", ticker)
    return 0.0
