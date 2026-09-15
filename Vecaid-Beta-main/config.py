"""
config.py — all constants, hyperparameters, and environment variables.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
TRAIN_YEARS = 7               # years of daily history used for training
LOOKBACK_WINDOW = 60          # rows fed into each feature-vector computation
PREDICTION_LOOKBACK_DAYS = 90 # days of recent data downloaded for live inference

# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------
N_CV_SPLITS = 3
CV_YEAR_LEN = 504   # ~2 trading years per fold (expanding train window)
CV_MONTH_LEN = 42   # ~2 months held-out per fold

# ---------------------------------------------------------------------------
# Neural networks
# GPU note: batch size 256 keeps the RTX 3060 Ti well-saturated.
# On CPU, drop NN_BATCH_SIZE back to 32 and NN_EPOCHS to 50.
# ---------------------------------------------------------------------------
NN_EPOCHS = 100
NN_BATCH_SIZE = 256   # larger batches = better GPU utilisation
NN_VALIDATION_SPLIT = 0.1

# ---------------------------------------------------------------------------
# XGBoost Bayesian hyper-parameter search
# ---------------------------------------------------------------------------
BAYES_INIT_POINTS = 5
BAYES_N_ITER = 20
XGB_N_ESTIMATORS = 150

# ---------------------------------------------------------------------------
# Prediction adjustments (small weights for options/fundamentals overlay)
# ---------------------------------------------------------------------------
FUND_WEIGHT = 0.15
OPTIONS_WEIGHT = 0.05

# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
PREDICTION_CACHE_TTL_SECONDS = 600   # 10 minutes
RATE_LIMIT = "10 per minute"
PORT = int(os.environ.get("PORT", 5000))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
