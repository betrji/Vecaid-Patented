"""
models/registry.py — save, load, and version model artifacts.

Directory layout:
    saved_models/
        <TICKER>/
            latest.txt            (contains version string of current live model)
            <YYYYMMDD_HHMMSS>/    (version directory)
                gru/              (Keras SavedModel format)
                cnn_lstm/
                cnn_lstm_v2/
                bilstm/
                dnn/
                transformer/
                xgb_0.ubj         (XGBoost binary)
                xgb_1.ubj
                xgb_2.ubj
                meta_model.ubj
                scaler.pkl        (joblib)
                pca.pkl
                metadata.json
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from config import SAVED_MODELS_DIR
from models.builders import BASE_MODEL_BUILDERS

logger = logging.getLogger(__name__)


def _version_dir(ticker: str, version: str) -> Path:
    d = SAVED_MODELS_DIR / ticker.upper() / version
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_models(
    ticker: str,
    models: Dict[str, Any],
    scaler,
    pca,
) -> str:
    """Persist all model artefacts.  Returns the version string."""
    version = time.strftime("%Y%m%d_%H%M%S")
    d = _version_dir(ticker, version)

    # Keras base models
    for name in BASE_MODEL_BUILDERS:
        if name in models:
            models[name].save(str(d / name))
            logger.debug("Saved Keras model: %s", name)

    # XGBoost ensemble
    for i, xm in enumerate(models.get("xgb_models", [])):
        xm.save_model(str(d / f"xgb_{i}.ubj"))

    # Meta-model
    if "meta_model" in models:
        models["meta_model"].save_model(str(d / "meta_model.ubj"))

    # Preprocessors
    joblib.dump(scaler, d / "scaler.pkl")
    joblib.dump(pca, d / "pca.pkl")

    # Metadata (metrics + version info)
    metadata = {
        "ticker": ticker.upper(),
        "version": version,
        "input_dim": models.get("input_dim"),
        "model_metrics": models.get("model_metrics", {}),
        "trained_at": version,
    }
    (d / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Write latest pointer
    (SAVED_MODELS_DIR / ticker.upper() / "latest.txt").write_text(version)

    logger.info("Saved models for %s — version %s", ticker.upper(), version)
    return version


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_models(ticker: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load all model artefacts for a given ticker.
    Defaults to the latest saved version.
    Returns None if no models have been trained for this ticker.
    """
    import tensorflow as tf
    from xgboost import XGBRegressor

    ticker = ticker.upper()
    ticker_dir = SAVED_MODELS_DIR / ticker

    if not ticker_dir.exists():
        logger.warning("No saved models directory for %s", ticker)
        return None

    if version is None:
        latest_file = ticker_dir / "latest.txt"
        if not latest_file.exists():
            logger.warning("No latest.txt for %s", ticker)
            return None
        version = latest_file.read_text().strip()

    d = ticker_dir / version
    if not d.exists():
        logger.error("Version directory not found: %s", d)
        return None

    models: Dict[str, Any] = {}

    # Keras base models
    for name in BASE_MODEL_BUILDERS:
        path = d / name
        if path.exists():
            models[name] = tf.keras.models.load_model(str(path))
        else:
            logger.warning("Keras model not found: %s", path)

    # XGBoost ensemble
    xgb_models = []
    for i in range(10):
        p = d / f"xgb_{i}.ubj"
        if p.exists():
            xm = XGBRegressor()
            xm.load_model(str(p))
            xgb_models.append(xm)
        else:
            break
    models["xgb_models"] = xgb_models

    # Meta-model
    meta_path = d / "meta_model.ubj"
    if meta_path.exists():
        mm = XGBRegressor()
        mm.load_model(str(meta_path))
        models["meta_model"] = mm

    # Preprocessors
    models["scaler"] = joblib.load(d / "scaler.pkl")
    models["pca"] = joblib.load(d / "pca.pkl")

    # Metadata
    meta = json.loads((d / "metadata.json").read_text())
    models["model_metrics"] = meta.get("model_metrics", {})
    models["input_dim"] = meta.get("input_dim")
    models["trained_at"] = meta.get("trained_at")
    models["version"] = version

    logger.info("Loaded models for %s — version %s", ticker, version)
    return models


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def list_versions(ticker: str) -> List[str]:
    """Return all available version strings for a ticker, newest first."""
    ticker_dir = SAVED_MODELS_DIR / ticker.upper()
    if not ticker_dir.exists():
        return []
    return sorted(
        [d.name for d in ticker_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
