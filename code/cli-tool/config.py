"""Configuration and paths for adversarial stylometry testing."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
STUFF_ROOT = Path(__file__).parent

# Model paths
MODELS_DIR = PROJECT_ROOT / "saved_models" / "attribution"
MODEL_PATHS = {
    "random_forest": MODELS_DIR / "random_forest",
    "sgd": MODELS_DIR / "sgd_classifier",
    "naive_bayes": MODELS_DIR / "naive_bayes",
    "neural_network": MODELS_DIR / "neural_network",
}

# Data paths
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_SPLITS_DIR = PROJECT_ROOT / "dataset_splits"
ADVERSARIAL_SAMPLES_DIR = PROJECT_ROOT / "adversarial_samples"

# Output paths
RESULTS_DIR = STUFF_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Authors in training set
AUTHORS = [
    "RasaHQ",
    "aleju",
    "gradio-app",
    "ivy-llc",
    "lazyprogrammer",
    "microsoft",
    "onnx",
    "scikit-learn",
    "scikit-learn-contrib",
    "sktime",
    "Trusted-AI",
    "aladdinpersson",
    "automl",
    "clips",
    "cvat-ai",
    "doccano",
    "harvard-edge",
    "huggingface",
    "lawlite19",
    "uber",
]

# Model file extensions
MODEL_EXTENSIONS = {
    "model": "_model.joblib",
    "vectorizer": "_vectorizer.joblib",
    "style_vectorizer": "_style_vectorizer.joblib",
    "scaler": "_scaler.joblib",
    "style_scaler": "_style_scaler.joblib",
    "selector": "_selector.joblib",
    "metadata": "_metadata.joblib",
}
