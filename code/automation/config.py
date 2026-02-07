"""
Configuration for the Adversarial Stylometry Automation System.

This module defines categories, paths, and model configurations used
throughout the automation pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
AUTOMATION_DIR = Path(__file__).parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models" / "attribution"
RESULTS_DIR = AUTOMATION_DIR / "results"

# Transformation categories
CATEGORIES = [
    "restructuring",
    "renaming",
    "formatting",
    "comments",
]

# Model types and their configurations
MODEL_TYPES = {
    "random_forest": {
        "name": "Random Forest",
        "short_name": "rf",
        "is_neural_network": False,
        "confidence_threshold": 0.4,
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "short_name": "nb",
        "is_neural_network": False,
        "confidence_threshold": 0.85,
    },
    "sgd_classifier": {
        "name": "SGD Classifier",
        "short_name": "sgd",
        "is_neural_network": False,
        "confidence_threshold": 0.8,
    },
    "neural_network": {
        "name": "Neural Network",
        "short_name": "nn",
        "is_neural_network": True,
        "confidence_threshold": 0.6,
    },
}

# Spreadsheet column definitions
SPREADSHEET_COLUMNS = {
    "metadata": [
        "run_id",
        "timestamp",
        "parent_run_id",
        "category",
        "prompt_summary",
        "author",
        "file_name",
        "original_file",
        "modified_file",
        "ai_tool",
    ],
    "model_results": [
        # Generated dynamically per model: {short_name}_orig_pred, {short_name}_orig_conf, etc.
    ],
    "stealthiness": [
        "l1_distance",
        "l2_distance",
        "cosine_distance",
        "relative_l2",
        "js_divergence",
        "features_changed_pct",
        "max_feature_change",
        "avg_feature_change",
        "stealth_score",
        "stealth_category",
    ],
    "aggregate": [
        "orig_accuracy_rate",
        "mod_accuracy_rate",
        "evasion_count",
        "evasion_rate_pct",
        "consensus_prediction",
        "result_type",
    ],
    "reference": [
        "json_path",
        "notes",
    ],
}

# Index columns for master index
INDEX_COLUMNS = [
    "run_id",
    "category",
    "timestamp",
    "author",
    "file_name",
    "evasion_rate",
    "category_csv_path",
    "json_path",
]


def get_all_columns():
    """Generate the complete list of spreadsheet columns including per-model columns."""
    columns = SPREADSHEET_COLUMNS["metadata"].copy()

    # Add per-model columns
    for model_type, config in MODEL_TYPES.items():
        short = config["short_name"]
        columns.extend([
            f"{short}_orig_pred",
            f"{short}_orig_conf",
            f"{short}_mod_pred",
            f"{short}_mod_conf",
            f"{short}_evasion",
        ])

    columns.extend(SPREADSHEET_COLUMNS["stealthiness"])
    columns.extend(SPREADSHEET_COLUMNS["aggregate"])
    columns.extend(SPREADSHEET_COLUMNS["reference"])

    return columns


def get_category_results_dir(category: str) -> Path:
    """Get the results directory for a specific category."""
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {CATEGORIES}")
    return RESULTS_DIR / category


def get_category_csv_path(category: str) -> Path:
    """Get the path to the runs.csv file for a specific category."""
    return get_category_results_dir(category) / "runs.csv"


def get_category_json_dir(category: str) -> Path:
    """Get the JSON directory for a specific category."""
    return get_category_results_dir(category) / "json"


def get_index_path() -> Path:
    """Get the path to the master index.csv file."""
    return RESULTS_DIR / "index.csv"


def get_model_dir(model_type: str) -> Path:
    """Get the directory for a specific model type."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type '{model_type}'. Must be one of: {list(MODEL_TYPES.keys())}")
    return SAVED_MODELS_DIR / model_type


# Ensure results directories exist
def ensure_results_dirs():
    """Create results directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        get_category_results_dir(category).mkdir(parents=True, exist_ok=True)
        get_category_json_dir(category).mkdir(parents=True, exist_ok=True)
