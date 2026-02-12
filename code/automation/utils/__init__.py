"""
Utilities for the Adversarial Stylometry Automation System.

This package provides:
- model_loader: Load saved attribution models
- feature_extractor: Extract features from code files
- metrics: Calculate stealthiness and evasion metrics
- results_tracker: Manage spreadsheets and JSON results
- dataset_scanner: Discover authors and files in dataset_splits/
- batch_tracker: Batch and evolution result tracking
"""

from .model_loader import load_all_models, load_model_components, load_neural_network_model
from .feature_extractor import process_code_file, extract_python_features_from_code
from .metrics import calculate_stealthiness, calculate_evasion, classify_result_type
from .results_tracker import ResultsTracker, generate_run_id
from .dataset_scanner import get_all_authors, select_files_for_batch, resolve_author_from_path
from .batch_tracker import BatchTracker, generate_batch_id, generate_evolution_id

__all__ = [
    "load_all_models",
    "load_model_components",
    "load_neural_network_model",
    "process_code_file",
    "extract_python_features_from_code",
    "calculate_stealthiness",
    "calculate_evasion",
    "classify_result_type",
    "ResultsTracker",
    "generate_run_id",
    "get_all_authors",
    "select_files_for_batch",
    "resolve_author_from_path",
    "BatchTracker",
    "generate_batch_id",
    "generate_evolution_id",
]
