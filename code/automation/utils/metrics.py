"""
Metrics utilities for the Adversarial Stylometry Automation System.

This module provides functions to calculate:
- Stealthiness metrics (how similar modified code is to original)
- Evasion metrics (whether models were fooled)
- Result classification
"""

from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import entropy


def calculate_stealthiness(
    original_vector: np.ndarray,
    modified_vector: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Calculate various distance metrics between original and modified code vectors
    to assess the stealthiness of adversarial modifications.

    Lower values indicate more stealthy modifications (closer to original).

    Args:
        original_vector: Feature vector of original code
        modified_vector: Feature vector of modified code

    Returns:
        Dictionary of stealthiness metrics, or None if calculation failed
    """
    if original_vector is None or modified_vector is None:
        return None

    # Ensure vectors are 1D
    original_vector = np.asarray(original_vector).flatten()
    modified_vector = np.asarray(modified_vector).flatten()

    metrics = {}

    try:
        # L1 norm (Manhattan distance)
        metrics['l1_distance'] = float(np.sum(np.abs(original_vector - modified_vector)))

        # L2 norm (Euclidean distance)
        metrics['l2_distance'] = float(euclidean(original_vector, modified_vector))

        # Cosine distance (1 - cosine similarity)
        # Handle zero vectors
        orig_norm = np.linalg.norm(original_vector)
        mod_norm = np.linalg.norm(modified_vector)
        if orig_norm > 0 and mod_norm > 0:
            metrics['cosine_distance'] = float(cosine(original_vector, modified_vector))
        else:
            metrics['cosine_distance'] = 0.0 if orig_norm == mod_norm else 1.0

        # Relative L2 distance (normalized by original vector magnitude)
        if orig_norm > 0:
            metrics['relative_l2'] = float(metrics['l2_distance'] / orig_norm)
        else:
            metrics['relative_l2'] = float('inf') if metrics['l2_distance'] > 0 else 0.0

        # Jensen-Shannon divergence (for probability distributions)
        # Add small epsilon to avoid division by zero
        orig_prob = np.abs(original_vector) + 1e-10
        mod_prob = np.abs(modified_vector) + 1e-10
        orig_prob = orig_prob / np.sum(orig_prob)
        mod_prob = mod_prob / np.sum(mod_prob)

        m = 0.5 * (orig_prob + mod_prob)
        js_div = 0.5 * entropy(orig_prob, m) + 0.5 * entropy(mod_prob, m)
        metrics['js_divergence'] = float(js_div)

        # Feature-wise changes
        threshold = 0.01
        changes = np.abs(original_vector - modified_vector)
        significant_changes = np.sum(changes > threshold)
        metrics['features_changed_pct'] = float((significant_changes / len(original_vector)) * 100)
        metrics['max_feature_change'] = float(np.max(changes))
        metrics['avg_feature_change'] = float(np.mean(changes))

        # Calculate composite stealth score (lower is more stealthy)
        # Weighted combination of normalized metrics
        stealth_score = (
            0.3 * min(metrics['cosine_distance'], 1.0) +
            0.3 * min(metrics['relative_l2'], 1.0) +
            0.2 * min(metrics['js_divergence'], 1.0) +
            0.2 * min(metrics['features_changed_pct'] / 100, 1.0)
        )
        metrics['stealth_score'] = float(stealth_score)

        # Categorize stealthiness
        if stealth_score < 0.2:
            metrics['stealth_category'] = 'high'
        elif stealth_score < 0.5:
            metrics['stealth_category'] = 'medium'
        else:
            metrics['stealth_category'] = 'low'

        return metrics

    except Exception as e:
        print(f"Error calculating stealthiness metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_evasion(
    original_pred: str,
    modified_pred: str,
    true_author: str,
    original_conf: float,
    modified_conf: float
) -> Dict[str, Any]:
    """
    Calculate evasion metrics for a single model.

    Args:
        original_pred: Model's prediction on original code
        modified_pred: Model's prediction on modified code
        true_author: The actual author of the code
        original_conf: Confidence score on original
        modified_conf: Confidence score on modified

    Returns:
        Dictionary with evasion metrics
    """
    # Original was correctly identified
    original_correct = (original_pred == true_author)

    # Modified was incorrectly identified (evasion successful)
    modified_correct = (modified_pred == true_author)
    evasion_successful = original_correct and not modified_correct

    return {
        'original_correct': original_correct,
        'modified_correct': modified_correct,
        'evasion': evasion_successful,
        'confidence_drop': float(original_conf - modified_conf),
    }


def calculate_aggregate_metrics(
    model_results: Dict[str, Dict[str, Any]],
    true_author: str
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all models.

    Args:
        model_results: Dictionary mapping model names to their results
        true_author: The actual author of the code

    Returns:
        Dictionary with aggregate metrics
    """
    total_models = len(model_results)
    if total_models == 0:
        return {}

    # Count correct predictions
    orig_correct = sum(1 for r in model_results.values() if r.get('original_correct', False))
    mod_correct = sum(1 for r in model_results.values() if r.get('modified_correct', False))
    evasion_count = sum(1 for r in model_results.values() if r.get('evasion', False))

    # Get all predictions for consensus
    orig_preds = [r.get('original_pred') for r in model_results.values() if r.get('original_pred')]
    mod_preds = [r.get('modified_pred') for r in model_results.values() if r.get('modified_pred')]

    # Calculate consensus (most common prediction)
    if mod_preds:
        pred_counts = Counter(mod_preds)
        consensus_prediction = pred_counts.most_common(1)[0][0]
    else:
        consensus_prediction = None

    return {
        'orig_accuracy_rate': float(orig_correct / total_models * 100),
        'mod_accuracy_rate': float(mod_correct / total_models * 100),
        'evasion_count': evasion_count,
        'evasion_rate_pct': float(evasion_count / total_models * 100),
        'consensus_prediction': consensus_prediction,
    }


def classify_result_type(
    evasion_count: int,
    total_models: int,
    orig_accuracy_rate: float
) -> str:
    """
    Classify the result type based on evasion metrics.

    Args:
        evasion_count: Number of models evaded
        total_models: Total number of models tested
        orig_accuracy_rate: Percentage of models correct on original

    Returns:
        Result type classification string
    """
    if total_models == 0:
        return "no_models"

    evasion_rate = evasion_count / total_models

    # Original wasn't correctly identified by most models
    if orig_accuracy_rate < 50:
        return "original_unknown"

    # All models evaded
    if evasion_rate >= 1.0:
        return "full_evasion"

    # Most models evaded
    if evasion_rate >= 0.75:
        return "strong_evasion"

    # Some models evaded
    if evasion_rate >= 0.25:
        return "partial_evasion"

    # Few or no models evaded
    if evasion_rate > 0:
        return "weak_evasion"

    return "failed"


def get_model_predictions(
    model: Any,
    original_vector: np.ndarray,
    modified_vector: np.ndarray,
    true_author: str
) -> Dict[str, Any]:
    """
    Get predictions from a model for both original and modified code.

    Args:
        model: The classifier model (must have predict_with_confidence method)
        original_vector: Feature vector of original code
        modified_vector: Feature vector of modified code
        true_author: The actual author of the code

    Returns:
        Dictionary with predictions, confidences, and evasion status
    """
    try:
        # Ensure vectors are 2D for prediction
        if original_vector.ndim == 1:
            original_vector = original_vector.reshape(1, -1)
        if modified_vector.ndim == 1:
            modified_vector = modified_vector.reshape(1, -1)

        # Get predictions
        orig_pred, orig_conf, _ = model.predict_with_confidence(original_vector)
        mod_pred, mod_conf, _ = model.predict_with_confidence(modified_vector)

        # Convert to scalar values
        orig_pred = orig_pred[0] if hasattr(orig_pred, '__len__') else orig_pred
        mod_pred = mod_pred[0] if hasattr(mod_pred, '__len__') else mod_pred
        orig_conf = float(orig_conf[0]) if hasattr(orig_conf, '__len__') else float(orig_conf)
        mod_conf = float(mod_conf[0]) if hasattr(mod_conf, '__len__') else float(mod_conf)

        # Calculate evasion
        evasion_metrics = calculate_evasion(
            orig_pred, mod_pred, true_author, orig_conf, mod_conf
        )

        return {
            'original_pred': str(orig_pred),
            'original_conf': orig_conf,
            'modified_pred': str(mod_pred),
            'modified_conf': mod_conf,
            **evasion_metrics
        }

    except Exception as e:
        print(f"Error getting model predictions: {e}")
        import traceback
        traceback.print_exc()
        return {
            'original_pred': None,
            'original_conf': None,
            'modified_pred': None,
            'modified_conf': None,
            'original_correct': False,
            'modified_correct': False,
            'evasion': False,
            'confidence_drop': 0.0,
            'error': str(e),
        }
