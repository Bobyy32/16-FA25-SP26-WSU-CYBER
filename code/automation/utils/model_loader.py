"""
Model loading utilities for the Adversarial Stylometry Automation System.

This module handles loading saved attribution models from disk, including:
- Scikit-learn models (Random Forest, Naive Bayes, SGD)
- Neural Network models (Keras/TensorFlow)
- Associated preprocessing components (vectorizers, selectors, normalizers)
"""

import os
import re
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from automation.config import MODEL_TYPES, get_model_dir


# Flag to track if classes have been registered for unpickling
_classes_registered = False


class ConfidenceBasedClassifier:
    """
    Wrapper for any scikit-learn classifier that adds unknown author detection
    based on prediction confidence thresholds.

    This class must be defined here to properly unpickle saved models.
    """

    def __init__(self, base_classifier, confidence_threshold=0.6, unknown_label='UNKNOWN'):
        self.base_classifier = base_classifier
        self.confidence_threshold = confidence_threshold
        self.unknown_label = unknown_label
        self.is_fitted = True  # Assume loaded models are fitted

    def predict_proba(self, X):
        """Get prediction probabilities from base classifier."""
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        """Predict with confidence-based unknown detection."""
        probabilities = self.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)
        base_predictions = self.base_classifier.predict(X)

        final_predictions = []
        for pred, max_prob in zip(base_predictions, max_probs):
            if max_prob >= self.confidence_threshold:
                final_predictions.append(pred)
            else:
                final_predictions.append(self.unknown_label)

        return np.array(final_predictions)

    def predict_with_confidence(self, X):
        """Return predictions along with confidence scores."""
        probabilities = self.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)
        base_predictions = self.base_classifier.predict(X)
        final_predictions = self.predict(X)

        return final_predictions, max_probs, base_predictions


class ConfidenceBasedNeuralNetwork:
    """
    Wrapper for neural networks that adds unknown author detection
    based on prediction confidence thresholds.
    """

    def __init__(self, model, confidence_threshold=0.6, unknown_label='UNKNOWN', label_encoder=None):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.unknown_label = unknown_label
        self.label_encoder = label_encoder
        self.is_fitted = True  # Assume loaded models are fitted

    def predict_proba(self, X):
        """Get prediction probabilities from neural network."""
        return self.model.predict(X, verbose=0)

    def predict(self, X):
        """Predict with confidence-based unknown detection."""
        probabilities = self.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)

        base_predictions_encoded = np.argmax(probabilities, axis=1)
        if self.label_encoder:
            base_predictions = self.label_encoder.inverse_transform(base_predictions_encoded)
        else:
            base_predictions = base_predictions_encoded

        final_predictions = []
        for pred, max_prob in zip(base_predictions, max_probs):
            if max_prob >= self.confidence_threshold:
                final_predictions.append(pred)
            else:
                final_predictions.append(self.unknown_label)

        return np.array(final_predictions)

    def predict_with_confidence(self, X):
        """Return predictions along with confidence scores."""
        probabilities = self.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)

        base_predictions_encoded = np.argmax(probabilities, axis=1)
        if self.label_encoder:
            base_predictions = self.label_encoder.inverse_transform(base_predictions_encoded)
        else:
            base_predictions = base_predictions_encoded

        final_predictions = self.predict(X)

        return final_predictions, max_probs, base_predictions


def _register_classes_for_unpickling():
    """
    Register ConfidenceBasedClassifier in __main__ module for joblib unpickling.

    Models saved from Jupyter notebooks have their classes stored with __main__
    as the module, so we need to make the class available there for unpickling.
    """
    global _classes_registered
    if _classes_registered:
        return

    import __main__
    if not hasattr(__main__, 'ConfidenceBasedClassifier'):
        __main__.ConfidenceBasedClassifier = ConfidenceBasedClassifier
    if not hasattr(__main__, 'ConfidenceBasedNeuralNetwork'):
        __main__.ConfidenceBasedNeuralNetwork = ConfidenceBasedNeuralNetwork

    _classes_registered = True


def get_latest_timestamp(model_dir: Path) -> Optional[str]:
    """
    Find the most recent model timestamp in a directory.

    Model files are named like: model_20260203_180706.joblib

    Args:
        model_dir: Path to the model directory

    Returns:
        Most recent timestamp string or None if no models found
    """
    model_files = list(model_dir.glob("model_*.joblib")) + list(model_dir.glob("model_*.h5"))

    if not model_files:
        return None

    timestamps = []
    for f in model_files:
        match = re.search(r'model_(\d{8}_\d{6})', f.name)
        if match:
            timestamps.append(match.group(1))

    if not timestamps:
        return None

    return sorted(timestamps)[-1]


def load_model_components(model_dir: Path, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all components for a scikit-learn model.

    Args:
        model_dir: Path to the model directory
        timestamp: Specific timestamp to load, or None for most recent

    Returns:
        Dictionary with model, vectorizer, selector, normalizer, metadata,
        style_vectorizer, and style_scaler
    """
    # Register classes for unpickling models saved from notebooks
    _register_classes_for_unpickling()

    if timestamp is None:
        timestamp = get_latest_timestamp(model_dir)
        if timestamp is None:
            raise FileNotFoundError(f"No model files found in {model_dir}")

    components = {}

    # Load main model
    model_path = model_dir / f"model_{timestamp}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    components['model'] = joblib.load(model_path)

    # Load vectorizer
    vectorizer_path = model_dir / f"vectorizer_{timestamp}.joblib"
    if vectorizer_path.exists():
        components['vectorizer'] = joblib.load(vectorizer_path)
    else:
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    # Load selector
    selector_path = model_dir / f"selector_{timestamp}.joblib"
    if selector_path.exists():
        components['selector'] = joblib.load(selector_path)
    else:
        raise FileNotFoundError(f"Selector not found: {selector_path}")

    # Load normalizer
    normalizer_path = model_dir / f"normalizer_{timestamp}.joblib"
    if normalizer_path.exists():
        components['normalizer'] = joblib.load(normalizer_path)
    else:
        components['normalizer'] = None

    # Load metadata
    metadata_path = model_dir / f"metadata_{timestamp}.joblib"
    if metadata_path.exists():
        components['metadata'] = joblib.load(metadata_path)
    else:
        components['metadata'] = {}

    # Load style components
    style_vectorizer_path = model_dir / f"style_vectorizer_{timestamp}.joblib"
    if style_vectorizer_path.exists():
        components['style_vectorizer'] = joblib.load(style_vectorizer_path)
    else:
        components['style_vectorizer'] = None

    style_scaler_path = model_dir / f"style_scaler_{timestamp}.joblib"
    if style_scaler_path.exists():
        components['style_scaler'] = joblib.load(style_scaler_path)
    else:
        components['style_scaler'] = None

    components['timestamp'] = timestamp

    return components


def load_neural_network_model(model_dir: Path, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all components for a neural network model.

    Args:
        model_dir: Path to the model directory
        timestamp: Specific timestamp to load, or None for most recent

    Returns:
        Dictionary with model, vectorizer, selector, normalizer, metadata,
        style_vectorizer, style_scaler, and label_encoder
    """
    # Register classes for unpickling models saved from notebooks
    _register_classes_for_unpickling()

    # Import TensorFlow only when needed
    import tensorflow as tf

    if timestamp is None:
        timestamp = get_latest_timestamp(model_dir)
        if timestamp is None:
            raise FileNotFoundError(f"No model files found in {model_dir}")

    # Load non-model components first
    components = {}

    # Load vectorizer
    vectorizer_path = model_dir / f"vectorizer_{timestamp}.joblib"
    if vectorizer_path.exists():
        components['vectorizer'] = joblib.load(vectorizer_path)
    else:
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    # Load selector
    selector_path = model_dir / f"selector_{timestamp}.joblib"
    if selector_path.exists():
        components['selector'] = joblib.load(selector_path)
    else:
        raise FileNotFoundError(f"Selector not found: {selector_path}")

    # Load normalizer
    normalizer_path = model_dir / f"normalizer_{timestamp}.joblib"
    if normalizer_path.exists():
        components['normalizer'] = joblib.load(normalizer_path)
    else:
        components['normalizer'] = None

    # Load metadata
    metadata_path = model_dir / f"metadata_{timestamp}.joblib"
    if metadata_path.exists():
        components['metadata'] = joblib.load(metadata_path)
    else:
        components['metadata'] = {}

    # Load style components
    style_vectorizer_path = model_dir / f"style_vectorizer_{timestamp}.joblib"
    if style_vectorizer_path.exists():
        components['style_vectorizer'] = joblib.load(style_vectorizer_path)
    else:
        components['style_vectorizer'] = None

    style_scaler_path = model_dir / f"style_scaler_{timestamp}.joblib"
    if style_scaler_path.exists():
        components['style_scaler'] = joblib.load(style_scaler_path)
    else:
        components['style_scaler'] = None

    # Load label encoder
    label_encoder_path = model_dir / f"label_encoder_{timestamp}.joblib"
    if label_encoder_path.exists():
        components['label_encoder'] = joblib.load(label_encoder_path)
    else:
        components['label_encoder'] = None

    # Load Keras model
    model_path = model_dir / f"model_{timestamp}.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Neural network model not found: {model_path}")

    keras_model = tf.keras.models.load_model(model_path)

    # Wrap in confidence-based wrapper
    confidence_threshold = MODEL_TYPES['neural_network']['confidence_threshold']
    components['model'] = ConfidenceBasedNeuralNetwork(
        keras_model,
        confidence_threshold=confidence_threshold,
        label_encoder=components['label_encoder']
    )

    components['timestamp'] = timestamp

    return components


def load_all_models(timestamp: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all attribution models.

    Args:
        timestamp: Specific timestamp to load, or None for most recent

    Returns:
        Dictionary mapping model type to component dictionary
    """
    models = {}

    for model_type, config in MODEL_TYPES.items():
        model_dir = get_model_dir(model_type)

        if not model_dir.exists():
            print(f"Warning: Model directory not found: {model_dir}")
            continue

        try:
            if config['is_neural_network']:
                components = load_neural_network_model(model_dir, timestamp)
            else:
                components = load_model_components(model_dir, timestamp)

            models[model_type] = {
                'components': components,
                'config': config,
            }

        except Exception as e:
            print(f"Warning: Failed to load {model_type}: {e}")
            continue

    return models
