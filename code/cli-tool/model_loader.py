"""Load and manage trained classifiers."""

import joblib
from pathlib import Path
from typing import Dict, Tuple
from config import MODEL_PATHS, MODEL_EXTENSIONS


class ConfidenceBasedClassifier:
    """
    Wrapper for any scikit-learn classifier that adds unknown author detection
    based on prediction confidence thresholds.
    """

    def __init__(self, base_classifier, confidence_threshold=0.6, unknown_label='UNKNOWN'):
        self.base_classifier = base_classifier
        self.confidence_threshold = confidence_threshold
        self.unknown_label = unknown_label
        self.is_fitted = False

    def fit(self, X, y):
        """Fit the base classifier"""
        self.base_classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Get prediction probabilities from base classifier"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        """Predict with confidence threshold"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        predictions = self.base_classifier.predict(X)
        probabilities = self.base_classifier.predict_proba(X)

        max_probs = probabilities.max(axis=1)
        predictions = [self.unknown_label if prob < self.confidence_threshold else pred
                       for pred, prob in zip(predictions, max_probs)]
        return predictions

    def get_params(self, deep=True):
        """Get parameters for grid search compatibility"""
        return self.base_classifier.get_params(deep=deep)

    def set_params(self, **params):
        """Set parameters for grid search compatibility"""
        self.base_classifier.set_params(**params)
        return self


class ModelLoader:
    """Load and manage trained models with their preprocessing pipelines."""

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.style_vectorizers = {}
        self.scalers = {}
        self.selectors = {}
        self.metadata = {}

    def load_model(self, model_name: str) -> bool:
        """
        Load a complete model pipeline (model + vectorizers + scalers + selector).

        Args:
            model_name: One of "random_forest", "sgd", "naive_bayes", "neural_network"

        Returns:
            True if loaded successfully, False otherwise
        """
        if model_name not in MODEL_PATHS:
            print(f"Unknown model: {model_name}")
            return False

        model_dir = MODEL_PATHS[model_name]

        try:
            # Handle neural network separately (uses .h5 files)
            if model_name == "neural_network":
                model_files = list(model_dir.glob("model_*.h5"))
            else:
                model_files = list(model_dir.glob("model_*.joblib"))

            if not model_files:
                print(f"No model files found in {model_dir}")
                return False

            # Get the most recent file
            model_file = max(model_files, key=lambda p: p.stat().st_mtime)
            timestamp = model_file.stem.replace("model_", "")

            # Load model based on type
            if model_name == "neural_network":
                import tensorflow as tf
                self.models[model_name] = tf.keras.models.load_model(str(model_file))
            else:
                self.models[model_name] = joblib.load(model_file)

            vectorizer_file = model_dir / f"vectorizer_{timestamp}.joblib"
            if vectorizer_file.exists():
                self.vectorizers[model_name] = joblib.load(vectorizer_file)

            style_vec_file = model_dir / f"style_vectorizer_{timestamp}.joblib"
            if style_vec_file.exists():
                self.style_vectorizers[model_name] = joblib.load(style_vec_file)

            style_scaler_file = model_dir / f"style_scaler_{timestamp}.joblib"
            if style_scaler_file.exists():
                self.scalers[model_name] = joblib.load(style_scaler_file)

            selector_file = model_dir / f"selector_{timestamp}.joblib"
            if selector_file.exists():
                self.selectors[model_name] = joblib.load(selector_file)

            metadata_file = model_dir / f"metadata_{timestamp}.joblib"
            if metadata_file.exists():
                self.metadata[model_name] = joblib.load(metadata_file)

            # Load label encoder for neural network
            if model_name == "neural_network":
                label_encoder_file = model_dir / f"label_encoder_{timestamp}.joblib"
                if label_encoder_file.exists():
                    self.metadata[model_name + "_label_encoder"] = joblib.load(label_encoder_file)

            print(f"Loaded {model_name} (timestamp: {timestamp})")
            return True

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return False

    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models.

        Returns:
            Dict mapping model names to load success status
        """
        results = {}
        for model_name in MODEL_PATHS.keys():
            results[model_name] = self.load_model(model_name)
        return results

    def get_model(self, model_name: str):
        """Get a loaded model."""
        return self.models.get(model_name)

    def get_vectorizer(self, model_name: str):
        """Get vectorizer for a model."""
        return self.vectorizers.get(model_name)

    def get_style_vectorizer(self, model_name: str):
        """Get style vectorizer for a model."""
        return self.style_vectorizers.get(model_name)

    def get_scaler(self, model_name: str):
        """Get scaler for a model."""
        return self.scalers.get(model_name)

    def get_selector(self, model_name: str):
        """Get feature selector for a model."""
        return self.selectors.get(model_name)

    def predict(self, model_name: str, code_content: str) -> Tuple[str, float]:
        """
        Get prediction for code using a specific model.

        Args:
            model_name: Model to use
            code_content: Python code as string

        Returns:
            Tuple of (predicted_author, confidence_score)
        """
        import numpy as np
        from sklearn.feature_extraction import DictVectorizer

        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return None, 0.0

        try:
            model = self.models[model_name]
            vectorizer = self.vectorizers.get(model_name)
            style_vectorizer = self.style_vectorizers.get(model_name)
            selector = self.selectors.get(model_name)
            scaler = self.scalers.get(model_name)

            if not vectorizer:
                print(f"No vectorizer for {model_name}")
                return None, 0.0

            # Extract TF-IDF features from code text
            tfidf_features = vectorizer.transform([code_content])
            if hasattr(tfidf_features, 'toarray'):
                tfidf_features = tfidf_features.toarray()

            # Extract and combine style features if available
            if style_vectorizer:
                from code_analyzer import CodeAnalyzer
                analyzer = CodeAnalyzer()
                style_dict = analyzer.full_analysis(code_content)

                # Flatten the nested analysis dict to match training format
                flat_style = {}
                for category, metrics in style_dict.items():
                    for metric, value in metrics.items():
                        flat_style[f"{category}_{metric}"] = value

                # Vectorize style features
                style_features = style_vectorizer.transform([flat_style])
                if hasattr(style_features, 'toarray'):
                    style_features = style_features.toarray()

                # Concatenate TF-IDF and style features
                features = np.concatenate([tfidf_features, style_features], axis=1)
            else:
                features = tfidf_features

            # Apply feature selection (dimensionality reduction)
            if selector:
                try:
                    features = selector.transform(features)
                except ValueError as e:
                    print(f"Warning: Feature selection failed for {model_name}: {e}")
                    return None, 0.0

            # Apply scaling
            if scaler:
                try:
                    features = scaler.transform(features)
                except ValueError:
                    pass

            # Make prediction based on model type
            if model_name == "neural_network":
                # Neural network outputs probabilities directly
                import tensorflow as tf
                probabilities = model.predict(features, verbose=0)[0]
                confidence = max(probabilities)
                # Get the class with highest probability
                class_idx = int(probabilities.argmax())

                # Convert class index to author name using label encoder
                label_encoder = self.metadata.get(model_name + "_label_encoder")
                if label_encoder:
                    prediction = label_encoder.inverse_transform([class_idx])[0]
                else:
                    prediction = class_idx
                return prediction, float(confidence)
            else:
                # Sklearn models
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
                return prediction, confidence

        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
