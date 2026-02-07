"""
Feature extraction utilities for the Adversarial Stylometry Automation System.

This module handles extracting features from Python code files using the
same pipeline as training:
1. Text vectorization (TF-IDF/Count with n-grams)
2. Style feature extraction (Python-specific stylometric features)
3. Feature selection (SelectKBest)
4. Normalization
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix, hstack


def extract_python_features_from_code(code: str) -> Dict[str, int]:
    """
    Extract Python-specific stylometric features from code.

    Counts the occurrences of various stylistic elements specific to Python.
    This MUST match EXACTLY the function used during model training.

    Args:
        code: Python source code as a string

    Returns:
        Dictionary mapping feature names to counts
    """
    features = {
        'inline_comments': 0,
        'block_comments': 0,
        'single_line_comments': 0,
        'function_comments': 0,
        'camel_case': 0,
        'snake_case': 0,
        'variable_name_length': 0,
        'function_name_length': 0,
        'use_of_tabs': 0,
        'use_of_spaces': 0,
        'line_length': 0,
        'import_style': 0
    }

    lines = code.split('\n')
    for i, line in enumerate(lines):
        # Inline comments (comments at end of a line)
        if '#' in line:
            if line.strip().startswith('#'):
                features['single_line_comments'] += 1
            else:
                features['inline_comments'] += 1

        # Block comments (""" or ''')
        if '"""' in line or "'''" in line:
            features['block_comments'] += 1

        # Function comments (Comments just above function def)
        if re.match(r'\s*def\s+\w+\s*\(.*\)\s*:', line):
            if i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line.startswith('#') or prev_line.startswith('"""') or prev_line.startswith("'''"):
                    features['function_comments'] += 1

        # Camel case and snake case
        words = re.findall(r'\b\w+\b', line)
        for word in words:
            if re.match(r'^[a-z]+([A-Z][a-z]*)+$', word):
                features['camel_case'] += 1
            if re.match(r'^[a-z]+(_[a-z]+)+$', word):
                features['snake_case'] += 1

        # Variable and function name lengths
        if re.match(r'\s*\w+\s*=\s*', line):
            var_name = line.split('=')[0].strip()
            features['variable_name_length'] += len(var_name)
        if re.match(r'\s*def\s+\w+\s*\(.*\)\s*:', line):
            func_match = re.findall(r'def\s+(\w+)\s*\(', line)
            if func_match:
                features['function_name_length'] += len(func_match[0])

        # Tabs vs spaces
        if '\t' in line:
            features['use_of_tabs'] += 1
        if '    ' in line:
            features['use_of_spaces'] += 1

        # Line length
        features['line_length'] += len(line)

        # Import style
        if re.match(r'\s*import\s+\w+', line) or re.match(r'\s*from\s+\w+\s+import\s+\w+', line):
            features['import_style'] += 1

    return features


def process_code_file(
    file_path: Union[str, Path],
    model_components: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Process a code file using saved model components.

    This function replicates the EXACT preprocessing pipeline from training:
    1. Text vectorization (TF-IDF/Count with n-grams)
    2. Style feature extraction (if enabled) and combination via hstack
    3. Feature selection (SelectKBest)
    4. Normalization

    Args:
        file_path: Path to the Python file to analyze
        model_components: Dictionary with vectorizer, selector, normalizer,
                         metadata, and optional style components

    Returns:
        Processed feature vector ready for model prediction, or None if failed
    """
    try:
        file_path = Path(file_path)

        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_text = f.read()

        return process_code_string(code_text, model_components)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_code_string(
    code_text: str,
    model_components: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Process a code string using saved model components.

    Args:
        code_text: Python source code as a string
        model_components: Dictionary with vectorizer, selector, normalizer,
                         metadata, and optional style components

    Returns:
        Processed feature vector ready for model prediction, or None if failed
    """
    try:
        # STEP 1: Vectorize text using saved vectorizer
        vectorizer = model_components['vectorizer']
        text_vector = vectorizer.transform([code_text])

        # STEP 2: Add style features if they were used during training
        metadata = model_components.get('metadata', {})
        has_style_components = metadata.get('has_style_components', False)

        if has_style_components:
            style_features = extract_python_features_from_code(code_text)

            style_vectorizer = model_components.get('style_vectorizer')
            style_scaler = model_components.get('style_scaler')

            if style_vectorizer is not None and style_scaler is not None:
                style_features_vector = style_vectorizer.transform([style_features])
                style_features_scaled = style_scaler.transform(style_features_vector)

                style_sparse = csr_matrix(style_features_scaled)
                combined_vector = hstack([text_vector, style_sparse])
                feature_vector = combined_vector
            else:
                print("Warning: Style components expected but not loaded, using text features only")
                feature_vector = text_vector
        else:
            feature_vector = text_vector

        # STEP 3: Apply saved feature selector
        selector = model_components['selector']
        selected_vector = selector.transform(feature_vector)

        # STEP 4: Apply saved normalizer
        normalizer = model_components.get('normalizer')
        if normalizer is not None:
            final_vector = normalizer.transform(selected_vector)
        else:
            final_vector = selected_vector

        # Convert to dense array if needed
        if hasattr(final_vector, 'toarray'):
            final_vector = final_vector.toarray()

        return final_vector

    except Exception as e:
        print(f"Error processing code string: {e}")
        import traceback
        traceback.print_exc()
        return None


def vectorize_file_pair(
    original_path: Union[str, Path],
    modified_path: Union[str, Path],
    model_components: Dict[str, Any]
) -> tuple:
    """
    Vectorize a pair of files using saved model components.

    Args:
        original_path: Path to original file
        modified_path: Path to modified file
        model_components: Dictionary with all saved components

    Returns:
        Tuple of (original_vector, modified_vector) or (None, None) if failed
    """
    try:
        original_vector = process_code_file(original_path, model_components)
        modified_vector = process_code_file(modified_path, model_components)

        if original_vector is None or modified_vector is None:
            return None, None

        return original_vector.flatten(), modified_vector.flatten()

    except Exception as e:
        print(f"Error vectorizing file pair: {e}")
        import traceback
        traceback.print_exc()
        return None, None
