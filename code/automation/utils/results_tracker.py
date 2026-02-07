"""
Results tracking utilities for the Adversarial Stylometry Automation System.

This module provides:
- ResultsTracker class for managing spreadsheets and JSON files
- Run ID generation with lineage tracking
- CSV and JSON file management
"""

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from automation.config import (
    CATEGORIES,
    MODEL_TYPES,
    get_all_columns,
    get_category_csv_path,
    get_category_json_dir,
    get_index_path,
    INDEX_COLUMNS,
    ensure_results_dirs,
)


def generate_run_id(category: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique run ID.

    Format: {category}_{YYYYMMDD}_{short_hash}

    Args:
        category: The transformation category
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Unique run ID string
    """
    if timestamp is None:
        timestamp = datetime.now()

    date_str = timestamp.strftime("%Y%m%d")

    # Create short hash from timestamp + random component
    hash_input = f"{timestamp.isoformat()}{os.urandom(4).hex()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:4]

    return f"{category}_{date_str}_{short_hash}"


class ResultsTracker:
    """
    Manages result tracking via CSV spreadsheets and JSON files.

    Handles:
    - Creating and appending to category-specific CSV files
    - Saving detailed JSON results
    - Maintaining the master index
    - Tracking lineage between runs
    """

    def __init__(self):
        """Initialize the results tracker and ensure directories exist."""
        ensure_results_dirs()
        self._columns = get_all_columns()

    def _ensure_csv_exists(self, csv_path: Path, columns: List[str]) -> None:
        """Create a CSV file with headers if it doesn't exist."""
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

    def _append_to_csv(self, csv_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
        """Append a row to a CSV file, creating it if necessary."""
        self._ensure_csv_exists(csv_path, columns)

        # Ensure all columns have values (use empty string for missing)
        row_data = {col: row.get(col, '') for col in columns}

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow(row_data)

    def save_result(
        self,
        run_id: str,
        category: str,
        result_data: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Save a test result to both CSV and JSON.

        Args:
            run_id: Unique identifier for this run
            category: Transformation category
            result_data: Dictionary containing all result data

        Returns:
            Dictionary with paths to saved files
        """
        if category not in CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {CATEGORIES}")

        timestamp = datetime.now()

        # Build CSV row data
        csv_row = self._build_csv_row(run_id, category, timestamp, result_data)

        # Save to category CSV
        category_csv_path = get_category_csv_path(category)
        self._append_to_csv(category_csv_path, csv_row, self._columns)

        # Save detailed JSON
        json_path = self._save_json(run_id, category, result_data)

        # Update relative path in csv_row
        csv_row['json_path'] = str(json_path.relative_to(get_category_json_dir(category).parent))

        # Update master index
        self._update_index(run_id, category, timestamp, result_data, category_csv_path, json_path)

        return {
            'category_csv': category_csv_path,
            'json': json_path,
            'index': get_index_path(),
        }

    def _build_csv_row(
        self,
        run_id: str,
        category: str,
        timestamp: datetime,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a CSV row from result data."""
        row = {
            'run_id': run_id,
            'timestamp': timestamp.isoformat(),
            'parent_run_id': data.get('parent_run_id', ''),
            'category': category,
            'prompt_summary': data.get('prompt_summary', ''),
            'author': data.get('author', ''),
            'file_name': data.get('file_name', ''),
            'original_file': data.get('original_file', ''),
            'modified_file': data.get('modified_file', ''),
            'ai_tool': data.get('ai_tool', ''),
        }

        # Add per-model results
        model_results = data.get('model_results', {})
        for model_type, config in MODEL_TYPES.items():
            short = config['short_name']
            model_data = model_results.get(model_type, {})

            row[f'{short}_orig_pred'] = model_data.get('original_pred', '')
            row[f'{short}_orig_conf'] = self._format_float(model_data.get('original_conf'))
            row[f'{short}_mod_pred'] = model_data.get('modified_pred', '')
            row[f'{short}_mod_conf'] = self._format_float(model_data.get('modified_conf'))
            row[f'{short}_evasion'] = model_data.get('evasion', '')

        # Add stealthiness metrics
        stealthiness = data.get('stealthiness', {})
        row['l1_distance'] = self._format_float(stealthiness.get('l1_distance'))
        row['l2_distance'] = self._format_float(stealthiness.get('l2_distance'))
        row['cosine_distance'] = self._format_float(stealthiness.get('cosine_distance'))
        row['relative_l2'] = self._format_float(stealthiness.get('relative_l2'))
        row['js_divergence'] = self._format_float(stealthiness.get('js_divergence'))
        row['features_changed_pct'] = self._format_float(stealthiness.get('features_changed_pct'))
        row['max_feature_change'] = self._format_float(stealthiness.get('max_feature_change'))
        row['avg_feature_change'] = self._format_float(stealthiness.get('avg_feature_change'))
        row['stealth_score'] = self._format_float(stealthiness.get('stealth_score'))
        row['stealth_category'] = stealthiness.get('stealth_category', '')

        # Add aggregate metrics
        aggregate = data.get('aggregate', {})
        row['orig_accuracy_rate'] = self._format_float(aggregate.get('orig_accuracy_rate'))
        row['mod_accuracy_rate'] = self._format_float(aggregate.get('mod_accuracy_rate'))
        row['evasion_count'] = aggregate.get('evasion_count', '')
        row['evasion_rate_pct'] = self._format_float(aggregate.get('evasion_rate_pct'))
        row['consensus_prediction'] = aggregate.get('consensus_prediction', '')
        row['result_type'] = data.get('result_type', '')

        # Reference fields
        row['json_path'] = ''  # Will be updated after JSON is saved
        row['notes'] = data.get('notes', '')

        return row

    def _format_float(self, value: Any, precision: int = 4) -> str:
        """Format a float value for CSV output."""
        if value is None or value == '':
            return ''
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return str(value)

    def _save_json(
        self,
        run_id: str,
        category: str,
        data: Dict[str, Any]
    ) -> Path:
        """Save detailed result data to a JSON file."""
        json_dir = get_category_json_dir(category)
        json_path = json_dir / f"{run_id}.json"

        # Add run_id to data
        json_data = {
            'run_id': run_id,
            'category': category,
            'saved_at': datetime.now().isoformat(),
            **data
        }

        # Convert any numpy types to native Python types
        json_data = self._convert_to_json_serializable(json_data)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)

        return json_path

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy and other types to JSON-serializable types."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def _update_index(
        self,
        run_id: str,
        category: str,
        timestamp: datetime,
        data: Dict[str, Any],
        category_csv_path: Path,
        json_path: Path
    ) -> None:
        """Update the master index with this run."""
        index_path = get_index_path()

        aggregate = data.get('aggregate', {})
        evasion_rate = aggregate.get('evasion_rate_pct', 0)

        index_row = {
            'run_id': run_id,
            'category': category,
            'timestamp': timestamp.isoformat(),
            'author': data.get('author', ''),
            'file_name': data.get('file_name', ''),
            'evasion_rate': self._format_float(evasion_rate),
            'category_csv_path': str(category_csv_path.relative_to(get_index_path().parent)),
            'json_path': str(json_path.relative_to(get_index_path().parent)),
        }

        self._append_to_csv(index_path, index_row, INDEX_COLUMNS)

    def get_run_lineage(self, run_id: str, category: str) -> List[Dict[str, Any]]:
        """
        Trace the lineage of a run back to its root.

        Args:
            run_id: The run ID to trace
            category: The category to search in

        Returns:
            List of run records from root to the specified run
        """
        csv_path = get_category_csv_path(category)
        if not csv_path.exists():
            return []

        # Load all runs for this category
        runs = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                runs[row['run_id']] = row

        # Trace lineage
        lineage = []
        current_id = run_id

        while current_id and current_id in runs:
            run_data = runs[current_id]
            lineage.insert(0, run_data)  # Insert at beginning to maintain order
            current_id = run_data.get('parent_run_id', '').strip()

        return lineage

    def get_run(self, run_id: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by ID.

        Args:
            run_id: The run ID to retrieve
            category: The category to search in

        Returns:
            Run data dictionary or None if not found
        """
        csv_path = get_category_csv_path(category)
        if not csv_path.exists():
            return None

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['run_id'] == run_id:
                    return row

        return None

    def get_run_json(self, run_id: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Get the detailed JSON data for a run.

        Args:
            run_id: The run ID to retrieve
            category: The category to search in

        Returns:
            JSON data dictionary or None if not found
        """
        json_path = get_category_json_dir(category) / f"{run_id}.json"
        if not json_path.exists():
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_runs(
        self,
        category: Optional[str] = None,
        author: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List runs with optional filtering.

        Args:
            category: Filter by category (None for all)
            author: Filter by author (None for all)
            limit: Maximum number of runs to return

        Returns:
            List of run records
        """
        runs = []
        categories = [category] if category else CATEGORIES

        for cat in categories:
            csv_path = get_category_csv_path(cat)
            if not csv_path.exists():
                continue

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if author and row.get('author') != author:
                        continue
                    runs.append(row)

        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        if limit:
            runs = runs[:limit]

        return runs
