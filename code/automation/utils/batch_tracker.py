"""
Batch and evolution tracking for the Adversarial Stylometry system.

Manages:
- Batch IDs and evolution IDs
- batch_index.csv for batch-level results
- Evolution JSON files for full evolution chains
- Links between evolutions -> batches -> individual runs
"""

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from automation.config import (
    BATCH_INDEX_COLUMNS,
    ensure_results_dirs,
    get_batch_index_path,
    get_batch_json_dir,
    get_evolution_json_dir,
)


def generate_batch_id(category: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique batch ID.

    Format: batch_{category}_{YYYYMMDD}_{short_hash}
    """
    if timestamp is None:
        timestamp = datetime.now()
    date_str = timestamp.strftime("%Y%m%d")
    hash_input = f"batch_{timestamp.isoformat()}{os.urandom(4).hex()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:4]
    return f"batch_{category}_{date_str}_{short_hash}"


def generate_evolution_id(category: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique evolution ID.

    Format: evo_{category}_{YYYYMMDD}_{short_hash}
    """
    if timestamp is None:
        timestamp = datetime.now()
    date_str = timestamp.strftime("%Y%m%d")
    hash_input = f"evo_{timestamp.isoformat()}{os.urandom(4).hex()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:4]
    return f"evo_{category}_{date_str}_{short_hash}"


class BatchTracker:
    """Tracks batch test results and evolution chains."""

    def __init__(self):
        ensure_results_dirs()

    def save_batch_result(
        self,
        batch_id: str,
        batch_data: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Save a batch result to CSV index and JSON detail file.

        Args:
            batch_id: Unique batch identifier.
            batch_data: Dictionary containing batch results. Expected keys:
                evolution_id, round_number, timestamp, category, author,
                prompt_text, ai_provider, ai_model, num_files,
                avg_evasion_rate, avg_stealth_score, best_evasion_rate,
                worst_evasion_rate, full_evasion_count, individual_results, ...

        Returns:
            Dictionary with paths to saved files.
        """
        # Save detailed JSON
        json_path = self._save_batch_json(batch_id, batch_data)

        # Build and append CSV row
        csv_row = {
            "batch_id": batch_id,
            "evolution_id": batch_data.get("evolution_id", ""),
            "round_number": batch_data.get("round_number", 0),
            "timestamp": batch_data.get("timestamp", datetime.now().isoformat()),
            "category": batch_data.get("category", ""),
            "author": batch_data.get("author", ""),
            "prompt_text": _truncate(batch_data.get("prompt_text", ""), 200),
            "ai_provider": batch_data.get("ai_provider", ""),
            "ai_model": batch_data.get("ai_model", ""),
            "num_files": batch_data.get("num_files", 0),
            "avg_evasion_rate": _fmt(batch_data.get("avg_evasion_rate")),
            "avg_stealth_score": _fmt(batch_data.get("avg_stealth_score")),
            "best_evasion_rate": _fmt(batch_data.get("best_evasion_rate")),
            "worst_evasion_rate": _fmt(batch_data.get("worst_evasion_rate")),
            "full_evasion_count": batch_data.get("full_evasion_count", 0),
            "json_path": str(json_path.relative_to(get_batch_json_dir().parent)),
        }

        index_path = get_batch_index_path()
        self._append_to_csv(index_path, csv_row, BATCH_INDEX_COLUMNS)

        return {"batch_csv": index_path, "batch_json": json_path}

    def save_evolution_result(
        self,
        evolution_id: str,
        evolution_data: Dict[str, Any],
    ) -> Path:
        """
        Save a complete evolution result to JSON.

        Args:
            evolution_id: Unique evolution identifier.
            evolution_data: Full evolution data including all rounds.

        Returns:
            Path to the saved JSON file.
        """
        json_dir = get_evolution_json_dir()
        json_path = json_dir / f"{evolution_id}.json"

        serializable = _make_serializable(evolution_data)
        serializable["evolution_id"] = evolution_id
        serializable["saved_at"] = datetime.now().isoformat()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        return json_path

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load a batch result JSON by ID."""
        json_path = get_batch_json_dir() / f"{batch_id}.json"
        if not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_evolution(self, evolution_id: str) -> Optional[Dict[str, Any]]:
        """Load an evolution result JSON by ID."""
        json_path = get_evolution_json_dir() / f"{evolution_id}.json"
        if not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_batches(
        self,
        category: Optional[str] = None,
        evolution_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """List batches from the index CSV with optional filtering."""
        index_path = get_batch_index_path()
        if not index_path.exists():
            return []

        rows = []
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if category and row.get("category") != category:
                    continue
                if evolution_id and row.get("evolution_id") != evolution_id:
                    continue
                rows.append(row)

        rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        if limit:
            rows = rows[:limit]
        return rows

    def _save_batch_json(self, batch_id: str, data: Dict[str, Any]) -> Path:
        """Save detailed batch data to JSON."""
        json_dir = get_batch_json_dir()
        json_path = json_dir / f"{batch_id}.json"

        serializable = _make_serializable(data)
        serializable["batch_id"] = batch_id
        serializable["saved_at"] = datetime.now().isoformat()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        return json_path

    def _append_to_csv(
        self, csv_path: Path, row: Dict[str, Any], columns: List[str]
    ) -> None:
        """Append a row to a CSV file, creating with headers if necessary."""
        write_header = not csv_path.exists()
        row_data = {col: row.get(col, "") for col in columns}

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row_data)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _fmt(value: Any, precision: int = 4) -> str:
    """Format a numeric value for CSV."""
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        return obj
