#!/usr/bin/env python3
"""
Main test runner for the Adversarial Stylometry Automation System.

This module provides the primary interface for running adversarial tests:
- Load models and extract features
- Calculate stealthiness and evasion metrics
- Save results to spreadsheets and JSON files
- Track lineage between runs

Usage:
    from automation.run_tests import run_adversarial_test

    result = run_adversarial_test(
        original_file="path/to/original.py",
        modified_file="path/to/modified.py",
        category="restructuring",
        author="author_name",
        ai_tool="Claude Sonnet 4",
        prompt_summary="Restructure code with better variable names"
    )
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure the automation package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from automation.config import (
    CATEGORIES,
    MODEL_TYPES,
    ensure_results_dirs,
)
from automation.utils.model_loader import load_all_models
from automation.utils.feature_extractor import process_code_file
from automation.utils.metrics import (
    calculate_stealthiness,
    get_model_predictions,
    calculate_aggregate_metrics,
    classify_result_type,
)
from automation.utils.results_tracker import ResultsTracker, generate_run_id


# Global model cache to avoid reloading
_model_cache: Optional[Dict[str, Any]] = None


def get_models(reload: bool = False) -> Dict[str, Any]:
    """
    Get loaded models, using cache if available.

    Args:
        reload: Force reload models even if cached

    Returns:
        Dictionary of loaded models
    """
    global _model_cache

    if _model_cache is None or reload:
        print("Loading attribution models...")
        _model_cache = load_all_models()
        print(f"Loaded {len(_model_cache)} models: {list(_model_cache.keys())}")

    return _model_cache


def run_adversarial_test(
    original_file: str,
    modified_file: str,
    category: str,
    author: str,
    ai_tool: str = "",
    prompt_summary: str = "",
    parent_run_id: Optional[str] = None,
    notes: str = "",
    reload_models: bool = False,
) -> Dict[str, Any]:
    """
    Run an adversarial test on a pair of code files.

    This is the main entry point for the automation system. It:
    1. Loads models (cached for performance)
    2. Extracts features from both files
    3. Gets predictions from all models
    4. Calculates stealthiness metrics
    5. Saves results to CSV and JSON
    6. Returns the result summary

    Args:
        original_file: Path to the original Python file
        modified_file: Path to the modified Python file
        category: Transformation category (restructuring, renaming, formatting, comments)
        author: True author of the code
        ai_tool: AI tool used for modification (e.g., "Claude Sonnet 4")
        prompt_summary: Brief description of the transformation prompt
        parent_run_id: ID of the run this builds upon (for lineage tracking)
        notes: Optional notes about this run
        reload_models: Force reload models

    Returns:
        Dictionary with run_id, paths to saved files, and summary metrics

    Raises:
        ValueError: If category is invalid or files don't exist
        FileNotFoundError: If original or modified file not found
    """
    # Validate inputs
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {CATEGORIES}")

    original_path = Path(original_file)
    modified_path = Path(modified_file)

    if not original_path.exists():
        raise FileNotFoundError(f"Original file not found: {original_path}")
    if not modified_path.exists():
        raise FileNotFoundError(f"Modified file not found: {modified_path}")

    # Ensure results directories exist
    ensure_results_dirs()

    # Generate run ID
    timestamp = datetime.now()
    run_id = generate_run_id(category, timestamp)
    print(f"\n{'='*60}")
    print(f"Starting adversarial test: {run_id}")
    print(f"{'='*60}")
    print(f"Category: {category}")
    print(f"Author: {author}")
    print(f"Original: {original_path.name}")
    print(f"Modified: {modified_path.name}")
    if parent_run_id:
        print(f"Parent run: {parent_run_id}")
    print()

    # Load models
    models = get_models(reload=reload_models)

    if not models:
        raise RuntimeError("No models loaded. Check saved_models directory.")

    # Process files and get predictions for each model
    print("Processing files and getting predictions...")
    model_results = {}
    original_vector = None
    modified_vector = None

    for model_type, model_data in models.items():
        config = model_data['config']
        components = model_data['components']
        short_name = config['short_name']

        print(f"  {config['name']}...", end=" ")

        # Extract features using this model's components
        orig_vec = process_code_file(original_path, components)
        mod_vec = process_code_file(modified_path, components)

        if orig_vec is None or mod_vec is None:
            print("FAILED (feature extraction)")
            model_results[model_type] = {
                'original_pred': None,
                'original_conf': None,
                'modified_pred': None,
                'modified_conf': None,
                'evasion': False,
                'error': 'Feature extraction failed',
            }
            continue

        # Store vectors for stealthiness calculation (use first successful extraction)
        if original_vector is None:
            original_vector = orig_vec.flatten()
            modified_vector = mod_vec.flatten()

        # Get predictions
        model = components['model']
        result = get_model_predictions(model, orig_vec, mod_vec, author)
        model_results[model_type] = result

        # Print result
        evasion_marker = "EVADED" if result.get('evasion') else "detected"
        print(f"{result.get('modified_pred', '?')} ({result.get('modified_conf', 0):.2%}) - {evasion_marker}")

    # Calculate stealthiness metrics
    print("\nCalculating stealthiness metrics...")
    stealthiness = calculate_stealthiness(original_vector, modified_vector)
    if stealthiness:
        print(f"  Stealth score: {stealthiness['stealth_score']:.4f} ({stealthiness['stealth_category']})")
        print(f"  Cosine distance: {stealthiness['cosine_distance']:.4f}")
        print(f"  Features changed: {stealthiness['features_changed_pct']:.2f}%")
    else:
        print("  FAILED to calculate stealthiness metrics")
        stealthiness = {}

    # Calculate aggregate metrics
    aggregate = calculate_aggregate_metrics(model_results, author)
    result_type = classify_result_type(
        aggregate.get('evasion_count', 0),
        len(model_results),
        aggregate.get('orig_accuracy_rate', 0)
    )

    print(f"\nResults:")
    print(f"  Original accuracy: {aggregate.get('orig_accuracy_rate', 0):.1f}%")
    print(f"  Modified accuracy: {aggregate.get('mod_accuracy_rate', 0):.1f}%")
    print(f"  Evasion rate: {aggregate.get('evasion_rate_pct', 0):.1f}% ({aggregate.get('evasion_count', 0)}/{len(model_results)} models)")
    print(f"  Result type: {result_type}")

    # Prepare result data
    result_data = {
        'parent_run_id': parent_run_id or '',
        'prompt_summary': prompt_summary,
        'author': author,
        'file_name': original_path.name,
        'original_file': str(original_path.absolute()),
        'modified_file': str(modified_path.absolute()),
        'ai_tool': ai_tool,
        'model_results': model_results,
        'stealthiness': stealthiness,
        'aggregate': aggregate,
        'result_type': result_type,
        'notes': notes,
        # Include raw vectors in JSON for detailed analysis (converted to list)
        'original_vector': original_vector.tolist() if original_vector is not None else None,
        'modified_vector': modified_vector.tolist() if modified_vector is not None else None,
    }

    # Save results
    print("\nSaving results...")
    tracker = ResultsTracker()
    saved_paths = tracker.save_result(run_id, category, result_data)

    print(f"  CSV: {saved_paths['category_csv']}")
    print(f"  JSON: {saved_paths['json']}")
    print(f"  Index: {saved_paths['index']}")

    print(f"\n{'='*60}")
    print(f"Test complete: {run_id}")
    print(f"{'='*60}\n")

    return {
        'run_id': run_id,
        'category': category,
        'author': author,
        'evasion_rate': aggregate.get('evasion_rate_pct', 0),
        'evasion_count': aggregate.get('evasion_count', 0),
        'result_type': result_type,
        'stealth_score': stealthiness.get('stealth_score'),
        'stealth_category': stealthiness.get('stealth_category'),
        'saved_paths': saved_paths,
        'model_results': model_results,
        'aggregate': aggregate,
    }


def get_run_lineage(run_id: str, category: str) -> list:
    """
    Get the lineage of a run (chain of parent runs).

    Args:
        run_id: The run ID to trace
        category: The category to search in

    Returns:
        List of run records from root to the specified run
    """
    tracker = ResultsTracker()
    return tracker.get_run_lineage(run_id, category)


def list_recent_runs(category: Optional[str] = None, limit: int = 10) -> list:
    """
    List recent runs.

    Args:
        category: Filter by category (None for all)
        limit: Maximum number of runs to return

    Returns:
        List of run records
    """
    tracker = ResultsTracker()
    return tracker.list_runs(category=category, limit=limit)


def main():
    """CLI entry point for running tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run adversarial stylometry tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s original.py modified.py -c restructuring -a "John Doe"
    %(prog)s orig.py mod.py -c renaming -a "Jane Smith" --ai-tool "Claude Sonnet 4"
    %(prog)s orig.py mod.py -c formatting -a "Bob" --parent prev_run_id
        """
    )

    parser.add_argument("original", help="Path to original Python file")
    parser.add_argument("modified", help="Path to modified Python file")
    parser.add_argument("-c", "--category", required=True, choices=CATEGORIES,
                        help="Transformation category")
    parser.add_argument("-a", "--author", required=True,
                        help="True author of the code")
    parser.add_argument("--ai-tool", default="",
                        help="AI tool used for modification")
    parser.add_argument("--prompt", default="",
                        help="Brief description of transformation prompt")
    parser.add_argument("--parent", default=None,
                        help="Parent run ID for lineage tracking")
    parser.add_argument("--notes", default="",
                        help="Optional notes about this run")
    parser.add_argument("--reload-models", action="store_true",
                        help="Force reload models")

    args = parser.parse_args()

    try:
        result = run_adversarial_test(
            original_file=args.original,
            modified_file=args.modified,
            category=args.category,
            author=args.author,
            ai_tool=args.ai_tool,
            prompt_summary=args.prompt,
            parent_run_id=args.parent,
            notes=args.notes,
            reload_models=args.reload_models,
        )

        print(f"\nRun ID: {result['run_id']}")
        print(f"Evasion rate: {result['evasion_rate']:.1f}%")
        print(f"Result type: {result['result_type']}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
