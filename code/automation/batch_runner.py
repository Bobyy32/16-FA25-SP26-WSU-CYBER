"""
Batch test runner for the Adversarial Stylometry system.

Orchestrates running one prompt across multiple files at once:
1. Select or validate files
2. Transform each file using an AI provider
3. Save modified files
4. Run adversarial tests on each pair
5. Aggregate and save batch-level results
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from automation.config import (
    BATCH_DEFAULTS,
    MODIFIED_FILES_DIR,
    ensure_results_dirs,
)
from automation.providers import get_provider
from automation.utils.dataset_scanner import (
    get_all_authors,
    resolve_author_from_path,
    select_files_for_batch,
)
from automation.utils.batch_tracker import BatchTracker, generate_batch_id
from automation.run_tests import run_adversarial_test


def run_batch_test(
    prompt: str,
    category: str,
    provider: str = "ollama",
    model: Optional[str] = None,
    author: Optional[str] = None,
    files: Optional[List[str]] = None,
    batch_size: int = BATCH_DEFAULTS["batch_size"],
    evolution_id: str = "",
    round_number: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a batch adversarial test: one prompt across multiple files.

    Args:
        prompt: The transformation prompt to apply.
        category: Transformation category (restructuring, renaming, etc.).
        provider: AI provider name ('ollama', 'anthropic', 'openai').
        model: Override the default model for the provider.
        author: Author to test against. Required if files not provided.
        files: Explicit list of file paths. If provided, author is auto-detected.
        batch_size: Number of files to test (used when files not provided).
        evolution_id: Link to parent evolution (empty if standalone batch).
        round_number: Evolution round number (0 if standalone).
        seed: Random seed for file selection reproducibility.

    Returns:
        Dictionary with batch_id, aggregated results, and individual run details.
    """
    ensure_results_dirs()
    timestamp = datetime.now()
    batch_id = generate_batch_id(category, timestamp)

    print(f"\n{'='*60}")
    print(f"Batch Test: {batch_id}")
    print(f"{'='*60}")
    print(f"Category: {category}")
    print(f"Provider: {provider}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Initialize provider
    ai_provider = get_provider(provider, model=model)
    model_name = ai_provider.model if hasattr(ai_provider, "model") else "unknown"
    print(f"Model: {model_name}")

    # Resolve files
    selected_files = _resolve_files(author, files, batch_size, seed)
    # Determine author
    if author is None and files:
        author = resolve_author_from_path(files[0]) or "unknown"
    print(f"Author: {author}")
    print(f"Files: {len(selected_files)}")
    print()

    # Create batch output directory for modified files
    batch_dir = MODIFIED_FILES_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    individual_results = []
    for i, (filename, original_path) in enumerate(selected_files, 1):
        print(f"[{i}/{len(selected_files)}] {filename}")
        result = _process_single_file(
            original_path=original_path,
            filename=filename,
            prompt=prompt,
            category=category,
            author=author,
            ai_provider=ai_provider,
            batch_dir=batch_dir,
            batch_id=batch_id,
        )
        individual_results.append(result)

    # Aggregate results
    aggregates = _aggregate_results(individual_results)

    print(f"\n{'='*60}")
    print(f"Batch Results: {batch_id}")
    print(f"{'='*60}")
    print(f"Files tested: {aggregates['num_files']}")
    print(f"Successful transforms: {aggregates['successful_transforms']}")
    print(f"Avg evasion rate: {aggregates['avg_evasion_rate']:.1f}%")
    print(f"Avg stealth score: {aggregates['avg_stealth_score']:.4f}")
    print(f"Best evasion: {aggregates['best_evasion_rate']:.1f}%")
    print(f"Worst evasion: {aggregates['worst_evasion_rate']:.1f}%")
    print(f"Full evasion count: {aggregates['full_evasion_count']}")
    print()

    # Save batch results
    batch_data = {
        "evolution_id": evolution_id,
        "round_number": round_number,
        "timestamp": timestamp.isoformat(),
        "category": category,
        "author": author,
        "prompt_text": prompt,
        "ai_provider": provider,
        "ai_model": model_name,
        "num_files": aggregates["num_files"],
        "avg_evasion_rate": aggregates["avg_evasion_rate"],
        "avg_stealth_score": aggregates["avg_stealth_score"],
        "best_evasion_rate": aggregates["best_evasion_rate"],
        "worst_evasion_rate": aggregates["worst_evasion_rate"],
        "full_evasion_count": aggregates["full_evasion_count"],
        "individual_results": individual_results,
        "per_model_evasion_rates": aggregates["per_model_evasion_rates"],
        "files_tested": [f[0] for f in selected_files],
    }

    tracker = BatchTracker()
    saved_paths = tracker.save_batch_result(batch_id, batch_data)
    print(f"Saved: {saved_paths['batch_json']}")

    return {
        "batch_id": batch_id,
        "category": category,
        "author": author,
        "prompt": prompt,
        "provider": provider,
        "model": model_name,
        "aggregates": aggregates,
        "individual_results": individual_results,
        "saved_paths": saved_paths,
    }


def _resolve_files(
    author: Optional[str],
    files: Optional[List[str]],
    batch_size: int,
    seed: Optional[int],
) -> List[Tuple[str, Path]]:
    """Resolve the list of files to test."""
    if files:
        # Validate provided file paths
        resolved = []
        for f in files:
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {f}")
            resolved.append((p.name, p))
        return resolved

    if author is None:
        raise ValueError("Either 'author' or 'files' must be provided.")

    max_batch = BATCH_DEFAULTS["max_batch_size"]
    if batch_size > max_batch:
        raise ValueError(f"batch_size {batch_size} exceeds max {max_batch}")

    return select_files_for_batch(author, split="testing", count=batch_size, seed=seed)


def _process_single_file(
    original_path: Path,
    filename: str,
    prompt: str,
    category: str,
    author: str,
    ai_provider,
    batch_dir: Path,
    batch_id: str,
) -> Dict[str, Any]:
    """
    Transform a single file and run the adversarial test.

    Returns a result dict with transformation and test outcomes.
    """
    # Read original code
    original_code = original_path.read_text(encoding="utf-8")

    # Transform via AI provider
    print(f"  Transforming with {ai_provider.name}...", end=" ")
    transform_result = ai_provider.transform_code(original_code, prompt)

    if not transform_result.success:
        print(f"FAILED: {transform_result.error}")
        return {
            "filename": filename,
            "original_path": str(original_path),
            "transform_success": False,
            "transform_error": transform_result.error,
            "evasion_rate": 0.0,
            "stealth_score": None,
            "run_id": None,
        }

    # Check if model returned identical code
    if transform_result.modified_code.strip() == original_code.strip():
        print("FAILED: model returned identical code (no changes made)")
        return {
            "filename": filename,
            "original_path": str(original_path),
            "transform_success": False,
            "transform_error": "Model returned identical code",
            "evasion_rate": 0.0,
            "stealth_score": None,
            "run_id": None,
        }

    print("OK")

    # Save modified file
    modified_path = batch_dir / f"modified_{filename}"
    modified_path.write_text(transform_result.modified_code, encoding="utf-8")

    # Run adversarial test
    print(f"  Running adversarial test...", end=" ")
    try:
        test_result = run_adversarial_test(
            original_file=str(original_path),
            modified_file=str(modified_path),
            category=category,
            author=author,
            ai_tool=f"{ai_provider.name}/{transform_result.model_name}",
            prompt_summary=prompt[:200],
            notes=f"batch={batch_id}",
        )
        print(f"evasion={test_result['evasion_rate']:.1f}%")

        return {
            "filename": filename,
            "original_path": str(original_path),
            "modified_path": str(modified_path),
            "transform_success": True,
            "run_id": test_result["run_id"],
            "evasion_rate": test_result["evasion_rate"],
            "evasion_count": test_result["evasion_count"],
            "stealth_score": test_result["stealth_score"],
            "stealth_category": test_result.get("stealth_category"),
            "result_type": test_result["result_type"],
            "model_results": test_result.get("model_results", {}),
            "input_tokens": transform_result.input_tokens,
            "output_tokens": transform_result.output_tokens,
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "filename": filename,
            "original_path": str(original_path),
            "modified_path": str(modified_path),
            "transform_success": True,
            "test_error": str(e),
            "evasion_rate": 0.0,
            "stealth_score": None,
            "run_id": None,
        }


def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate individual file results into batch-level metrics."""
    successful = [r for r in results if r.get("transform_success") and r.get("run_id")]

    if not successful:
        return {
            "num_files": len(results),
            "successful_transforms": 0,
            "avg_evasion_rate": 0.0,
            "avg_stealth_score": 0.0,
            "best_evasion_rate": 0.0,
            "worst_evasion_rate": 0.0,
            "full_evasion_count": 0,
            "per_model_evasion_rates": {},
        }

    evasion_rates = [r["evasion_rate"] for r in successful]
    stealth_scores = [
        r["stealth_score"] for r in successful if r.get("stealth_score") is not None
    ]

    # Per-model evasion rates
    per_model = {}
    for r in successful:
        model_results = r.get("model_results", {})
        for model_type, mr in model_results.items():
            if model_type not in per_model:
                per_model[model_type] = {"evaded": 0, "total": 0}
            per_model[model_type]["total"] += 1
            if mr.get("evasion"):
                per_model[model_type]["evaded"] += 1

    per_model_rates = {
        mt: (counts["evaded"] / counts["total"] * 100) if counts["total"] > 0 else 0.0
        for mt, counts in per_model.items()
    }

    # Count files with 100% evasion
    full_evasion_count = sum(1 for r in evasion_rates if r >= 100.0)

    return {
        "num_files": len(results),
        "successful_transforms": len(successful),
        "avg_evasion_rate": sum(evasion_rates) / len(evasion_rates),
        "avg_stealth_score": (
            sum(stealth_scores) / len(stealth_scores) if stealth_scores else 0.0
        ),
        "best_evasion_rate": max(evasion_rates),
        "worst_evasion_rate": min(evasion_rates),
        "full_evasion_count": full_evasion_count,
        "per_model_evasion_rates": per_model_rates,
    }
