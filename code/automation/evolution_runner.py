"""
Multi-round automated prompt evolution for the Adversarial Stylometry system.

Runs an iterative loop:
1. Test a prompt across a batch of files
2. Analyze results
3. Generate an improved prompt
4. Repeat until targets are met or max rounds reached
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from automation.config import EVOLUTION_DEFAULTS, BATCH_DEFAULTS
from automation.batch_runner import run_batch_test
from automation.prompt_evolver import evolve_prompt
from automation.utils.batch_tracker import BatchTracker, generate_evolution_id
from automation.utils.dataset_scanner import select_files_for_batch


def run_evolution(
    initial_prompt: str,
    category: str,
    provider: str = "ollama",
    model: Optional[str] = None,
    author: Optional[str] = None,
    batch_size: int = BATCH_DEFAULTS["batch_size"],
    max_rounds: int = EVOLUTION_DEFAULTS["max_rounds"],
    target_evasion_rate: float = EVOLUTION_DEFAULTS["target_evasion_rate"],
    target_stealth_max: float = EVOLUTION_DEFAULTS["target_stealth_max"],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a multi-round prompt evolution loop.

    Selects files once in round 1 and reuses them across all rounds
    for fair comparison. Stops when targets are met or max rounds reached.

    Args:
        initial_prompt: Starting transformation prompt.
        category: Transformation category.
        provider: AI provider name ('ollama', 'anthropic', 'openai').
        model: Override model for the provider.
        author: Author to test against.
        batch_size: Number of files per batch.
        max_rounds: Maximum evolution rounds.
        target_evasion_rate: Stop when avg evasion >= this (percentage).
        target_stealth_max: Stop when avg stealth <= this.
        seed: Random seed for file selection.

    Returns:
        Dictionary with evolution_id, status, best_prompt, best results,
        and full round history.
    """
    if author is None:
        raise ValueError("'author' is required for evolution runs.")

    timestamp = datetime.now()
    evolution_id = generate_evolution_id(category, timestamp)

    print(f"\n{'#'*60}")
    print(f"EVOLUTION: {evolution_id}")
    print(f"{'#'*60}")
    print(f"Category: {category}")
    print(f"Author: {author}")
    print(f"Provider: {provider}")
    print(f"Batch size: {batch_size}")
    print(f"Max rounds: {max_rounds}")
    print(f"Target evasion: {target_evasion_rate}%")
    print(f"Target stealth: <= {target_stealth_max}")
    print(f"Initial prompt: {initial_prompt[:80]}{'...' if len(initial_prompt) > 80 else ''}")
    print()

    # Select files ONCE for all rounds (fair comparison)
    selected_files = select_files_for_batch(
        author, split="testing", count=batch_size, seed=seed
    )
    file_paths = [str(p) for _, p in selected_files]
    print(f"Selected {len(selected_files)} files for all rounds:")
    for fname, _ in selected_files:
        print(f"  - {fname}")
    print()

    # Evolution state
    current_prompt = initial_prompt
    prompts_used = [initial_prompt]
    rounds_history = []
    best_round = 0
    best_evasion = 0.0
    best_prompt = initial_prompt
    status = "max_rounds_reached"

    tracker = BatchTracker()

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{max_rounds}")
        print(f"{'='*60}")
        print(f"Prompt: {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")
        print()

        # Run batch test
        batch_result = run_batch_test(
            prompt=current_prompt,
            category=category,
            provider=provider,
            model=model,
            author=author,
            files=file_paths,
            evolution_id=evolution_id,
            round_number=round_num,
        )

        aggregates = batch_result["aggregates"]
        avg_evasion = aggregates["avg_evasion_rate"]
        avg_stealth = aggregates["avg_stealth_score"]

        # Track this round
        round_data = {
            "round": round_num,
            "prompt": current_prompt,
            "batch_id": batch_result["batch_id"],
            "avg_evasion_rate": avg_evasion,
            "avg_stealth_score": avg_stealth,
            "best_evasion_rate": aggregates["best_evasion_rate"],
            "worst_evasion_rate": aggregates["worst_evasion_rate"],
            "full_evasion_count": aggregates["full_evasion_count"],
            "per_model_evasion_rates": aggregates["per_model_evasion_rates"],
        }
        rounds_history.append(round_data)

        # Track best
        if avg_evasion > best_evasion:
            best_evasion = avg_evasion
            best_round = round_num
            best_prompt = current_prompt

        # Check stopping criteria
        if avg_evasion >= target_evasion_rate and avg_stealth <= target_stealth_max:
            status = "target_met"
            print(f"\n>>> TARGET MET in round {round_num}!")
            print(f"    Evasion: {avg_evasion:.1f}% >= {target_evasion_rate}%")
            print(f"    Stealth: {avg_stealth:.4f} <= {target_stealth_max}")
            break

        if avg_evasion >= target_evasion_rate:
            # Evasion met but stealth too high
            print(f"\n>>> Evasion target met ({avg_evasion:.1f}%) but stealth too high ({avg_stealth:.4f})")

        # Don't evolve after the last round
        if round_num >= max_rounds:
            break

        # Evolve prompt for next round
        print(f"\n--- Evolving prompt for round {round_num + 1} ---")
        try:
            new_prompt = evolve_prompt(
                batch_results=batch_result,
                previous_prompts=prompts_used,
                round_number=round_num,
                provider=provider,
                model=model,
            )
            current_prompt = new_prompt
            prompts_used.append(new_prompt)
            print(f"  New prompt: {new_prompt[:100]}{'...' if len(new_prompt) > 100 else ''}")
        except Exception as e:
            print(f"  Prompt evolution failed: {e}")
            print("  Reusing current prompt for next round.")

    # Save evolution results
    evolution_data = {
        "evolution_id": evolution_id,
        "status": status,
        "category": category,
        "author": author,
        "provider": provider,
        "model": batch_result.get("model", ""),
        "batch_size": batch_size,
        "max_rounds": max_rounds,
        "rounds_completed": len(rounds_history),
        "target_evasion_rate": target_evasion_rate,
        "target_stealth_max": target_stealth_max,
        "best_round": best_round,
        "best_evasion_rate": best_evasion,
        "best_prompt": best_prompt,
        "initial_prompt": initial_prompt,
        "all_prompts": prompts_used,
        "files_tested": [f[0] for f in selected_files],
        "rounds": rounds_history,
        "started_at": timestamp.isoformat(),
        "finished_at": datetime.now().isoformat(),
    }

    evo_path = tracker.save_evolution_result(evolution_id, evolution_data)

    print(f"\n{'#'*60}")
    print(f"EVOLUTION COMPLETE: {evolution_id}")
    print(f"{'#'*60}")
    print(f"Status: {status}")
    print(f"Rounds: {len(rounds_history)}")
    print(f"Best round: {best_round}")
    print(f"Best evasion: {best_evasion:.1f}%")
    print(f"Best prompt: {best_prompt[:100]}{'...' if len(best_prompt) > 100 else ''}")
    print(f"Saved: {evo_path}")
    print()

    return {
        "evolution_id": evolution_id,
        "status": status,
        "best_round": best_round,
        "best_evasion_rate": best_evasion,
        "best_prompt": best_prompt,
        "rounds_completed": len(rounds_history),
        "rounds": rounds_history,
        "all_prompts": prompts_used,
        "saved_path": str(evo_path),
    }
