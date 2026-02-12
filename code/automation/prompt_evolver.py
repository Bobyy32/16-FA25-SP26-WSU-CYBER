"""
Prompt evolution for the Adversarial Stylometry system.

Analyzes batch test results and generates improved prompts that target
specific weaknesses in the current approach. Uses knowledge of the
attribution models' features to craft targeted transformations.
"""

from typing import Any, Dict, List, Optional

from automation.providers import get_provider
from automation.config import MODEL_TYPES


# Description of what the attribution models analyze - used in the meta-prompt
# so the AI understands what features to target
FEATURE_DESCRIPTION = """
The authorship attribution system uses 4 models (Random Forest, Naive Bayes, SGD Classifier, Neural Network).
Each model analyzes code using:

1. TF-IDF n-grams: Token-level patterns in the source code (variable names, keywords, operators, whitespace patterns)
2. 12 stylometric features:
   - inline_comments: Count of inline comments (# after code)
   - block_comments: Count of block/multiline comments
   - single_line_comments: Count of standalone comment lines
   - function_comments: Count of docstrings in functions
   - camel_case: Count of camelCase identifiers
   - snake_case: Count of snake_case identifiers
   - variable_name_length: Average variable name length
   - function_name_length: Average function name length
   - use_of_tabs: Whether tabs are used for indentation
   - use_of_spaces: Whether spaces are used for indentation
   - line_length: Average line length
   - import_style: Import style pattern (from...import vs import)

The models are sensitive to changes in naming conventions, comment styles, code structure patterns,
and whitespace usage. Changes that alter the TF-IDF token distribution (like renaming variables,
restructuring control flow, or changing import patterns) are most effective at evading detection.
"""


def build_analysis_context(
    batch_results: Dict[str, Any],
    previous_prompts: List[str],
    round_number: int,
) -> str:
    """
    Build the analysis context string that will be sent to the AI
    for prompt evolution.

    Args:
        batch_results: Results from the most recent batch test.
        previous_prompts: List of all prompts used so far (oldest first).
        round_number: Current round number.

    Returns:
        Formatted analysis context string.
    """
    aggregates = batch_results.get("aggregates", {})
    individual = batch_results.get("individual_results", [])
    per_model = aggregates.get("per_model_evasion_rates", {})

    # Build per-model breakdown
    model_lines = []
    for model_type, config in MODEL_TYPES.items():
        rate = per_model.get(model_type, 0.0)
        status = "EVADED" if rate >= 75.0 else "PARTIALLY EVADED" if rate >= 25.0 else "NOT EVADED"
        model_lines.append(f"  - {config['name']}: {rate:.1f}% evasion ({status})")

    model_breakdown = "\n".join(model_lines)

    # Build per-file summary
    file_lines = []
    for r in individual:
        if r.get("transform_success") and r.get("run_id"):
            stealth = r.get("stealth_score")
            stealth_str = f"{stealth:.4f}" if stealth is not None else "N/A"
            file_lines.append(
                f"  - {r['filename']}: evasion={r['evasion_rate']:.1f}%, "
                f"stealth={stealth_str}, type={r.get('result_type', 'unknown')}"
            )
        else:
            error = r.get("transform_error") or r.get("test_error", "unknown")
            file_lines.append(f"  - {r['filename']}: FAILED ({error})")

    file_summary = "\n".join(file_lines) if file_lines else "  No files processed"

    # Build prompt history
    prompt_history_lines = []
    for i, p in enumerate(previous_prompts, 1):
        prompt_history_lines.append(f"  Round {i}: {p}")
    prompt_history = "\n".join(prompt_history_lines)

    # Detect trends
    trend_notes = _analyze_trends(batch_results, round_number)

    context = f"""
{FEATURE_DESCRIPTION}

=== CURRENT RESULTS (Round {round_number}) ===

Overall:
  - Average evasion rate: {aggregates.get('avg_evasion_rate', 0):.1f}%
  - Average stealth score: {aggregates.get('avg_stealth_score', 0):.4f} (lower = stealthier, target < 0.5)
  - Files with 100% evasion: {aggregates.get('full_evasion_count', 0)}/{aggregates.get('num_files', 0)}

Per-model breakdown:
{model_breakdown}

Per-file results:
{file_summary}

=== PROMPT HISTORY ===
{prompt_history}

=== ANALYSIS NOTES ===
{trend_notes}

=== TASK ===
Based on the above results and the attribution system's features, generate an improved
transformation prompt. The prompt should instruct an AI to modify Python code in a way that:
1. Maximizes evasion of the attribution models (target: 75%+ evasion rate)
2. Keeps changes stealthy (stealth score < 0.5)
3. Preserves code functionality
4. Specifically targets the models that were NOT evaded

Focus on changes that alter the TF-IDF token distribution and stylometric features
without breaking the code. Be specific and actionable.

Return ONLY the new transformation prompt, nothing else.
"""
    return context.strip()


def evolve_prompt(
    batch_results: Dict[str, Any],
    previous_prompts: List[str],
    round_number: int,
    provider: str = "ollama",
    model: Optional[str] = None,
) -> str:
    """
    Generate an evolved prompt based on batch test results.

    Args:
        batch_results: Results from the most recent batch test.
        previous_prompts: List of all prompts used so far.
        round_number: Current round number.
        provider: AI provider to use for prompt generation.
        model: Override model for the provider.

    Returns:
        The new evolved prompt string.
    """
    ai = get_provider(provider, model=model)

    context = build_analysis_context(batch_results, previous_prompts, round_number)

    print(f"  Generating evolved prompt with {ai.name}...", end=" ")
    new_prompt = ai.generate_evolved_prompt(context)
    print("OK")

    if not new_prompt:
        raise RuntimeError("AI provider returned empty prompt")

    return new_prompt


def _analyze_trends(batch_results: Dict[str, Any], round_number: int) -> str:
    """Generate trend analysis notes to help guide evolution."""
    notes = []
    aggregates = batch_results.get("aggregates", {})
    per_model = aggregates.get("per_model_evasion_rates", {})

    avg_evasion = aggregates.get("avg_evasion_rate", 0)
    avg_stealth = aggregates.get("avg_stealth_score", 0)

    # Identify weakest models
    weak_models = [
        MODEL_TYPES[mt]["name"]
        for mt, rate in per_model.items()
        if rate < 50.0
    ]
    if weak_models:
        notes.append(
            f"These models are hardest to evade: {', '.join(weak_models)}. "
            "Focus transformations on features these models rely on most."
        )

    # Stealth-evasion tradeoff
    if avg_evasion > 50 and avg_stealth > 0.5:
        notes.append(
            "Evasion is moderate but stealth is poor. Make changes more subtle - "
            "smaller, distributed modifications rather than large rewrites."
        )
    elif avg_evasion < 25:
        notes.append(
            "Evasion is very low. Try more aggressive transformations: "
            "change naming conventions, restructure control flow, alter import patterns."
        )

    # Plateauing detection (only meaningful after round 1)
    if round_number > 1 and avg_evasion < 30:
        notes.append(
            "Results remain low after multiple rounds. Consider a fundamentally "
            "different approach: change the type of transformation entirely."
        )

    if not notes:
        notes.append("No specific trends detected. Continue iterating.")

    return "\n".join(f"- {n}" for n in notes)
