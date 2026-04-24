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
   - inline_comments, block_comments, single_line_comments, function_comments
   - camel_case count, snake_case count
   - variable_name_length (avg), function_name_length (avg)
   - use_of_tabs, use_of_spaces
   - line_length (avg), import_style

STEALTH SCORING (how "stealthy" the changes are — lower = better, target < 0.5):
  stealth_score = 0.3 * cosine_distance + 0.3 * relative_l2 + 0.2 * js_divergence + 0.2 * features_changed_pct
  - cosine_distance: measures angle between original and modified feature vectors (0 = identical direction)
  - relative_l2: euclidean distance normalized by original magnitude
  - js_divergence: statistical divergence between token distributions
  - features_changed_pct: percentage of features that changed significantly

KEY INSIGHT: To stay stealthy while evading, focus on TARGETED changes to the features each model
relies on most, rather than rewriting everything. Change the RIGHT features, not ALL features.
Renaming variables changes TF-IDF tokens (high evasion impact) with low cosine distance cost.
Massive rewrites change many features at once (high stealth cost) for diminishing evasion returns.
"""


CATEGORY_CONSTRAINTS = {
    "restructuring": (
        "CATEGORY CONSTRAINT — RESTRUCTURING ONLY:\n"
        "You may ONLY use these techniques: reorder statements, extract/inline helper functions,\n"
        "split/merge functions, change class hierarchy, move code between modules.\n"
        "You MUST NOT rename variables/functions, change comments, change formatting/whitespace,\n"
        "or change import styles. Those belong to other categories."
    ),
    "renaming": (
        "CATEGORY CONSTRAINT — RENAMING ONLY:\n"
        "You may ONLY use these techniques: rename variables, rename functions, rename parameters,\n"
        "rename classes, change naming conventions (snake_case/camelCase).\n"
        "You MUST NOT restructure control flow, change comments, change formatting/whitespace,\n"
        "or change import styles. Those belong to other categories."
    ),
    "formatting": (
        "CATEGORY CONSTRAINT — FORMATTING ONLY:\n"
        "You may ONLY use these techniques: change indentation style, change line lengths,\n"
        "change whitespace around operators, change blank line usage, change bracket placement.\n"
        "You MUST NOT rename variables/functions, restructure control flow, change comments,\n"
        "or change import styles. Those belong to other categories."
    ),
    "comments": (
        "CATEGORY CONSTRAINT — COMMENTS ONLY:\n"
        "You may ONLY use these techniques: add/remove/rewrite inline comments, add/remove/rewrite\n"
        "block comments, add/remove/rewrite docstrings, change comment style (# vs triple-quote).\n"
        "You MUST NOT rename variables/functions, restructure control flow, change formatting,\n"
        "or change import styles. Those belong to other categories."
    ),
    "control_flow": (
        "CATEGORY CONSTRAINT — CONTROL FLOW ONLY:\n"
        "You may ONLY use these techniques: convert for loops to comprehensions or vice versa,\n"
        "replace if/elif with dict dispatch or match/case, swap ternary vs full if/else,\n"
        "convert early returns to nested conditionals, invert boolean conditions,\n"
        "swap while/for loops, merge/split conditionals.\n"
        "You MUST NOT rename variables/functions, change comments, change formatting/whitespace,\n"
        "or change import styles. Those belong to other categories."
    ),
    "type_hints": (
        "CATEGORY CONSTRAINT — TYPE HINTS ONLY:\n"
        "You may ONLY use these techniques: add or remove type annotations on function parameters,\n"
        "return types, and variable assignments. Add/remove imports from typing (Optional, List,\n"
        "Dict, Tuple, Union, Any, etc.).\n"
        "You MUST NOT rename variables/functions, change comments, change formatting/whitespace,\n"
        "restructure control flow, or change import styles. Those belong to other categories."
    ),
}


def build_analysis_context(
    batch_results: Dict[str, Any],
    previous_prompts: List[str],
    round_number: int,
    category: str = "",
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

    # Build prompt history — only include last 3 to keep context manageable
    recent_prompts = previous_prompts[-3:]
    start_idx = len(previous_prompts) - len(recent_prompts) + 1
    prompt_history_lines = []
    for i, p in enumerate(recent_prompts, start_idx):
        prompt_history_lines.append(f"  Round {i}: {p}")
    if len(previous_prompts) > 3:
        prompt_history_lines.insert(0, f"  (showing last 3 of {len(previous_prompts)} prompts)")
    prompt_history = "\n".join(prompt_history_lines)

    # Detect trends
    trend_notes = _analyze_trends(batch_results, round_number)

    # Category constraint
    category_upper = category.upper().replace("_", " ")
    category_constraint = CATEGORY_CONSTRAINTS.get(
        category,
        f"CATEGORY CONSTRAINT — {category_upper}:\n"
        f"Only use techniques that belong to the '{category}' transformation category.\n"
        "Do NOT mix in techniques from other categories (renaming, restructuring, formatting, comments, etc.)."
    )

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

=== CATEGORY: {category_upper} ===
{category_constraint}

=== TASK ===
Generate an improved transformation prompt based on the results above.
The prompt MUST only use techniques allowed by the category constraint above.

RULES:
1. STAY IN YOUR LANE. Only use techniques from the {category_upper} category.
   Do NOT add renaming, restructuring, comment changes, or other techniques
   that belong to different categories, even if they might help evasion.
2. ITERATE, don't restart. If the last prompt had good evasion, keep its core strategy
   and refine it. Only change what isn't working.
3. If evasion was high but stealth was poor: keep the same transformations but make them
   more surgical and natural.
4. If evasion was low: try a different strategy WITHIN THIS CATEGORY targeting the
   specific un-evaded models.
5. Keep the prompt SHORT — 2-3 sentences MAX. No bullet lists, no numbered steps.
6. Preserve code functionality.
7. DO NOT add dummy/dead code, random whitespace, or pass statements — these inflate
   stealth scores without helping evasion.

The prompt should instruct an AI to modify Python source code. Be specific and actionable.

CRITICAL: Return ONLY the new transformation prompt. It must be 2-3 sentences, no more.
Do NOT return a list, do NOT use bullet points, do NOT explain your reasoning.
"""
    return context.strip()


def evolve_prompt(
    batch_results: Dict[str, Any],
    previous_prompts: List[str],
    round_number: int,
    provider: str = "ollama",
    model: Optional[str] = None,
    category: str = "",
) -> str:
    """
    Generate an evolved prompt based on batch test results.

    Args:
        batch_results: Results from the most recent batch test.
        previous_prompts: List of all prompts used so far.
        round_number: Current round number.
        provider: AI provider to use for prompt generation.
        model: Override model for the provider.
        category: Transformation category to constrain the evolver.

    Returns:
        The new evolved prompt string.
    """
    ai = get_provider(provider, model=model)

    context = build_analysis_context(batch_results, previous_prompts, round_number, category)

    print(f"  Generating evolved prompt with {ai.name}...", end=" ")
    new_prompt = ai.generate_evolved_prompt(context)
    print("OK")

    if not new_prompt:
        raise RuntimeError("AI provider returned empty prompt")

    new_prompt = _sanitize_prompt(new_prompt)

    return new_prompt


def _sanitize_prompt(prompt: str) -> str:
    """Strip rule violations from an evolved prompt as a safety net."""
    import re

    lines = prompt.strip().splitlines()

    # Drop lines that look like bullet points, numbered steps, or headers
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(\*|-|•|\d+[.)]) ", stripped):
            continue
        if stripped.startswith("#"):
            continue
        cleaned.append(stripped)

    # Rejoin into sentences and enforce 3-sentence max
    text = " ".join(cleaned)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if len(sentences) > 10:
        sentences = sentences[:10]

    return " ".join(sentences)


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

    # Stealth-evasion tradeoff — give specific, actionable guidance
    if avg_evasion >= 75 and avg_stealth > 0.5:
        notes.append(
            "Evasion is GOOD but stealth is too high. DO NOT reduce the scope of changes. "
            "Instead, make the SAME types of changes but more surgically: rename variables "
            "to realistic alternatives (not random gibberish), use natural-looking comments, "
            "and avoid adding dead code or dummy statements that inflate feature distances."
        )
    elif avg_evasion > 40 and avg_stealth > 0.5:
        notes.append(
            "Evasion is promising but stealth is too high. Keep the evasion strategy but "
            "reduce unnecessary bulk: avoid dummy variables, dead code, and random whitespace. "
            "Focus on renaming identifiers to common alternatives and subtle comment changes."
        )
    elif avg_evasion < 25:
        notes.append(
            "Evasion is very low. Try more aggressive transformations WITHIN the allowed "
            "category. Push harder on the permitted techniques — be bolder with the scope "
            "and intensity of changes, but stay within the category constraint."
        )

    # Plateauing detection
    if round_number > 3 and avg_evasion < 30:
        notes.append(
            "Results remain low after multiple rounds. Try a fundamentally different "
            "approach WITHIN the allowed category — vary the intensity, target different "
            "code patterns, or apply changes more/less aggressively."
        )

    if not notes:
        notes.append("No specific trends detected. Continue iterating.")

    return "\n".join(f"- {n}" for n in notes)
