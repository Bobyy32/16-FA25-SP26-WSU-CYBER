from automation.evolution_runner import run_evolution

# Dead code injection evolution against aleju
# Insert unused variables, dummy imports, no-op assignments, and decoy functions
# to poison TF-IDF token distributions without altering real functionality.

evo = run_evolution(
    initial_prompt=(
        "Inject dead code into this Python file to disguise the author's style. "
        "Add unused variable assignments with generic names (e.g. _tmp, _unused, _val). "
        "Insert no-op statements like redundant reassignments (x = x) after real assignments. "
        "Add unused imports for common stdlib modules (e.g. os, sys, math, itertools). "
        "Insert unreachable code blocks after return statements. "
        "Add decoy helper functions that are defined but never called. "
        "Sprinkle dummy constants at module level (e.g. _SENTINEL = None, _FLAG = False). "
        "Do NOT rename any existing variables or functions. "
        "Do NOT change control flow, formatting, or comments. "
        "Preserve all real functionality exactly."
    ),
    category="dead_code",
    provider="ollama",
    model="qwen3.5:9b",
    author="aleju",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    auto_resume=True,
)

print(f"\nStatus: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
