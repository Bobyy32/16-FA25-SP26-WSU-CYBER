from automation.evolution_runner import run_evolution

# Control flow evolution against aleju
# Rewrite logic structures: loops, conditionals, comprehensions, ternaries, etc.

evo = run_evolution(
    initial_prompt=(
        "Rewrite the control flow of this Python code to disguise the author's style. "
        "Convert for loops to list comprehensions or vice versa. "
        "Replace if/elif chains with dictionary dispatch or match/case. "
        "Swap ternary expressions for full if/else blocks. "
        "Convert early returns to nested conditionals or vice versa. "
        "Replace while loops with for loops where possible. "
        "Do NOT change variable names, comments, formatting, or add dead code. "
        "Preserve all functionality exactly."
    ),
    category="control_flow",
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
