from automation.evolution_runner import run_evolution

# Type hints evolution against aleju
# Add/remove type annotations to shift TF-IDF token distributions

evo = run_evolution(
    initial_prompt=(
        "Add type annotations to all function parameters, return types, and key variable "
        "assignments in this Python file. Import necessary types from the typing module. "
        "Do NOT rename, reformat, change comments, or alter control flow."
    ),
    category="type_hints",
    provider="ollama",
    model="qwen3.5:4b",
    author="aleju",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    auto_resume=True,
)

print(f"\nStatus: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
