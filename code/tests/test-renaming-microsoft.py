from automation.evolution_runner import run_evolution

# auto_resume=True will pick up the best renaming prompt from evo_renaming_20260212_3750 (41.7%)
# Testing against a new author to see if the prompt generalizes

evo = run_evolution(
    initial_prompt="",  # ignored when auto_resume finds a previous best
    category="renaming",
    provider="ollama",
    model="qwen3-coder:30b",
    author="microsoft",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    auto_resume=True,
)

print(f"\nStatus: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
