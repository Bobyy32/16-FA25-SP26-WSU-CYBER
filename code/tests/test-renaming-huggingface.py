from automation.evolution_runner import run_evolution

# Renaming evolution against huggingface (fresh author)
# auto_resume will pick up the 100% evasion prompt from evo_renaming_20260301_9ef8
# Testing whether that result was a fluke or if it generalizes to a 3rd author

evo = run_evolution(
    initial_prompt="",  # ignored when auto_resume finds a previous best
    category="renaming",
    provider="ollama",
    model="qwen3-coder:30b",
    author="huggingface",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    auto_resume=True,
)

print(f"\nStatus: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
