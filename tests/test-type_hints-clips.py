from automation.evolution_runner import run_evolution

# Type hints evolution against clips

evo = run_evolution(
    initial_prompt="",
    category="type_hints",
    provider="ollama",
    model="qwen3.5:9b",
    author="clips",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    auto_resume=True,
)

print(f"\nStatus: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
