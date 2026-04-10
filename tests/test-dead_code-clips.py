from automation.evolution_runner import run_evolution

# Dead code injection evolution against clips
# Starting from best dead_code strategy found in aleju runs (90% evasion, round 4)

evo = run_evolution(
    initial_prompt="",
    category="dead_code",
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
