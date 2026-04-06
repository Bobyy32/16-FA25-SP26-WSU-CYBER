from automation.evolution_runner import run_evolution

# Control flow evolution against huggingface
# auto_resume will pick up the best control_flow prompt from previous runs

evo = run_evolution(
    initial_prompt="",  # ignored when auto_resume finds a previous best
    category="control_flow",
    provider="ollama",
    model="qwen3.5:9b",
    author="huggingface",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    auto_resume=True,
)

print(f"\nStatus: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
