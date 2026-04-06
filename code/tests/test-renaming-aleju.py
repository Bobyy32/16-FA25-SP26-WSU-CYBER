from automation.batch_runner import run_batch_test
from automation.evolution_runner import run_evolution

prompt_message = "Rename all variables, functions, parameters, and class names to generic, convention-neutral alternatives. Use names like data, result, value, item, process, handle, compute. Replace author-specific naming patterns (Hungarian notation, abbreviations, verbose descriptive names) with plain, common alternatives. Preserve all functionality and imports."

evo = run_evolution(
    initial_prompt=prompt_message,
    category="renaming",
    provider="ollama",
    model="qwen3-coder:30b",
    author="aleju",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
)

print(f"Status: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")