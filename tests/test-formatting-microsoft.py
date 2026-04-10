from automation.evolution_runner import run_evolution

# First evolution for the "formatting" category
# No previous best exists, so auto_resume will fall through to the initial_prompt

evo = run_evolution(
    initial_prompt=(
        "Reformat the Python code to disrupt stylometric fingerprints while keeping it natural: "
        "1) Convert all indentation from spaces to tabs (or vice versa). "
        "2) Normalize line lengths to 80-100 chars by breaking long lines or joining short ones. "
        "3) Switch import style: change 'import X' to 'from X import *' and vice versa. "
        "4) Standardize all string literals to double quotes. "
        "Preserve all functionality and logic exactly."
    ),
    category="formatting",
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
