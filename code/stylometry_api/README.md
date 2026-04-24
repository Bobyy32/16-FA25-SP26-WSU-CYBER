# Stylometry API

Adversarial stylometry testing framework — transform code with an LLM, measure how well it evades authorship attribution models, and evolve prompts automatically.

## Overview

- **Single-file testing** — compare one original/modified pair against all 4 attribution models
- **Batch testing** — run one prompt across multiple files for statistically meaningful results
- **Prompt evolution** — automated loop that analyzes results and generates improved prompts
- **AI provider support** — Ollama (local, free), Anthropic (Claude), OpenAI (GPT), Google (Gemini)
- **Metrics** — evasion rate per model, composite stealth score, per-feature divergence

## Setup

### 1. Install Miniconda

Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html  
The project uses **Python 3.13** and stores its environment in `.conda/` at the repo root.

### 2. Create the conda environment

```bash
cd Adversarial_Stylometry
conda create -p ./.conda python=3.13 -y
conda activate ./.conda
```

### 3. Install dependencies

```bash
pip install numpy scipy scikit-learn joblib tensorflow \
            pandas matplotlib seaborn requests \
            anthropic openai google-genai
```

**Core** (always required):

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 2.4+ | Array operations |
| `scipy` | 1.17+ | Distance metrics |
| `scikit-learn` | 1.8+ | Attribution models + TF-IDF |
| `joblib` | 1.5+ | Model serialization |
| `tensorflow` | 2.20+ | Neural network model |
| `pandas` | 3.0+ | CSV result storage |
| `matplotlib` | 3.10+ | Figure generation |
| `requests` | 2.32+ | Ollama HTTP calls |

**Optional** (only needed for non-Ollama providers):

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API (`provider="anthropic"`) |
| `openai` | GPT API (`provider="openai"`) |
| `google-genai` | Gemini API (`provider="google"`) |

### 4. Install and start Ollama (default provider)

Ollama runs LLMs locally for free. Download from https://ollama.com

```bash
ollama serve                              # start the server (leave running)
ollama pull qwen3.5:4b
```

### 5. Verify everything works

```bash
.conda/python.exe -c "from stylometry_api.batch_runner import run_batch_test; print('OK')"
```

---

## Quick Start

```python
from stylometry_api.batch_runner import run_batch_test

result = run_batch_test(
    prompt="Inject dead code to obscure the author's style.",
    category="dead_code",
    author="aleju",
    batch_size=5,
    seed=42,
)

print(f"Avg evasion : {result['aggregates']['avg_evasion_rate']:.1f}%")
print(f"Avg stealth : {result['aggregates']['avg_stealth_score']:.4f}")
```

## Folder Structure

```
stylometry_api/
├── run_tests.py              # Single-file test runner
├── batch_runner.py           # Batch testing (1 prompt × N files)
├── prompt_evolver.py         # Analyze results + generate improved prompts
├── evolution_runner.py       # Multi-round automated evolution loop
├── config.py                 # Categories, paths, model config
├── providers/
│   ├── __init__.py           # get_provider() factory
│   ├── base.py               # AIProvider ABC + TransformationResult
│   ├── ollama_provider.py    # Ollama local inference (default)
│   ├── anthropic_provider.py # Claude API
│   ├── openai_provider.py    # GPT API
│   └── google_provider.py    # Gemini API
├── utils/
│   ├── model_loader.py       # Load saved models from ../saved_models/
│   ├── feature_extractor.py  # TF-IDF vectorization and feature extraction
│   ├── metrics.py            # Stealth score and evasion metrics
│   ├── results_tracker.py    # CSV and JSON result storage
│   ├── dataset_scanner.py    # Discover authors/files in dataset_splits/
│   └── batch_tracker.py      # Batch and evolution tracking
├── results/
│   ├── index.csv
│   ├── {category}/
│   │   ├── runs.csv
│   │   └── json/
│   └── batches/
│       ├── batch_index.csv
│       ├── json/
│       └── evolutions/
│           └── json/
└── modified_files/           # AI-generated modified files per batch
```

## Categories

| Category | Description |
|----------|-------------|
| `renaming` | Variable, function, and class renaming |
| `dead_code` | Injecting unused variables, unreachable blocks |
| `type_hints` | Adding or altering type annotations |
| `restructuring` | Function organization, code structure |
| `control_flow` | Loop/conditional restructuring |
| `formatting` | Whitespace, indentation, line breaks |
| `comments` | Adding, removing, or modifying comments |

## API Reference

### `run_batch_test()`

Run one prompt across multiple files and get aggregated evasion and stealth metrics.

```python
from stylometry_api.batch_runner import run_batch_test

result = run_batch_test(
    prompt: str,              # Transformation prompt
    category: str,            # See categories above
    provider: str = "ollama", # 'ollama', 'anthropic', 'openai', 'google'
    model: str = None,        # Override default model for the provider
    author: str = None,       # Author to pull files from (required if files not given)
    files: list = None,       # Explicit file paths (author auto-detected)
    batch_size: int = 5,      # Number of files (when using author)
    seed: int = None,         # Random seed for reproducible file selection
    evolution_id: str = "",   # Link to a parent evolution run
    round_number: int = 0,    # Round number within an evolution
)
```

Returns:
```python
{
    'batch_id': str,
    'aggregates': {
        'avg_evasion_rate': float,       # % of files where attribution failed
        'avg_stealth_score': float,      # composite stealth (lower = stealthier)
        'best_evasion_rate': float,
        'worst_evasion_rate': float,
        'full_evasion_count': int,       # files where all 4 models were evaded
        'per_model_evasion_rates': dict, # {model_name: evasion_%}
        'num_files': int,
        'successful_transforms': int,
    },
    'individual_results': list,
    'saved_paths': dict,
}
```

---

### `run_evolution()`

Automated loop: test prompt → analyze → generate improved prompt → repeat.

```python
from stylometry_api.evolution_runner import run_evolution

result = run_evolution(
    initial_prompt: str,
    category: str,
    provider: str = "ollama",
    model: str = None,
    author: str = None,
    batch_size: int = 5,
    max_rounds: int = 10,
    target_evasion_rate: float = 75.0,  # stop when avg evasion >= this %
    target_stealth_max: float = 0.75,   # stop when avg stealth <= this
    seed: int = None,
)
```

Returns:
```python
{
    'evolution_id': str,
    'status': str,            # 'target_met' or 'max_rounds_reached'
    'best_round': int,
    'best_evasion_rate': float,
    'best_prompt': str,
    'rounds_completed': int,
    'rounds': list,           # per-round details
    'all_prompts': list,
    'saved_path': str,
}
```

---

### `run_adversarial_test()`

Low-level: compare a single original/modified file pair against all 4 models.

```python
from stylometry_api.run_tests import run_adversarial_test

result = run_adversarial_test(
    original_file: str,
    modified_file: str,
    category: str,
    author: str,
    ai_tool: str = "",
    prompt_summary: str = "",
    notes: str = "",
)
```

Returns:
```python
{
    'run_id': str,
    'evasion_rate': float,
    'evasion_count': int,
    'stealth_score': float,
    'stealth_category': str,
    'result_type': str,
    'model_results': dict,
}
```

---

## Workflow Example

### 1. Test a prompt idea

```python
from stylometry_api.batch_runner import run_batch_test

result = run_batch_test(
    prompt="Inject unreachable blocks after return statements to add token noise.",
    category="dead_code",
    author="aleju",
    batch_size=5,
    seed=42,
)
print(f"Evasion: {result['aggregates']['avg_evasion_rate']:.1f}%")
print(f"Stealth: {result['aggregates']['avg_stealth_score']:.4f}")
```

### 2. Refine the prompt manually (recommended)

Look at the per-model breakdown and write an improved prompt yourself based on what you observe — which models were evaded, which weren't, and what the stealth score looks like. Call `run_batch_test()` again with the new prompt using the same `seed` so you're comparing against the same files.

```python
result2 = run_batch_test(
    prompt="Target unreachable conditional blocks with deeply nested arithmetic "
           "that evaluates to false, altering token patterns without touching active logic.",
    category="dead_code",
    author="aleju",
    batch_size=5,
    seed=42,   # same seed = same files as round 1, fair comparison
)
print(f"Evasion: {result2['aggregates']['avg_evasion_rate']:.1f}%")
print(f"Stealth: {result2['aggregates']['avg_stealth_score']:.4f}")
```

Repeat this loop — observe results, rewrite the prompt, test again. Human-written prompts tend to produce better and more interpretable results than AI-generated ones because you can reason directly about what the attribution models are sensitive to.

### 2b. Or run an automated evolution (optional)

If you want the system to generate and iterate prompts automatically, use `run_evolution()`. This is faster but gives up control over what the prompts actually say.

```python
from stylometry_api.evolution_runner import run_evolution

evo = run_evolution(
    initial_prompt="Inject unreachable blocks after return statements.",
    category="dead_code",
    author="aleju",
    batch_size=5,
    max_rounds=15,
    target_evasion_rate=75.0,
    target_stealth_max=0.75,
    seed=42,
)
print(f"Status     : {evo['status']}")
print(f"Best round : {evo['best_round']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}%")
print(f"Best prompt : {evo['best_prompt']}")
```

### 3. Try the best prompt on a different author

```python
result2 = run_batch_test(
    prompt=evo['best_prompt'],
    category="dead_code",
    author="clips",
    batch_size=5,
)
print(f"Evasion on clips: {result2['aggregates']['avg_evasion_rate']:.1f}%")
```

### 4. Explore what authors and files are available

```python
from stylometry_api.utils.dataset_scanner import get_all_authors, select_files_for_batch

authors = get_all_authors()
print(f"{len(authors)} authors available")

files = select_files_for_batch("aleju", split="testing", count=5, seed=42)
for name, path in files:
    print(name)
```

---

## Generated Data Layout

Every time you run a batch or evolution, results are saved automatically. Here's where everything ends up:

```
stylometry_api/
├── results/
│   ├── batches/
│   │   ├── batch_index.csv                        # one row per batch run (summary)
│   │   ├── json/
│   │   │   └── batch_dead_code_20260423_5fa1.json # full detail for one batch
│   │   └── evolutions/
│   │       └── json/
│   │           └── evo_dead_code_20260423_5fa1.json # full detail for one evolution
└── modified_files/
    └── batch_dead_code_20260423_5fa1/
        ├── modified_arithmetic.py                 # AI-transformed version
        └── modified_check_canny.py
```

### `batch_index.csv`

One row per batch run. Useful for quickly comparing prompts across runs.

| Column | Description |
|--------|-------------|
| `batch_id` | Unique ID for this batch |
| `category` | Transformation category |
| `author` | Author whose files were tested |
| `prompt_text` | The full prompt used |
| `ai_provider` / `ai_model` | Which LLM did the transformation |
| `avg_evasion_rate` | Average evasion % across all files |
| `avg_stealth_score` | Average stealth score across all files |
| `best_evasion_rate` | Highest evasion among the files |
| `per_model_evasion_rates` | Evasion % per attribution model |
| `timestamp` | When the batch ran |

### `results/batches/json/{batch_id}.json`

Full detail for one batch. Contains everything in the CSV row plus:

- `individual_results` — per-file breakdown (evasion rate, stealth score, transform success/failure, `run_id`)
- `files_tested` — list of filenames used
- `evolution_id` / `round_number` — set if this batch was part of an evolution

### `results/batches/evolutions/json/{evo_id}.json`

Full detail for one evolution run. Key fields:

- `status` — `"target_met"` or `"max_rounds_reached"`
- `best_round` / `best_evasion_rate` / `best_prompt` — the winning round
- `rounds` — list of every round with its prompt, `batch_id`, evasion, stealth, and per-model rates
- `all_prompts` — every prompt tried in order
- `files_tested` — the fixed file list used across all rounds
- `target_evasion_rate` / `target_stealth_max` — the thresholds the run was chasing

### `modified_files/{batch_id}/`

The actual AI-transformed Python files, named `modified_{original_filename}.py`. One folder per batch. These are what get tested against the attribution models.

---

## AI Providers

| Provider | Key required | Cost | Default model |
|----------|-------------|------|---------------|
| `ollama` (default) | None | Free (local) | `qwen3.5:4b` |
| `anthropic` | `ANTHROPIC_API_KEY` | Paid | `claude-sonnet-4-20250514` |
| `openai` | `OPENAI_API_KEY` | Paid | `gpt-4o` |
| `google` | `GOOGLE_API_KEY` | Free tier | `gemini-2.5-pro` |

### Ollama setup

```bash
ollama serve
ollama pull qwen3.5:4b
```

### Switching models

```python
result = run_batch_test(
    prompt="...",
    category="dead_code",
    provider="ollama",
    model="qwen3.5:4b",   # any pulled Ollama model
    author="aleju",
)
```

---

## Stealth Score

Composite metric measuring how different the modified code looks from the original (lower = harder to notice):

```
stealth = 0.3 × cosine_distance
        + 0.3 × relative_l2
        + 0.2 × js_divergence
        + 0.2 × features_changed_pct
```

Target: `< 0.75`

## Result Types

| Type | Evasion |
|------|---------|
| `full_evasion` | 100% — all 4 models evaded |
| `strong_evasion` | ≥ 75% |
| `partial_evasion` | ≥ 25% |
| `weak_evasion` | < 25% |
| `failed` | 0% |

## Run ID Format

`{category}_{YYYYMMDD}_{short_hash}` — e.g. `dead_code_20260423_5fa1`
