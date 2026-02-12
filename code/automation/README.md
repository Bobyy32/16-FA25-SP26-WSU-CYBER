# Adversarial Stylometry Automation System

Automated testing framework for adversarial stylometry experiments with comprehensive result tracking.

## Overview

This system provides:
- **Single-file testing** of adversarial code modifications against 4 attribution models
- **Batch testing** - test one prompt across 5-10 files at once for statistically meaningful results
- **Prompt evolution** - fully automatic loop that analyzes results and generates improved prompts
- **AI provider integration** - Ollama (local, free), Anthropic (Claude API), OpenAI (GPT API)
- **Category-based organization** for different transformation types
- **Lineage tracking** to trace how modifications evolve across iterations
- **Detailed metrics** including stealthiness scores and per-model evasion rates
- **Dual storage** with CSV spreadsheets for quick analysis and JSON files for detailed inspection

## Quick Start

### Python API

```python
from automation.run_tests import run_adversarial_test

# Run a test
result = run_adversarial_test(
    original_file="path/to/original.py",
    modified_file="path/to/modified.py",
    category="restructuring",
    author="author_name",
    ai_tool="Claude Sonnet 4",
    prompt_summary="Restructure with better variable names"
)

print(f"Run ID: {result['run_id']}")
print(f"Evasion rate: {result['evasion_rate']:.1f}%")
```

### Command Line

```bash
# Basic usage
python -m automation.run_tests original.py modified.py -c restructuring -a "author_name"

# With all options
python -m automation.run_tests original.py modified.py \
    -c restructuring \
    -a "author_name" \
    --ai-tool "Claude Sonnet 4" \
    --prompt "Restructure with better variable names" \
    --parent "restructuring_20260205_a3f2"
```

## Folder Structure

```
automation/
├── run_tests.py              # Single-file test runner
├── batch_runner.py           # Batch testing (1 prompt × N files)
├── prompt_evolver.py         # Analyze results + generate improved prompts
├── evolution_runner.py       # Multi-round automated evolution loop
├── config.py                 # Configuration (categories, paths, models)
├── providers/
│   ├── __init__.py           # Provider registry + get_provider() factory
│   ├── base.py               # AIProvider ABC + TransformationResult
│   ├── ollama_provider.py    # Ollama local inference (DEFAULT)
│   ├── anthropic_provider.py # Claude API integration
│   └── openai_provider.py    # GPT API integration
├── utils/
│   ├── __init__.py
│   ├── model_loader.py       # Load saved models from ../saved_models/
│   ├── feature_extractor.py  # Vectorization and feature extraction
│   ├── metrics.py            # Stealthiness and evasion metrics
│   ├── results_tracker.py    # CSV and JSON result management
│   ├── dataset_scanner.py    # Discover authors/files in dataset_splits/
│   └── batch_tracker.py      # Batch + evolution CSV/JSON tracking
├── results/
│   ├── index.csv             # Master index linking all runs
│   ├── restructuring/
│   │   ├── runs.csv          # All runs for this category
│   │   └── json/             # Detailed JSON for each run
│   ├── renaming/ ...
│   ├── formatting/ ...
│   ├── comments/ ...
│   └── batches/
│       ├── batch_index.csv   # Index of all batch runs
│       ├── json/             # Per-batch detail JSON
│       └── evolutions/
│           └── json/         # Per-evolution detail JSON
├── modified_files/           # AI-generated modified files
│   └── {batch_id}/           # One directory per batch
└── README.md
```

## Categories

| Category | Description |
|----------|-------------|
| `restructuring` | Code structure changes (function organization, control flow) |
| `renaming` | Variable, function, and class renaming |
| `formatting` | Whitespace, indentation, line breaks |
| `comments` | Adding, removing, or modifying comments |

## Run ID Format

Run IDs follow the pattern: `{category}_{YYYYMMDD}_{short_hash}`

Example: `restructuring_20260205_a3f2`

## Lineage Tracking

Track how modifications build on each other:

```python
from automation.run_tests import run_adversarial_test, get_run_lineage

# First run in a chain
result1 = run_adversarial_test(
    original_file="code.py",
    modified_file="code_v1.py",
    category="restructuring",
    author="alice",
    prompt_summary="Initial restructuring"
)

# Build on previous run
result2 = run_adversarial_test(
    original_file="code.py",
    modified_file="code_v2.py",
    category="restructuring",
    author="alice",
    parent_run_id=result1['run_id'],  # Links to previous
    prompt_summary="Further restructuring"
)

# Get lineage
lineage = get_run_lineage(result2['run_id'], "restructuring")
for run in lineage:
    print(f"{run['run_id']}: {run['prompt_summary']}")
```

## Spreadsheet Columns

### Metadata
- `run_id` - Unique identifier
- `timestamp` - When the test was run
- `parent_run_id` - ID of the run this builds upon
- `category` - Transformation category
- `prompt_summary` - Brief description of the transformation
- `author` - True author of the code
- `file_name` - Base file name
- `original_file` / `modified_file` - Full paths
- `ai_tool` - Tool used (Claude, GPT, etc.)

### Per-Model Results
For each model (rf, nb, sgd, nn):
- `{model}_orig_pred` - Prediction on original
- `{model}_orig_conf` - Confidence on original
- `{model}_mod_pred` - Prediction on modified
- `{model}_mod_conf` - Confidence on modified
- `{model}_evasion` - Whether evasion succeeded

### Stealthiness Metrics
- `l1_distance` - Manhattan distance
- `l2_distance` - Euclidean distance
- `cosine_distance` - Cosine distance (0=identical)
- `relative_l2` - L2 normalized by original magnitude
- `js_divergence` - Jensen-Shannon divergence
- `features_changed_pct` - % of features with significant change
- `max_feature_change` / `avg_feature_change` - Feature modification stats
- `stealth_score` - Composite score (lower = more stealthy)
- `stealth_category` - Category (high/medium/low)

### Aggregate Metrics
- `orig_accuracy_rate` - % of models correct on original
- `mod_accuracy_rate` - % of models correct on modified
- `evasion_count` - Number of models evaded
- `evasion_rate_pct` - % of models evaded
- `consensus_prediction` - Most common prediction
- `result_type` - Classification (full_evasion, partial_evasion, etc.)

## Result Types

| Type | Description |
|------|-------------|
| `full_evasion` | All models evaded (100%) |
| `strong_evasion` | Most models evaded (>=75%) |
| `partial_evasion` | Some models evaded (>=25%) |
| `weak_evasion` | Few models evaded (<25%) |
| `failed` | No models evaded |
| `original_unknown` | Original wasn't identified by most models |

## Models

The system tests against four attribution models:
- **Random Forest** (rf) - Ensemble tree-based classifier
- **Naive Bayes** (nb) - Probabilistic classifier
- **SGD Classifier** (sgd) - Linear classifier with stochastic gradient descent
- **Neural Network** (nn) - Deep learning classifier

Models are loaded from `../saved_models/attribution/`.

## API Reference

### `run_adversarial_test()`

```python
result = run_adversarial_test(
    original_file: str,          # Path to original file
    modified_file: str,          # Path to modified file
    category: str,               # Category (restructuring, renaming, etc.)
    author: str,                 # True author name
    ai_tool: str = "",           # AI tool used
    prompt_summary: str = "",    # Description of transformation
    parent_run_id: str = None,   # Parent run for lineage
    notes: str = "",             # Optional notes
    reload_models: bool = False  # Force reload models
) -> dict
```

Returns:
```python
{
    'run_id': str,
    'category': str,
    'author': str,
    'evasion_rate': float,
    'evasion_count': int,
    'result_type': str,
    'stealth_score': float,
    'stealth_category': str,
    'saved_paths': {
        'category_csv': Path,
        'json': Path,
        'index': Path
    },
    'model_results': dict,
    'aggregate': dict
}
```

### `get_run_lineage()`

```python
lineage = get_run_lineage(run_id: str, category: str) -> List[dict]
```

Returns list of run records from root to specified run.

### `list_recent_runs()`

```python
runs = list_recent_runs(category: str = None, limit: int = 10) -> List[dict]
```

Returns recent runs, optionally filtered by category.

### `run_batch_test()`

```python
result = run_batch_test(
    prompt: str,                     # Transformation prompt
    category: str,                   # Category (restructuring, renaming, etc.)
    provider: str = "ollama",        # AI provider ('ollama', 'anthropic', 'openai')
    model: str = None,               # Override default model
    author: str = None,              # Author to test (required if files not given)
    files: List[str] = None,         # Explicit file paths (auto-detects author)
    batch_size: int = 5,             # Files per batch (when using author)
    evolution_id: str = "",          # Link to parent evolution
    round_number: int = 0,           # Evolution round number
    seed: int = None,                # Random seed for file selection
) -> dict
```

Returns:
```python
{
    'batch_id': str,
    'category': str,
    'author': str,
    'prompt': str,
    'provider': str,
    'model': str,
    'aggregates': {
        'num_files': int,
        'successful_transforms': int,
        'avg_evasion_rate': float,
        'avg_stealth_score': float,
        'best_evasion_rate': float,
        'worst_evasion_rate': float,
        'full_evasion_count': int,
        'per_model_evasion_rates': dict,
    },
    'individual_results': List[dict],
    'saved_paths': dict,
}
```

### `run_evolution()`

```python
result = run_evolution(
    initial_prompt: str,                 # Starting prompt
    category: str,                       # Category
    provider: str = "ollama",            # AI provider
    model: str = None,                   # Override default model
    author: str = None,                  # Author (required)
    batch_size: int = 5,                 # Files per batch
    max_rounds: int = 10,                # Max evolution rounds
    target_evasion_rate: float = 75.0,   # Stop when avg evasion >= this %
    target_stealth_max: float = 0.5,     # Stop when avg stealth <= this
    seed: int = None,                    # Random seed
) -> dict
```

Returns:
```python
{
    'evolution_id': str,
    'status': str,               # 'target_met' or 'max_rounds_reached'
    'best_round': int,
    'best_evasion_rate': float,
    'best_prompt': str,
    'rounds_completed': int,
    'rounds': List[dict],        # Per-round details
    'all_prompts': List[str],    # All prompts tried
    'saved_path': str,
}
```

## Example: Full Category Workflow

Working through the `restructuring` category from start to finish.

### Step 1: Explore the dataset

```python
from automation.utils.dataset_scanner import get_all_authors, select_files_for_batch

# See what authors are available
authors = get_all_authors()
print(f"{len(authors)} authors: {list(authors.keys())}")

# Preview files for an author
files = select_files_for_batch("aleju", split="testing", count=5, seed=42)
for name, path in files:
    print(f"  {name}")
```

### Step 2: Quick-test a prompt idea with a single batch

```python
from automation.batch_runner import run_batch_test

result = run_batch_test(
    prompt="Restructure this code: extract repeated logic into helper functions, "
           "use generic names like process_data and handle_input for new functions",
    category="restructuring",
    author="aleju",
    batch_size=5,
    seed=42,
)

print(f"Avg evasion: {result['aggregates']['avg_evasion_rate']:.1f}%")
print(f"Avg stealth: {result['aggregates']['avg_stealth_score']:.4f}")

# See which models were evaded
for model, rate in result['aggregates']['per_model_evasion_rates'].items():
    print(f"  {model}: {rate:.1f}%")
```

### Step 3: If results are promising, run a full evolution

```python
from automation.evolution_runner import run_evolution

evo = run_evolution(
    initial_prompt="Restructure this code: extract repeated logic into helper "
                   "functions, use generic names like process_data and handle_input",
    category="restructuring",
    author="aleju",
    batch_size=5,
    max_rounds=10,
    target_evasion_rate=75.0,
    seed=42,                    # same seed = same files as step 2
)

print(f"Status: {evo['status']}")
print(f"Best evasion: {evo['best_evasion_rate']:.1f}% (round {evo['best_round']})")
print(f"Best prompt: {evo['best_prompt']}")
```

### Step 4: Try the winning prompt on a different author

```python
result2 = run_batch_test(
    prompt=evo['best_prompt'],   # reuse the evolved prompt
    category="restructuring",
    author="clips",              # different author
    batch_size=5,
)

print(f"Avg evasion on clips: {result2['aggregates']['avg_evasion_rate']:.1f}%")
```

### Step 5: Try a different model for comparison

```python
result3 = run_batch_test(
    prompt=evo['best_prompt'],
    category="restructuring",
    author="aleju",
    model="deepseek-coder-v2:16b",  # different Ollama model
    batch_size=5,
    seed=42,
)

print(f"deepseek evasion: {result3['aggregates']['avg_evasion_rate']:.1f}%")
```

### Step 6: Review results

```python
from automation.utils.batch_tracker import BatchTracker

tracker = BatchTracker()

# List all batches for this category
batches = tracker.list_batches(category="restructuring")
for b in batches:
    print(f"{b['batch_id']}: evasion={b['avg_evasion_rate']}%, "
          f"stealth={b['avg_stealth_score']}")

# Load full details for a specific evolution
evo_data = tracker.get_evolution(evo['evolution_id'])
for r in evo_data['rounds']:
    print(f"Round {r['round']}: {r['avg_evasion_rate']:.1f}% evasion, "
          f"prompt: {r['prompt'][:60]}...")
```

Results are also saved to CSV files you can open in any spreadsheet app:
- Individual runs: `results/restructuring/runs.csv`
- Batch summaries: `results/batches/batch_index.csv`
- Evolution details: `results/batches/evolutions/json/{evo_id}.json`

---

## Batch Testing

Test one prompt across multiple files at once for statistically meaningful results.

### Single Batch (local, free with Ollama)

```python
from automation.batch_runner import run_batch_test

result = run_batch_test(
    prompt="Restructure using helper functions and generic variable names",
    category="restructuring",
    provider="ollama",           # default - local, free
    author="aleju",
    batch_size=5,
)

print(f"Avg evasion: {result['aggregates']['avg_evasion_rate']:.1f}%")
print(f"Avg stealth: {result['aggregates']['avg_stealth_score']:.4f}")
print(f"Best evasion: {result['aggregates']['best_evasion_rate']:.1f}%")
```

### Batch with specific files

```python
result = run_batch_test(
    prompt="Rename all variables to single letters",
    category="renaming",
    files=[
        "dataset_splits/aleju/testing/check_canny.py",
        "dataset_splits/aleju/testing/check_clouds.py",
    ],
)
```

### Batch with Claude API

```python
result = run_batch_test(
    prompt="Restructure using helper functions",
    category="restructuring",
    provider="anthropic",        # requires ANTHROPIC_API_KEY env var
    author="aleju",
    batch_size=3,
)
```

### What happens during a batch

1. Selects files from `dataset_splits/{author}/testing/`
2. For each file: transforms it via the AI provider, saves the modified version to `modified_files/{batch_id}/`
3. Runs `run_adversarial_test()` on each original/modified pair
4. Aggregates results (avg evasion, avg stealth, per-model rates)
5. Saves batch results to `results/batches/batch_index.csv` + JSON

---

## Prompt Evolution

Fully automatic loop that tests a prompt, analyzes what worked, generates an improved prompt, and repeats.

### Basic Evolution (local, free)

```python
from automation.evolution_runner import run_evolution

result = run_evolution(
    initial_prompt="Restructure with generic variable names",
    category="restructuring",
    provider="ollama",           # default - local, free
    author="aleju",
    batch_size=5,
    max_rounds=10,
    target_evasion_rate=75.0,    # stop when >= 75% evasion
)

print(f"Status: {result['status']}")  # 'target_met' or 'max_rounds_reached'
print(f"Best prompt (round {result['best_round']}): {result['best_prompt']}")
print(f"Best evasion: {result['best_evasion_rate']:.1f}%")
```

### How evolution works

```
Round 1: Test initial prompt across 5 files
         → Avg evasion: 25%, stealth: 0.3
         → Analyze: RF and NB not evaded, need more name changes
         → AI generates improved prompt

Round 2: Test evolved prompt across same 5 files
         → Avg evasion: 50%, stealth: 0.35
         → Analyze: NB still not evaded, try import changes
         → AI generates improved prompt

Round 3: Test evolved prompt
         → Avg evasion: 80%, stealth: 0.4
         → TARGET MET! Stop.
```

Key design: files are selected **once** in round 1 and reused across all rounds for fair comparison.

### Evolution parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_rounds` | 10 | Maximum evolution rounds |
| `target_evasion_rate` | 75.0 | Stop when avg evasion >= this % |
| `target_stealth_max` | 0.5 | Stop when avg stealth <= this |
| `batch_size` | 5 | Files per batch |
| `seed` | None | Random seed for file selection |

### What the prompt evolver knows

The evolution system tells the AI about:
- The 4 attribution models and which ones were/weren't evaded
- The 12 stylometric features the models analyze (comments, naming, whitespace, etc.)
- TF-IDF n-gram sensitivity
- Stealth-evasion tradeoff in current results
- All previous prompts and their results
- Trend analysis (plateauing detection, weak model targeting)

---

## AI Providers

| Provider | API Key | Cost | Default Model |
|----------|---------|------|---------------|
| `ollama` (default) | None needed | Free (local) | `qwen2.5-coder:14b` |
| `anthropic` | `ANTHROPIC_API_KEY` env var | Paid | `claude-sonnet-4-20250514` |
| `openai` | `OPENAI_API_KEY` env var | Paid | `gpt-4o` |

### Ollama setup

```bash
# Install Ollama: https://ollama.ai
ollama serve                     # start the server
ollama pull qwen2.5-coder:14b   # pull the default model
```

### Using a custom model

```python
result = run_batch_test(
    prompt="...",
    category="restructuring",
    provider="ollama",
    model="codellama:13b",       # any Ollama model
    author="aleju",
)
```

---

## Dataset Scanner

Discover what's available in `dataset_splits/`:

```python
from automation.utils.dataset_scanner import get_all_authors, select_files_for_batch

# List all 20 authors
authors = get_all_authors()
for name, path in authors.items():
    print(f"{name}: {path}")

# Select random files for testing
files = select_files_for_batch("aleju", split="testing", count=5, seed=42)
for filename, filepath in files:
    print(filename)
```

---

## Viewing Results

### CSV Files
Open `results/{category}/runs.csv` in any spreadsheet application (Excel, Google Sheets, etc.)

### JSON Files
Detailed results are stored in `results/{category}/json/{run_id}.json`

```python
from automation.utils.results_tracker import ResultsTracker

tracker = ResultsTracker()
details = tracker.get_run_json("restructuring_20260205_a3f2", "restructuring")
print(details['model_results'])
```

## Requirements

**Core (all features):**
- Python 3.8+
- numpy, scipy, scikit-learn, joblib
- tensorflow (for neural network model)
- requests (for Ollama provider)

**Optional (API providers):**
- `anthropic` - for Claude API (`pip install anthropic`)
- `openai` - for GPT API (`pip install openai`)

**Ollama (default provider):**
- Ollama installed and running (`ollama serve`)
- A model pulled (`ollama pull qwen2.5-coder:14b`)

The system uses saved models from `../saved_models/attribution/` and the conda env at `.conda/`.
