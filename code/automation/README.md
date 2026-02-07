# Adversarial Stylometry Automation System

Automated testing framework for adversarial stylometry experiments with comprehensive result tracking.

## Overview

This system provides:
- **Automated testing** of adversarial code modifications against multiple attribution models
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
├── run_tests.py              # Main test runner script
├── config.py                 # Configuration (categories, paths, models)
├── utils/
│   ├── __init__.py
│   ├── model_loader.py       # Load saved models from ../saved_models/
│   ├── feature_extractor.py  # Vectorization and feature extraction
│   ├── metrics.py            # Stealthiness and evasion metrics
│   └── results_tracker.py    # CSV and JSON result management
├── results/
│   ├── index.csv             # Master index linking all runs
│   ├── restructuring/
│   │   ├── runs.csv          # All runs for this category
│   │   └── json/             # Detailed JSON for each run
│   ├── renaming/
│   │   ├── runs.csv
│   │   └── json/
│   ├── formatting/
│   │   ├── runs.csv
│   │   └── json/
│   └── comments/
│       ├── runs.csv
│       └── json/
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

## Example Workflow

```python
from automation.run_tests import run_adversarial_test, list_recent_runs

# Test multiple transformation iterations
base_file = "samples/alice/code.py"

# Iteration 1: Basic restructuring
result1 = run_adversarial_test(
    original_file=base_file,
    modified_file="samples/alice/code_v1.py",
    category="restructuring",
    author="alice",
    ai_tool="Claude Sonnet 4",
    prompt_summary="Extract helper functions"
)

# Iteration 2: Build on previous
result2 = run_adversarial_test(
    original_file=base_file,
    modified_file="samples/alice/code_v2.py",
    category="restructuring",
    author="alice",
    parent_run_id=result1['run_id'],
    ai_tool="Claude Sonnet 4",
    prompt_summary="Rename extracted functions to generic names"
)

# Check recent runs
for run in list_recent_runs(category="restructuring", limit=5):
    print(f"{run['run_id']}: {run['evasion_rate_pct']}% evasion")
```

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

- Python 3.8+
- numpy
- scipy
- scikit-learn
- tensorflow (for neural network model)
- joblib

The system uses saved models from `../saved_models/attribution/`.
