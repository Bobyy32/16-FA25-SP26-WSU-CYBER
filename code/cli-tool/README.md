# Adversarial Stylometry Testing Tool

A modular CLI tool for testing adversarial attacks against code authorship attribution models.

## Setup

```bash
cd temp
pip install -r requirements.txt
```

## Usage

### 1. Analyze Code Stylometry

Analyze the stylometric features of a Python file:

```bash
python main.py analyze path/to/code.py
```

Output includes:
- Naming conventions (camelCase vs. snake_case ratios)
- Indentation patterns
- Comment density and style
- Whitespace patterns
- Import organization
- Function definition style

### 2. Predict Author

Predict the author of a code file using a specific classifier:

```bash
python main.py predict path/to/code.py --model random_forest
```

Available models:
- `random_forest` (default)
- `sgd`
- `naive_bayes`
- `neural_network`

### 3. Test Evasion

Compare predictions on original vs. modified code:

```bash
python main.py evasion original.py modified.py \
  --true-author alice \
  --target-author bob \
  --save
```

### 4. Batch Testing

Run multiple test cases from a JSON file:

```bash
python main.py batch test_cases.json --save
```

## File Structure

- config.py - Configuration and paths
- model_loader.py - Load trained models
- code_analyzer.py - Extract stylometric features
- evaluator.py - Test and measure attacks
- main.py - CLI interface
