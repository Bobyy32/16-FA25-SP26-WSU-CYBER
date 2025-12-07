"""CLI entry point for adversarial stylometry testing."""

import argparse
import sys
from pathlib import Path
from model_loader import ModelLoader, ConfidenceBasedClassifier
from code_analyzer import CodeAnalyzer
from evaluator import Evaluator


def load_code_file(filepath: str) -> str:
    """Load code from a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def analyze_code(args):
    """Analyze stylometry of a code file."""
    code = load_code_file(args.file)
    analyzer = CodeAnalyzer()
    analysis = analyzer.full_analysis(code)
    analyzer.print_analysis(analysis)


def predict_author(args):
    """Predict author of a code file."""
    code = load_code_file(args.file)

    loader = ModelLoader()
    if not loader.load_model(args.model):
        print(f"Failed to load model: {args.model}")
        sys.exit(1)

    author, confidence = loader.predict(args.model, code)

    if author:
        print(f"\nPrediction using {args.model}:")
        print(f"  Predicted Author: {author}")
        print(f"  Confidence: {confidence:.4f}")
    else:
        print("Failed to make prediction")
        sys.exit(1)


def test_evasion(args):
    """Test evasion by comparing original and modified code."""
    original_code = load_code_file(args.original)
    modified_code = load_code_file(args.modified)

    loader = ModelLoader()
    loader.load_all_models()

    evaluator = Evaluator(loader)
    result = evaluator.test_code_pair(original_code, modified_code, args.true_author, args.target_author)

    # Print results
    print("\n" + "="*60)
    print("EVASION TEST RESULTS")
    print("="*60)
    print(f"True Author: {args.true_author}")
    if args.target_author:
        print(f"Target Author (Spoofing): {args.target_author}")

    print("\n--- ORIGINAL CODE PREDICTIONS ---")
    for model_name, pred in result["original_predictions"].items():
        correct_str = "(CORRECT)" if pred["correct"] else "(WRONG)"
        print(f"{model_name}: {pred['predicted_author']} ({pred['confidence']:.4f}) {correct_str}")

    print("\n--- MODIFIED CODE PREDICTIONS ---")
    for model_name, pred in result["modified_predictions"].items():
        correct_str = "(CORRECT)" if pred["correct"] else "(WRONG)"
        print(f"{model_name}: {pred['predicted_author']} ({pred['confidence']:.4f}) {correct_str}")

    print("\n--- EVASION SUCCESS ---")
    for model_name, evasion in result["evasion_results"].items():
        print(f"{model_name}:")
        print(f"  Evasion: {'SUCCESS' if evasion['evasion_success'] else 'FAILED'}")
        if args.target_author:
            print(f"  Misattribution: {'SUCCESS' if evasion['misattribution_success'] else 'FAILED'}")
        print(f"  Confidence Change: {evasion['confidence_change']:+.4f}")

    # Save results
    if args.save:
        evaluator.save_results()


def batch_test(args):
    """Run batch testing from a test file."""
    try:
        import json
        with open(args.file, 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading test file: {e}")
        sys.exit(1)

    loader = ModelLoader()
    loader.load_all_models()

    evaluator = Evaluator(loader)

    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Testing: {test_case.get('name', 'Unknown')}")

        original_code = load_code_file(test_case["original_file"])
        modified_code = load_code_file(test_case["modified_file"])

        evaluator.test_code_pair(
            original_code,
            modified_code,
            test_case["true_author"],
            test_case.get("target_author"),
        )

    evaluator.print_summary()

    if args.save:
        evaluator.save_results()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adversarial Stylometry Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze code stylometry
  python main.py analyze code.py

  # Predict author using Random Forest
  python main.py predict code.py --model random_forest

  # Test evasion attempt
  python main.py evasion original.py modified.py --true-author alice --target-author bob --save

  # Batch test multiple pairs
  python main.py batch tests.json --save
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code stylometry")
    analyze_parser.add_argument("file", help="Path to Python code file")
    analyze_parser.set_defaults(func=analyze_code)

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict author of code")
    predict_parser.add_argument("file", help="Path to Python code file")
    predict_parser.add_argument("--model", default="random_forest",
                               choices=["random_forest", "sgd", "naive_bayes", "neural_network"],
                               help="Model to use for prediction")
    predict_parser.set_defaults(func=predict_author)

    # Evasion test command
    evasion_parser = subparsers.add_parser("evasion", help="Test evasion on a code pair")
    evasion_parser.add_argument("original", help="Original code file")
    evasion_parser.add_argument("modified", help="Modified code file")
    evasion_parser.add_argument("--true-author", required=True, help="True author of the code")
    evasion_parser.add_argument("--target-author", help="Target author for spoofing (optional)")
    evasion_parser.add_argument("--save", action="store_true", help="Save results to JSON")
    evasion_parser.set_defaults(func=test_evasion)

    # Batch test command
    batch_parser = subparsers.add_parser("batch", help="Run batch tests from JSON file")
    batch_parser.add_argument("file", help="JSON file with test cases")
    batch_parser.add_argument("--save", action="store_true", help="Save results to JSON")
    batch_parser.set_defaults(func=batch_test)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
