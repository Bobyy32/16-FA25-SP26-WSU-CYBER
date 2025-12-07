"""Evaluate adversarial attacks and measure evasion success."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from model_loader import ModelLoader
from code_analyzer import CodeAnalyzer
from config import RESULTS_DIR, AUTHORS


class Evaluator:
    """Evaluate adversarial attacks across all classifiers."""

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.analyzer = CodeAnalyzer()
        self.results = []

    def test_code_pair(self, original_code: str, modified_code: str, true_author: str,
                      target_author: str = None) -> Dict:
        """
        Test a code pair (original vs. modified) against all models.

        Args:
            original_code: Original code
            modified_code: Adversarially modified code
            true_author: Ground truth author
            target_author: If spoofing, the author being impersonated

        Returns:
            Results dictionary with predictions and metrics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "true_author": true_author,
            "target_author": target_author,
            "original_predictions": {},
            "modified_predictions": {},
            "evasion_results": {},
            "stylometry_analysis": {},
        }

        # Get predictions from all models
        for model_name in self.model_loader.models.keys():
            orig_author, orig_confidence = self.model_loader.predict(model_name, original_code)
            mod_author, mod_confidence = self.model_loader.predict(model_name, modified_code)

            results["original_predictions"][model_name] = {
                "predicted_author": orig_author,
                "confidence": float(orig_confidence) if orig_confidence else 0.0,
                "correct": orig_author == true_author,
            }

            results["modified_predictions"][model_name] = {
                "predicted_author": mod_author,
                "confidence": float(mod_confidence) if mod_confidence else 0.0,
                "correct": mod_author == true_author,
            }

            # Calculate evasion success
            evasion_success = orig_author == true_author and mod_author != true_author
            misattribution_success = target_author and mod_author == target_author

            results["evasion_results"][model_name] = {
                "evasion_success": evasion_success,
                "misattribution_success": misattribution_success,
                "confidence_change": float(mod_confidence - orig_confidence) if orig_confidence and mod_confidence else 0.0,
            }

        # Analyze stylometry
        original_analysis = self.analyzer.full_analysis(original_code)
        modified_analysis = self.analyzer.full_analysis(modified_code)

        results["stylometry_analysis"] = {
            "original": original_analysis,
            "modified": modified_analysis,
        }

        self.results.append(results)
        return results

    def batch_test(self, code_pairs: List[Tuple[str, str, str, str]]) -> List[Dict]:
        """
        Test multiple code pairs.

        Args:
            code_pairs: List of (original_code, modified_code, true_author, target_author)

        Returns:
            List of result dictionaries
        """
        batch_results = []
        for original, modified, true_author, target_author in code_pairs:
            result = self.test_code_pair(original, modified, true_author, target_author)
            batch_results.append(result)
        return batch_results

    def get_summary_stats(self, results: List[Dict] = None) -> Dict:
        """
        Calculate summary statistics for evaluation.

        Args:
            results: Results to summarize (default: all stored results)

        Returns:
            Summary statistics dictionary
        """
        if results is None:
            results = self.results

        if not results:
            return {}

        summary = {
            "total_tests": len(results),
            "per_model_stats": {},
        }

        for model_name in self.model_loader.models.keys():
            evasion_count = sum(1 for r in results if r["evasion_results"][model_name]["evasion_success"])
            misattribution_count = sum(1 for r in results if r["evasion_results"][model_name]["misattribution_success"])
            avg_confidence_change = sum(r["evasion_results"][model_name]["confidence_change"] for r in results) / len(results)

            summary["per_model_stats"][model_name] = {
                "evasion_success_rate": evasion_count / len(results),
                "misattribution_success_rate": misattribution_count / len(results),
                "avg_confidence_change": avg_confidence_change,
                "total_evasion_successes": evasion_count,
            }

        return summary

    def save_results(self, filename: str = None) -> str:
        """
        Save results to JSON file.

        Args:
            filename: Output filename (default: auto-generated with timestamp)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"adversarial_evaluation_{timestamp}.json"

        filepath = RESULTS_DIR / filename

        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary_stats(),
            "detailed_results": self.results,
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {filepath}")
        return str(filepath)

    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.get_summary_stats()

        if not summary:
            print("No results to summarize")
            return

        print("\n" + "="*60)
        print("ADVERSARIAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print("\nPer-Model Results:")

        for model_name, stats in summary["per_model_stats"].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Evasion Success Rate: {stats['evasion_success_rate']:.2%}")
            print(f"  Misattribution Success Rate: {stats['misattribution_success_rate']:.2%}")
            print(f"  Avg Confidence Change: {stats['avg_confidence_change']:.4f}")
            print(f"  Total Evasion Successes: {stats['total_evasion_successes']}/{summary['total_tests']}")

        print("\n" + "="*60)
