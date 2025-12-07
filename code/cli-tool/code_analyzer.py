"""Analyze and extract stylometric features from code."""

import re
from typing import Dict, List


class CodeAnalyzer:
    """Extract stylometric features from Python code."""

    @staticmethod
    def analyze_naming_conventions(code: str) -> Dict:
        """Analyze naming style (camelCase, snake_case, etc.)."""
        # Extract identifiers
        identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)

        camel_case_count = sum(1 for id in identifiers if re.match(r'[a-z]+([A-Z][a-z]*)+', id))
        snake_case_count = sum(1 for id in identifiers if '_' in id and id.islower())
        upper_case_count = sum(1 for id in identifiers if id.isupper())

        return {
            "camel_case_ratio": camel_case_count / len(identifiers) if identifiers else 0,
            "snake_case_ratio": snake_case_count / len(identifiers) if identifiers else 0,
            "upper_case_ratio": upper_case_count / len(identifiers) if identifiers else 0,
            "avg_identifier_length": sum(len(id) for id in identifiers) / len(identifiers) if identifiers else 0,
        }

    @staticmethod
    def analyze_indentation(code: str) -> Dict:
        """Analyze indentation style (spaces, tabs, indent width)."""
        lines = code.split('\n')
        indents = []

        for line in lines:
            if line and line[0] in (' ', '\t'):
                indent_chars = len(line) - len(line.lstrip())
                indents.append(indent_chars)

        return {
            "has_tabs": any('\t' in line for line in lines),
            "has_spaces": any(line.startswith(' ') for line in lines),
            "avg_indent_width": sum(indents) / len(indents) if indents else 0,
            "indent_consistency": 1 if len(set(indents)) <= 2 else 0,
        }

    @staticmethod
    def analyze_comments(code: str) -> Dict:
        """Analyze comment style and density."""
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        docstring_count = code.count('"""') // 2 + code.count("'''") // 2

        return {
            "comment_density": comment_lines / len(lines) if lines else 0,
            "docstring_count": docstring_count,
            "inline_comment_count": sum(1 for line in lines if '#' in line and not line.strip().startswith('#')),
        }

    @staticmethod
    def analyze_whitespace(code: str) -> Dict:
        """Analyze whitespace and formatting patterns."""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]

        return {
            "avg_line_length": sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0,
            "blank_line_ratio": (len(lines) - len(non_empty_lines)) / len(lines) if lines else 0,
            "max_line_length": max(len(line) for line in lines) if lines else 0,
        }

    @staticmethod
    def analyze_imports(code: str) -> Dict:
        """Analyze import statement style."""
        import_lines = [line for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]

        from_imports = sum(1 for line in import_lines if line.strip().startswith('from'))
        direct_imports = sum(1 for line in import_lines if line.strip().startswith('import'))

        return {
            "from_import_ratio": from_imports / len(import_lines) if import_lines else 0,
            "direct_import_ratio": direct_imports / len(import_lines) if import_lines else 0,
            "total_import_count": len(import_lines),
        }

    @staticmethod
    def analyze_functions(code: str) -> Dict:
        """Analyze function definition style."""
        functions = re.findall(r'def\s+(\w+)\s*\(', code)
        function_bodies = re.findall(r'def\s+\w+\s*\([^)]*\):\s*\n((?:.*\n?)+?)(?=\ndef|\nclass|\Z)', code, re.MULTILINE)

        return {
            "function_count": len(functions),
            "avg_function_name_length": sum(len(f) for f in functions) / len(functions) if functions else 0,
            "avg_function_body_length": sum(len(body) for body in function_bodies) / len(function_bodies) if function_bodies else 0,
        }

    def full_analysis(self, code: str) -> Dict:
        """Perform complete stylometric analysis."""
        return {
            "naming": self.analyze_naming_conventions(code),
            "indentation": self.analyze_indentation(code),
            "comments": self.analyze_comments(code),
            "whitespace": self.analyze_whitespace(code),
            "imports": self.analyze_imports(code),
            "functions": self.analyze_functions(code),
        }

    @staticmethod
    def print_analysis(analysis: Dict):
        """Pretty print stylometric analysis."""
        print("\n=== STYLOMETRIC ANALYSIS ===")
        for category, metrics in analysis.items():
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")
