{
  "unused_imports": [
    {
      "line_number": 4,
      "original_line": "from builtins import str, bytes, dict, int",
      "reason": "The 'builtins' module imports (str, bytes, dict, int) are not used anywhere in the code.",
      "action": "Remove this import"
    },
    {
      "line_number": 11,
      "original_line": "from pattern.graph import Graph, WEIGHT, CENTRALITY, DEGREE, DEFAULT",
      "reason": "Only 'Graph' is used from this import. 'WEIGHT', 'CENTRALITY', 'DEGREE', and 'DEFAULT' are unused.",
      "action": "Remove unused symbols from this import"
    }
  ]
}