"""
Chain arithmetic exercise that evaluates expressions with operator precedence.
"""

from typing import Dict, Any
import operator
import numpy as np

class ChainSumExercise:
    """Exercise generator for chain arithmetic problems."""
    def __init__(self):
        # Define operator mappings
        self.pedmas = {
            '**': (operator.pow, 3),    # (function, precedence)
            '*': (operator.mul, 2),
            '/': (operator.truediv, 2),
            '+': (operator.add, 1),
            '-': (operator.sub, 1)
        }
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a problem using the curriculum's template system.

        Returns:
            Dict containing:
                - question: str (e.g. "What is 2 + 3 * 4?")
                - answer: float (the computed result)
                - metadata: dict with parsed expression details
        """
        self.curriculum = curriculum
        max_attempts = 10

        for _ in range(max_attempts):
            try:
                template = curriculum.get_template(curriculum.rng)
                return template.eval(self, curriculum.rng)  # Pass self to use exercise's parse/evaluate methods
            except ValueError as e:
                if "Invalid operation" in str(e):
                    continue
                raise

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the template metadata into structured data.

        Args:
            metadata: Raw metadata from template evaluation
            Structure:
            {
                "expression": {
                    "term_0": {"sign": "", "value": "3"},
                    "op_0": "+",
                    "term_1": {"sign": "", "value": "4"}
                }
            }
        Returns:
            Dictionary containing:
                - values: List of numeric values
                - operators: List of operators
                - structure: Expression structure info
        """
        expr_parts = metadata["expression"]

        parsed = {
            "values": [],
            "operators": [],
            "structure": {
                "num_terms": 0,
                "notations": []
            }
        }

        # Extract values
        i = 0
        while f"term_{i}" in expr_parts:
            term = expr_parts[f"term_{i}"]
            sign = term["sign"] if term["sign"] else ""
            val = term["value"]

            # Parse the value based on its format
            try:
                num = val.lstrip('-')
                if num.startswith(('0b', '0x')):
                    base = 2 if num.startswith('0b') else 16 if num.startswith('0x') else 10
                    num_val = float(int(num[2:], base))
                    notation = f"base{base}"
                else:
                    num_val = float(val)
                    notation = "scientific" if 'e' in num.lower() else "regular"

                # Apply sign
                if sign == '-':
                    num_val = -num_val

                parsed["values"].append(num_val)
                parsed["structure"]["notations"].append(notation)
            except ValueError:
                raise ValueError(f"Failed to parse value: {val}")

            i += 1

        parsed["structure"]["num_terms"] = i

        # Extract operators
        for i in range(len(parsed["values"]) - 1):
            if f"op_{i}" in expr_parts:
                parsed["operators"].append(expr_parts[f"op_{i}"])

        return parsed

    def _evaluate_expression(self, parsed: Dict[str, Any]) -> float:
        """
        Evaluate expression respecting operator precedence.

        Args:
            parsed: Dictionary containing parsed expression data
        Returns:
            float: The computed result
        """
        values = parsed["values"]
        operators = parsed["operators"]

        if not operators:
            return values[0] if values else 0

        vals, ops = list(values), list(operators)

        def handle_edge(op, a, b):
            # Handle division first
            if op == '/':
                if np.isclose(b, 0):
                    raise ValueError("chain_sum.py: Invalid operation, division by zero")
            # Handle exponentiation edge cases
            if op == '**':
                if np.isclose(a, 0) and b < 0:
                    raise ValueError("chain_sum.py: Invalid operation, zero with negative exponent")
                if a < 0 and not isinstance(b, int) and not b.is_integer():
                    raise ValueError("chain_sum.py: Invalid operation, fractional exponent of negative base")

            # Handle potential overflows
            try:
                result = self.pedmas[op][0](a, b)
                if abs(result) > np.finfo(float).max:
                    raise OverflowError
                return result
            except OverflowError:
                raise ValueError("chain_sum.py: Invalid operation, overflow in calculation")

        for precedence in sorted({self.pedmas[op][1] for op in ops}, reverse=True):
            i = 0
            while i < len(ops):
                if self.pedmas[ops[i]][1] != precedence:
                    i += 1
                    continue
                op = ops[i]
                a, b = vals[i], vals[i + 1]
                result = handle_edge(op, a, b)
                vals[i] = result  # Replace first value with result
                del vals[i + 1]   # Remove second value
                del ops[i]        # Remove used operator

        return vals[0]