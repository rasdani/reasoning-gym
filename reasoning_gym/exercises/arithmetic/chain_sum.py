from dataclasses import dataclass
from typing import Dict, Any
import operator
import numpy as np
from reasoning_gym.core.base_curriculum import BaseCurriculum

@dataclass
class ChainSumDataset:
    """Dataset generator for chain arithmetic problems."""
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

    def generate(self, curriculum: BaseCurriculum) -> Dict[str, Any]:
        """Generate a problem using the curriculum's template system"""
        self.curriculum = curriculum
        max_attempts = 10

        for _ in range(max_attempts):
            try:
                template = curriculum.get_template(curriculum.rng)
                return template.eval(self, curriculum.rng)
            except ValueError as e:
                if "Invalid operation" in str(e):
                    continue
                raise

    def _parse_expression(self, executed_parts: Dict[str, str]) -> tuple[list, list]:
        """Extract values and operators from executed parts"""
        values = []
        operators = []

        i = 0
        while f"term_{i}" in executed_parts:
            val = executed_parts[f"term_{i}"].lstrip('+')
            try:
                num = val.lstrip('-')
                if num.startswith(('0b', '0x')):
                    sign = -1 if val.startswith('-') else 1
                    base = 2 if num.startswith('0b') else 16 if num.startswith('0x') else 10
                    values.append(sign * float(int(num[2:], base)))
                else:
                    values.append(float(val))
            except ValueError:
                values.append(val)
            i += 1

        # Extract operators
        for i in range(len(values) - 1):
            if f"op_{i}" in executed_parts:
                operators.append(executed_parts[f"op_{i}"])

        return values, operators

    def _evaluate_expression(self, values: list, operators: list) -> float:
        """Evaluate expression respecting operator precedence"""
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