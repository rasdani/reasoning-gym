"""
Simple equations exercise that generates and solves linear equations with one variable.
"""

from typing import Dict, Any
from sympy import Symbol, solve, parse_expr, Eq

class SimpleEquationsExercise:
    """Exercise generator for simple equations with one variable."""

    def __init__(self):
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a simple equation problem using the curriculum.

        Returns:
            Dict containing:
                - question: str (e.g. "Find the value of x in the equation: 3*x + 2 = 4*x - 1")
                - answer: str (the solution value, e.g. "3")
                - metadata: dict with details (equation, variable, etc.)
        """
        self.curriculum = curriculum
        template = curriculum.get_template(curriculum.rng)
        return template.eval(self, curriculum.rng)

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the template metadata into structured data.

        The metadata structure is expected to be:
        {
            "lhs": {
                "term_0": {
                    "sign": str,      # "" or "-"
                    "coeff": str,     # coefficient value with "*" if needed
                    "variable": str   # variable name or empty
                },
                "term_1": {...},      # Same structure as term_0
                ...,
                "op_0": str,         # "+" or "-" between terms
                "op_1": str,         # More operators if needed
                ...
            },
            "rhs": {                 # Same structure as lhs
                ...
            },
            "variable": {
                "var": str           # The variable name used in the equation
            }
        }

        Args:
            metadata: Raw metadata from template evaluation
        Returns:
            Dictionary containing:
                - lhs_terms: List[str] of formatted term strings for left side
                - rhs_terms: List[str] of formatted term strings for right side
                - lhs_operators: List[str] of operators between left terms
                - rhs_operators: List[str] of operators between right terms
                - variable: str, the variable name used
        """
        def parse_side(side_parts: Dict[str, Any]) -> tuple[list, list]:
            """Helper to parse one side of the equation."""
            terms = []
            operators = []
            i = 0
            while f"term_{i}" in side_parts:
                term_dict = side_parts[f"term_{i}"]
                terms.append("".join(term_dict[k] for k in ("sign", "coeff", "variable")))
                if f"op_{i}" in side_parts:
                    operators.append(side_parts[f"op_{i}"])
                i += 1
            return terms, operators

        # Parse both sides of the equation
        lhs_terms, lhs_operators = parse_side(metadata["lhs"])
        rhs_terms, rhs_operators = parse_side(metadata["rhs"])

        return {
            "lhs_terms": lhs_terms,
            "rhs_terms": rhs_terms,
            "lhs_operators": lhs_operators,
            "rhs_operators": rhs_operators,
            "variable": metadata["variable"]["var"]
        }

    def _evaluate_expression(self, parsed: Dict[str, Any]) -> str:
        """
        Evaluate the equation and find its solution.

        Args:
            parsed: Dictionary containing parsed expression data
        Returns:
            String representation of the solution
        """
        # Create sympy symbol from parsed variable
        var = Symbol(parsed["variable"])

        # Build left and right expressions
        def build_expr(terms: list, operators: list) -> str:
            """Helper to build expression string from terms and operators."""
            expr = terms[0]
            for i, op in enumerate(operators):
                expr = f"{expr} {op} {terms[i + 1]}"
            return expr

        lhs_expr = build_expr(parsed["lhs_terms"], parsed["lhs_operators"])
        rhs_expr = build_expr(parsed["rhs_terms"], parsed["rhs_operators"])

        try:
            # Parse both sides into sympy expressions
            lhs = parse_expr(lhs_expr, local_dict={parsed["variable"]: var})
            rhs = parse_expr(rhs_expr, local_dict={parsed["variable"]: var})

            # Solve the equation
            solution = solve(Eq(lhs, rhs), var)

            # Convert to float and return as string
            if solution:
                return str(float(solution[0]))
            return ""
        except Exception as e:
            return f"Error solving equation: {lhs_expr} = {rhs_expr}\nError: {str(e)}"
