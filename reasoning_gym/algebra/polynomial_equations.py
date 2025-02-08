"""
Polynomial equation exercise that generates equations and finds their real solutions.
"""

from typing import Dict, Any
from sympy import Symbol, expand, solve, Eq, parse_expr

class PolynomialEquationsExercise:
    """
    Generates random polynomial equations and finds their real solutions.
    The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    Then we solve "polynomial_expr = 0" using Sympy.
    """

    def __init__(self):
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a polynomial equation problem using the curriculum.

        Returns:
            Dict containing:
                - question: str (e.g. "Solve the polynomial equation: 2*x^2 - 3*x + 1 = 0")
                - answer: str (the sorted list of real solutions, e.g. "[0.5, 1.0]")
                - metadata: dict with details (polynomial_expr, symbolic_info, etc.)
        """
        self.curriculum = curriculum
        template = curriculum.get_template(curriculum.rng)
        return template.eval(self, curriculum.rng)

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the template metadata into structured data.

        The metadata structure is expected to be:
        {
            "expression": {
                "term_0": {
                    "sign": str,      # "" or "-"
                    "coeff": str,     # coefficient value with "*" if needed
                    "variable": str,  # variable name or empty
                    "exponent": str   # "**N" for degree N > 1, or empty
                },
                "term_1": {...},      # Same structure as term_0
                ...,
                "op_0": str,         # "+" or "-" between terms
                "op_1": str,         # More operators if needed
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
                - terms: List[str] of formatted term strings
                - operators: List[str] of operators between terms
                - variable: str, the variable name used
        """
        expr_parts = metadata["expression"]

        # Extract terms and operators
        terms = []
        operators = []
        i = 0
        while f"term_{i}" in expr_parts:
            term_dict = expr_parts[f"term_{i}"]
            terms.append("".join(term_dict[k] for k in ("sign", "coeff", "variable", "exponent")))
            # Get operator if it exists
            if f"op_{i}" in expr_parts:
                operators.append(expr_parts[f"op_{i}"])
            i += 1

        return {
            "terms": terms,
            "operators": operators,
            "variable": metadata["variable"]["var"]
        }

    def _evaluate_expression(self, parsed: Dict[str, Any]) -> str:
        """
        Evaluate the polynomial equation and find its real solutions.

        Args:
            parsed: Dictionary containing parsed expression data
        Returns:
            String representation of the sorted list of real solutions
        """
        # Create sympy symbol from parsed variable
        var = Symbol(parsed["variable"])

        # Build expression from parsed terms and operators
        expr = parsed["terms"][0]
        for i, op in enumerate(parsed["operators"]):
            expr = f"{expr} {op} {parsed['terms'][i + 1]}"

        try:
            sympy_expr = parse_expr(expr, local_dict={parsed["variable"]: var})
            expanded = expand(sympy_expr)
            solutions = solve(Eq(expanded, 0), var, dict=False)

            # Filter and sort real solutions
            real_solutions = []
            for sol in solutions:
                if sol.is_real:
                    real_solutions.append(float(sol.evalf()))
            real_solutions.sort()

            return str(real_solutions)
        except Exception as e:
            return f"Error evaluating expression: {expr}\nError: {str(e)}"
