import random
from dataclasses import dataclass
from typing import Any, Optional

import sympy as sp
from sympy.polys.monomials import itermonomials

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "polynomial_multiplication"


@dataclass
class PolynomialMultiplicationConfig:
    """
    Configuration for polynomial multiplication task generation.
    """

    min_terms: int = 2  # Minimum number of polynomial terms
    max_terms: int = 4  # Maximum number of polynomial terms
    min_value: int = 1  # Minimum value for coefficients
    max_value: int = 100  # Maximum value for coefficients
    min_degree: int = 0  # Minimum polynomial degree
    max_degree: int = 3  # Maximum polynomial degree
    min_polynomials: int = 2  # Minimum number of polynomials being multiplied
    max_polynomials: int = 3  # Maximum number of polynomials being multiplied
    variables: tuple[str] = ("x", "y", "z")  # Tuple of variable names, that will be chosen randomly
    allow_cross_variable_product: bool = False  # Generate tasks like "Multiply (x^2+3x-1)*(y^2-5)"
    allow_multivariate_polynomials: bool = False  # Generate multivariate tasks like "Multiply (2x^2 + 3y)*(5x^2+3x-1)"
    operators: tuple[str, ...] = (
        "+",
        "-",
    )  # Allowed operators between terms, Avoid adding '*' or '/' because they will affect the degree
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_terms > 0, "min_terms must be positive."
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms."

        assert self.min_value > 0, "min_value must be positive."
        assert self.max_value >= self.min_value, "max_value must be >= min_value."

        assert self.min_degree >= 0, "min_degree must be >= 0."
        assert self.max_degree >= self.min_degree, "max_degree must be >= min_degree."

        assert self.min_polynomials >= 2, "min_polynomials must be >= 2."
        assert self.max_polynomials >= self.min_polynomials, "max_polynomials must be >= min_polynomials."

        assert len(self.variables) > 0, "The variable tuple is empty."
        assert not (
            self.allow_multivariate_polynomials and not self.allow_cross_variable_product
        ), "Multivariate polynomials require cross product."

        allowed_ops = {"+", "-"}
        assert len(self.operators) > 0, "operators tuple cannot be empty."
        assert all(op in allowed_ops for op in self.operators), "Invalid operator found. Must be a subset of {+, -}."


class PolynomialMultiplicationDataset(ProceduralDataset):
    """
    Generates [min_polynomials, max_polynomials] random polynomials of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we find "F = P_0 * ... * P_1" using Sympy.
    """

    def __init__(self, config: PolynomialMultiplicationConfig):
        self._prompt_templates = [
            "Simplify this expression: {polynomial_expr}",
            "Calculate the following: {polynomial_expr}",
        ]
        self.added_instruction = """
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.
"""
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single polynomial multiplication item.

        Returns:
            A dict with:
                - question: str (e.g. "Multiply polynomials: (8x^3 + x + 2)*(x - 3)")
                - answer: str (Product, e.g. "8x^4 - 24x^3 + x^2 - x - 6")
                - metadata: dict with details (polynomial_expr, result, variables)
        """

        rng = random.Random(self.seed + idx)

        """
        Three Monomial States:
            - allow_multivariate_polynomials == 1: list of multivariate monomials (e.g "xy" --> [x, y, xy, x**2, y**2])
            - allow_cross_variable_product   == 1: None. Will generate a unique list of single variable monomials for each term
            - allow_cross_variable_product   == 0: A shared list of monomials for each term (e.g "x" --> [x, x**2, 1])
        """
        monomials = self._get_monomials(rng) if self.config.allow_cross_variable_product else None
        monomials = None if self.config.allow_cross_variable_product else self._get_monomials(rng)

        number_polynomials = rng.randint(self.config.min_polynomials, self.config.max_polynomials)

        polynomial_terms = [self._generate_polynomial(rng, monomials) for _ in range(number_polynomials)]
        polynomial_expr = sp.prod(polynomial_terms)
        product = sp.expand(polynomial_expr)
        question = rng.choice(self._prompt_templates).format(polynomial_expr=polynomial_expr) + self.added_instruction

        return {
            "question": question,
            "answer": str(product),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "polynomial_expr": str(polynomial_expr),
                "variables": list(product.free_symbols),
                "difficulty": {
                    "min_terms": self.config.min_terms,
                    "max_terms": self.config.max_terms,
                    "min_value": self.config.min_value,
                    "max_value": self.config.max_value,
                    "min_degree": self.config.min_degree,
                    "max_degree": self.config.max_degree,
                    "min_polynomials": self.config.min_polynomials,
                    "max_polynomials": self.config.max_polynomials,
                },
            },
        }

    def _get_monomials(self, rng: random.Random) -> str:
        """Get a list of monomials"""
        if self.config.allow_multivariate_polynomials:
            sym = sp.symbols(self.config.variables)
        else:
            sym = [sp.symbols(rng.choice(self.config.variables))]
        monomials = list(itermonomials(sym, self.config.max_degree, self.config.min_degree))
        return monomials

    def _generate_polynomial(self, rng: random.Random, monomials: Optional[list]):
        """Generates a random polynomial, returns expression."""
        # Choose the number of terms and their respective degrees
        monomials = monomials if monomials else self._get_monomials(rng)
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)

        polynomial_expr = 0
        for _ in range(num_terms):
            # Pick a nonzero random coefficient between min_value and max_value.
            coeff = rng.randint(self.config.min_value, self.config.max_value)

            # Pick a random monomial
            var = rng.choice(monomials)

            # If '-' in operators, we can randomly flip the sign
            if "-" in self.config.operators and rng.random() < 0.5:
                coeff = -coeff

            polynomial_expr += coeff * var

        return polynomial_expr

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        if answer is not None:
            try:
                predicted_poly = sp.parse_expr(answer)
                target_poly = sp.parse_expr(entry["answer"])

                # Check if the difference simplifies to zero (i.e. they are equivalent).
                if predicted_poly == target_poly:
                    reward = 1.0
            except Exception:
                reward = 0.0
        return reward


class PolynomialMultiplicationCurriculum(BaseCurriculum):
    """Curriculum for complex number arithmetic problems."""

    def __init__(self):
        super().__init__(PolynomialMultiplicationCurriculum.__name__, PolynomialMultiplicationConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="min_terms",
                field_name="min_terms",
                levels=[2, 4, 6, 8],
                description="Minimum number of terms in polynomial expressions",
            ),
            ScalarAttributeDefinition(
                name="max_terms",
                field_name="max_terms",
                levels=[4, 8, 12, 16],
                description="Maximum number of terms in polynomial expressions",
            ),
            ScalarAttributeDefinition(
                name="min_value",
                field_name="min_value",
                levels=[1, 10, 100, 1000],
                description="Minimum value for coefficients",
            ),
            ScalarAttributeDefinition(
                name="max_value",
                field_name="max_value",
                levels=[100, 10000, 1000000, 100000000],
                description="Maximum value for coefficients",
            ),
            ScalarAttributeDefinition(
                name="min_degree",
                field_name="min_degree",
                levels=[0, 1, 2, 3],
                description="Minimum polynomial degree",
            ),
            ScalarAttributeDefinition(
                name="max_degree",
                field_name="max_degree",
                levels=[2, 4, 8, 10],
                description="Maximum polynomial degree",
            ),
            ScalarAttributeDefinition(
                name="min_polynomials",
                field_name="min_polynomials",
                levels=[2, 3, 4, 5],
                description="Minimum number of polynomials being multiplied",
            ),
            ScalarAttributeDefinition(
                name="max_polynomials",
                field_name="max_polynomials",
                levels=[4, 6, 8, 10],
                description="Maximum number of polynomials being multiplied",
            ),
        )


register_dataset(
    DATASET_NAME,
    PolynomialMultiplicationDataset,
    PolynomialMultiplicationConfig,
    PolynomialMultiplicationCurriculum,
)
