import random
from dataclasses import dataclass
from typing import Any, Optional

from reasoning_gym import utils

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "products"


@dataclass
class ProductsConfig:
    """Configuration for products task generation"""

    min_terms: int = 2
    max_terms: int = 2
    min_digits: int = 1
    max_digits: int = 5
    allow_negation: bool = False
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_digits > 0, "min_digits must be positive"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"


class ProductsDataset(ProceduralDataset):
    """Generates multiplication tasks with configurable number of terms"""

    def __init__(self, config: ProductsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single multiplication task

        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the formatted multiplication expression
                - answer: str, the ground truth result
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        rng = random.Random(self.seed + idx)

        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        num_digits = rng.randint(self.config.min_digits, self.config.max_digits)

        # Calculate value ranges based on number of digits
        if self.config.allow_negation:
            min_value = -1 * 10 ** (num_digits) + 1
        else:
            min_value = 0 if num_digits == 1 else 10 ** (num_digits - 1)  # Special case for 1 digit
        max_value = (10**num_digits) - 1  # e.g., 999 for 3 digits

        expression, result = self._generate_task(rng, num_terms, min_value, max_value)

        return {
            "question": f"Solve the following multiplication: {expression}. Give only the result as your final answer.",
            "answer": str(result),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "expression": expression,
                "num_terms": num_terms,
                "num_digits": num_digits,
                "difficulty": {
                    "num_terms": (self.config.min_terms, self.config.max_terms),
                    "num_digits": (self.config.min_digits, self.config.max_digits),
                },
            },
        }

    def _generate_task(self, rng: random.Random, num_terms: int, min_value: int, max_value: int) -> tuple[str, int]:
        """Generate a multiplication task

        Args:
            rng: Random number generator
            num_terms: Number of terms in the expression
            min_value: Minimum value for generated numbers
            max_value: Maximum value for generated numbers

        Returns:
            Tuple of (expression string, result integer)
        """
        # Generate random numbers within the specified range
        constants = [rng.randint(min_value, max_value) for _ in range(num_terms)]

        # Build expression and compute result
        expression_parts = []
        result = constants[0]

        expression_parts.append(str(constants[0]))
        for i in range(1, len(constants)):
            expression_parts.append("*")
            expression_parts.append(str(constants[i]))
            result *= constants[i]

        expression = " ".join(expression_parts)
        return expression, result

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        # tolerate sign, leading zeros and trailing decimals, strip commas "+01,000.00" == "1000"
        return utils.compute_decimal_reward(answer, oracle_answer=entry["answer"], strip_commas=True)


class ProductsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ProductsCurriculum.__name__, ProductsConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_terms",
                levels=list(range(2, 13)),
                default_level=0,  # Start with 2 terms
                description="Maximum number of terms in the expression",
                lower_field_name="min_terms",
                upper_field_name="max_terms",
            ),
            RangeAttributeDefinition(
                name="num_digits",
                levels=list(range(1, 11)),
                default_level=0,  # Start with 1-digit numbers
                description="Number of digits in each operand",
                lower_field_name="min_digits",
                upper_field_name="max_digits",
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, ProductsDataset, ProductsConfig, ProductsCurriculum)
