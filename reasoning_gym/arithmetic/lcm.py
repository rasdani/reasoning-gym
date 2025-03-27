"""Least Common Multiple (LCM) task generator"""

from dataclasses import dataclass
from functools import reduce
from math import lcm
from random import Random
from typing import Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "lcm"


@dataclass
class LCMConfig:
    """Configuration for LCM task generation"""

    min_numbers: int = 2  # Minimum numbers to find LCM of
    max_numbers: int = 2  # Maximum numbers to find LCM of
    min_value: int = 1  # Minimum value for each number
    max_value: int = 100  # Maximum value for each number (kept smaller than GCD default since LCM grows fast)
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers >= 2, "min_numbers must be at least 2"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value >= 1, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class LCMDataset(ProceduralDataset):
    """Generates Least Common Multiple (LCM) tasks"""

    def __init__(self, config: LCMConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_numbers(self, rng: Random) -> tuple[list[int], int]:
        """Generate a list of random positive integers and their LCM.
        Will try up to 3 times to find numbers with LCM < product."""

        def calculate_product(nums: list[int]) -> int:
            return reduce(lambda x, y: x * y, nums)

        # Try up to 3 times to get LCM < product
        for _ in range(3):
            num_count = rng.randint(self.config.min_numbers, self.config.max_numbers)
            numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_count)]
            result = reduce(lcm, numbers)
            if result < calculate_product(numbers):
                break

        # Return the last generated numbers, whether they met the criteria or not
        return numbers, result

    def __getitem__(self, idx: int) -> dict:
        """Generate a single LCM task"""
        rng = Random(self.seed + idx)

        numbers, result = self._generate_numbers(rng)
        numbers_str = ", ".join(str(n) for n in numbers)

        return {
            "question": f"Find the Least Common Multiple (LCM) of these numbers: {numbers_str}",
            "answer": str(result),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "numbers": numbers,
                "result": result,
                "difficulty": {
                    "numbers": (self.config.min_numbers, self.config.max_numbers),
                    "value": (self.config.min_value, self.config.max_value),
                },
            },
        }


class LCMCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LCMCurriculum.__name__, LCMConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="numbers",
                levels=[2, 4, 6, 8, 10],
                description="Number of integers to find LCM of",
                lower_field_name="min_numbers",
                upper_field_name="max_numbers",
            ),
            RangeAttributeDefinition(
                name="value",
                levels=[1, 100, 500, 1000, 5000],
                description="Range of values for each integer",
                lower_field_name="min_value",
                upper_field_name="max_value",
                ensure_interval=True,
            ),
        )


register_dataset(DATASET_NAME, LCMDataset, LCMConfig, LCMCurriculum)
