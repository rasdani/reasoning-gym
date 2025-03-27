"""Letter counting task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Optional

from reasoning_gym.data import read_data_file

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "letter_counting"


@dataclass
class LetterCountingConfig:
    """Configuration for letter counting task generation"""

    min_words: int = 5  # Minimum words in span
    max_words: int = 15  # Maximum words in span
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"


class LetterCountingDataset(ProceduralDataset):
    """Generates letter counting tasks from text spans"""

    def __init__(self, config: LetterCountingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and clean them to contain only alphanumeric characters
        self.words = [word for word in re.findall(r"\b\w+\b", text) if word.isalnum()]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single letter counting task"""
        rng = Random(self.seed + idx)

        # Select random span of words
        span_length = min(
            rng.randint(self.config.min_words, self.config.max_words),
            len(self.words),
        )
        start_idx = rng.randint(0, len(self.words) - span_length)
        span = self.words[start_idx : start_idx + span_length]

        # Get all unique letters from span
        letters = set("".join(span).lower())
        if not letters:
            letters = {"a"}  # Fallback if span has no letters

        # Select random letter that appears in the span
        target_letter = rng.choice(sorted(letters))

        # Count occurrences
        count = sum(word.lower().count(target_letter) for word in span)

        return {
            "question": f'How many times does the letter "{target_letter}" appear in the text: "{" ".join(span)}"?',
            "answer": str(count),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "span_length": span_length,
                "target_letter": target_letter,
                "span": span,
                "difficulty": {
                    "words": (self.config.min_words, self.config.max_words),
                },
            },
        }


class LetterCountingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LetterCountingCurriculum.__name__, LetterCountingConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="words",
                levels=[10, 50, 100, 1000],
                description="Number of words in the span",
                lower_field_name="min_words",
                upper_field_name="max_words",
                ensure_interval=True,
            ),
        )


register_dataset(DATASET_NAME, LetterCountingDataset, LetterCountingConfig, LetterCountingCurriculum)
