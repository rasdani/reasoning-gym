"""Insert into string according to a pattern

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_string_insertion.py
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Your output should be a string that has been modified according to the pattern.

Given the following string, provide the answer after inserting the characters according to the pattern: {string}
"""


DATASET_NAME = "string_insertion"


@dataclass
class StringInsertionConfig:
    """Configuration for String Insertion dataset generation"""

    min_string_length: int = 5  # Minimum string length
    max_string_length: int = 20  # Maximum string length

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 5 <= self.min_string_length, "Minimum string length should be at least 5"
        assert self.min_string_length <= self.max_string_length, "Minimum string length should be less than maximum"


class StringInsertionDataset(ProceduralDataset):
    """Generates String Insertion exercises with configurable difficulty"""

    def __init__(self, config: StringInsertionConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.vocabulary = ["A", "B", "C", "D", "E"]
        self.insertion_rules = [
            ("ABCD", "A"),
            ("BCDE", "B"),
            ("CDEA", "C"),
            ("DEAB", "D"),
            ("EABC", "E"),
        ]

    def _get_answer(self, string: str) -> str:
        """Apply insertion rules to a string"""
        output = []
        i = 0
        while i < len(string):
            inserted = False
            for pattern, char in self.insertion_rules:
                substring = string[i : i + len(pattern)]
                if substring == pattern:
                    output.append(substring + char)
                    i += len(pattern)
                    inserted = True
                    break
            if not inserted:
                output.append(string[i])
                i += 1
        return "".join(output)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        if isinstance(answer, str):
            if answer == oracle_answer:
                return 1.0
            else:
                try:
                    # check if answer is python list of characters
                    answer = "".join(eval(answer))
                    if answer == oracle_answer:
                        return 0.1
                except Exception:
                    pass
        return 0.0

    def __getitem__(self, idx: int) -> dict:
        """Generate a single String Insertion question"""
        rng = Random(self.seed + idx)

        string_length = rng.randint(self.config.min_string_length, self.config.max_string_length)
        string = "".join(rng.choice(self.vocabulary) for _ in range(string_length))

        answer = self._get_answer(string)

        return {
            "question": QUESTION_TEMPLATE.format(string=string),
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "string": string,
                "solution": answer,
                "string_length": string_length,
                "difficulty": {
                    "string_length": (self.config.min_string_length, self.config.max_string_length),
                },
            },
        }


class StringInsertionCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(StringInsertionCurriculum.__name__, StringInsertionConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="string_length",
                levels=[10, 50, 100, 1000],
                description="Length of the string",
                lower_field_name="min_string_length",
                upper_field_name="max_string_length",
                ensure_interval=True,
            ),
        )


register_dataset(DATASET_NAME, StringInsertionDataset, StringInsertionConfig, StringInsertionCurriculum)
