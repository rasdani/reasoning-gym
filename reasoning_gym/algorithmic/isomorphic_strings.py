"""Check if two strings are isomorphic.

Two strings are isomorphic if the characters in one string can be replaced to get the second string.

A popular Leetcode problem:
https://leetcode.com/problems/isomorphic-strings/description/
"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Return True if the following two strings are isomorphic, or False otherwise:
{s} {t}
"""


DATASET_NAME = "isomorphic_strings"


@dataclass
class IsomorphicStringsConfig:
    """Configuration for Isomorphic Strings dataset generation"""

    min_string_length: int = 2  # Minimum length of the strings
    max_string_length: int = 10  # Maximum length of the strings
    p_solvable: float = 0.5  # Probability that the generated question is solvable

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert (
            2 <= self.min_string_length <= self.max_string_length
        ), "min_string_length must be between 2 and max_string_length"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"


class IsomorphicStringsDataset(ProceduralDataset):
    """Generates Isomorphic Strings exercises with configurable difficulty"""

    def __init__(self, config: IsomorphicStringsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}

    def _check_isomorphic(self, s: str, t: str) -> bool:
        """Check if two strings are isomorphic"""
        if len(s) != len(t):
            return False

        mapping, inverse_mapping = {}, {}  # s -> t, t -> s
        for i in range(len(s)):
            if (s[i] in mapping and mapping[s[i]] != t[i]) or (
                t[i] in inverse_mapping and s[i] != inverse_mapping[t[i]]
            ):
                return False
            mapping[s[i]] = t[i]
            inverse_mapping[t[i]] = s[i]

        return True

    def _generate_inputs(self, rng: Random, string_length: int, solvable: bool) -> tuple[str, str]:
        """Generate the two input strings"""
        s, t = [], []
        mapping = {}

        # Generate a valid isomorphic pair first (leave one character for potential conflict)
        for _ in range(string_length - 1):
            char_s = rng.choice(list(self.letters))
            if char_s not in mapping:
                # Choose a random character that is not already mapped
                char_t = rng.choice(list(self.letters - set(mapping.values())))
                mapping[char_s] = char_t
            else:
                # Use the existing mapping
                char_t = mapping[char_s]
            s.append(char_s)
            t.append(char_t)

        if not solvable:
            # Solution should be unsolvable, create conflict
            letter = rng.choice(list(mapping.keys()))
            conflict = rng.choice(list(self.letters - {mapping[letter]}))
            insert_idx = rng.randint(0, len(s))
            s.insert(insert_idx, letter)
            t.insert(insert_idx, conflict)

        return "".join(s), "".join(t)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Isomorphic Strings question"""
        rng = Random(self.seed + idx)

        string_length = rng.randint(self.config.min_string_length, self.config.max_string_length)
        solvable = rng.random() < self.config.p_solvable
        s, t = self._generate_inputs(rng, string_length, solvable)
        answer = self._check_isomorphic(s, t)

        return {
            "question": QUESTION_TEMPLATE.format(s=s, t=t),
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "words": [s, t],
                "solution": answer,
                "solvable": solvable,
                "string_length": string_length,
                "difficulty": {
                    "string_length": (self.config.min_string_length, self.config.max_string_length),
                },
            },
        }


class IsomorphicStringsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(IsomorphicStringsCurriculum.__name__, IsomorphicStringsConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="string_length",
                levels=[10, 50, 100, 1000],
                description="Length of the strings",
                lower_field_name="min_string_length",
                upper_field_name="max_string_length",
            )
        )


register_dataset(DATASET_NAME, IsomorphicStringsDataset, IsomorphicStringsConfig, IsomorphicStringsCurriculum)
