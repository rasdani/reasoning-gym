"""Given a string, return all possible partitions of the string such that each substring is a palindrome.

A popular Leetcode problem:
https://leetcode.com/problems/palindrome-partitioning/description/
"""

import json
import string
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Your output should be a list of lists, where each list represents a palindrome partition, e.g. [["a","a","b"],["aa","b"]].

Partition the following string into palindromes: {string}
"""

DATASET_NAME = "palindrome_partitioning"


@dataclass
class PalindromePartitioningConfig:
    """Configuration for Palindrome Partitioning dataset generation"""

    min_string_len: int = 5
    max_string_len: int = 15
    min_substring_palindrome_len: int = 1
    max_substring_palindrome_len: int = 5

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_string_len, "Minimum string length must be at least 1"
        assert self.min_string_len <= self.max_string_len, "Minimum string length must be less than or equal to maximum"
        assert 1 <= self.min_substring_palindrome_len, "Minimum substring palindrome length must be at least 1"
        assert (
            self.min_substring_palindrome_len <= self.max_substring_palindrome_len
        ), "Minimum substring palindrome length must be less than or equal to maximum"
        assert (
            self.max_substring_palindrome_len <= self.max_string_len
        ), "Maximum substring palindrome length must be less than or equal to maximum string length"


class PalindromePartitioningDataset(ProceduralDataset):
    """Generates Palindrome Partitioning exercises with configurable difficulty"""

    def __init__(self, config: PalindromePartitioningConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _sort_list(self, lst: list[list[str]]) -> list[list[str]]:
        """Sort the list of palindrome partitions"""
        return sorted(lst, key=lambda x: x[0] if x else "")

    def to_set_of_tuples(self, list_of_lists: list[list[str]]) -> set[tuple[str]]:
        """Convert a list of lists to a set of tuples"""
        return {tuple(lst) for lst in list_of_lists}

    def _palindrome_partitioning(self, string: str) -> list[list[str]]:
        """Return all possible palindrome partitions of a string"""
        if not string:
            return []
        dp = {}

        def is_palindrome(i, j) -> bool:
            if i >= j:
                return True
            if (i, j) in dp:
                return dp[(i, j)]
            dp[(i, j)] = string[i] == string[j] and is_palindrome(i + 1, j - 1)
            return dp[(i, j)]

        res, temp = [], []

        def _partition(idx) -> None:
            if idx >= len(string):
                res.append(temp[:])
            for i in range(idx, len(string)):
                if is_palindrome(idx, i):
                    temp.append(string[idx : i + 1])
                    _partition(i + 1)
                    temp.pop()

        _partition(0)
        return self._sort_list(res)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a single Palindrome Partitioning question"""
        if answer is not None:
            try:
                answer = self.to_set_of_tuples(json.loads(answer))
                oracle = self.to_set_of_tuples(entry["metadata"]["solution"])
                if answer == oracle:
                    return 1.0
            except Exception:
                pass
        return 0.0

    def _generate_palindrome_letters(self, rng: Random, length: int) -> list[str]:
        """Generate a set of letters that can form a palindrome."""
        half_length = length // 2
        letters = rng.choices(string.ascii_lowercase, k=half_length)
        if length % 2 == 1:
            middle_letter = rng.choice(string.ascii_lowercase)
            return letters + [middle_letter] + letters[::-1]
        return letters + letters[::-1]

    def _get_string(self, rng: Random, string_len: int) -> str:
        """Generate a random string"""
        output = ""
        while len(output) < string_len:
            palindrome_len = min(
                string_len - len(output),
                rng.randint(self.config.min_substring_palindrome_len, self.config.max_substring_palindrome_len),
            )
            substring = "".join(self._generate_palindrome_letters(rng, palindrome_len))
            output += substring
        return output

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Palindrome Partitioning question"""
        rng = Random(self.seed + idx)

        string_len = rng.randint(self.config.min_string_len, self.config.max_string_len)
        string = self._get_string(rng, string_len)
        answer = self._palindrome_partitioning(string)
        answer_str = json.dumps(answer)

        return {
            "question": QUESTION_TEMPLATE.format(string=string),
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "string": string,
                "solution": answer,
                "string_len": string_len,
                "difficulty": {
                    "string_len": (self.config.min_string_len, self.config.max_string_len),
                    "substring_palindrome_len": (
                        self.config.min_substring_palindrome_len,
                        self.config.max_substring_palindrome_len,
                    ),
                },
            },
        }


class PalindromePartitioningCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PalindromePartitioningCurriculum.__name__, PalindromePartitioningConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="string_len",
                levels=[10, 100, 500, 1000],
                description="Length of the string",
                lower_field_name="min_string_len",
                upper_field_name="max_string_len",
            ),
            RangeAttributeDefinition(
                name="substring_palindrome_len",
                levels=[5, 10, 50, 100],
                description="Length of the substring palindrome",
                lower_field_name="min_substring_palindrome_len",
                upper_field_name="max_substring_palindrome_len",
            ),
        )


register_dataset(
    DATASET_NAME,
    PalindromePartitioningDataset,
    PalindromePartitioningConfig,
    PalindromePartitioningCurriculum,
)
