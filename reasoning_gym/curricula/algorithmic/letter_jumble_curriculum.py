"""
Curriculum definition for letter jumble exercises.
"""

from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template
from reasoning_gym.data import read_data_file

class LetterJumbleCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("LetterJumbleCurriculum")
        import re
        self.words = [word for word in re.findall(r"\b\w+\b", read_data_file("in_the_year_2889.txt")) if word.isalpha()]

    def _init_curriculum(self) -> None:
        """Initialize the letter jumble curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,   # For boolean flags
            AttributeType.UBOUND,   # For ranges like word length, num words
            AttributeType.APPEND    # For accumulating options
        }

        # Define attributes
        self._attributes = {
            "word_length": AttributeDefinition(
                levels=[7, 12, 64],  # From min_word_len/max_word_len
                default_level=0,
                description="Maximum word length",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure at least 2 chars for scrambling
            ),
            "preserve_length": AttributeDefinition(
                levels=[4, 2],
                default_level=0,
                description="Word length to preserve",
                attr_type=AttributeType.STATIC
            ),
            "num_words": AttributeDefinition(
                levels=[3, 5, 20],  # From min_words/max_words
                default_level=0,
                description="Number of words to scramble",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure at least 1 word
            ),
            "corruption_level": AttributeDefinition(
                levels=[0.1, 0.3, 0.9],  # From min/max_corruption_level
                default_level=0,
                description="Fraction of characters to swap",
                attr_type=AttributeType.UBOUND,
                min_value=0.1
            ),
            "consecutive_words": AttributeDefinition(
                levels=[True, False],
                default_level=0,
                description="Whether to select consecutive words",
                attr_type=AttributeType.APPEND
            )
        }

        # Define templates with symbolic placeholders
        self._templates = [
            Template(
                template="Unscramble these words: \"{scrambled}\"",
                parts={"scrambled": "word_list"}
            ),
            Template(
                template="What are the original words? \"{scrambled}\"",
                parts={"scrambled": "word_list"}
            ),
            Template(
                template="Rearrange the letters to find the original words: \"{scrambled}\"",
                parts={"scrambled": "word_list"}
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Shared variables that need to be consistent across templates
            "shared_vars": {
                # Selected original words that will be scrambled
                "selected_words": lambda refs: (
                    n_words := refs["num_words"](),
                    pool := self.words,
                    refs["dataset_rng"].sample(pool, n_words) if not refs["consecutive_words"]() else
						(
                            start := refs["dataset_rng"].randint(0, max(0, len(pool)-n_words)),
                            pool[start:start + n_words]
                        )[-1]
                )[-1]
            },
            # Value generators for dynamic content
            "generators": {
                # Scramble a single word based on corruption level
                "scramble_word": lambda refs: lambda lst: (
                    [
                        (i, j, lst.__setitem__(i, lst[j]), lst.__setitem__(j, temp)) # Debugging: keep track of indices and assignments
                        for _ in range(max(0, int(len(lst) * refs["corruption_level"]())))
                        for i, j in [refs["dataset_rng"].sample(range(len(lst)), 2)]
                        for temp in [lst[i]] # Introduce temp variable for correct swap
                    ],
                    "".join(lst)
                )[-1],
                # Generate scrambled version of all selected words
                "scramble_all": lambda refs: lambda: [
                    refs["scramble_word"](refs)(list(word)) if len(word) > refs["preserve_length"]() else word
                    for word in refs["selected_words"](refs)
                ]
            },
            # Template composition
            "templates": {
                "word_list": lambda refs: {
                    "template": "{scrambled_words}",
                    "parts": {
                        "scrambled_words": lambda refs=refs: " ".join(refs["scramble_all"](refs)()),
                        "original_words": lambda refs=refs: refs["selected_words"](refs)
                    }
                }
            }
        }