"""Curriculum definition for letter counting exercises."""

from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template
from reasoning_gym.data import read_data_file


class LetterCountingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("LetterCountingCurriculum")
        import re
        self.words = [word for word in re.findall(r"\b\w+\b", read_data_file("in_the_year_2889.txt"))
                      if word.isalnum()]

    def _init_curriculum(self) -> None:
        """Initialize the letter counting curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,   # For fixed values
            AttributeType.UBOUND,   # For ranges like span length
            AttributeType.APPEND    # For accumulating options
        }

        # Define attributes
        self._attributes = {
            "num_words": AttributeDefinition(
                levels=[5, 10, 15],  # From min_words/max_words
                default_level=0,
                description="Number of words in the text span",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure at least 1 word
            ),
            "case_sensitivity": AttributeDefinition(
                levels=[False, True],
                default_level=0,
                description="Whether letter counting is case sensitive",
                attr_type=AttributeType.STATIC
            ),
            "letter_selection": AttributeDefinition(
                levels=["common", "all", "rare"],
                default_level=0,
                description="Strategy for selecting target letter",
                attr_type=AttributeType.APPEND
            )
        }

        # Define templates with symbolic placeholders
        self._templates = [
            Template(
                template='How many times {case_sensitivity} does the letter "{letter}" appear in the text: "{text}"?',
                parts={"text": "text_span", "letter": "target_letter", "case_sensitivity": "case_sensitivity"}
            ),
            Template(
                template='Count the occurrences of "{letter}" in: "{text}" {case_sensitivity}',
                parts={"text": "text_span", "letter": "target_letter", "case_sensitivity": "case_sensitivity"}
            ),
            Template(
                template='In the text "{text}", how many times {case_sensitivity} does the letter "{letter}" appear?',
                parts={"text": "text_span", "letter": "target_letter", "case_sensitivity": "case_sensitivity"}
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Define shared variables that need to be consistent
            "shared_vars": {
                "selected_span": lambda refs: (
                    n_words := refs["num_words"](),
                    idx := refs["dataset_rng"].randint(0, len(self.words) - n_words),
                    span := self.words[idx:idx+n_words],
                    " ".join(span)
                )[-1],
                "is_case_sensitive": lambda refs: refs["case_sensitivity"](),
            },
            # Define value generators
            "generators": {
                "get_letter": lambda refs: (
                    text := refs["selected_span"](refs),
                    text := text.lower() if not refs["is_case_sensitive"](refs) else text,
                    strategy := refs["letter_selection"](),
                    letters := set(c for c in text if c.isalpha()),
                    freqs := {c: text.count(c) for c in letters},
                    sorted_letters := sorted(letters, key=lambda c: (-freqs[c] if strategy == "common" else freqs[c])),
                    refs["dataset_rng"].choice(sorted_letters if strategy == "all" else sorted_letters[:2])
                )[-1]
            },
            # Define composition templates
            "templates": {
                "text_span": lambda refs: {
                    "template": "{text}",
                    "parts": {
                        "text": lambda refs=refs: refs["selected_span"](refs)
                    }
                },
                "target_letter": lambda refs: {
                    "template": "{letter}",
                    "parts": {
                        "letter": lambda refs=refs: refs["get_letter"](refs)
                    }
                },
                "case_sensitivity": lambda refs: {
                    "template": "(case {sensitivity})",
                    "parts": {
                        "sensitivity": lambda refs=refs: "sensitive" if refs["is_case_sensitive"](refs) else "insensitive"
                    }
                }
            }
        }