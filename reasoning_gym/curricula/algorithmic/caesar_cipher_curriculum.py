"""Curriculum definition for Caesar cipher exercises."""

from typing import Dict, Any, List
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template
from reasoning_gym.data import read_data_file


class CaesarCipherCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("CaesarCipherCurriculum")
        import re
        self.text_data = re.findall(r"[aA-zZ]+", read_data_file("in_the_year_2889.txt"))

    def _init_curriculum(self) -> None:
        """Initialize the Caesar cipher curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,   # For fixed values like delimiter
            AttributeType.UBOUND,   # For ranges like words, rotation
            AttributeType.APPEND    # For accumulating options
        }

        # Define attributes
        self._attributes = {
            "num_words": AttributeDefinition(
                levels=[5, 10, 20],
                default_level=0,
                description="Number of words in the sentence",
                attr_type=AttributeType.UBOUND,
                min_value=3  # Ensure at least 3 words
            ),
            "rotation": AttributeDefinition(
                levels=[1, 3, 10, 15, 25],
                default_level=0,
                description="Caesar cipher rotation value",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure at least rotation of 1
            ),
            "text_case": AttributeDefinition(
                levels=["UPPER", "lower", "Mixed"],
                default_level=0,
                description="Text case style",
                attr_type=AttributeType.APPEND
            )
        }

        # Define templates with symbolic placeholders
        self._templates = [
            Template(
                template="Decrypt this Caesar cipher text: {cipher_text}",
                parts={"cipher_text": "cipher_text"}
            ),
            Template(
                template="What is the original text for this Caesar cipher: {cipher_text}",
                parts={"cipher_text": "cipher_text"}
            ),
            Template(
                template="This text was encrypted using a Caesar cipher with rotation {rotation}:\n{cipher_text}\nWhat was the original text?",
                parts={"cipher_text": "cipher_text", "rotation": "rotation_value"}
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Define composition templates
            "templates": {
                "cipher_text": lambda refs: {
                    "template": "{encrypted_text}",
                    "parts": {
                        "encrypted_text": lambda refs=refs: refs["encrypt"](refs),
                        "clear_text": lambda refs=refs: refs["clear_text"](refs),
                        "rotation": lambda refs=refs: refs["rot"](refs)
                    }
                },
                # Rotation value template
                "rotation_value": lambda refs: {
                    "template": "{value}",
                    "parts": {
                        "value": lambda refs=refs: refs["rot"](refs)
                    }
                }
            },
            # Define shared variables that need to be consistent
            "shared_vars": {
                "clear_text": lambda refs: (
                    case := refs["txt_case"](refs),
                    "".join(c.lower() if (case!="UPPER" and (case=="lower" or (refs["dataset_rng"].random() < 0.5))) else c.upper()
                           for c in refs["read_text"](refs))
                )[-1],
                "txt_case": lambda refs: refs["text_case"](),
                "rot": lambda refs: refs["rotation"]()
            },
            # Define value generators
            "generators": {
                "encrypt": lambda refs: (
                    rot := refs["rot"](refs),
                    case := refs["txt_case"](refs),
                    "".join(
                        chr(((ord(c.upper()) - ord("A") + rot) % 26) +
                            (ord("A") if (case!="UPPER" and (case=="lower" or (refs["dataset_rng"].random() < 0.5))) else ord("a")))
                        if c.isalpha() else c
                        for c in refs["clear_text"](refs)
                    )
                )[-1],
                "read_text": lambda refs: (
                    n_words := refs["num_words"](),
                    idx := refs["dataset_rng"].randint(0, len(self.text_data) - n_words),
                    " ".join(self.text_data[idx:idx+n_words])
                )[-1]
            }
        } 