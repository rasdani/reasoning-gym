"""
Curriculum definition for base conversion exercises.
"""

from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template


class BaseConversionCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("BaseConversionCurriculum")

    def _init_curriculum(self) -> None:
        """Initialize the base conversion curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,   # For base names
            AttributeType.UBOUND,   # For ranges like value, base
            AttributeType.APPEND    # For accumulating options
        }

        # Define attributes
        self._attributes = {
            "value": AttributeDefinition(
                levels=[100, 1000, 10000],
                default_level=0,
                description="Maximum decimal value to convert",
                attr_type=AttributeType.UBOUND,
                min_value=1
            ),
            "base_range": AttributeDefinition(
                levels=[16, 26, 36],
                default_level=0,
                description="Maximum base value (2 is minimum)",
                attr_type=AttributeType.UBOUND,
                min_value=2  # Ensure at least binary
            ),
            "base_names": AttributeDefinition(
                levels=[{"2": "binary", "16": "hexadecimal"},
                        {"8": "octal", "10": "decimal"}],
                default_level=0,
                description="Special names for bases",
                attr_type=AttributeType.APPEND
            ),
            "hint": AttributeDefinition(
                levels=[True, False],
                default_level=0,
                description="Whether to include a hint",
                attr_type=AttributeType.STATIC
            )
        }

        # Define templates with symbolic placeholders
        self._templates = [
            Template(
                template="Convert {source_value} from {source_base} to {target_base}",
                parts={
                    "source_value": "value",
                    "source_base": "base_src",
                    "target_base": "base_trg"
                }
            ),
            Template(
                template="What is {source_value} ({source_base}) in {target_base}",
                parts={
                    "source_value": "value",
                    "source_base": "base_src",
                    "target_base": "base_trg"
                }
            ),
            Template(
                template="Express the {source_base} number {source_value} in {target_base}",
                parts={
                    "source_value": "value",
                    "source_base": "base_src",
                    "target_base": "base_trg"
                }
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Define shared variables that need to be consistent
            "shared_vars": {
                "src_base": lambda refs: refs["base_range"]()
            },
            "generators": {
                "trg_base": lambda refs: (
                    base := refs["base_range"](),
                    base if base != refs["src_base"](refs) else refs["trg_base"](refs)
                )[-1],
                "format_base": lambda refs: lambda base: (
                    names := refs["base_names"](),
                    names.get(str(base), f"base-{base}") if refs["dataset_rng"].random() < 0.5 else f"base-{base}"
                )[-1],
                # Convert decimal to any base
                "convert_decimal": lambda refs: lambda base: (
                    n := refs["value"](),
                    digits := [],
                    [digits.append(int(n % base)) or (n := n // base) for _ in range(32) if n > 0],
                    "".join(str(d) if d < 10 else chr(ord("a") + d - 10) for d in reversed(digits) or [0])
                )[-1],
                "format_hint": lambda refs: lambda base: " (use lowercase letters a-z for digits above 9)" if base > 10 and refs["hint"]() else ""
            },
            # Define composition templates
            "templates": {
                "value": lambda refs: {
                    "template": "{val}",
                    "parts": {
                        "val": lambda refs=refs: refs["convert_decimal"](refs)(refs["src_base"](refs))
                    }
                },
                "base_src": lambda refs: (
                    base := refs["src_base"](refs),
                    {
                        "template": "{base}",
                        "parts": {
                            "base": lambda refs=refs: refs["format_base"](refs)(base)
                        }
                    }
                )[-1],
                "base_trg": lambda refs: (
                    base := refs["trg_base"](refs),
                    {
                        "template": "{base}{hint}",
                        "parts": {
                            "base": lambda refs=refs: refs["format_base"](refs)(base),
                            "hint": lambda refs=refs: refs["format_hint"](refs)(base)
                        }
                    }
                )[-1]
            }
        }