"""
Curriculum definition for the ChainSum exercise.
"""

from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template, Placeholder

# TODO: Brackets
class ChainSumCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("ChainSumCurriculum")

    def _init_curriculum(self) -> None:
        """Initialize the ChainSum curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,  # For base numbers
            AttributeType.UBOUND,  # For ranges like digits and terms
            AttributeType.APPEND   # For operators and notations
        }

        # Define attributes
        self._attributes = {
            "num_digits": AttributeDefinition(
                levels=[2, 4, 10],
                default_level=0,  # Start with 1-digit numbers
                description="Number of digits in each operand",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure numbers are at least 1 digit
                # distribution_type
            ),
            "num_decimals": AttributeDefinition(
                levels=[0, 1, 2, 10],
                default_level=0,  # Start with integers
                description="Number of decimal places in operands",
                attr_type=AttributeType.UBOUND
            ),
            "operators": AttributeDefinition(
                levels=['+', '-', '*', '/', '**'], # //, %, ^, &, |
                default_level=0,  # Start with basic operators
                description="Set of operators that can be used",
                attr_type=AttributeType.APPEND
            ),
            "num_terms": AttributeDefinition(
                levels=[2, 3, 4, 5],
                default_level=0,  # Start with 2 terms
                description="Maximum number of terms in the expression",
                attr_type=AttributeType.UBOUND,
                min_value=2  # Ensure at least 2 terms
            ),
            "sign": AttributeDefinition(
                levels=['', '+', '-'],
                default_level=0,  # Start without negatives
                description="Whether negative numbers are allowed",
                attr_type=AttributeType.APPEND
            ),
            "notation": AttributeDefinition(
                levels=["regular", "scientific", "base2", "base16"],
                default_level=0,
                description="The notation to use for the expression",
                attr_type=AttributeType.APPEND
            )
        }

        # Define templates with symbolic placeholders
        expression = Placeholder("expression", "symbolic_expression")
        self._templates = [
            Template(
                question="What is {expression} ?",
                placeholders={"expression": expression}
            ),
            Template(
                question="Calculate the following: {expression}",
                placeholders={"expression": expression}
            ),
            Template(
                question="Solve {expression}",
                placeholders={"expression": expression}
            ),
            Template(
                question="{expression} = ?",
                placeholders={"expression": expression}
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Define composition templates
            "templates": {
                # Expression structure - this key matches the generator name in Placeholder
                "symbolic_expression": lambda refs: (
                    n_terms := refs["num_terms"](),
                    {
                        "template": "{term_0}" + "".join(f" {{op_{i}}} {{term_{i+1}}}" 
                                                        for i in range(n_terms - 1)),
                        "parts": {
                            **{f"term_{i}": "term" for i in range(n_terms)},
                            **{f"op_{i}": lambda refs=refs: refs["operators"]()
                               for i in range(n_terms - 1)}
                        }
                    }
                )[-1],

                # Term structure
                "term": lambda refs: {
                    "template": "{sign}{value}",
                    "parts": {
                        "sign": lambda refs=refs: refs["sign"](),
                        "value": "notation"
                    }
                },

                # Notation structure
                "notation": lambda refs: {
                    "template": {
                        "regular": str(refs["number"](refs)()),
                        "scientific": f"{float(refs['number'](refs)()):e}",
                        "base2": f"0b{int(refs['number'](refs)()):b}",
                        "base16": f"0x{int(refs['number'](refs)()):X}"
                    }[refs["notation"]()],
                    "parts": {}
                }
            },

            # Define value generators
            "generators": {
                "number": lambda refs: (
                    lambda: (
                        max_val := (10 ** refs["num_digits"]()) - 1,
                        base_num := refs["dataset_rng"].randint(0, max_val),
                        base_num / (10 ** refs["num_decimals"]())
                            if refs["num_decimals"]() > 0 and refs["notation"]() in ["regular", "scientific"]
                            else base_num
                    )[-1]
                )
            }
        }