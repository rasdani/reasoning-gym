"""
Curriculum definition for the ChainSum exercise.
"""

from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template

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

        # Define templates using the new system
        self._templates = [
            Template(
                template="What is {expression} ?",
                parts={"expression": "symbolic_expression"}
            ),
            Template(
                template="Calculate the following: {expression}",
                parts={"expression": "symbolic_expression"}
            ),
            Template(
                template="Solve {expression}",
                parts={"expression": "symbolic_expression"}
            ),
            Template(
                template="{expression} = ?",
                parts={"expression": "symbolic_expression"}
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Define composition templates
            "templates": {
                # Expression structure
                "symbolic_expression": lambda refs: (
                    n_terms := refs["num_terms"](),
                    {
                        "template": "{term_0}" + "".join(f" {{op_{i}}} {{term_{i+1}}}" 
                                                        for i in range(n_terms - 1)),
                        "parts": {
                            **{f"term_{i}": "term" for i in range(n_terms)},
                            **{f"op_{i}": lambda refs=refs: refs["operator"](refs)()
                               for i in range(n_terms - 1)}
                        }
                    }
                )[-1],

                # Term structure
                "term": lambda refs: {
                    "template": "{sign}{value}",
                    "parts": {
                        "sign": lambda refs=refs: refs["sign_term"](refs)(),
                        "value": lambda refs=refs: refs["format_number"](refs)(refs["number"](refs)())
                    }
                }
            },
            # Define value generators
            "generators": {
                # Generate a number based on current settings
                "number": lambda refs: (
                    lambda: (
                        max_val := (10 ** refs["num_digits"]()) - 1,
                        base_num := refs["dataset_rng"].randint(0, max_val),
                        base_num / (10 ** refs["num_decimals"]())
                            if refs["num_decimals"]() > 0 and refs["notation"]() in ["regular", "scientific"]
                            else base_num
                    )[-1]
                ),
                # Generate an operator from available options
                "operator": lambda refs: lambda: refs["operators"](),
                # Generate a sign based on current settings
                "sign_term": lambda refs: lambda: refs["sign"](),
                # Format a number according to notation
                "format_number": lambda refs: lambda value: {
                    "regular": str(value),
                    "scientific": f"{float(value):e}",
                    "base2": f"0b{int(value):b}",
                    "base16": f"0x{int(value):X}"
                }[refs["notation"]()]
            },
        }