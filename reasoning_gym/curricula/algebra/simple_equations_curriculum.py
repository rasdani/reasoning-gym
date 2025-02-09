"""
Curriculum definition for simple equation exercises.
"""

import string
from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template


class SimpleEquationsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("SimpleEquationsCurriculum")

    def _init_curriculum(self) -> None:
        """Initialize the simple equations curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,   # For operators
            AttributeType.UBOUND,   # For ranges like num_terms, value
            AttributeType.APPEND,   # For operators that accumulate
        }

        # Define attributes
        self._attributes = {
            "num_terms": AttributeDefinition(
                levels=[2, 3, 4],  # From min_terms/max_terms
                default_level=0,
                description="Number of terms in the equation",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure at least 1 term
            ),
            "value": AttributeDefinition(
                levels=[10, 50, 100],  # From min_value/max_value
                default_level=0,
                description="Maximum value for constants and coefficients",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure non-zero values
            ),
            "operators": AttributeDefinition(
                levels=["+", "-", "*"],  # Keep original operators
                default_level=0,
                description="Allowed operators between terms",
                attr_type=AttributeType.APPEND
            ),
            "sign": AttributeDefinition(
                levels=["", "-"],  # Remove explicit + sign
                default_level=0,
                description="Sign of the coefficient",
                attr_type=AttributeType.APPEND
            ),
            "var_name": AttributeDefinition(
                levels=[list("xyz"), list(string.ascii_lowercase), list("αβγρθφψω")],
                default_level=0,
                description="Variables to use in equations",
                attr_type=AttributeType.APPEND
            )
        }

        # Define templates with symbolic placeholders
        self._templates = [
            Template(
                template="Find the value of {variable} in the equation: {lhs} = {rhs}",
                parts={"lhs": "eq_lhs", "rhs": "eq_rhs", "variable": "variable_name"}
            ),
            Template(
                template="Solve for {variable}: {lhs} = {rhs}",
                parts={"lhs": "eq_lhs", "rhs": "eq_rhs", "variable": "variable_name"}
            ),
            Template(
                template="Determine the value of {variable} that satisfies: {lhs} = {rhs}",
                parts={"lhs": "eq_lhs", "rhs": "eq_rhs", "variable": "variable_name"}
            )
        ]

        # Define symbolic structure
        self._symbolic = {
            # Define composition templates
            "templates": {
                "variable_name": lambda refs: {
                    "template": "{var}",
                    "parts": {"var": lambda refs=refs: refs["var"](refs)}
                },
                # Use shared template for both sides
                "eq_lhs": lambda refs: refs["templates"]["equation_side"](refs, "lhs"),
                "eq_rhs": lambda refs: refs["templates"]["equation_side"](refs, "rhs"),
                # Shared equation side template
                "equation_side": lambda refs, side: (
                    n_terms := refs["num_terms"](),
                    var_side := refs["var_side"](refs),
                    var_term := refs["dataset_rng"].randrange(n_terms) if var_side == side else None,
                    {
                        "template": "{term_0}" + "".join(f" {{op_{i}}} {{term_{i+1}}}" 
                                                      for i in range(n_terms - 1)),
                        "parts": {
                            **{f"term_{i}": lambda i=i: refs["templates"]["term"](refs, i == var_term)
                               for i in range(n_terms)},
                            **{f"op_{i}": lambda refs=refs: refs["operator"](refs)()
                               for i in range(n_terms - 1)}
                        }
                    }
                )[-1],
                # Term template
                "term": lambda refs, has_var: {
                    "template": "{sign}{coeff}{variable}",
                    "parts": {
                        "sign": lambda refs=refs: refs["sign_term"](refs)(),
                        "coeff": lambda refs=refs: (
                            coeff := refs["term_value"](refs)(),
                            f"{coeff}*" if has_var and coeff != 1 else
                            f"{coeff}" if not has_var else
                            ""
                        )[-1],
                        "variable": lambda refs=refs: refs["var"](refs) if has_var else ""
                    }
                },
            },
            # Define shared variables that need to be consistent across templates
            "shared_vars": {
                "var": lambda refs: refs["var_name"](),
                "var_side": lambda refs: refs["dataset_rng"].choice(["lhs", "rhs"])
            },
            # Define value generators
            "generators": {
                "term_value": lambda refs: lambda: refs["value"](),
                "operator": lambda refs: lambda: refs["operators"](),
                "sign_term": lambda refs: lambda: refs["sign"]()
            }
        }