"""
Curriculum definition for polynomial equation exercises.
"""

import string
from typing import Dict, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType
from reasoning_gym.core.template import Template


class PolynomialEquationsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("PolynomialEquationsCurriculum")

    def _init_curriculum(self) -> None:
        """Initialize the polynomial equations curriculum configuration"""
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,   # For operators
            AttributeType.UBOUND,   # For ranges like num_terms, degree, value
            AttributeType.APPEND,    # For operators that accumulate
            AttributeType.APPEND_LIST    # For variables that accumulate
        }

        # Define attributes
        self._attributes = {
            "num_terms": AttributeDefinition(
                levels=[2, 3, 4],  # From min_terms/max_terms
                default_level=0,
                description="Number of polynomial terms",
                attr_type=AttributeType.UBOUND,
                min_value=2  # Ensure at least 2 terms
            ),
            "coefficient_value": AttributeDefinition(
                levels=[10, 50, 100],  # From min_value/max_value
                default_level=0,
                description="Maximum value for coefficients",
                attr_type=AttributeType.UBOUND,
                min_value=1  # Ensure non-zero coefficients
            ),
            "max_degree": AttributeDefinition(
                levels=[1, 2, 3],  # From min_degree/max_degree
                default_level=0,
                description="Maximum polynomial degree",
                attr_type=AttributeType.UBOUND
            ),
            "operators": AttributeDefinition(
                levels=["+", "-"],  # Keep original operators
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
                levels=[list("xyz"), list(string.ascii_lowercase + string.ascii_uppercase), list("αβγρθφψω")],
                default_level=0,
                description="Variables to use in polynomials",
                attr_type=AttributeType.APPEND_LIST
            )
        }

        # Define templates with symbolic placeholders
        self._templates = [
            Template(
                template="Find the real value(s) of {variable} in the equation: {expression} = 0",
                parts={"expression": "polynomial_expression", "variable": "variable_name"}
            ),
            Template(
                template="Solve for real {variable}: {expression} = 0",
                parts={"expression": "polynomial_expression", "variable": "variable_name"}
            ),
            Template(
                template="Determine the real value(s) of {variable} that satisfies: {expression} = 0",
                parts={"expression": "polynomial_expression", "variable": "variable_name"}
            ),
            Template(
                template="Solve the polynomial equation for real {variable}:\n{expression} = 0",
                parts={"expression": "polynomial_expression", "variable": "variable_name"}
            )
        ]

		# TODO: must always be at least one var
        # Define symbolic structure
        self._symbolic = {
            # Define composition templates
            "templates": {
                # Variable name template
                "variable_name": lambda refs: {
                    "template": "{var}",
                    "parts": {
                        "var": lambda refs=refs: refs["var"](refs)
                    }
                },
                # Expression structure
                "polynomial_expression": lambda refs: (
                    n_terms := refs["num_terms"](),
                    {
                        "template": "{term_0}" + "".join(f" {{op_{i}}} {{term_{i+1}}}" 
                                                        for i in range(n_terms - 1)),
                        "parts": {
                            **{f"term_{i}": "polynomial_term" for i in range(n_terms)},
                            **{f"op_{i}": lambda refs=refs: refs["operator"](refs)()
                               for i in range(n_terms - 1)}
                        }
                    }
                )[-1],
                # Term structure
                "polynomial_term": lambda refs: (
                    coeff := refs["coefficient"](refs)(),
                    deg := refs["degree"](refs)(),
                    var := refs["var"](refs),
                    {
                        "template": "{sign}{coeff}{variable}{exponent}",
                        "parts": {
                            "sign": lambda refs=refs: refs["sign_term"](refs)(),
                            "coeff": lambda: (
                                f"{coeff}*" if (deg > 0 and coeff != 1) else
                                f"{coeff}" if deg == 0 else
                                ""  # No coefficient if 1 and has variable
                            ),
                            "variable": lambda: (
                                "" if deg == 0 else
                                f"{var}"
                            ),
                            "exponent": lambda: (
                                "" if deg <= 1 else
                                f"**{deg}"
                            )
                        }
                    }
                )[-1]
            },
            # Define shared variables that need to be consistent across templates
            "shared_vars": {
                "var": lambda refs: refs["var_name"]()
            },
            # Define value generators
            "generators": {
                "coefficient": lambda refs: lambda: refs["coefficient_value"](),
                "degree": lambda refs: lambda: refs["max_degree"](),
                "operator": lambda refs: lambda: refs["operators"](),
                "sign_term": lambda refs: lambda: refs["sign"]()
            }
        }