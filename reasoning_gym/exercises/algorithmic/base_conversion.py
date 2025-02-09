"""Base conversion exercise that converts numbers between different bases."""

from typing import Dict, Any

class BaseConversionExercise:
    """Exercise generator for base conversion problems."""

    def __init__(self):
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a base conversion problem using the curriculum.

        Returns:
            Dict containing:
                - question: str (e.g. "Convert the binary number 1010 to hexadecimal")
                - answer: str (the converted number in target base)
                - metadata: dict with details (value, source_base, target_base, etc.)
        """
        self.curriculum = curriculum
        template = curriculum.get_template(curriculum.rng)
        return template.eval(self, curriculum.rng)

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the template metadata into structured data.

        The metadata structure from the curriculum:
        {
            "source_value": {"val": str},  # e.g. "1010" or "a5"
            "source_base": {"base": str},  # e.g. "binary" or "base-3"
            "target_base": {"base": str, "hint": str},  # e.g. "hexadecimal" or "base-8" with optional hint
        }

        Returns:
            Dictionary containing:
                - source_value: str (value to convert)
                - source_base: int (base to convert from)
                - target_base: int (base to convert to)
        """
        def parse_base_name(name: str) -> int:
            """Convert base name to numeric value."""
            name = name.lower()
            if name == "binary":
                return 2
            elif name == "octal":
                return 8
            elif name == "decimal":
                return 10
            elif name == "hexadecimal":
                return 16
            elif name.startswith("base-"):
                return int(name[5:])
            raise ValueError(f"Unknown base name: {name}")

        return {
            "source_value": metadata["source_value"]["val"],
            "source_base": parse_base_name(metadata["source_base"]["base"]),
            "target_base": parse_base_name(metadata["target_base"]["base"])
        }

    def _evaluate_expression(self, parsed: Dict[str, Any]) -> str:
        """
        Convert the number between bases.

        Args:
            parsed: Dictionary containing:
                - source_base: int (base to convert from)
                - target_base: int (base to convert to)
                - source_value: str (value to convert)
        Returns:
            String representation of the number in target base
        """
        try:
            # Convert source value to decimal, handling letter digits
            source_value = parsed["source_value"].lower()
            decimal_value = 0
            for digit in source_value:
                if digit.isdigit():
                    digit_val = int(digit)
                else:
                    digit_val = ord(digit) - ord('a') + 10
                if digit_val >= parsed["source_base"]:
                    raise ValueError(f"Digit {digit} is invalid for base {parsed['source_base']}")
                decimal_value = decimal_value * parsed["source_base"] + digit_val

            # Convert decimal to target base
            if decimal_value == 0:
                return "0"

            # Manual conversion for all bases
            digits = []
            n = decimal_value
            while n:
                digits.append(int(n % parsed["target_base"]))
                n //= parsed["target_base"]
            # Convert to string with letters for digits > 9
            result = "".join(str(d) if d < 10 else chr(ord("a") + d - 10) 
                           for d in reversed(digits))
            return result

        except ValueError as e:
            return f"Error converting number: {str(e)}"
