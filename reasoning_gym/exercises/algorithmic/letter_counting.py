"""Letter counting exercise that generates tasks to count letter occurrences in text."""

from typing import Dict, Any

class LetterCountingExercise:
    """Exercise generator for letter counting tasks."""

    def __init__(self):
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a letter counting problem using the curriculum.

        Returns:
            Dict containing:
                - question: str (e.g. "How many times does 'a' appear in 'banana'?")
                - answer: str (the count as a string)
                - metadata: dict with details (text, target_letter, etc.)
        """
        self.curriculum = curriculum
        template = curriculum.get_template(curriculum.rng)
        return template.eval(self, curriculum.rng)

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the template metadata into structured data.

        The metadata structure from the template system:
        {
            "text": {"text": str},  # The text span to analyze
            "letter": {"letter": str},  # The letter to count
            "case_sensitivity": {"sensitivity": str}  # "sensitive" or "insensitive"
        }

        Returns:
            Dictionary containing:
                - text: str (the text to analyze)
                - target_letter: str (the letter to count)
                - case_sensitive: bool (whether to count case sensitively)
        """
        return {
            "text": metadata["text"]["text"],
            "target_letter": metadata["letter"]["letter"],
            "case_sensitive": metadata["case_sensitivity"]["sensitivity"] == "sensitive"
        }

    def _evaluate_expression(self, parsed: Dict[str, Any]) -> str:
        """
        Count occurrences of the target letter in the text.

        Args:
            parsed: Dictionary containing:
                - text: str (the text to analyze)
                - target_letter: str (the letter to count)
                - case_sensitive: bool (whether to count case sensitively)
        Returns:
            String representation of the count
        """
        if parsed["case_sensitive"]:
            return str(parsed["text"].count(parsed["target_letter"]))
        else:
            return str(parsed["text"].lower().count(parsed["target_letter"].lower()))
