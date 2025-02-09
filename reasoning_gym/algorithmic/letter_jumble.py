"""Exercise definition for letter jumble exercises."""

from typing import Dict, Any
from reasoning_gym.core.template import Template

class LetterJumbleExercise:
    """Exercise generator for word jumbling tasks."""

    def __init__(self):
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a word jumbling problem using the curriculum.

        Returns:
            Dict containing:
                - question: str (e.g. "Unscramble these words: OLHEL DLWOR")
                - answer: str (the original words)
                - metadata: dict with details (scrambled_words, original_words, etc.)
        """
        self.curriculum = curriculum
        template = curriculum.get_template(curriculum.rng)
        return template.eval(self, curriculum.rng)

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the expression from the metadata.

        The metadata structure from the template system:
        {
            "scrambled": {
                "scrambled_words": str,  # Space-separated scrambled words
                "original_words": List[str]  # List of original words
            }
        }

        Args:
            metadata: The metadata containing the expression information.

        Returns:
            A dictionary containing:
                - scrambled_words: List[str] of scrambled words
                - original_words: List[str] of original words
        """
        # Extract the scrambled and original words from metadata
        template_data = metadata["scrambled"]
        scrambled_words = template_data["scrambled_words"].split()
        original_words = template_data["original_words"]

        return {
            "scrambled_words": scrambled_words,
            "original_words": original_words
        }

    def _evaluate_expression(self, parsed_data: Dict[str, Any]) -> str:
        """Evaluate the expression using the parsed data.

        Args:
            parsed_data: Dictionary containing:
                - scrambled_words: List[str] of scrambled words
                - original_words: List[str] of original words

        Returns:
            The answer string (space-separated original words).
        """
        return " ".join(parsed_data["original_words"])
