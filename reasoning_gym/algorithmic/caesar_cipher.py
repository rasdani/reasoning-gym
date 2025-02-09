"""Caesar cipher exercise that generates encryption/decryption tasks."""

from typing import Dict, Any

class CaesarCipherExercise:
    """Exercise generator for Caesar cipher encryption/decryption tasks."""

    def __init__(self):
        self.curriculum = None

    def generate(self, curriculum: Any) -> Dict[str, Any]:
        """
        Generate a Caesar cipher problem using the curriculum.

        Returns:
            Dict containing:
                - question: str (e.g. "Decrypt this Caesar cipher text: KHOOR")
                - answer: str (the decrypted text)
                - metadata: dict with details (rotation, cipher_text, clear_text)
        """
        self.curriculum = curriculum
        template = curriculum.get_template(curriculum.rng)
        return template.eval(self, curriculum.rng)

    def _parse_expression(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the template metadata into structured data.

        The metadata structure is expected to be:
        {
            "cipher_text": {
                "encrypted_text": str,  # The encrypted text
                "clear_text": str,      # The original text
                "rotation": int         # The rotation value
            }
        }
        Returns:
            Dictionary containing parsed data for evaluation
        """
        return {
            "cipher_text": metadata["cipher_text"]["encrypted_text"],
            "clear_text": metadata["cipher_text"]["clear_text"],
            "rotation": metadata["cipher_text"]["rotation"]
        }

    def _evaluate_expression(self, parsed: Dict[str, Any]) -> str:
        """
        Return the clear text for decryption problems.

        Args:
            parsed: Dictionary containing:
                - cipher_text: str (the encrypted text)
                - clear_text: str (the original text)
                - rotation: int (the rotation value)
        Returns:
            String with the decrypted text (clear_text)
        """
        # For the current curriculum, we only handle decryption
        # and the clear_text is already provided in the metadata
        return parsed["clear_text"]
