"""Examples of generated problems from the CaesarCipher exercise.

This file demonstrates different types of Caesar cipher problems that can be generated
at various difficulty levels.
"""

from reasoning_gym.curricula.algorithmic.caesar_cipher_curriculum import CaesarCipherCurriculum
from reasoning_gym.exercises.algorithmic.caesar_cipher import CaesarCipherExercise
import random

def main():
    # Initialize with fixed seed for reproducibility
    curriculum = CaesarCipherCurriculum()
    exercise = CaesarCipherExercise()
    curriculum.rng = random.Random(42)

    print("\n========================================\n")

    # Level 0: Basic decryption with short text and small rotation
    curriculum.set_attr_level("num_words", 0)  # Short text (5 words)
    curriculum.set_attr_level("rotation", 0)  # Small rotation (1-3)
    curriculum.set_attr_level("text_case", 0)  # UPPER case only
    problem = exercise.generate(curriculum)
    print("Level 0 (Basic Decryption):")
    print(problem)

    print("\n========================================\n")

    # Level 1: Medium length text with larger rotation
    curriculum.set_attr_level("num_words", 1)  # Medium text (10 words)
    curriculum.set_attr_level("rotation", 2)  # Medium rotation (10-15)
    curriculum.set_attr_level("text_case", 1)  # lower case only
    problem = exercise.generate(curriculum)
    print("Level 1 (Medium Length Text):")
    print(problem)

    print("\n========================================\n")

    # Level 2: Long text with mixed case and large rotation
    curriculum.set_attr_level("num_words", 2)  # Long text (20 words)
    curriculum.set_attr_level("rotation", 4)  # Large rotation (20-25)
    curriculum.set_attr_level("text_case", 2)  # Mixed case with preserved capitalization
    problem = exercise.generate(curriculum)
    print("Level 2 (Complex Text):")
    print(problem)

    print("\n========================================\n")

    # Random Examples with Different Seeds
    print("Random Examples (Different Seeds):")
    for seed in range(10, 15):
        curriculum.rng = random.Random(seed)
        # Randomly set curriculum levels
        curriculum.set_attr_level("num_words", random.randint(0, 2))
        curriculum.set_attr_level("rotation", random.randint(0, 4))
        curriculum.set_attr_level("text_case", random.randint(0, 2))
        problem = exercise.generate(curriculum)
        print(f"\nRandom Example (Seed {seed}):")
        print(problem)

    print("\n========================================\n")

    # Special Cases
    print("Special Cases:")

    # Case 1: Maximum length with small rotation
    for attr in curriculum.attributes:  # Reset all attributes
        curriculum.set_attr_level(attr, 0)
    curriculum.set_attr_level("num_words", 2)  # Maximum words (20)
    curriculum.set_attr_level("rotation", 0)  # Small rotation (1-3)
    curriculum.set_attr_level("text_case", 0)  # UPPER case
    problem = exercise.generate(curriculum)
    print("\nLong Text with Small Rotation:")
    print(problem)

    # Case 2: Short text with maximum rotation
    for attr in curriculum.attributes:  # Reset all attributes
        curriculum.set_attr_level(attr, 0)
    curriculum.set_attr_level("num_words", 0)  # Minimum words (5)
    curriculum.set_attr_level("rotation", 4)  # Maximum rotation (20-25)
    curriculum.set_attr_level("text_case", 2)  # Mixed case
    problem = exercise.generate(curriculum)
    print("\nShort Text with Large Rotation:")
    print(problem)

    # Case 3: Medium text with mixed case
    for attr in curriculum.attributes:  # Reset all attributes
        curriculum.set_attr_level(attr, 0)
    curriculum.set_attr_level("num_words", 1)  # Medium length (10 words)
    curriculum.set_attr_level("rotation", 2)  # Medium rotation (10-15)
    curriculum.set_attr_level("text_case", 2)  # Mixed case
    problem = exercise.generate(curriculum)
    print("\nMedium Text with Mixed Case:")
    print(problem)

if __name__ == "__main__":
    main() 