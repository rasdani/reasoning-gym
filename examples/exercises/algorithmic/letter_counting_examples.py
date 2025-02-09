"""Examples of generated problems from the LetterCounting exercise.

This file demonstrates different types of letter counting problems that can be generated
at various difficulty levels.
"""

from reasoning_gym.curricula.algorithmic.letter_counting_curriculum import LetterCountingCurriculum
from reasoning_gym.exercises.algorithmic.letter_counting import LetterCountingExercise
import random

def main():
    # Initialize with fixed seed for reproducibility
    curriculum = LetterCountingCurriculum()
    exercise = LetterCountingExercise()
    curriculum.rng = random.Random(42)

    print("\n========================================\n")

    # Level 0: Basic counting with short text and case insensitive
    curriculum.set_attr_level("num_words", 0)  # Short text (5 words)
    curriculum.set_attr_level("case_sensitivity", 0)  # Case insensitive
    curriculum.set_attr_level("letter_selection", 0)  # Common letters
    problem = exercise.generate(curriculum)
    print("Level 0 (Basic Counting):")
    print(problem)

    print("\n========================================\n")

    # Level 1: Medium length text with case sensitivity
    curriculum.set_attr_level("num_words", 1)  # Medium text (10 words)
    curriculum.set_attr_level("case_sensitivity", 1)  # Case sensitive
    curriculum.set_attr_level("letter_selection", 1)  # All letters
    problem = exercise.generate(curriculum)
    print("Level 1 (Case Sensitive Counting):")
    print(problem)

    print("\n========================================\n")

    # Level 2: Long text with rare letters
    curriculum.set_attr_level("num_words", 2)  # Long text (15 words)
    curriculum.set_attr_level("case_sensitivity", 1)  # Case sensitive
    curriculum.set_attr_level("letter_selection", 2)  # Rare letters
    problem = exercise.generate(curriculum)
    print("Level 2 (Rare Letters):")
    print(problem)

    print("\n========================================\n")

    # Random Examples with Different Seeds
    print("Random Examples (Different Seeds):")
    for seed in range(10, 15):
        curriculum.rng = random.Random(seed)
        # Randomly set curriculum levels
        curriculum.set_attr_level("num_words", random.randint(0, 2))
        curriculum.set_attr_level("case_sensitivity", random.randint(0, 1))
        curriculum.set_attr_level("letter_selection", random.randint(0, 2))
        problem = exercise.generate(curriculum)
        print(f"\nRandom Example (Seed {seed}):")
        print(problem)

    print("\n========================================\n")

    # Special Cases
    print("Special Cases:")

    # Case 1: Maximum length with case insensitive common letters
    for attr in curriculum.attributes:  # Reset all attributes
        curriculum.set_attr_level(attr, 0)
    curriculum.set_attr_level("num_words", 2)  # Maximum words (15)
    curriculum.set_attr_level("case_sensitivity", 0)  # Case insensitive
    curriculum.set_attr_level("letter_selection", 0)  # Common letters
    problem = exercise.generate(curriculum)
    print("\nLong Text with Common Letters:")
    print(problem)

    # Case 2: Short text with case sensitive rare letters
    for attr in curriculum.attributes:  # Reset all attributes
        curriculum.set_attr_level(attr, 0)
    curriculum.set_attr_level("num_words", 0)  # Minimum words (5)
    curriculum.set_attr_level("case_sensitivity", 1)  # Case sensitive
    curriculum.set_attr_level("letter_selection", 2)  # Rare letters
    problem = exercise.generate(curriculum)
    print("\nShort Text with Rare Letters:")
    print(problem)

    # Case 3: Medium text with all letters
    for attr in curriculum.attributes:  # Reset all attributes
        curriculum.set_attr_level(attr, 0)
    curriculum.set_attr_level("num_words", 1)  # Medium length (10 words)
    curriculum.set_attr_level("case_sensitivity", 1)  # Case sensitive
    curriculum.set_attr_level("letter_selection", 1)  # All letters
    problem = exercise.generate(curriculum)
    print("\nMedium Text with All Letters:")
    print(problem)

if __name__ == "__main__":
    main()