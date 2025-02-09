"""Examples of generated problems from the BaseConversion exercise.

This file demonstrates different types of base conversion problems that can be generated
at various difficulty levels.
"""

from reasoning_gym.curricula.algorithmic.base_conversion_curriculum import BaseConversionCurriculum
from reasoning_gym.exercises.algorithmic.base_conversion import BaseConversionExercise
import random

def main():
    # Initialize with fixed seed for reproducibility
    curriculum = BaseConversionCurriculum()
    exercise = BaseConversionExercise()
    curriculum.rng = random.Random(42)

    print("\n========================================\n")

    # Level 0: Basic binary/decimal conversions
    curriculum.set_attr_level("value", 0)  # Small values (up to 100)
    curriculum.set_attr_level("base_range", 0)  # Up to base-16
    curriculum.set_attr_level("base_names", 0)  # Basic names (binary, hexadecimal)
    curriculum.set_attr_level("hint", 0)  # Include hints
    problem = exercise.generate(curriculum)
    print("Level 0 (Basic Binary/Decimal):")
    print(problem)

    print("\n========================================\n")

    # Level 1: Medium difficulty with octal/decimal
    curriculum.set_attr_level("value", 1)  # Medium values (up to 1000)
    curriculum.set_attr_level("base_range", 0)  # Up to base-16
    curriculum.set_attr_level("base_names", 1)  # Add octal/decimal names
    curriculum.set_attr_level("hint", 0)  # Include hints
    problem = exercise.generate(curriculum)
    print("Level 1 (Medium with Octal):")
    print(problem)

    print("\n========================================\n")

    # Level 2: Advanced with higher bases
    curriculum.set_attr_level("value", 2)  # Large values (up to 10000)
    curriculum.set_attr_level("base_range", 1)  # Up to base-26
    curriculum.set_attr_level("base_names", 1)  # All base names
    curriculum.set_attr_level("hint", 1)  # No hints
    problem = exercise.generate(curriculum)
    print("Level 2 (Advanced High Bases):")
    print(problem)

    print("\n========================================\n")

    # Random Examples with Different Seeds
    print("Random Examples (Different Seeds):")
    for seed in range(10, 15):
        curriculum.rng = random.Random(seed)
        # Randomly set curriculum levels
        curriculum.set_attr_level("value", random.randint(0, 2))
        curriculum.set_attr_level("base_range", random.randint(0, 2))
        curriculum.set_attr_level("base_names", random.randint(0, 1))
        curriculum.set_attr_level("hint", random.randint(0, 1))
        problem = exercise.generate(curriculum)
        print(f"\nRandom Example (Seed {seed}):")
        print(problem)

    print("\n========================================\n")

    # Special Cases
    print("Special Cases:")

    # Case 1: Maximum value in binary
    curriculum.set_attr_level("value", 2)  # Large values
    curriculum.set_attr_level("base_range", 0)  # Basic bases
    curriculum.set_attr_level("base_names", 0)  # Basic names
    curriculum.set_attr_level("hint", 0)  # With hints
    problem = exercise.generate(curriculum)
    print("\nLarge Binary Conversion:")
    print(problem)

    # Case 2: High base with small value
    curriculum.set_attr_level("value", 0)  # Small values
    curriculum.set_attr_level("base_range", 2)  # Up to base-36
    curriculum.set_attr_level("base_names", 1)  # All names
    curriculum.set_attr_level("hint", 0)  # With hints
    problem = exercise.generate(curriculum)
    print("\nHigh Base with Small Value:")
    print(problem)

    # Case 3: Medium value with no hints
    curriculum.set_attr_level("value", 1)  # Medium values
    curriculum.set_attr_level("base_range", 1)  # Up to base-26
    curriculum.set_attr_level("base_names", 1)  # All names
    curriculum.set_attr_level("hint", 1)  # No hints
    problem = exercise.generate(curriculum)
    print("\nMedium Value No Hints:")
    print(problem)

if __name__ == "__main__":
    main() 