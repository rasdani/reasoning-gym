"""Examples of generated problems from the SimpleEquations exercise.

This file demonstrates different types of linear equation problems that can be generated
at various difficulty levels.
"""

from reasoning_gym.curricula.algebra.simple_equations_curriculum import SimpleEquationsCurriculum
from reasoning_gym.exercises.algebra.simple_equations import SimpleEquationsExercise
import random

def main():
    # Initialize with fixed seed for reproducibility
    curriculum = SimpleEquationsCurriculum()
    exercise = SimpleEquationsExercise()
    curriculum.rng = random.Random(42)

    print("\n========================================\n")

    # Level 0: Basic equations (ax = b)
    curriculum.set_attr_level("num_terms", 0)  # 2 terms
    curriculum.set_attr_level("value", 0)  # Small values (1-10)
    curriculum.set_attr_level("operators", 0)  # Just + operator
    curriculum.set_attr_level("sign", 0)  # No negative signs
    curriculum.set_attr_level("var_name", 0)  # Basic variables (x, y, z)
    problem = exercise.generate(curriculum)
    print("Level 0 (Basic Equations):")
    print(problem)

    print("\n========================================\n")

    # Level 1: Two-term equations with negatives (ax + b = c)
    curriculum.set_attr_level("num_terms", 1)  # 3 terms
    curriculum.set_attr_level("value", 1)  # Medium values (1-50)
    curriculum.set_attr_level("operators", 1)  # +, - operators
    curriculum.set_attr_level("sign", 1)  # Allow negative signs
    curriculum.set_attr_level("var_name", 0)  # Basic variables
    problem = exercise.generate(curriculum)
    print("Level 1 (Two-term Equations):")
    print(problem)

    print("\n========================================\n")

    # Level 2: Complex equations with multiple terms
    curriculum.set_attr_level("num_terms", 2)  # 4 terms
    curriculum.set_attr_level("value", 2)  # Large values (1-100)
    curriculum.set_attr_level("operators", 2)  # +, - operators
    curriculum.set_attr_level("sign", 1)  # Allow negative signs
    curriculum.set_attr_level("var_name", 1)  # All lowercase letters
    problem = exercise.generate(curriculum)
    print("Level 2 (Complex Equations):")
    print(problem)

    print("\n========================================\n")

    # Random Examples with Different Seeds
    print("Random Examples (Different Seeds):")
    for seed in range(10, 15):
        curriculum.rng = random.Random(seed)
        # Randomly set curriculum levels
        curriculum.set_attr_level("num_terms", random.randint(0, 2))
        curriculum.set_attr_level("value", random.randint(0, 2))
        curriculum.set_attr_level("operators", random.randint(0, 2))
        curriculum.set_attr_level("sign", random.randint(0, 1))
        curriculum.set_attr_level("var_name", random.randint(0, 2))
        problem = exercise.generate(curriculum)
        print(f"\nRandom Example (Seed {seed}):")
        print(problem)

    print("\n========================================\n")

    # Special Cases
    print("Special Cases:")

    # Case 1: Greek variable names with complex terms
    curriculum.set_attr_level("num_terms", 2)  # 4 terms
    curriculum.set_attr_level("value", 1)  # Medium values
    curriculum.set_attr_level("var_name", 2)  # Greek letters
    problem = exercise.generate(curriculum)
    print("\nGreek Variables with Complex Terms:")
    print(problem)

    # Case 2: Maximum terms with small values
    curriculum.set_attr_level("num_terms", 2)  # Maximum terms
    curriculum.set_attr_level("value", 0)  # Small values
    curriculum.set_attr_level("var_name", 0)  # Basic variables
    problem = exercise.generate(curriculum)
    print("\nMaximum Terms with Small Values:")
    print(problem)

    # Case 3: Simple equation with large values
    curriculum.set_attr_level("num_terms", 0)  # 2 terms
    curriculum.set_attr_level("value", 2)  # Large values
    curriculum.set_attr_level("var_name", 0)  # Basic variables
    problem = exercise.generate(curriculum)
    print("\nSimple Equation with Large Values:")
    print(problem)

if __name__ == "__main__":
    main()
