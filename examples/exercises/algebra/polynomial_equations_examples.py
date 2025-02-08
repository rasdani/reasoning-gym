"""Examples of generated problems from the PolynomialEquations exercise.

This file demonstrates different types of polynomial equation problems that can be generated
at various difficulty levels.
"""

from reasoning_gym.curricula.algebra.polynomial_equations_curriculum import PolynomialEquationsCurriculum
from reasoning_gym.exercises.algebra.polynomial_equations import PolynomialEquationsExercise
import random

def main():
    # Initialize with fixed seed for reproducibility
    curriculum = PolynomialEquationsCurriculum()
    exercise = PolynomialEquationsExercise()
    curriculum.rng = random.Random(42)

    print("\n========================================\n")

    # Level 0: Linear equations (ax + b = 0)
    curriculum.set_attr_level("num_terms", 0)  # 2 terms
    curriculum.set_attr_level("coefficient_value", 0)  # Small coefficients (1-10)
    curriculum.set_attr_level("max_degree", 0)  # Linear equations
    curriculum.set_attr_level("operators", 0)  # Just + operator
    curriculum.set_attr_level("sign", 0)  # No signs
    curriculum.set_attr_level("var_name", 0)  # Basic variables (x, y, z)
    problem = exercise.generate(curriculum)
    print("Level 0 (Linear Equations):")
    print(problem)

    print("\n========================================\n")

    # Level 1: Quadratic equations (ax² + bx + c = 0)
    curriculum.set_attr_level("num_terms", 1)  # 3 terms
    curriculum.set_attr_level("coefficient_value", 1)  # Medium coefficients (1-50)
    curriculum.set_attr_level("max_degree", 1)  # Quadratic equations
    curriculum.set_attr_level("operators", 1)  # +, - operators
    curriculum.set_attr_level("sign", 1)  # Allow +/-
    curriculum.set_attr_level("var_name", 0)  # Basic variables
    problem = exercise.generate(curriculum)
    print("Level 1 (Quadratic Equations):")
    print(problem)

    print("\n========================================\n")

    # Level 2: Cubic equations with larger coefficients
    curriculum.set_attr_level("num_terms", 2)  # 4 terms
    curriculum.set_attr_level("coefficient_value", 2)  # Large coefficients (1-100)
    curriculum.set_attr_level("max_degree", 2)  # Cubic equations
    curriculum.set_attr_level("operators", 1)  # +, - operators
    curriculum.set_attr_level("sign", 1)  # Allow +/-
    curriculum.set_attr_level("var_name", 1)  # All ASCII letters
    problem = exercise.generate(curriculum)
    print("Level 2 (Cubic Equations):")
    print(problem)

    print("\n========================================\n")

    # Random Examples with Different Seeds
    print("Random Examples (Different Seeds):")
    for seed in range(10, 15):
        curriculum.rng = random.Random(seed)
        # Randomly set curriculum levels
        curriculum.set_attr_level("num_terms", random.randint(0, 2))
        curriculum.set_attr_level("coefficient_value", random.randint(0, 2))
        curriculum.set_attr_level("max_degree", random.randint(0, 2))
        curriculum.set_attr_level("operators", random.randint(0, 1))
        curriculum.set_attr_level("sign", random.randint(0, 1))
        curriculum.set_attr_level("var_name", random.randint(0, 2))
        problem = exercise.generate(curriculum)
        print(f"\nRandom Example (Seed {seed}):")
        print(problem)

    print("\n========================================\n")

    # Special Cases
    print("Special Cases:")

    # Case 1: Greek variable names with high degree
    curriculum.set_attr_level("num_terms", 2)  # 4 terms
    curriculum.set_attr_level("coefficient_value", 1)  # Medium coefficients
    curriculum.set_attr_level("max_degree", 2)  # Cubic equations
    curriculum.set_attr_level("var_name", 2)  # Greek letters
    problem = exercise.generate(curriculum)
    print("\nGreek Variables with High Degree:")
    print(problem)

    # Case 2: Maximum terms with small coefficients
    curriculum.set_attr_level("num_terms", 2)  # Maximum terms
    curriculum.set_attr_level("coefficient_value", 0)  # Small coefficients
    curriculum.set_attr_level("max_degree", 1)  # Quadratic equations
    problem = exercise.generate(curriculum)
    print("\nMaximum Terms with Small Coefficients:")
    print(problem)

    # Case 3: Linear equation with large coefficients
    curriculum.set_attr_level("num_terms", 0)  # 2 terms
    curriculum.set_attr_level("coefficient_value", 2)  # Large coefficients
    curriculum.set_attr_level("max_degree", 0)  # Linear equations
    curriculum.set_attr_level("var_name", 0)  # Basic variables
    problem = exercise.generate(curriculum)
    print("\nLinear Equation with Large Coefficients:")
    print(problem)

if __name__ == "__main__":
    main() 