"""Examples of generated problems from the ChainSum exercise.

This file demonstrates different types of problems that can be generated
by the ChainSum exercise at various difficulty levels.
"""

from reasoning_gym.curricula.arithmetic.chain_sum_curriculum import ChainSumCurriculum
from reasoning_gym.exercises.arithmetic.chain_sum import ChainSumExercise
import random
import numpy as np

def main():
	# Initialize with fixed seed for reproducibility
	curriculum = ChainSumCurriculum()
	exercise = ChainSumExercise()
	curriculum.rng = random.Random(42)

	print("\n========================================\n")

	# Level 0: Basic addition
	curriculum.set_attr_level("operators", 0)  # Only +
	curriculum.set_attr_level("num_terms", 0)  # 2 terms
	curriculum.set_attr_level("num_digits", 0)  # 1-2 digits
	curriculum.set_attr_level("num_decimals", 0)  # No decimals
	curriculum.set_attr_level("sign", 0)  # No signs
	curriculum.set_attr_level("notation", 0)  # Regular notation
	problem = exercise.generate(curriculum)
	print("Level 0 (Basic Addition):")
	print(problem)

	print("\n========================================\n")

	# Level 1: Addition/subtraction with decimals
	curriculum.set_attr_level("operators", 1)  # +, -
	curriculum.set_attr_level("num_terms", 0)  # 2 terms
	curriculum.set_attr_level("num_digits", 1)  # 1-4 digits
	curriculum.set_attr_level("num_decimals", 1)  # 1 decimal place
	curriculum.set_attr_level("sign", 2)  # Allow +/-
	curriculum.set_attr_level("notation", 0)  # Regular notation
	problem = exercise.generate(curriculum)
	print("\nLevel 1 (Addition/Subtraction with Decimals):")
	print(problem)

	print("\n========================================\n")

	# Level 2: Mixed operations with scientific notation
	curriculum.set_attr_level("operators", 2)  # +, -, *, /
	curriculum.set_attr_level("num_terms", 2)  # 2-4 terms
	curriculum.set_attr_level("num_digits", 2)  # 1-10 digits
	curriculum.set_attr_level("sign", 2)  # Allow +/-
	curriculum.set_attr_level("notation", 1)  # Scientific notation
	problem = exercise.generate(curriculum)
	print("\nLevel 2 (Mixed Operations with Scientific Notation):")
	print(problem)

	print("\n========================================\n")

	# Level 3: Complex expressions with different notations
	curriculum.set_attr_level("operators", 2)  # +, -, *, /
	curriculum.set_attr_level("num_terms", 3)  # 2-5 terms
	curriculum.set_attr_level("num_digits", 2)  # 1-10 digits
	curriculum.set_attr_level("sign", 2)  # Allow +/-
	curriculum.set_attr_level("notation", 3)  # All notations
	problem = exercise.generate(curriculum)
	print("\nLevel 3 (Complex Expressions with Mixed Notations):")
	print(problem)

	print("\n========================================\n")

	# Random Examples with Different Seeds
	print("Random Examples (Different Seeds):")
	for seed in range(10, 15):
		curriculum.rng = random.Random(seed)
		# Randomly set curriculum levels
		curriculum.set_attr_level("operators", random.randint(0, 4))
		curriculum.set_attr_level("num_terms", random.randint(0, 3))
		curriculum.set_attr_level("num_digits", random.randint(0, 2))
		curriculum.set_attr_level("num_decimals", random.randint(0, 3))
		curriculum.set_attr_level("sign", random.randint(0, 2))
		curriculum.set_attr_level("notation", random.randint(0, 3))
		problem = exercise.generate(curriculum)
		print(f"\nRandom Example (Seed {seed}):")
		print(problem)

	print("\n========================================\n")

	# Special Cases
	print("Special Cases:")

	# Case 1: Large number arithmetic with mixed notations
	curriculum.set_attr_level("operators", 2)  # +, -, *, /
	curriculum.set_attr_level("num_terms", 2)  # 2-4 terms
	curriculum.set_attr_level("num_digits", 2)  # Large numbers
	curriculum.set_attr_level("notation", 3)  # Mixed notations
	problem = exercise.generate(curriculum)
	print("\nLarge Numbers with Mixed Notation:")
	print(problem)

	# Case 2: Maximum terms with all operators
	curriculum.set_attr_level("operators", 3)  # All operators including **
	curriculum.set_attr_level("num_terms", 3)  # Maximum terms
	curriculum.set_attr_level("num_digits", 1)  # Medium numbers
	curriculum.set_attr_level("notation", 0)  # Regular notation
	problem = exercise.generate(curriculum)
	print("\nMaximum Terms with All Operators:")
	print(problem)

	# Case 3: Binary and Hex mixed
	curriculum.set_attr_level("operators", 1)  # +, -
	curriculum.set_attr_level("num_terms", 2)  # 3-4 terms
	curriculum.set_attr_level("notation", 3)  # All notations
	problem = exercise.generate(curriculum)
	print("\nBinary and Hex Mixed:")
	print(problem)

if __name__ == "__main__":
	main()