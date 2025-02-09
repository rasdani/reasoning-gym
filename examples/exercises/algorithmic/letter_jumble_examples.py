"""Examples of generated problems from the LetterJumble exercise.

This file demonstrates different types of letter jumble problems that can be generated
at various difficulty levels.
"""

import random
from reasoning_gym.curricula.algorithmic.letter_jumble_curriculum import LetterJumbleCurriculum
from reasoning_gym.exercises.algorithmic.letter_jumble import LetterJumbleExercise

def main():
    # Initialize with fixed seed for reproducibility
    curriculum = LetterJumbleCurriculum()
    exercise = LetterJumbleExercise()
    curriculum.rng = random.Random(42)

    print("\n========================================\n")

    # Level 0: Basic word scrambling
    curriculum.set_attr_level("word_length", 0)  # Short words (up to 5 chars)
    curriculum.set_attr_level("num_words", 0)  # Few words (up to 3)
    curriculum.set_attr_level("corruption_level", 0)  # Light scrambling (0.3)
    curriculum.set_attr_level("consecutive_words", 0)  # Consecutive words
    curriculum.set_attr_level("preserve_length", 0)  # Preserve first 4 chars
    problem = exercise.generate(curriculum)
    print("Level 0 (Basic Word Scrambling):")
    print(problem)

    print("\n========================================\n")

    # Level 1: Medium difficulty
    curriculum.set_attr_level("word_length", 1)  # Medium words (up to 8 chars)
    curriculum.set_attr_level("num_words", 1)  # More words (up to 5)
    curriculum.set_attr_level("corruption_level", 1)  # Medium scrambling (0.6)
    curriculum.set_attr_level("consecutive_words", 0)  # Consecutive words
    curriculum.set_attr_level("preserve_length", 0)  # Preserve first 4 chars
    problem = exercise.generate(curriculum)
    print("Level 1 (Medium Difficulty):")
    print(problem)

    print("\n========================================\n")

    # Level 2: Advanced scrambling
    curriculum.set_attr_level("word_length", 2)  # Long words (up to 64 chars)
    curriculum.set_attr_level("num_words", 2)  # Many words (up to 20)
    curriculum.set_attr_level("corruption_level", 2)  # Heavy scrambling (0.9)
    curriculum.set_attr_level("consecutive_words", 1)  # Non-consecutive words
    curriculum.set_attr_level("preserve_length", 1)  # Preserve first 2 chars
    problem = exercise.generate(curriculum)
    print("Level 2 (Advanced Scrambling):")
    print(problem)

    print("\n========================================\n")

    # Random Examples with Different Seeds
    print("Random Examples (Different Seeds):")
    for seed in range(10, 15):
        curriculum.rng = random.Random(seed)
        # Randomly set curriculum levels
        curriculum.set_attr_level("word_length", random.randint(0, 2))
        curriculum.set_attr_level("num_words", random.randint(0, 2))
        curriculum.set_attr_level("corruption_level", random.randint(0, 2))
        curriculum.set_attr_level("consecutive_words", random.randint(0, 1))
        curriculum.set_attr_level("preserve_length", random.randint(0, 1))
        problem = exercise.generate(curriculum)
        print(f"\nRandom Example (Seed {seed}):")
        print(problem)

    print("\n========================================\n")

    # Special Cases
    print("Special Cases:")

    # Case 1: Maximum length single word with minimal preservation
    curriculum.set_attr_level("word_length", 2)  # Long words
    curriculum.set_attr_level("num_words", 0)  # Single word
    curriculum.set_attr_level("corruption_level", 2)  # Heavy scrambling
    curriculum.set_attr_level("consecutive_words", 0)  # Consecutive (doesn't matter for single word)
    curriculum.set_attr_level("preserve_length", 1)  # Preserve first 2 chars
    problem = exercise.generate(curriculum)
    print("\nLong Single Word (Minimal Preservation):")
    print(problem)

    # Case 2: Many short words with maximum preservation
    curriculum.set_attr_level("word_length", 0)  # Short words
    curriculum.set_attr_level("num_words", 2)  # Many words
    curriculum.set_attr_level("corruption_level", 1)  # Medium scrambling
    curriculum.set_attr_level("consecutive_words", 1)  # Non-consecutive
    curriculum.set_attr_level("preserve_length", 0)  # Preserve first 4 chars
    problem = exercise.generate(curriculum)
    print("\nMany Short Words (Maximum Preservation):")
    print(problem)

    # Case 3: Medium words with balanced preservation
    curriculum.set_attr_level("word_length", 1)  # Medium words
    curriculum.set_attr_level("num_words", 1)  # Medium number of words
    curriculum.set_attr_level("corruption_level", 0)  # Light scrambling
    curriculum.set_attr_level("consecutive_words", 0)  # Consecutive
    curriculum.set_attr_level("preserve_length", 1)  # Preserve first 2 chars
    problem = exercise.generate(curriculum)
    print("\nMedium Words (Balanced Preservation):")
    print(problem)

if __name__ == "__main__":
    main() 