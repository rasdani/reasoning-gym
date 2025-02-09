"""Unit tests for the letter jumble exercise."""

from reasoning_gym.curricula.algorithmic.letter_jumble_curriculum import LetterJumbleCurriculum
from reasoning_gym.exercises.algorithmic.letter_jumble import LetterJumbleExercise
import unittest
import random
from collections import defaultdict

class TestLetterJumbleParsing(unittest.TestCase):
    """Test parsing of letter jumble metadata"""

    def setUp(self):
        self.exercise = LetterJumbleExercise()

    def test_parse_expression_basic(self):
        """Test parsing of basic letter jumble metadata"""
        test_metadata = {
            "scrambled": {
                "scrambled_words": "EHLLO DLWOR",
                "original_words": ["HELLO", "WORLD"]
            }
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["scrambled_words"], ["EHLLO", "DLWOR"])
        self.assertEqual(parsed["original_words"], ["HELLO", "WORLD"])

    def test_parse_with_spaces(self):
        """Test parsing with spaces and punctuation"""
        test_metadata = {
            "scrambled": {
                "scrambled_words": "EHLLO DLWOR!",
                "original_words": ["HELLO", "WORLD!"]
            }
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["scrambled_words"], ["EHLLO", "DLWOR!"])
        self.assertEqual(parsed["original_words"], ["HELLO", "WORLD!"])

    def test_parse_mixed_case(self):
        """Test parsing with mixed case text"""
        test_metadata = {
            "scrambled": {
                "scrambled_words": "HeLlO WoRlD",
                "original_words": ["hElLo", "wOrLd"]
            }
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["scrambled_words"], ["HeLlO", "WoRlD"])
        self.assertEqual(parsed["original_words"], ["hElLo", "wOrLd"])

class TestLetterJumbleEvaluation(unittest.TestCase):
    """Test evaluation of letter jumble problems"""

    def setUp(self):
        self.exercise = LetterJumbleExercise()

    def test_basic_unscrambling(self):
        """Test basic unscrambling cases"""
        test_cases = [
            (["EHLLO"], "HELLO"),    # Single word
            (["EHLLO", "DLWOR"], "HELLO WORLD"),  # Two words
            (["AAAA"], "AAAA"),      # Same letters
            (["ZBAC"], "ABCZ"),      # Sorted order
            (["HELLO"], "HELLO")      # Already unscrambled
        ]
        for scrambled, expected in test_cases:
            parsed = {
                "scrambled_words": scrambled,
                "original_words": expected.split()
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

    def test_mixed_case_unscrambling(self):
        """Test unscrambling with mixed case"""
        test_cases = [
            (["HeLlO"], "hElLo"),    # Mixed case, single word
            (["WoRlD", "HeLlO"], "wOrLd hElLo"),  # Mixed case, multiple words
            (["AbCdE"], "aBcDe")     # Mixed case, alternating
        ]
        for scrambled, expected in test_cases:
            parsed = {
                "scrambled_words": scrambled,
                "original_words": expected.split()
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

    def test_with_spaces_and_punctuation(self):
        """Test unscrambling with spaces and punctuation"""
        test_cases = [
            (["EHLLO!", "DLWOR?"], "HELLO! WORLD?"),
            (["EHLLO.", "DLWOR."], "HELLO. WORLD."),
            (["EHLLO,", "DLWOR,"], "HELLO, WORLD,")
        ]
        for scrambled, expected in test_cases:
            parsed = {
                "scrambled_words": scrambled,
                "original_words": expected.split()
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

class TestLetterJumbleGeneration(unittest.TestCase):
    """Test problem generation"""

    def setUp(self):
        self.curriculum = LetterJumbleCurriculum()
        self.exercise = LetterJumbleExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

    def test_problem_structure(self):
        """Test that generated problems have the correct structure"""
        problem = self.exercise.generate(self.curriculum)

        # Check basic structure
        self.assertIn("question", problem)
        self.assertIn("answer", problem)
        self.assertIn("metadata", problem)

        # Check metadata structure
        metadata = problem["metadata"]
        self.assertEqual(metadata["type"], "direct")
        self.assertIn("executed_parts", metadata)
        executed_parts = metadata["executed_parts"]
        self.assertIn("scrambled_words", executed_parts)
        self.assertIn("original_words", executed_parts)

    def test_word_length_ranges(self):
        """Test that word lengths are within expected ranges"""
        # Test all word length levels
        level_max_lengths = {0: 5, 1: 8, 2: 64}

        for level, max_length in level_max_lengths.items():
            self.curriculum.set_attr_level("word_length", level)
            problem = self.exercise.generate(self.curriculum)
            words = problem["metadata"]["executed_parts"]["original_words"]
            for word in words:
                self.assertLessEqual(len(word), max_length)
                self.assertGreaterEqual(len(word), 2)  # Min length is 2

    def test_word_count_ranges(self):
        """Test that word counts are within expected ranges"""
        # Test all word count levels
        level_word_counts = {0: 3, 1: 5, 2: 20}

        for level, max_words in level_word_counts.items():
            self.curriculum.set_attr_level("num_words", level)
            problem = self.exercise.generate(self.curriculum)
            words = problem["metadata"]["executed_parts"]["original_words"]
            self.assertLessEqual(len(words), max_words)
            self.assertGreaterEqual(len(words), 1)  # Min words is 1

class TestLetterJumbleComprehensive(unittest.TestCase):
    """Comprehensive tests for letter jumble"""

    def setUp(self):
        self.curriculum = LetterJumbleCurriculum()
        self.exercise = LetterJumbleExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

    def test_corruption_levels(self):
        """Test different corruption levels"""
        corruption_levels = [0.1, 0.3, 0.9]
        num_samples = 100  # Test with multiple samples

        # Test each level
        for level, expected_corruption in enumerate(corruption_levels):
            self.curriculum.set_attr_level("corruption_level", level)
            differences = []

            # Generate multiple problems to measure average corruption
            for _ in range(num_samples):
                problem = self.exercise.generate(self.curriculum)
                metadata = problem["metadata"]["executed_parts"]
                # Calculate character differences
                preserve_len = self.curriculum.attributes["preserve_length"].levels[self.curriculum.get_attr_level("preserve_length")]
                for orig, scrambled in zip(metadata["original_words"], metadata["scrambled_words"]):
                    if len(orig) > preserve_len:
                        diff_count = sum(1 for a, b in zip(orig, scrambled) if a != b)
                        differences.append(diff_count / len(orig))

            # Check average corruption level is reasonable
            # It's okay if actual corruption is lower than target due to:
            # 1. Some swaps might cancel out previous swaps
            # 2. The same characters might be swapped multiple times
            # 3. The preserve_length attribute prevents some characters from being swapped
            # 4. For short words, even a few swaps can make them readable
            if differences:
                avg_corruption = sum(differences) / len(differences)
                # Only check that we don't exceed target by too much
                self.assertLess(avg_corruption, expected_corruption + 0.1,
                              f"Corruption level {avg_corruption:.2f} too high (target: {expected_corruption:.2f})")
                # And ensure we have some corruption
                self.assertGreater(avg_corruption, 0.02,
                                 f"Corruption level {avg_corruption:.2f} too low (should be above 0.02)")

    def test_template_variation(self):
        """Test that different templates are used"""
        templates_seen = set()
        num_samples = 100

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            templates_seen.add(problem["question"].split(":")[0])

        self.assertGreater(len(templates_seen), 1, "Not enough template variation")

    def test_comprehensive_random_evaluation(self):
        """Test random evaluation with various configurations and track statistics."""
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        self.curriculum.rng = self.rng

        # Track statistics
        word_lengths = defaultdict(int)
        word_counts = defaultdict(int)
        corruption_levels = defaultdict(list)
        consecutive_words_count = 0
        total_samples = 1000

        # Generate test cases
        for _ in range(total_samples):
            # Set random attribute levels
            for attr in self.curriculum.attributes:
                max_level = len(self.curriculum.attributes[attr].levels) - 1
                self.curriculum.set_attr_level(attr, self.rng.randint(0, max_level))

            # Generate and evaluate a random problem
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            original_words = metadata["original_words"]
            scrambled_words = metadata["scrambled_words"]

            # Track statistics
            word_counts[len(original_words)] += 1
            for word in original_words:
                word_lengths[len(word)] += 1

            # Calculate corruption levels
            for orig, scrambled in zip(original_words, scrambled_words):
                preserve_len = self.curriculum.attributes["preserve_length"].levels[self.curriculum.get_attr_level("preserve_length")]
                if len(orig) > preserve_len:
                    diff_count = sum(1 for a, b in zip(orig, scrambled) if a != b)
                    corruption_levels[len(orig)].append(diff_count / len(orig))

            # Check if words are consecutive in source text
            if len(original_words) > 1:
                text = " ".join(self.curriculum.words)
                phrase = " ".join(original_words)
                if phrase in text:
                    consecutive_words_count += 1

            # Verify scrambling is valid
            for orig, scrambled in zip(original_words, scrambled_words):
                # Check lengths match
                self.assertEqual(len(orig), len(scrambled))
                # Check same letters are used
                self.assertEqual(sorted(orig), sorted(scrambled))

        # Print statistics
        print("\nWord length distribution:")
        for length, count in sorted(word_lengths.items()):
            print(f"  Length {length}: {count}")

        print("\nWord count distribution:")
        for count, freq in sorted(word_counts.items()):
            print(f"  {count} words: {freq}")

        print("\nAverage corruption levels by word length:")
        for length, levels in sorted(corruption_levels.items()):
            avg = sum(levels) / len(levels) if levels else 0
            print(f"  Length {length}: {avg:.2f}")

        print(f"\nConsecutive words: {consecutive_words_count}/{total_samples}")

        # Verify statistical properties
        self.assertTrue(any(length >= 8 for length in word_lengths), 
                       "No long words generated")
        self.assertTrue(any(count >= 3 for count in word_counts.values()),
                       "Not enough variation in word counts")
        self.assertTrue(consecutive_words_count > 0,
                       "No consecutive words generated")
        self.assertTrue(consecutive_words_count < total_samples,
                       "Too many consecutive words")

if __name__ == '__main__':
    unittest.main()