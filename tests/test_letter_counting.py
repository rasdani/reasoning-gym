"""Tests for the letter counting exercise."""

from reasoning_gym.curricula.algorithmic.letter_counting_curriculum import LetterCountingCurriculum
from reasoning_gym.exercises.algorithmic.letter_counting import LetterCountingExercise
import unittest
import random


class TestLetterCountingParsing(unittest.TestCase):
    """Test parsing of expressions and metadata."""

    def setUp(self):
        self.exercise = LetterCountingExercise()

    def test_parse_expression(self):
        """Test parsing of metadata into structured data."""
        test_metadata = {
            "text": {"text": "hello world"},
            "letter": {"letter": "l"},
            "case_sensitivity": {"sensitivity": "sensitive"}
        }

        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["text"], "hello world")
        self.assertEqual(parsed["target_letter"], "l")
        self.assertTrue(parsed["case_sensitive"])

    def test_parse_case_insensitive(self):
        """Test parsing with case insensitive setting."""
        test_metadata = {
            "text": {"text": "Hello World"},
            "letter": {"letter": "L"},
            "case_sensitivity": {"sensitivity": "insensitive"}
        }

        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["text"], "Hello World")
        self.assertEqual(parsed["target_letter"], "L")
        self.assertFalse(parsed["case_sensitive"])


class TestLetterCountingEvaluation(unittest.TestCase):
    """Test evaluation of letter counting expressions."""

    def setUp(self):
        self.exercise = LetterCountingExercise()

    def test_case_sensitive_counting(self):
        """Test counting letters with case sensitivity."""
        test_cases = [
            {
                "text": "hello",
                "target_letter": "l",
                "case_sensitive": True,
                "expected": "2"
            },
            {
                "text": "Hello",
                "target_letter": "l",
                "case_sensitive": True,
                "expected": "2"
            },
            {
                "text": "HELLO",
                "target_letter": "l",
                "case_sensitive": True,
                "expected": "0"
            }
        ]

        for case in test_cases:
            result = self.exercise._evaluate_expression(case)
            self.assertEqual(result, case["expected"], 
                           f"Failed to count '{case['target_letter']}' in '{case['text']}' case sensitively")

    def test_case_insensitive_counting(self):
        """Test counting letters without case sensitivity."""
        test_cases = [
            {
                "text": "hello",
                "target_letter": "l",
                "case_sensitive": False,
                "expected": "2"
            },
            {
                "text": "Hello",
                "target_letter": "L",
                "case_sensitive": False,
                "expected": "2"
            },
            {
                "text": "HELLO",
                "target_letter": "l",
                "case_sensitive": False,
                "expected": "2"
            }
        ]

        for case in test_cases:
            result = self.exercise._evaluate_expression(case)
            self.assertEqual(result, case["expected"],
                           f"Failed to count '{case['target_letter']}' in '{case['text']}' case insensitively")

    def test_empty_string(self):
        """Test counting in empty string."""
        parsed = {
            "text": "",
            "target_letter": "a",
            "case_sensitive": True
        }
        result = self.exercise._evaluate_expression(parsed)
        self.assertEqual(result, "0")


class TestLetterCountingGeneration(unittest.TestCase):
    """Test problem generation."""

    def setUp(self):
        self.curriculum = LetterCountingCurriculum()
        self.exercise = LetterCountingExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

        # Add some test words to ensure we have content
        self.curriculum.words = ["hello", "world", "test", "example", "python", 
                               "programming", "language", "code", "algorithm", "data"]

    def test_problem_structure(self):
        """Test that generated problems have the correct structure."""
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
        self.assertIn("text", executed_parts)
        self.assertIn("target_letter", executed_parts)
        self.assertIn("case_sensitive", executed_parts)

    def test_text_generation(self):
        """Test generation of text spans."""
        num_samples = 50
        texts_seen = set()

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            text = problem["metadata"]["executed_parts"]["text"]
            texts_seen.add(text)

            # Verify text is not empty
            self.assertTrue(len(text) > 0, "Empty text generated")
            # Verify text contains only valid characters
            self.assertTrue(all(c.isalnum() or c.isspace() for c in text),
                          f"Invalid characters in text: {text}")

        # Verify we get different texts
        self.assertTrue(len(texts_seen) > 1, "Only one text pattern generated")

    def test_letter_selection(self):
        """Test selection of target letters."""
        num_samples = 50
        letters_seen = set()

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            letter = problem["metadata"]["executed_parts"]["target_letter"]
            letters_seen.add(letter)

            # Verify letter is a single character
            self.assertEqual(len(letter), 1, f"Invalid letter length: {letter}")
            # Verify letter is alphabetic
            self.assertTrue(letter.isalpha(), f"Non-alphabetic letter: {letter}")

        # Verify we get different letters
        self.assertTrue(len(letters_seen) > 1, "Only one letter generated")


class TestLetterCountingComprehensive(unittest.TestCase):
    """Comprehensive tests for letter counting."""

    def setUp(self):
        self.curriculum = LetterCountingCurriculum()
        self.exercise = LetterCountingExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

        # Add more test words to ensure we have enough content
        self.curriculum.words = [
            "hello", "world", "test", "example", "python", "programming", "language",
            "code", "algorithm", "data", "computer", "science", "software", "development",
            "testing", "debugging", "function", "variable", "constant", "loop", "condition",
            "string", "integer", "float", "boolean", "array", "list", "dictionary", "set",
            "class", "object", "method", "inheritance", "polymorphism", "encapsulation",
            "abstraction", "interface", "implementation", "module", "package", "library",
            "framework", "application", "system", "network", "database", "security",
            "authentication", "authorization", "validation", "verification"
        ]

    def test_case_sensitivity_levels(self):
        """Test that both case sensitivity levels are used."""
        num_samples = 100
        sensitivities_seen = set()

        # Set other attributes to stable values
        self.curriculum.set_attr_level("num_words", 0)  # 5 words
        self.curriculum.set_attr_level("letter_selection", 0)  # common letters

        # Try both sensitivity levels
        for level in [0, 1]:  # False, True
            self.curriculum.set_attr_level("case_sensitivity", level)

            for _ in range(num_samples // 2):
                problem = self.exercise.generate(self.curriculum)
                case_sensitive = problem["metadata"]["executed_parts"]["case_sensitive"]
                sensitivities_seen.add(case_sensitive)

        # Verify we see both sensitivity settings
        self.assertEqual(len(sensitivities_seen), 2, 
                        f"Only saw case sensitivities: {sensitivities_seen}")

    def test_comprehensive_random_evaluation(self):
        """Test 1000 problems with varying attribute levels."""
        num_samples = 1000

        # Statistics tracking
        stats = {
            'text_lengths': {},         # Distribution of text lengths
            'letter_frequencies': {},    # Frequency of target letters
            'case_sensitivity': {        # Count of case sensitive vs insensitive
                'sensitive': 0,
                'insensitive': 0
            },
            'answer_distribution': {},   # Distribution of letter counts
            'word_counts': {},          # Distribution of word counts
            'attribute_levels': {        # Track curriculum levels used
                'num_words': set(),
                'case_sensitivity': set(),
                'letter_selection': set()
            }
        }

        for _ in range(num_samples):
            # Randomly vary all attribute levels
            self.curriculum.set_attr_level("num_words", self.rng.randint(0, 2))  # 5, 10, or 15 words
            self.curriculum.set_attr_level("case_sensitivity", self.rng.randint(0, 1))  # False or True
            self.curriculum.set_attr_level("letter_selection", self.rng.randint(0, 2))  # common, all, or rare

            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            text = metadata["text"]
            letter = metadata["target_letter"]
            case_sensitive = metadata["case_sensitive"]

            # Update text length statistics
            text_len = len(text)
            stats['text_lengths'][text_len] = stats['text_lengths'].get(text_len, 0) + 1

            # Update letter frequency statistics
            stats['letter_frequencies'][letter] = stats['letter_frequencies'].get(letter, 0) + 1

            # Update case sensitivity statistics
            if case_sensitive:
                stats['case_sensitivity']['sensitive'] += 1
            else:
                stats['case_sensitivity']['insensitive'] += 1

            # Update answer distribution
            answer = int(problem["answer"])
            stats['answer_distribution'][answer] = stats['answer_distribution'].get(answer, 0) + 1

            # Update word count statistics
            word_count = len(text.split())
            stats['word_counts'][word_count] = stats['word_counts'].get(word_count, 0) + 1

            # Verify answer correctness
            parsed = {
                "text": {"text": text},
                "letter": {"letter": letter},
                "case_sensitivity": {"sensitivity": "sensitive" if case_sensitive else "insensitive"}
            }
            expected = self.exercise._evaluate_expression(self.exercise._parse_expression(parsed))
            self.assertEqual(problem["answer"], expected,
                           f"Wrong answer for counting '{letter}' in '{text}' (case_sensitive={case_sensitive})")

        # Print statistics
        print("\nComprehensive Random Evaluation Statistics:")
        print("-" * 50)

        print("\nText Length Distribution:")
        for length, count in sorted(stats['text_lengths'].items()):
            print(f"  Length {length}: {count} ({count/num_samples*100:.1f}%)")

        print("\nLetter Frequency Distribution:")
        total_letters = sum(stats['letter_frequencies'].values())
        for letter, count in sorted(stats['letter_frequencies'].items()):
            print(f"  '{letter}': {count} ({count/total_letters*100:.1f}%)")

        print("\nCase Sensitivity Distribution:")
        for sensitivity, count in stats['case_sensitivity'].items():
            print(f"  {sensitivity}: {count} ({count/num_samples*100:.1f}%)")
            # Verify we see case sensitive problems
            if sensitivity == 'sensitive':
                self.assertGreater(count, 0, "No case sensitive problems generated")

        print("\nAnswer Distribution:")
        for count, freq in sorted(stats['answer_distribution'].items()):
            print(f"  Count {count}: {freq} ({freq/num_samples*100:.1f}%)")

        print("\nWord Count Distribution:")
        for words, count in sorted(stats['word_counts'].items()):
            print(f"  {words} words: {count} ({count/num_samples*100:.1f}%)")


if __name__ == '__main__':
    unittest.main()