"""Unit tests for the Caesar cipher exercise."""

from reasoning_gym.curricula.algorithmic.caesar_cipher_curriculum import CaesarCipherCurriculum
from reasoning_gym.exercises.algorithmic.caesar_cipher import CaesarCipherExercise
import unittest
import random
from collections import defaultdict

class TestCaesarCipherParsing(unittest.TestCase):
    """Test parsing of Caesar cipher metadata"""

    def setUp(self):
        self.exercise = CaesarCipherExercise()

    def test_parse_expression_basic(self):
        """Test parsing of basic Caesar cipher metadata"""
        test_metadata = {
            "cipher_text": {
                "encrypted_text": "KHOOR",
                "clear_text": "HELLO",
                "rotation": 3
            }
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["cipher_text"], "KHOOR")
        self.assertEqual(parsed["clear_text"], "HELLO")
        self.assertEqual(parsed["rotation"], 3)

    def test_parse_with_spaces(self):
        """Test parsing with spaces and punctuation"""
        test_metadata = {
            "cipher_text": {
                "encrypted_text": "KHOOR ZRUOG!",
                "clear_text": "HELLO WORLD!",
                "rotation": 3
            }
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["cipher_text"], "KHOOR ZRUOG!")
        self.assertEqual(parsed["clear_text"], "HELLO WORLD!")
        self.assertEqual(parsed["rotation"], 3)

    def test_parse_mixed_case(self):
        """Test parsing with mixed case text"""
        test_metadata = {
            "cipher_text": {
                "encrypted_text": "KhOoR",
                "clear_text": "HeLlO",
                "rotation": 3
            }
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["cipher_text"], "KhOoR")
        self.assertEqual(parsed["clear_text"], "HeLlO")
        self.assertEqual(parsed["rotation"], 3)

class TestCaesarCipherEvaluation(unittest.TestCase):
    """Test evaluation of Caesar cipher problems"""

    def setUp(self):
        self.exercise = CaesarCipherExercise()

    def test_basic_decryption(self):
        """Test basic decryption cases"""
        test_cases = [
            ("KHOOR", "HELLO", 3),    # Basic uppercase
            ("khoor", "hello", 3),    # Basic lowercase
            ("WORLD", "WORLD", 0),    # No rotation
            ("ABCDE", "ZABCD", 1),    # Wrap around
            ("hello", "hello", 26)     # Full rotation
        ]
        for cipher_text, clear_text, rotation in test_cases:
            parsed = {
                "cipher_text": cipher_text,
                "clear_text": clear_text,
                "rotation": rotation
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, clear_text)

    def test_mixed_case_decryption(self):
        """Test decryption with mixed case"""
        test_cases = [
            ("HeLlO", "HeLlO", 26),    # Mixed case, full rotation
            ("WoRlD", "WoRlD", 0),     # Mixed case, no rotation
            ("AbCdE", "ZaBcD", 1)      # Mixed case, wrap around
        ]
        for cipher_text, clear_text, rotation in test_cases:
            parsed = {
                "cipher_text": cipher_text,
                "clear_text": clear_text,
                "rotation": rotation
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, clear_text)

    def test_with_spaces_and_punctuation(self):
        """Test decryption with spaces and punctuation"""
        test_cases = [
            ("KHOOR ZRUOG!", "HELLO WORLD!", 3),
            ("Pb Pbvwhub!", "My Mystery!", 3),
            ("ABCDE. FGHIJ?", "ZABCD. EFGHI?", 1)
        ]
        for cipher_text, clear_text, rotation in test_cases:
            parsed = {
                "cipher_text": cipher_text,
                "clear_text": clear_text,
                "rotation": rotation
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, clear_text)

class TestCaesarCipherGeneration(unittest.TestCase):
    """Test problem generation"""

    def setUp(self):
        self.curriculum = CaesarCipherCurriculum()
        self.exercise = CaesarCipherExercise()
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
        self.assertIn("cipher_text", executed_parts)
        self.assertIn("clear_text", executed_parts)
        self.assertIn("rotation", executed_parts)

    def test_rotation_ranges(self):
        """Test that rotation values are within expected ranges"""
        # Test all rotation levels
        level_max_rotations = {0: 1, 1: 3, 2: 10, 3: 15, 4: 25}

        for level, max_rotation in level_max_rotations.items():
            self.curriculum.set_attr_level("rotation", level)
            problem = self.exercise.generate(self.curriculum)
            rotation = problem["metadata"]["executed_parts"]["rotation"]
            self.assertLessEqual(rotation, max_rotation)
            self.assertGreaterEqual(rotation, 1)  # Min rotation is 1

    def test_word_count_ranges(self):
        """Test that word counts are within expected ranges"""
        # Test all word count levels
        level_word_counts = {0: 5, 1: 10, 2: 20}

        for level, max_words in level_word_counts.items():
            self.curriculum.set_attr_level("num_words", level)
            problem = self.exercise.generate(self.curriculum)
            clear_text = problem["metadata"]["executed_parts"]["clear_text"]
            word_count = len(clear_text.split())
            self.assertLessEqual(word_count, max_words)
            self.assertGreaterEqual(word_count, 3)  # Min words is 3

class TestCaesarCipherComprehensive(unittest.TestCase):
    """Comprehensive tests for Caesar cipher"""

    def setUp(self):
        self.curriculum = CaesarCipherCurriculum()
        self.exercise = CaesarCipherExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

    def test_text_case_styles(self):
        """Test different text case styles"""
        case_styles = ["UPPER", "lower", "Mixed"]
        num_samples = 100  # Test with multiple samples to ensure we see all styles

        # Test each level
        for level, expected_styles in enumerate(case_styles):
            self.curriculum.set_attr_level("text_case", level)
            styles_seen = set()

            # Generate multiple problems to catch all possible styles
            for _ in range(num_samples):
                problem = self.exercise.generate(self.curriculum)
                text = problem["metadata"]["executed_parts"]["clear_text"]

                # Determine the style of this text
                if text.isupper():
                    styles_seen.add("UPPER")
                elif text.islower():
                    styles_seen.add("lower")
                else:
                    styles_seen.add("Mixed")

            # At each level, we should see all styles up to that level
            expected_styles_set = set(case_styles[:level + 1])
            self.assertEqual(styles_seen, expected_styles_set,
                           f"At level {level}, expected to see styles {expected_styles_set} but saw {styles_seen}")

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
        rotations_used = defaultdict(int)
        word_counts = defaultdict(int)
        case_styles = defaultdict(int)
        total_samples = 1000

        # Generate test cases
        for _ in range(total_samples):
            # Set random attribute levels
            self.curriculum.set_attr_level("rotation", self.rng.randint(0, 4))
            self.curriculum.set_attr_level("num_words", self.rng.randint(0, 2))
            self.curriculum.set_attr_level("text_case", self.rng.randint(0, 2))

            # Generate and evaluate a random problem
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]

            # Track statistics
            rotations_used[metadata["rotation"]] += 1
            word_counts[len(metadata["clear_text"].split())] += 1

            # Determine case style
            text = metadata["clear_text"]
            if text.isupper():
                case_styles["UPPER"] += 1
            elif text.islower():
                case_styles["lower"] += 1
            else:
                case_styles["Mixed"] += 1

            # Verify encryption is correct
            cipher_text = metadata["cipher_text"]
            clear_text = metadata["clear_text"]
            rotation = metadata["rotation"]

            # Verify each character is correctly encrypted
            for c1, c2 in zip(cipher_text, clear_text):
                if c1.isalpha():
                    expected = chr((ord(c2.upper()) - ord('A') + rotation) % 26 + ord('A'))
                    self.assertEqual(c1.upper(), expected)
                else:
                    self.assertEqual(c1, c2)

        # Print statistics
        print("\nRotations used:")
        for rotation, count in sorted(rotations_used.items()):
            print(f"  Rotation {rotation}: {count}")

        print("\nWord counts:")
        for words, count in sorted(word_counts.items()):
            print(f"  {words} words: {count}")

        print("\nCase styles:")
        for style, count in case_styles.items():
            print(f"  {style}: {count}")
