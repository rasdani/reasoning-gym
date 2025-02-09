"""Unit tests for the base conversion exercise."""

from enum import verify
from reasoning_gym.curricula.algorithmic.base_conversion_curriculum import BaseConversionCurriculum
from reasoning_gym.exercises.algorithmic.base_conversion import BaseConversionExercise
import unittest
import random
from collections import defaultdict

class TestBaseConversionParsing(unittest.TestCase):
    """Test parsing of base conversion metadata"""

    def setUp(self):
        self.exercise = BaseConversionExercise()

    def test_parse_expression_basic(self):
        """Test parsing of basic base conversion metadata"""
        test_metadata = {
            "source_value": {"val": "1010"},
            "source_base": {"base": "binary"},
            "target_base": {"base": "hexadecimal", "hint": ""}
        }
        parsed = self.exercise._parse_expression(test_metadata)
        self.assertEqual(parsed["source_value"], "1010")
        self.assertEqual(parsed["source_base"], 2)
        self.assertEqual(parsed["target_base"], 16)

    def test_parse_base_names(self):
        """Test parsing of different base names"""
        test_cases = [
            ({"base": "binary"}, 2),
            ({"base": "octal"}, 8),
            ({"base": "decimal"}, 10),
            ({"base": "hexadecimal"}, 16),
            ({"base": "base-3"}, 3),
            ({"base": "base-36"}, 36)
        ]
        for base_dict, expected in test_cases:
            metadata = {
                "source_value": {"val": "0"},
                "source_base": base_dict,
                "target_base": {"base": "decimal", "hint": ""}
            }
            parsed = self.exercise._parse_expression(metadata)
            self.assertEqual(parsed["source_base"], expected)

    def test_invalid_base_name(self):
        """Test handling of invalid base names"""
        metadata = {
            "source_value": {"val": "0"},
            "source_base": {"base": "invalid"},
            "target_base": {"base": "decimal", "hint": ""}
        }
        with self.assertRaises(ValueError):
            self.exercise._parse_expression(metadata)

    def test_parse_with_hints(self):
        """Test parsing with different hint configurations"""
        test_cases = [
            ({"hint": ""}, ""),
            ({"hint": " (use lowercase letters a-z for digits above 9)"}, " (use lowercase letters a-z for digits above 9)"),
            ({"hint": " (hint: convert to decimal first)"}, " (hint: convert to decimal first)")
        ]
        for hint_dict, expected in test_cases:
            metadata = {
                "source_value": {"val": "0"},
                "source_base": {"base": "binary"},
                "target_base": {"base": "hexadecimal", "hint": hint_dict["hint"]}
            }
            parsed = self.exercise._parse_expression(metadata)
            self.assertEqual(parsed["source_base"], 2)
            self.assertEqual(parsed["target_base"], 16)

class TestBaseConversionEvaluation(unittest.TestCase):
    """Test evaluation of base conversion problems"""

    def setUp(self):
        self.exercise = BaseConversionExercise()

    def test_binary_to_decimal(self):
        """Test binary to decimal conversion"""
        test_cases = [
            ("1010", "10"),    # 10 in decimal
            ("1111", "15"),    # 15 in decimal
            ("10000", "16"),   # 16 in decimal
            ("0", "0"),        # 0 in any base is 0
            ("1", "1")         # 1 in any base is 1
        ]
        for binary, expected in test_cases:
            parsed = {
                "source_value": binary,
                "source_base": 2,
                "target_base": 10
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

    def test_decimal_to_hex(self):
        """Test decimal to hexadecimal conversion"""
        test_cases = [
            ("255", "ff"),     # Max 8-bit value
            ("16", "10"),      # Power of 16
            ("10", "a"),       # Single hex digit
            ("0", "0"),        # Zero
            ("4096", "1000")   # Power of 16
        ]
        for decimal, expected in test_cases:
            parsed = {
                "source_value": decimal,
                "source_base": 10,
                "target_base": 16
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

    def test_hex_to_octal(self):
        """Test hexadecimal to octal conversion"""
        test_cases = [
            ("ff", "377"),     # Max 8-bit value
            ("10", "20"),      # Simple conversion
            ("a5", "245"),     # Mixed digits and letters
            ("0", "0"),        # Zero
            ("100", "400")     # Power of 16
        ]
        for hex_val, expected in test_cases:
            parsed = {
                "source_value": hex_val,
                "source_base": 16,
                "target_base": 8
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

    def test_zero_value(self):
        """Test conversion of zero in any base"""
        bases = [2, 3, 8, 10, 16, 36]  # Test more bases
        for source_base in bases:
            for target_base in bases:
                parsed = {
                    "source_value": "0",
                    "source_base": source_base,
                    "target_base": target_base
                }
                result = self.exercise._evaluate_expression(parsed)
                self.assertEqual(result, "0")

    def test_invalid_digits(self):
        """Test handling of invalid digits for given base"""
        test_cases = [
            ("123", 2),    # Invalid binary
            ("9", 8),      # Invalid octal
            ("g", 16),     # Invalid hex
            ("z", 35)      # Invalid for base-35
        ]
        for value, base in test_cases:
            parsed = {
                "source_value": value,
                "source_base": base,
                "target_base": 10
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertTrue(result.startswith("Error"))

    def test_edge_cases(self):
        """Test edge cases and boundary values"""
        test_cases = [
            # Max values for different bases
            ("11111111", 2, 16, "ff"),          # Max 8-bit binary to hex
            ("77777777", 8, 16, "ffffff"),      # Large octal to hex
            ("ffffff", 16, 2, "111111111111111111111111"),  # Large hex to binary
            # Single digits
            ("1", 2, 36, "1"),
            ("z", 36, 2, "100011"),             # Corrected: 'z' in base-36 is 35, which is 100011 in binary
            # Alternating patterns
            ("101010", 2, 8, "52"),
            ("aaaaaa", 16, 10, "11184810")
        ]
        for value, source_base, target_base, expected in test_cases:
            parsed = {
                "source_value": value,
                "source_base": source_base,
                "target_base": target_base
            }
            result = self.exercise._evaluate_expression(parsed)
            self.assertEqual(result, expected)

class TestBaseConversionGeneration(unittest.TestCase):
    """Test problem generation"""

    def setUp(self):
        self.curriculum = BaseConversionCurriculum()
        self.exercise = BaseConversionExercise()
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
        self.assertIn("source_value", executed_parts)
        self.assertIn("source_base", executed_parts)
        self.assertIn("target_base", executed_parts)

    def test_value_ranges(self):
        """Test that generated values are within expected ranges"""
        # Test all value levels
        level_max_values = {0: 100, 1: 1000, 2: 10000}
        
        for level, max_value in level_max_values.items():
            self.curriculum.set_attr_level("value", level)
            problem = self.exercise.generate(self.curriculum)
            decimal_val = int(problem["metadata"]["executed_parts"]["source_value"], 
                            problem["metadata"]["executed_parts"]["source_base"])
            self.assertLessEqual(decimal_val, max_value)

    def test_base_ranges(self):
        """Test that bases are within expected ranges"""
        # Test all base range levels
        level_max_bases = {0: 16, 1: 26, 2: 36}
        
        for level, max_base in level_max_bases.items():
            self.curriculum.set_attr_level("base_range", level)
            problem = self.exercise.generate(self.curriculum)
            source_base = problem["metadata"]["executed_parts"]["source_base"]
            target_base = problem["metadata"]["executed_parts"]["target_base"]
            self.assertLessEqual(source_base, max_base)
            self.assertLessEqual(target_base, max_base)
            self.assertGreaterEqual(source_base, 2)
            self.assertGreaterEqual(target_base, 2)

    def test_template_variation(self):
        """Test that different templates are used"""
        templates_seen = set()
        num_samples = 100

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            templates_seen.add(problem["question"].split(":")[0])  # Get the question pattern

        self.assertGreater(len(templates_seen), 1, "Not enough template variation")

class TestBaseConversionComprehensive(unittest.TestCase):
    """Comprehensive tests for base conversion"""

    def setUp(self):
        self.curriculum = BaseConversionCurriculum()
        self.exercise = BaseConversionExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

    def _extract_base(self, text):
        """Helper method to extract base from problem text."""
        if "binary" in text.lower():
            return 2
        if "octal" in text.lower():
            return 8
        if "decimal" in text.lower():
            return 10
        if "hexadecimal" in text.lower():
            return 16

        # Try to find base-N pattern
        import re
        match = re.search(r'base-(\d+)', text.lower())
        if match:
            return int(match.group(1))
        return None

    def test_all_base_combinations(self):
        """Test conversion between all possible base combinations"""
        bases = [2, 8, 10, 16, 36]  # Test common bases
        test_values = ["10", "ff", "xyz", "777", "42"]  # Test values

        for source_base in bases:
            for target_base in bases:
                for value in test_values:
                    try:
                        # Skip if value is invalid for source base
                        int(value, min(source_base, 36))
                    except ValueError:
                        continue

                    parsed = {
                        "source_value": value,
                        "source_base": source_base,
                        "target_base": target_base
                    }
                    result = self.exercise._evaluate_expression(parsed)

                    # Verify result by converting back
                    try:
                        decimal = int(result, target_base)
                        original = int(value, source_base)
                        self.assertEqual(decimal, original)
                    except ValueError:
                        self.fail(f"Invalid conversion: {value} from base {source_base} to base {target_base}")

    def test_hint_inclusion(self):
        """Test that hints are included appropriately"""
        # Test with hints enabled
        self.curriculum.set_attr_level("hint", 0)
        problem = self.exercise.generate(self.curriculum)
        if problem["metadata"]["executed_parts"]["target_base"] > 10:
            self.assertIn("use lowercase letters", problem["question"].lower())

        # Test with hints disabled
        self.curriculum.set_attr_level("hint", 1)
        problem = self.exercise.generate(self.curriculum)
        self.assertNotIn("use lowercase letters", problem["question"].lower())

    def test_base_names(self):
        """Test that base names are used correctly"""
        # Test with basic names
        self.curriculum.set_attr_level("base_names", 0)
        problem = self.exercise.generate(self.curriculum)
        question = problem["question"].lower()
        self.assertTrue(any(name in question for name in ["binary", "hexadecimal", "base-"]))

        # Test with extended names
        self.curriculum.set_attr_level("base_names", 1)
        problem = self.exercise.generate(self.curriculum)
        question = problem["question"].lower()
        self.assertTrue(any(name in question for name in ["octal", "decimal", "base-"]))

    def test_comprehensive_random_evaluation(self):
        """Test random evaluation with all base combinations and track statistics."""
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        self.curriculum.rng = self.rng

        # Track statistics
        base_name_usage = defaultdict(int)
        source_bases = defaultdict(int)
        target_bases = defaultdict(int)
        values = []
        hint_count = 0
        total_samples = 1000

        # Generate test cases
        for _ in range(total_samples):
            # Set random attribute levels
            for attr in ["value", "base_range"]:
                self.curriculum.set_attr_level(attr, self.rng.randint(0, 2))
            for attr in ["base_names", "hint"]:
                self.curriculum.set_attr_level(attr, self.rng.randint(0, 1))

            # Generate and evaluate a random problem
            problem = self.exercise.generate(self.curriculum)

            # Track statistics
            if "binary" in problem["question"].lower():
                base_name_usage["binary"] += 1
            elif "octal" in problem["question"].lower():
                base_name_usage["octal"] += 1
            elif "hexadecimal" in problem["question"].lower():
                base_name_usage["hexadecimal"] += 1
            elif "decimal" in problem["question"].lower():
                base_name_usage["decimal"] += 1
            else:
                base_name_usage["other"] += 1

            # Track source and target bases
            metadata = problem["metadata"]["executed_parts"]
            source_base = metadata["source_base"]
            target_base = metadata["target_base"]

            if source_base:
                source_bases[source_base] += 1
            if target_base:
                target_bases[target_base] += 1

            # Track if hints are included
            if "(use lowercase letters a-z for digits above 9)" in problem["question"]:
                hint_count += 1

            # Track value statistics
            try:
                value = int(metadata["source_value"], source_base)
                values.append(value)
            except ValueError:
                pass

        # Print statistics
        print("\nBase name usage:")
        for name, count in base_name_usage.items():
            print(f"  {name}: {count}")

        print("\nSource bases used (35 bases):")
        for base in range(2, 37):
            if source_bases[base] > 0:
                print(f"  base-{base}: {source_bases[base]}")

        print("\nTarget bases used (35 bases):")
        for base in range(2, 37):
            if target_bases[base] > 0:
                print(f"  base-{base}: {target_bases[base]}")

        print("\nValue statistics:")
        if values:
            print(f"  Min value: {min(values)}")
            print(f"  Max value: {max(values)}")
            print(f"  Average value: {sum(values) / len(values):.2f}")
        print(f"  Total samples with hints: {hint_count} / {total_samples}")

        # verify statistics
        self.assertTrue(base_name_usage["hexadecimal"] >= 4, "Hexadecimal base name was not used enough")
        self.assertTrue(len(source_bases) >= 10, "Not enough different source bases used")
        self.assertTrue(len(target_bases) >= 10, "Not enough different target bases used")
        self.assertTrue(hint_count > 0, "No hints were included")
        self.assertTrue(hint_count < total_samples, "Too many hints were included")

if __name__ == '__main__':
    unittest.main()
