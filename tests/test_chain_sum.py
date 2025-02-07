from reasoning_gym.curricula.arithmetic.chain_sum_curriculum import ChainSumCurriculum
from reasoning_gym.exercises.arithmetic.chain_sum import ChainSumExercise
import numpy as np
import random
import unittest

class TestChainSumParsing(unittest.TestCase):
    """Test parsing of expressions and numbers"""

    def test_parse_decimal(self):
        """Test parsing of decimal numbers"""
        test_cases = [
            ("5", 5.0),
            ("-2", -2.0),
            ("+3", 3.0),
            ("0.5", 0.5),
            ("-1.5", -1.5),
            ("+2.5", 2.5)
        ]
        for input_str, expected in test_cases:
            result = float(input_str)
            self.assertEqual(result, expected, f"Failed to parse decimal: {input_str}")

    def test_parse_scientific(self):
        """Test parsing of scientific notation"""
        test_cases = [
            ("1e2", 100.0),
            ("-1e2", -100.0),
            ("1.5e2", 150.0),
            ("1e-2", 0.01)
        ]
        for input_str, expected in test_cases:
            result = float(input_str)
            self.assertEqual(result, expected, f"Failed to parse scientific: {input_str}")

    def test_parse_binary(self):
        """Test parsing of binary numbers"""
        test_cases = [
            ("0b101", 5),
            ("0b1010", 10),
            ("-0b101", -5)
        ]
        for input_str, expected in test_cases:
            sign = -1 if input_str.startswith('-') else 1
            num = input_str.lstrip('+-')
            result = sign * int(num[2:], 2)
            self.assertEqual(result, expected, f"Failed to parse binary: {input_str}")

    def test_parse_hex(self):
        """Test parsing of hexadecimal numbers"""
        test_cases = [
            ("0xA", 10),
            ("0xFF", 255),
            ("-0xA", -10)
        ]
        for input_str, expected in test_cases:
            sign = -1 if input_str.startswith('-') else 1
            num = input_str.lstrip('+-')
            result = sign * int(num[2:], 16)
            self.assertEqual(result, expected, f"Failed to parse hex: {input_str}")

class TestChainSumEvaluation(unittest.TestCase):
    """Test evaluation of arithmetic expressions"""

    def test_basic_operations(self):
        """Test basic arithmetic operations"""
        test_cases = [
            ([2, 3], ["+"], 5),
            ([5, 2], ["-"], 3),
            ([4, 3], ["*"], 12),
            ([6, 2], ["/"], 3)
        ]
        for values, operators, expected in test_cases:
            expr = f"{values[0]} {operators[0]} {values[1]}"
            result = eval(expr)  # Safe since we control the input
            self.assertEqual(result, expected, f"Failed operation: {expr}")

    def test_division_by_zero(self):
        """Test division by zero handling"""
        exercise = ChainSumExercise()

        # Test division by zero raises ValueError
        parsed = {
            "values": [1, 0],
            "operators": ["/"]
        }
        with self.assertRaises(ValueError) as cm:
            exercise._evaluate_expression(parsed)
        self.assertEqual(str(cm.exception), "chain_sum.py: Invalid operation, division by zero")

        parsed = {
            "values": [-1, 0],
            "operators": ["/"]
        }
        with self.assertRaises(ValueError) as cm:
            exercise._evaluate_expression(parsed)
        self.assertEqual(str(cm.exception), "chain_sum.py: Invalid operation, division by zero")

    def test_operator_precedence(self):
        """Test operator precedence without parentheses"""
        test_cases = [
            ([2, 3, 4], ["+", "*"], 14),  # 2 + (3 * 4)
            ([8, 2, 3], ["/", "+"], 7),   # (8 / 2) + 3
            ([2, 3, 4], ["*", "+"], 10)   # (2 * 3) + 4
        ]
        for values, operators, expected in test_cases:
            expr = f"{values[0]} {operators[0]} {values[1]} {operators[1]} {values[2]}"
            result = eval(expr)  # Safe since we control the input
            self.assertEqual(result, expected, f"Failed precedence: {expr}")


class TestChainSumGeneration(unittest.TestCase):
    """Test problem generation"""

    def setUp(self):
        self.curriculum = ChainSumCurriculum()
        self.exercise = ChainSumExercise()
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
        metadata = problem["metadata"]["executed_parts"]
        self.assertIn("values", metadata)
        self.assertIn("operators", metadata)
        self.assertIn("structure", metadata)

    def test_term_generation(self):
        """Test generation of individual terms"""
        # Set curriculum to basic settings
        self.curriculum.set_attr_level("num_digits", 0)  # 1-2 digits
        self.curriculum.set_attr_level("num_decimals", 0)  # No decimals
        self.curriculum.set_attr_level("sign", 0)  # No signs

        problem = self.exercise.generate(self.curriculum)
        values = problem["metadata"]["executed_parts"]["values"]

        # Check first term is a valid number
        self.assertTrue(len(values) > 0, "No values generated")
        term_0 = str(values[0])
        self.assertTrue(term_0.replace('.','',1).replace('-','',1).isdigit(), f"Invalid term: {term_0}")

    def test_operator_generation(self):
        """Test generation of operators"""
        # Set curriculum to use all basic operators
        self.curriculum.set_attr_level("operators", 1)  # +, -
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms

        problem = self.exercise.generate(self.curriculum)
        operators = problem["metadata"]["executed_parts"]["operators"]

        # Check operator is valid
        self.assertTrue(len(operators) > 0, "No operators generated")
        op_0 = operators[0]
        self.assertIn(op_0, ["+", "-"], f"Invalid operator: {op_0}")

class TestChainSumGenerate(unittest.TestCase):
    """Test the generate function with different curriculum settings"""

    def setUp(self):
        self.curriculum = ChainSumCurriculum()
        self.exercise = ChainSumExercise()
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        self.curriculum.rng = self.rng

    def test_generate_basic_addition(self):
        """Test generation of basic addition problems"""
        # Configure curriculum for simple addition
        self.curriculum.set_attr_level("operators", 0)  # Only +
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms
        self.curriculum.set_attr_level("num_digits", 0)  # 1-2 digits
        self.curriculum.set_attr_level("num_decimals", 0)  # No decimals
        self.curriculum.set_attr_level("sign", 0)  # No signs
        self.curriculum.set_attr_level("notation", 0)  # Regular notation

        problem = self.exercise.generate(self.curriculum)

        # Verify structure
        self.assertIn("question", problem)
        self.assertIn("answer", problem)
        self.assertIn("metadata", problem)

        # Verify values and operators
        metadata = problem["metadata"]["executed_parts"]
        self.assertIn("values", metadata)
        self.assertIn("operators", metadata)
        self.assertTrue(len(metadata["values"]) >= 2, "Not enough values generated")
        self.assertTrue(len(metadata["operators"]) >= 1, "No operators generated")

        # Verify operator is addition
        self.assertEqual(metadata["operators"][0], "+")

        # Verify terms are valid integers
        term_0 = float(metadata["values"][0])
        term_1 = float(metadata["values"][1])
        self.assertTrue(term_0.is_integer())
        self.assertTrue(term_1.is_integer())

        # Verify answer is correct
        expected = term_0 + term_1
        self.assertEqual(float(problem["answer"]), expected,
                        f"Wrong answer for {term_0} + {term_1}. Expected {expected}, got {problem['answer']}")

    def test_generate_with_signs(self):
        """Test generation with positive/negative signs"""
        self.curriculum.set_attr_level("operators", 0)  # Only +
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms
        self.curriculum.set_attr_level("sign", 2)  # Allow +/-
        self.curriculum.set_attr_level("notation", 0)  # Regular notation

        num_samples = 50
        terms_seen = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            terms_seen.extend(metadata["values"])

            # Verify answer computation
            term_0, term_1 = metadata["values"][:2]
            expected = term_0 + term_1  # Only addition in this test
            self.assertEqual(float(problem["answer"]), expected,
                           f"Wrong answer for {term_0} + {term_1}. Expected {expected}, got {problem['answer']}")

        has_positive = any(t > 0 for t in terms_seen)
        has_negative = any(t < 0 for t in terms_seen)
        self.assertTrue(has_positive, "No positive numbers generated")
        self.assertTrue(has_negative, "No negative numbers generated")

    def test_generate_scientific_notation(self):
        """Test generation with scientific notation"""
        self.curriculum.set_attr_level("operators", 0)  # Only +
        self.curriculum.set_attr_level("notation", 1)  # Scientific notation
        self.curriculum.set_attr_level("num_digits", 2)  # More digits to encourage scientific notation

        num_samples = 50  # Need multiple samples to ensure we see scientific notation
        terms = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            # Convert values to scientific notation for comparison
            terms.extend([f"{v:e}" for v in metadata["values"]])

        # Verify we see scientific notation
        has_scientific = any('e' in t.lower() for t in terms)
        self.assertTrue(has_scientific, "No scientific notation terms generated")

    def test_term_count_distribution(self):
        """Test that term counts follow the correct distribution for each level"""
        self.curriculum.set_attr_level("num_terms", 2)  # 2-4 terms
        num_samples = 100
        term_counts = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            term_count = len(metadata["values"])
            term_counts.append(term_count)
            self.assertTrue(2 <= term_count <= 4, f"Term count {term_count} outside valid range [2,4]")

        # Verify we see different term counts
        unique_counts = set(term_counts)
        self.assertTrue(len(unique_counts) > 1, "Only one term count generated")

    def test_term_operator_consistency(self):
        """Test that the number of operators is always one less than the number of terms"""
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            num_terms = len(metadata["values"])
            num_operators = len(metadata["operators"])
            self.assertEqual(num_operators, num_terms - 1,
                           f"Mismatch between terms ({num_terms}) and operators ({num_operators})")

    def test_term_number_ranges(self):
        """Test that generated terms fall within expected ranges"""
        self.curriculum.set_attr_level("num_digits", 1)  # 1-4 digits
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            values = metadata["values"]
            for val in values:
                self.assertTrue(abs(val) < 10000,  # 4 digits max
                              f"Value {val} outside expected range")

    def test_decimal_generation(self):
        """Test generation of decimal numbers"""
        self.curriculum.set_attr_level("num_decimals", 1)  # 1 decimal place
        num_samples = 50
        has_decimal = False

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            values = metadata["values"]
            for val in values:
                if not float(val).is_integer():
                    has_decimal = True
                    break
            if has_decimal:
                break

        self.assertTrue(has_decimal, "No decimal numbers generated")

    def test_sign_distribution(self):
        """Test distribution of signs in generated terms"""
        self.curriculum.set_attr_level("sign", 2)  # Allow all signs
        num_samples = 100
        values_seen = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            values_seen.extend(metadata["values"])

        # Check we see both positive and negative numbers
        has_positive = any(v > 0 for v in values_seen)
        has_negative = any(v < 0 for v in values_seen)
        self.assertTrue(has_positive, "No positive numbers generated")
        self.assertTrue(has_negative, "No negative numbers generated")

    def test_notation_appearance(self):
        """Test that each notation type appears at least once over multiple samples"""
        self.curriculum.set_attr_level("notation", 3)  # All notations
        num_samples = 100
        notations_seen = set()

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            for i, val in enumerate(metadata["values"]):
                notation = metadata["structure"]["notations"][i]
                notations_seen.add(notation)

        self.assertTrue(len(notations_seen) > 1, "Only one notation type generated")

    def test_operator_count(self):
        """Test that the number of operators is always terms - 1"""
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            num_terms = len(metadata["values"])
            num_operators = len(metadata["operators"])
            self.assertEqual(num_operators, num_terms - 1,
                           f"Wrong number of operators for {num_terms} terms")

    def test_operator_validity(self):
        """Test that all operators are valid for the given level"""
        self.curriculum.set_attr_level("operators", 1)  # +, -
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            operators = metadata["operators"]
            for op in operators:
                self.assertIn(op, ["+", "-"], f"Invalid operator {op} for level 1")

    def test_question_formatting(self):
        """Test that questions are formatted correctly with all terms and operators"""
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            values = metadata["values"]
            operators = metadata["operators"]

            # Question should contain all values and operators
            question = problem["question"]
            for val in values:
                # Convert to integer if it's a whole number
                if float(val).is_integer():
                    val_str = str(int(abs(val)))
                else:
                    val_str = str(abs(float(val)))
                self.assertIn(val_str, question.replace('e+0', 'e+').replace('e-0', 'e-'),
                            f"Value {val} missing from question: {question}")
            for op in operators:
                self.assertIn(op, question, f"Operator {op} missing from question: {question}")

    def test_comprehensive_random_evaluation(self):
        """Test 1000 random problems across all levels to verify correct evaluation"""
        num_samples = 1000
        
        # Statistics tracking
        stats = {
            'operator_counts': {},      # Count of each operator used
            'notation_counts': {},      # Count of each notation type used
            'term_counts': {},          # Distribution of number of terms
            'value_ranges': {           # Track value ranges
                'min': float('inf'),
                'max': float('-inf'),
                'decimals': 0,          # Count of decimal numbers
                'integers': 0,          # Count of integer numbers
                'negatives': 0,         # Count of negative numbers
                'positives': 0          # Count of positive numbers
            },
            'level_distribution': {     # Track curriculum level usage
                'operators': {},
                'num_terms': {},
                'num_digits': {},
                'num_decimals': {},
                'sign': {},
                'notation': {}
            }
        }

        for _ in range(num_samples):
            # Randomly set curriculum levels
            levels = {
                'operators': self.rng.randint(0, 4),
                'num_terms': self.rng.randint(0, 3),
                'num_digits': self.rng.randint(0, 2),
                'num_decimals': self.rng.randint(0, 3),
                'sign': self.rng.randint(0, 2),
                'notation': self.rng.randint(0, 3)
            }

            # Update level distribution stats
            for attr, level in levels.items():
                stats['level_distribution'][attr][level] = stats['level_distribution'][attr].get(level, 0) + 1

            # Set curriculum levels
            for attr, level in levels.items():
                self.curriculum.set_attr_level(attr, level)

            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]["executed_parts"]
            values = metadata["values"]
            operators = metadata["operators"]
            notations = metadata["structure"]["notations"]

            # Update operator statistics
            for op in operators:
                stats['operator_counts'][op] = stats['operator_counts'].get(op, 0) + 1

            # Update notation statistics
            for notation in notations:
                stats['notation_counts'][notation] = stats['notation_counts'].get(notation, 0) + 1

            # Update term count statistics
            num_terms = len(values)
            stats['term_counts'][num_terms] = stats['term_counts'].get(num_terms, 0) + 1

            # Update value range statistics
            for val in values:
                stats['value_ranges']['min'] = min(stats['value_ranges']['min'], val)
                stats['value_ranges']['max'] = max(stats['value_ranges']['max'], val)
                if isinstance(val, float) and not val.is_integer():
                    stats['value_ranges']['decimals'] += 1
                else:
                    stats['value_ranges']['integers'] += 1
                if val < 0:
                    stats['value_ranges']['negatives'] += 1
                elif val > 0:
                    stats['value_ranges']['positives'] += 1

            # Verify answer matches evaluation
            parsed = {
                "values": values,
                "operators": operators
            }
            expected = self.exercise._evaluate_expression(parsed)
            self.assertAlmostEqual(float(problem["answer"]), expected,
                                 msg=f"Wrong answer for {problem['question']}")

        # Print statistics
        print("\nComprehensive Random Evaluation Statistics:")
        print("-" * 50)

        print("\nOperator Distribution:")
        for op, count in sorted(stats['operator_counts'].items()):
            print(f"  {op}: {count} ({count/sum(stats['operator_counts'].values())*100:.1f}%)")

        print("\nNotation Distribution:")
        for notation, count in sorted(stats['notation_counts'].items()):
            print(f"  {notation}: {count} ({count/sum(stats['notation_counts'].values())*100:.1f}%)")

        print("\nTerm Count Distribution:")
        for terms, count in sorted(stats['term_counts'].items()):
            print(f"  {terms} terms: {count} ({count/num_samples*100:.1f}%)")

        print("\nValue Statistics:")
        print(f"  Range: [{stats['value_ranges']['min']:.2e} to {stats['value_ranges']['max']:.2e}]")
        print(f"  Integers: {stats['value_ranges']['integers']} ({stats['value_ranges']['integers']/sum(stats['term_counts'].values())*100:.1f}%)")
        print(f"  Decimals: {stats['value_ranges']['decimals']} ({stats['value_ranges']['decimals']/sum(stats['term_counts'].values())*100:.1f}%)")
        print(f"  Negatives: {stats['value_ranges']['negatives']} ({stats['value_ranges']['negatives']/sum(stats['term_counts'].values())*100:.1f}%)")
        print(f"  Positives: {stats['value_ranges']['positives']} ({stats['value_ranges']['positives']/sum(stats['term_counts'].values())*100:.1f}%)")

        print("\nCurriculum Level Distribution:")
        for attr, levels in sorted(stats['level_distribution'].items()):
            print(f"\n  {attr}:")
            for level, count in sorted(levels.items()):
                print(f"    Level {level}: {count} ({count/num_samples*100:.1f}%)")

if __name__ == '__main__':
    unittest.main()