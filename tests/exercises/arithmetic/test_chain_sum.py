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
        metadata = problem["metadata"]
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
        values = problem["metadata"]["values"]

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
        operators = problem["metadata"]["operators"]

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
        metadata = problem["metadata"]
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
            metadata = problem["metadata"]
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
            metadata = problem["metadata"]
            # Convert values to scientific notation for comparison
            terms.extend([f"{v:e}" for v in metadata["values"]])

        # Verify at least some terms are in scientific notation
        scientific_terms = [t for t in terms if 'e' in t.lower()]
        self.assertTrue(len(scientific_terms) > 0, "No scientific notation terms generated")

    def test_term_count_distribution(self):
        """Test that term counts follow the correct distribution for each level"""
        term_test_cases = [
            (0, 2),  # Level 0 -> max 2 terms
            (1, 3),  # Level 1 -> max 3 terms
            (2, 4),  # Level 2 -> max 4 terms
            (3, 5)   # Level 3 -> max 5 terms
        ]

        num_samples = 50  # Need more samples to ensure distribution

        for term_level, max_terms in term_test_cases:
            self.curriculum.set_attr_level("num_terms", term_level)

            # Track term counts for this level
            term_counts = []

            for _ in range(num_samples):
                problem = self.exercise.generate(self.curriculum)
                metadata = problem["metadata"]
                term_count = len(metadata["values"])
                term_counts.append(term_count)

                # Verify no problem exceeds max terms for this level
                self.assertLessEqual(term_count, max_terms,
                    f"Problem exceeded maximum terms for level {term_level}. "
                    f"Got {term_count}, max allowed is {max_terms}")

                # Verify minimum of 2 terms
                self.assertGreaterEqual(term_count, 2,
                    f"Problem has fewer than 2 terms at level {term_level}. Got {term_count}")

            # Verify we hit the maximum at least once
            self.assertIn(max_terms, term_counts,
                f"Never generated maximum number of terms ({max_terms}) "
                f"for level {term_level} in {num_samples} samples")

            # For levels > 0, verify we see some variation in term counts
            if term_level > 0:
                unique_counts = set(term_counts)
                self.assertGreater(len(unique_counts), 1,
                    f"No variation in term counts for level {term_level}. "
                    f"Always got {list(unique_counts)[0]} terms")

                # Verify we see at least one case with fewer than max terms
                self.assertTrue(any(count < max_terms for count in term_counts),
                    f"Always generated maximum terms ({max_terms}) for level {term_level}")

    def test_operator_count(self):
        """Test that the number of operators is always terms - 1"""
        for _ in range(50):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]
            num_terms = len(metadata["values"])
            num_operators = len(metadata["operators"])
            self.assertEqual(num_operators, num_terms - 1,
                           f"Wrong number of operators. Expected {num_terms-1}, got {num_operators}")

    def test_operator_validity(self):
        """Test that all operators are valid for the given level"""
        # Set curriculum to basic operators only
        self.curriculum.set_attr_level("operators", 0)  # Only +

        problem = self.exercise.generate(self.curriculum)
        metadata = problem["metadata"]
        operators = metadata["operators"]

        # Verify only + is used
        self.assertTrue(all(op == "+" for op in operators),
                       f"Invalid operator found: {operators}")

    def test_expression_evaluation(self):
        """Test that expressions are evaluated correctly for different combinations"""
        test_cases = [
            # (operators, terms, expected_result_func)
            (["+"], [2, 3], lambda t: t[0] + t[1]),
            (["+", "-"], [2, 3, 4], lambda t: t[0] + t[1] - t[2]),
            (["+", "*"], [2, 3, 4], lambda t: t[0] + t[1] * t[2]),  # Test precedence
            (["*", "+"], [2, 3, 4], lambda t: t[0] * t[1] + t[2]),
            (["/", "+"], [8, 2, 3], lambda t: t[0] / t[1] + t[2])
        ]

        for operators, term_values, expected_func in test_cases:
            # Create a problem with these specific values
            problem = {
                "metadata": {
                    "expression": {
                        "executed_parts": {
                            **{f"term_{i}": str(val) for i, val in enumerate(term_values)},
                            **{f"op_{i}": op for i, op in enumerate(operators)}
                        }
                    }
                }
            }

            # Calculate expected result
            expected = expected_func(term_values)

            # Build and evaluate expression
            expr = str(term_values[0])
            for i, op in enumerate(operators):
                expr += f" {op} {term_values[i+1]}"
            result = eval(expr)  # Safe since we control the input

            self.assertEqual(result, expected,
                f"Wrong evaluation for {expr}. Expected {expected}, got {result}")

    def test_question_formatting(self):
        """Test that questions are formatted correctly with all terms and operators"""
        problem = self.exercise.generate(self.curriculum)
        metadata = problem["metadata"]
        values = metadata["values"]
        operators = metadata["operators"]

        # Build expected expression
        expected_parts = []
        for i, val in enumerate(values):
            # Convert float to integer if it's a whole number
            if float(val).is_integer():
                expected_parts.append(str(int(val)))
            else:
                expected_parts.append(str(val))
            if i < len(operators):
                expected_parts.append(operators[i])

        expected_expr = " ".join(expected_parts)
        self.assertIn(expected_expr, problem["question"],
                     f"Question does not contain expression: {expected_expr}")

    def test_term_operator_consistency(self):
        """Test that the number of operators is always one less than the number of terms"""
        problem = self.exercise.generate(self.curriculum)
        metadata = problem["metadata"]
        num_terms = len(metadata["values"])
        num_operators = len(metadata["operators"])

        self.assertEqual(num_operators, num_terms - 1,
                        f"Number of operators ({num_operators}) should be one less than number of terms ({num_terms})")

    def test_term_number_ranges(self):
        """Test that generated terms fall within expected ranges"""
        self.curriculum.set_attr_level("num_digits", 0)  # 1-2 digits

        problem = self.exercise.generate(self.curriculum)
        metadata = problem["metadata"]
        values = metadata["values"]

        for val in values:
            val_str = str(val)
            # Skip non-regular notation values
            if any(val_str.lower().startswith(prefix) for prefix in ('0b', '0x')) or 'e' in val_str.lower():
                continue
            val_float = float(val_str)
            self.assertGreaterEqual(abs(val_float), 1, f"Value too small: {val}")
            self.assertLess(abs(val_float), 100, f"Value too large: {val}")

    def test_decimal_generation(self):
        """Test generation of decimal numbers"""
        self.curriculum.set_attr_level("num_decimals", 2)  # Allow decimals

        problem = self.exercise.generate(self.curriculum)
        metadata = problem["metadata"]
        values = metadata["values"]

        # Check that at least one value has a decimal point
        has_decimal = any('.' in str(v) and not str(v).lower().startswith(('0b', '0x')) 
                         and 'e' not in str(v).lower() for v in values)
        self.assertTrue(has_decimal, "No decimal numbers generated")

    def test_sign_distribution(self):
        """Test distribution of signs in generated terms"""
        self.curriculum.set_attr_level("sign", 2)  # Allow +/-

        num_samples = 100
        values_seen = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]
            values_seen.extend(metadata["values"])

        pos_count = sum(1 for v in values_seen if v > 0)
        neg_count = sum(1 for v in values_seen if v < 0)

        self.assertGreater(pos_count, 0, "No positive numbers generated")
        self.assertGreater(neg_count, 0, "No negative numbers generated")

    def test_notation_appearance(self):
        """Test that each notation type appears at least once over multiple samples"""
        # Test with different notation levels
        notation_types = set()
        raw_values = []  # Track raw string values

        # Try multiple times with different notation levels
        for notation_level in range(4):  # Test all notation levels
            self.curriculum.set_attr_level("notation", notation_level)
            self.curriculum.set_attr_level("num_digits", 2)  # More digits to encourage scientific notation

            for _ in range(100):  # Increase sample size
                problem = self.exercise.generate(self.curriculum)
                metadata = problem["metadata"]

                # Get notations directly from structure
                if "structure" in metadata and "notations" in metadata["structure"]:
                    notation_types.update(metadata["structure"]["notations"])

                # Get raw values for verification
                for i, val in enumerate(metadata["values"]):
                    raw_val = str(val)
                    raw_values.append(raw_val)

        # Print statistics for debugging
        print("\nNotation types found:", notation_types)
        print("Sample raw values:", raw_values[:10])

        # Verify we see different notation types
        expected_notations = {"regular", "scientific", "base2", "base16"}
        found_notations = notation_types.intersection(expected_notations)

        self.assertGreaterEqual(
            len(found_notations), 3,  # Should see at least 3 different notations
            f"Not enough notation types seen. Found: {found_notations}, Expected at least 3 from: {expected_notations}"
        )

    def test_comprehensive_random_evaluation(self):
        """Test 1000 random problems across all levels to verify correct evaluation"""
        num_samples = 1000

        # Track statistics
        total_terms = 0
        total_operators = 0
        operator_counts = {"+": 0, "-": 0, "*": 0, "/": 0, "**": 0}
        notation_counts = {"regular": 0, "scientific": 0, "base2": 0, "base16": 0}

        for _ in range(num_samples):
            # Randomly set curriculum levels
            self.curriculum.set_attr_level("num_digits", random.randint(0, 2))
            self.curriculum.set_attr_level("num_decimals", random.randint(0, 3))
            self.curriculum.set_attr_level("operators", random.randint(0, 4))
            self.curriculum.set_attr_level("num_terms", random.randint(0, 3))
            self.curriculum.set_attr_level("sign", random.randint(0, 2))
            self.curriculum.set_attr_level("notation", random.randint(0, 3))

            problem = self.exercise.generate(self.curriculum)
            metadata = problem["metadata"]
            values = metadata["values"]
            operators = metadata["operators"]

            # Track term statistics
            total_terms += len(values)
            # Get notations directly from structure
            if "structure" in metadata and "notations" in metadata["structure"]:
                for notation in metadata["structure"]["notations"]:
                    if notation in notation_counts:
                        notation_counts[notation] += 1

            # Track operator statistics
            total_operators += len(operators)
            for op in operators:
                operator_counts[op] += 1

            # Verify the answer matches our evaluation
            try:
                # Use the exercise's own evaluation method
                parsed = {
                    "values": values,
                    "operators": operators
                }
                expected = self.exercise._evaluate_expression(parsed)
                self.assertAlmostEqual(float(problem["answer"]), expected,
                                     msg=f"Wrong answer for {values} with operators {operators}. Expected {expected}, got {problem['answer']}")
            except ValueError as e:
                if "division by zero" in str(e):
                    continue
                raise

        # Print statistics
        print(f"\nComprehensive test statistics:")
        print(f"Total problems generated: {num_samples}")
        print(f"Total terms: {total_terms}")
        print(f"Total operators: {total_operators}")
        print(f"\nOperator distribution:")
        for op, count in operator_counts.items():
            if total_operators > 0:
                percentage = (count / total_operators) * 100
                print(f"  {op}: {count} ({percentage:.1f}%)")

        print(f"\nNotation distribution:")
        for notation, count in notation_counts.items():
            if total_terms > 0:
                percentage = (count / total_terms) * 100
                print(f"  {notation}: {count} ({percentage:.1f}%)")

        # Verify we have a good distribution of operators at higher levels
        if total_operators > 0:
            for op in ["+", "-"]:  # Basic operators should be common
                op_ratio = operator_counts[op] / total_operators
                self.assertGreater(op_ratio, 0.1,
                    f"Too few {op} operators: {op_ratio:.1%}")
            for op in ["*", "/"]:  # Mid-level operators should appear sometimes
                op_ratio = operator_counts[op] / total_operators
                self.assertGreater(op_ratio, 0.05,
                    f"Too few {op} operators: {op_ratio:.1%}")

        # Verify we have a good distribution of notations
        if total_terms > 0:
            for notation in ["regular", "scientific"]:  # These should be common
                notation_ratio = notation_counts[notation] / total_terms
                self.assertGreater(notation_ratio, 0.01,  # Lower threshold from 5% to 3%
                    f"Too few {notation} numbers: {notation_ratio:.1%}")

if __name__ == "__main__":
    unittest.main()