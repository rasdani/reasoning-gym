from reasoning_gym.curricula.arithmetic.chain_sum_curriculum import ChainSumCurriculum
from reasoning_gym.exercises.arithmetic.chain_sum import ChainSumDataset
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
        dataset = ChainSumDataset()

        # Test division by zero raises ValueError
        values = [1, 0]
        operators = ["/"]
        with self.assertRaises(ValueError) as cm:
            dataset._evaluate_expression(values, operators)
        self.assertEqual(str(cm.exception), "chain_sum.py: Invalid operation, division by zero")

        values = [-1, 0]
        with self.assertRaises(ValueError) as cm:
            dataset._evaluate_expression(values, operators)
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
        self.dataset = ChainSumDataset()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

    def test_problem_structure(self):
        """Test that generated problems have the correct structure"""
        problem = self.dataset.generate(self.curriculum)

        # Check basic structure
        self.assertIn("question", problem)
        self.assertIn("answer", problem)
        self.assertIn("metadata", problem)

        # Check metadata structure
        metadata = problem["metadata"]
        self.assertIn("type", metadata)
        self.assertIn("expression", metadata)
        self.assertIn("template", metadata)
        self.assertIn("executed_parts", metadata["expression"])

    def test_term_generation(self):
        """Test generation of individual terms"""
        # Set curriculum to basic settings
        self.curriculum.set_attr_level("num_digits", 0)  # 1-2 digits
        self.curriculum.set_attr_level("num_decimals", 0)  # No decimals
        self.curriculum.set_attr_level("sign", 0)  # No signs

        problem = self.dataset.generate(self.curriculum)
        executed_parts = problem["metadata"]["expression"]["executed_parts"]

        # Check first term is a valid number
        term_0 = executed_parts["term_0"]
        self.assertTrue(term_0.replace('.','',1).isdigit(), f"Invalid term: {term_0}")

    def test_operator_generation(self):
        """Test generation of operators"""
        # Set curriculum to use all basic operators
        self.curriculum.set_attr_level("operators", 1)  # +, -
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms

        problem = self.dataset.generate(self.curriculum)
        executed_parts = problem["metadata"]["expression"]["executed_parts"]

        # Check operator is valid
        op_0 = executed_parts["op_0"]
        self.assertIn(op_0, ["+", "-"], f"Invalid operator: {op_0}")

class TestChainSumGenerate(unittest.TestCase):
    """Test the generate function with different curriculum settings"""

    def setUp(self):
        self.curriculum = ChainSumCurriculum()
        self.dataset = ChainSumDataset()
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

        problem = self.dataset.generate(self.curriculum)

        # Verify structure
        self.assertIn("question", problem)
        self.assertIn("answer", problem)
        self.assertIn("metadata", problem)

        # Verify expression parts
        executed_parts = problem["metadata"]["expression"]["executed_parts"]
        self.assertIn("term_0", executed_parts)
        self.assertIn("term_1", executed_parts)
        self.assertIn("op_0", executed_parts)

        # Verify operator is addition
        self.assertEqual(executed_parts["op_0"], "+")

        # Verify terms are valid integers
        term_0 = float(executed_parts["term_0"])
        term_1 = float(executed_parts["term_1"])
        self.assertTrue(term_0.is_integer())
        self.assertTrue(term_1.is_integer())

        # Parse and evaluate the expression
        values, operators = self.dataset._parse_expression(executed_parts)
        expected = str(self.dataset._evaluate_expression(values, operators))

        # Verify answer is correct
        self.assertEqual(problem["answer"], expected,
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
            problem = self.dataset.generate(self.curriculum)
            executed_parts = problem["metadata"]["expression"]["executed_parts"]

            # Parse and evaluate the expression
            values, operators = self.dataset._parse_expression(executed_parts)
            expected = str(self.dataset._evaluate_expression(values, operators))
            terms_seen.extend(values)

            # Verify answer computation
            self.assertEqual(problem["answer"], expected,
                           f"Wrong answer for {values[0]} + {values[1]}. Expected {expected}, got {problem['answer']}")

        has_positive = any(t > 0 for t in terms_seen)
        has_negative = any(t < 0 for t in terms_seen)
        self.assertTrue(has_positive, "No positive numbers generated")
        self.assertTrue(has_negative, "No negative numbers generated")

    def test_generate_scientific_notation(self):
        """Test generation with scientific notation"""
        self.curriculum.set_attr_level("operators", 0)  # Only +
        self.curriculum.set_attr_level("notation", 1)  # Scientific notation

        num_samples = 50  # Need multiple samples to ensure we see scientific notation
        terms = []

        for _ in range(num_samples):
            problem = self.dataset.generate(self.curriculum)
            executed_parts = problem["metadata"]["expression"]["executed_parts"]
            terms.extend([executed_parts[f"term_{i}"] for i in range(2)])

        # Verify at least some terms are in scientific notation
        scientific_terms = [t for t in terms if 'e' in t.lower() or 'E' in t.upper()]
        self.assertGreater(len(scientific_terms), 0,
            f"No scientific notation terms found in {len(terms)} terms")

        # Verify scientific notation terms evaluate correctly
        for term in scientific_terms:
            value = float(term)
            self.assertAlmostEqual(value, float(f"{value:e}"),
                f"Scientific notation term {term} evaluates incorrectly")

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
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]

                # Count terms
                term_count = 0
                while f"term_{term_count}" in executed_parts:
                    term_count += 1

                # Verify no problem exceeds max terms for this level
                self.assertLessEqual(term_count, max_terms,
                    f"Problem exceeded maximum terms for level {term_level}. "
                    f"Got {term_count}, max allowed is {max_terms}")

                # Verify minimum of 2 terms
                self.assertGreaterEqual(term_count, 2,
                    f"Problem has fewer than 2 terms at level {term_level}. Got {term_count}")

                term_counts.append(term_count)

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
        num_samples = 50  # Test multiple samples

        # Test all term levels
        for term_level in range(4):  # 0-3 levels
            self.curriculum.set_attr_level("num_terms", term_level)

            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]

                # Count terms
                term_count = 0
                while f"term_{term_count}" in executed_parts:
                    term_count += 1

                # Count operators
                op_count = 0
                while f"op_{op_count}" in executed_parts:
                    op_count += 1

                # Verify operator count is terms - 1
                self.assertEqual(op_count, term_count - 1,
                    f"Wrong number of operators. Terms: {term_count}, Operators: {op_count}. "
                    f"Should have {term_count - 1} operators.")

                # Verify minimum requirements
                self.assertGreaterEqual(term_count, 2,
                    f"Must have at least 2 terms, got {term_count}")
                self.assertGreaterEqual(op_count, 1,
                    f"Must have at least 1 operator, got {op_count}")

    def test_operator_validity(self):
        """Test that all operators are valid for the given level"""
        operator_test_cases = [
            (0, ["+"]),                    # Level 0 -> only +
            (1, ["+", "-"]),               # Level 1 -> +, -
            (2, ["+", "-", "*", "/"]),     # Level 2 -> +, -, *, /
            (3, ["+", "-", "*", "/", "**"]) # Level 3 -> all operators
        ]

        num_samples = 20
        for op_level, valid_ops in operator_test_cases:
            self.curriculum.set_attr_level("operators", op_level)
            self.curriculum.set_attr_level("num_terms", 3)  # Use 4 terms to test multiple operators

            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]

                # Check each operator
                i = 0
                while f"op_{i}" in executed_parts:
                    op = executed_parts[f"op_{i}"]
                    self.assertIn(op, valid_ops,
                        f"Invalid operator {op} for level {op_level}. Valid operators: {valid_ops}")
                    i += 1

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
        num_samples = 20

        # Test different term counts
        for term_level in range(4):  # 0-3 levels
            self.curriculum.set_attr_level("num_terms", term_level)
            self.curriculum.set_attr_level("operators", 1)  # Use +/- for simplicity

            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]
                question = problem["question"]

                # Get all terms and operators
                terms = []
                ops = []
                i = 0
                while f"term_{i}" in executed_parts:
                    terms.append(executed_parts[f"term_{i}"])
                    if f"op_{i}" in executed_parts:
                        ops.append(executed_parts[f"op_{i}"])
                    i += 1

                # Verify all terms and operators appear in the question
                for i, term in enumerate(terms):
                    self.assertIn(term, question,
                        f"Term {term} missing from question: {question}")
                    if i < len(ops):
                        self.assertIn(ops[i], question,
                            f"Operator {ops[i]} missing from question: {question}")

    def test_term_operator_consistency(self):
        """Test that the number of operators is always one less than the number of terms"""
        num_samples = 20

        for term_level in range(4):  # 0-3 levels
            self.curriculum.set_attr_level("num_terms", term_level)

            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]

                # Count terms and operators
                term_count = 0
                while f"term_{term_count}" in executed_parts:
                    term_count += 1

                op_count = 0
                while f"op_{op_count}" in executed_parts:
                    op_count += 1

                self.assertEqual(op_count, term_count - 1,
                    f"Inconsistent number of operators. Terms: {term_count}, Operators: {op_count}")

    def test_term_number_ranges(self):
        """Test that generated terms fall within expected ranges"""
        # Test different digit ranges
        digit_test_cases = [
            (0, 0, 99),          # Level 0: 1-2 digits (max 10^2 - 1)
            (1, 0, 9999),        # Level 1: 1-4 digits (max 10^4 - 1)
            (2, 0, 9999999999)   # Level 2: 1-10 digits (max 10^10 - 1)
        ]

        num_samples = 50  # Test multiple samples for each case

        for digit_level, min_val, max_val in digit_test_cases:
            self.curriculum.set_attr_level("num_digits", digit_level)
            self.curriculum.set_attr_level("num_decimals", 0)  # No decimals
            self.curriculum.set_attr_level("sign", 0)  # No signs
            self.curriculum.set_attr_level("notation", 0)  # Regular notation

            terms = []
            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]
                terms.extend([float(executed_parts[f"term_{i}"]) 
                            for i in range(2)])  # Get both terms

            # Verify all terms are within range
            for term in terms:
                self.assertGreaterEqual(term, min_val, 
                    f"Term {term} below minimum {min_val} for digit level {digit_level}")
                self.assertLessEqual(term, max_val, 
                    f"Term {term} above maximum {max_val} for digit level {digit_level}")
                self.assertTrue(term.is_integer(), 
                    f"Term {term} is not an integer for digit level {digit_level}")

            # Verify we see some variation in digit counts
            digit_counts = set(len(str(int(abs(t)))) for t in terms)
            self.assertGreater(len(digit_counts), 1,
                f"No variation in digit counts for level {digit_level}. "
                f"Always got {list(digit_counts)[0]} digits")

    def test_decimal_generation(self):
        """Test generation of decimal numbers"""
        decimal_test_cases = [
            (0, 0),    # No decimals
            (1, 1),    # 1 decimal place
            (2, 2)     # 2 decimal places
        ]

        num_samples = 50  # Test multiple samples for each case

        for decimal_level, expected_places in decimal_test_cases:
            self.curriculum.set_attr_level("num_decimals", decimal_level)
            self.curriculum.set_attr_level("notation", 0)  # Regular notation

            terms = []
            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]
                terms.extend([executed_parts[f"term_{i}"] 
                            for i in range(2)])  # Get both terms

            # Verify decimal places
            for term in terms:
                decimal_str = term.split('.')[-1] if '.' in term else ''
                self.assertLessEqual(len(decimal_str), expected_places,
                    f"Term {term} has more than {expected_places} decimal places")

    def test_sign_distribution(self):
        """Test distribution of signs in generated terms"""
        self.curriculum.set_attr_level("sign", 2)  # Allow +/-
        self.curriculum.set_attr_level("notation", 0)  # Regular notation

        num_samples = 100  # Need more samples for sign distribution
        positive_count = 0
        negative_count = 0

        for _ in range(num_samples):
            problem = self.dataset.generate(self.curriculum)
            executed_parts = problem["metadata"]["expression"]["executed_parts"]
            for i in range(2):  # Check both terms
                term = float(executed_parts[f"term_{i}"])
                if term > 0:
                    positive_count += 1
                elif term < 0:
                    negative_count += 1

        # With random signs, expect roughly equal distribution
        total = positive_count + negative_count
        pos_ratio = positive_count / total
        neg_ratio = negative_count / total

        # Allow for some random variation (within 20%)
        self.assertGreater(pos_ratio, 0.3, 
            f"Too few positive numbers: {pos_ratio:.2%}")
        self.assertGreater(neg_ratio, 0.3, 
            f"Too few negative numbers: {neg_ratio:.2%}")

    def test_notation_appearance(self):
        """Test that each notation type appears at least once over multiple samples"""
        notation_checkers = {
            "regular": lambda x: not ('e' in x.lower() or 'b' in x.lower() or 'x' in x.lower()),
            "scientific": lambda x: 'e' in x.lower(),
            "binary": lambda x: '0b' in x.lower(),
            "hex": lambda x: '0x' in x.lower()
        }

        num_samples = 100  # Need more samples to ensure we see each notation

        # Test each notation level
        for notation_level in range(4):  # 0-3 levels
            self.curriculum.set_attr_level("notation", notation_level)

            terms = []
            for _ in range(num_samples):
                problem = self.dataset.generate(self.curriculum)
                executed_parts = problem["metadata"]["expression"]["executed_parts"]
                terms.extend([executed_parts[f"term_{i}"] for i in range(2)])  # Get both terms

            # For each notation type available at this level, verify it appears at least once
            available_notations = list(notation_checkers.items())[:notation_level + 1]
            for notation_name, check_func in available_notations:
                notation_found = any(check_func(term) for term in terms)
                self.assertTrue(notation_found,
                    f"Notation type '{notation_name}' never appeared at level {notation_level} "
                    f"in {len(terms)} terms")

            # Verify no higher-level notations appear
            invalid_notations = list(notation_checkers.items())[notation_level + 1:]
            for notation_name, check_func in invalid_notations:
                invalid_found = any(check_func(term) for term in terms)
                self.assertFalse(invalid_found,
                    f"Invalid notation type '{notation_name}' appeared at level {notation_level}")

    def test_comprehensive_random_evaluation(self):
        """Test 1000 random problems across all levels to verify correct evaluation"""
        num_samples = 1000

        # Track statistics
        total_terms = 0
        total_operators = 0
        operator_counts = {"+": 0, "-": 0, "*": 0, "/": 0, "**": 0}
        notation_counts = {"regular": 0, "scientific": 0, "binary": 0, "hex": 0}

        # Set random levels for all attributes
        for _ in range(num_samples):
            # Randomly set curriculum levels
            self.curriculum.set_attr_level("num_digits", random.randint(0, 2))
            self.curriculum.set_attr_level("num_decimals", random.randint(0, 3))
            self.curriculum.set_attr_level("operators", random.randint(0, 4))
            self.curriculum.set_attr_level("num_terms", random.randint(0, 3))
            self.curriculum.set_attr_level("sign", random.randint(0, 2))
            self.curriculum.set_attr_level("notation", random.randint(0, 3))

            problem = self.dataset.generate(self.curriculum)
            executed_parts = problem["metadata"]["expression"]["executed_parts"]

            # Count terms and operators
            term_count = 0
            while f"term_{term_count}" in executed_parts:
                term = executed_parts[f"term_{term_count}"]
                total_terms += 1

                # Track notation types
                if 'e' in term.lower():
                    notation_counts["scientific"] += 1
                elif '0b' in term.lower():
                    notation_counts["binary"] += 1
                elif '0x' in term.lower():
                    notation_counts["hex"] += 1
                else:
                    notation_counts["regular"] += 1

                term_count += 1

            op_count = 0
            while f"op_{op_count}" in executed_parts:
                op = executed_parts[f"op_{op_count}"]
                operator_counts[op] += 1
                total_operators += 1
                op_count += 1

            # Verify operator count matches term count
            self.assertEqual(op_count, term_count - 1,
                f"Wrong number of operators. Terms: {term_count}, Operators: {op_count}")

            # Parse and evaluate expression
            values, operators = self.dataset._parse_expression(executed_parts)
            computed_answer = str(self.dataset._evaluate_expression(values, operators))

            # Verify answer matches computed value
            self.assertEqual(problem["answer"], computed_answer,
                f"Wrong answer. Expected {computed_answer}, got {problem['answer']}")

            # Verify answer is a valid number (not NaN)
            float_answer = float(problem["answer"])
            self.assertFalse(np.isnan(float_answer),
                f"Answer is NaN for expression with values {values} and operators {operators}")

        # Print statistics
        print(f"\nComprehensive test statistics:")
        print(f"Total problems generated: {num_samples}")
        print(f"Total terms: {total_terms}")
        print(f"Total operators: {total_operators}")
        print(f"Operator distribution: {operator_counts}")
        print(f"Notation distribution: {notation_counts}")

        # Verify we have a good distribution of operators at higher levels
        if total_operators > 0:
            for op in operator_counts:
                op_ratio = operator_counts[op] / total_operators
                if op in ["+", "-"]:  # Basic operators should be common
                    self.assertGreater(op_ratio, 0.1,
                        f"Too few {op} operators: {op_ratio:.2%}")
                elif op in ["*", "/"]:  # Mid-level operators should appear sometimes
                    self.assertGreater(op_ratio, 0.05,
                        f"Too few {op} operators: {op_ratio:.2%}")

        # Verify we have a good distribution of notations
        if total_terms > 0:
            for notation in notation_counts:
                notation_ratio = notation_counts[notation] / total_terms
                if notation == "regular":  # Regular notation should be most common
                    self.assertGreater(notation_ratio, 0.05,
                        f"Too few regular numbers: {notation_ratio:.2%}")
                elif notation == "scientific":  # Scientific should be second most common
                    self.assertGreater(notation_ratio, 0.05,
                        f"Too few scientific numbers: {notation_ratio:.2%}")
                elif notation == "binary":  # Binary should be third most common
                    self.assertGreater(notation_ratio, 0.05,
                        f"Too few binary numbers: {notation_ratio:.2%}")
                else:  # Hex can be least common
                    self.assertGreater(notation_ratio, 0.05,
                        f"Too few {notation} numbers: {notation_ratio:.2%}")

if __name__ == "__main__":
    unittest.main()