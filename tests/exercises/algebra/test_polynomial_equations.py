from reasoning_gym.curricula.algebra.polynomial_equations_curriculum import PolynomialEquationsCurriculum
from reasoning_gym.exercises.algebra.polynomial_equations import PolynomialEquationsExercise
import unittest
import random
from sympy import solve, Symbol, Eq, parse_expr

class TestPolynomialEquationsParsing(unittest.TestCase):
    """Test parsing of polynomial expressions and terms"""

    def setUp(self):
        self.exercise = PolynomialEquationsExercise()

    def test_parse_expression(self):
        """Test parsing of polynomial expressions"""
        test_metadata = {
            'type': 'direct',
            'executed_parts': {
                'terms': ['2*x**2', '3*x', '1'],
                'operators': ['+', '+'],
                'variable': 'x'
            }
        }

        parsed = test_metadata['executed_parts']
        self.assertEqual(parsed["terms"], ["2*x**2", "3*x", "1"])
        self.assertEqual(parsed["operators"], ["+", "+"])
        self.assertEqual(parsed["variable"], "x")

    def test_parse_negative_terms(self):
        """Test parsing of expressions with negative terms"""
        test_metadata = {
            'type': 'direct',
            'executed_parts': {
                'terms': ['-2*x**2', '4*x'],
                'operators': ['+'],
                'variable': 'x'
            }
        }

        parsed = test_metadata['executed_parts']
        self.assertEqual(parsed["terms"], ["-2*x**2", "4*x"])
        self.assertEqual(parsed["operators"], ["+"])
        self.assertEqual(parsed["variable"], "x")

class TestPolynomialEquationsEvaluation(unittest.TestCase):
    """Test evaluation of polynomial equations"""

    def setUp(self):
        self.exercise = PolynomialEquationsExercise()

    def test_quadratic_equation(self):
        """Test evaluation of quadratic equations"""
        parsed = {
            "terms": ["x**2", "-5*x", "6"],
            "operators": ["+", "+"],
            "variable": "x"
        }
        result = self.exercise._evaluate_expression(parsed)
        expected = "[2.0, 3.0]"  # x^2 - 5x + 6 = 0 has roots at x = 2 and x = 3
        self.assertEqual(result, expected)

    def test_linear_equation(self):
        """Test evaluation of linear equations"""
        parsed = {
            "terms": ["2*x", "-4"],
            "operators": ["+"],
            "variable": "x"
        }
        result = self.exercise._evaluate_expression(parsed)
        expected = "[2.0]"  # 2x - 4 = 0 has root at x = 2
        self.assertEqual(result, expected)

    def test_no_real_solutions(self):
        """Test equations with no real solutions"""
        parsed = {
            "terms": ["x**2", "1"],
            "operators": ["+"],
            "variable": "x"
        }
        result = self.exercise._evaluate_expression(parsed)
        expected = "[]"  # x^2 + 1 = 0 has no real solutions
        self.assertEqual(result, expected)

class TestPolynomialEquationsGeneration(unittest.TestCase):
    """Test problem generation"""

    def setUp(self):
        self.curriculum = PolynomialEquationsCurriculum()
        self.exercise = PolynomialEquationsExercise()
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
        self.assertIn("terms", executed_parts)
        self.assertIn("operators", executed_parts)
        self.assertIn("variable", executed_parts)

    def test_term_generation(self):
        """Test generation of polynomial terms"""
        # Set curriculum to basic settings
        self.curriculum.set_attr_level("coefficient_value", 0)  # 1-10
        self.curriculum.set_attr_level("max_degree", 0)  # degree 1
        self.curriculum.set_attr_level("sign", 0)  # No signs

        problem = self.exercise.generate(self.curriculum)
        executed_parts = problem["metadata"]["executed_parts"]

        # Check we have at least one term
        self.assertTrue(len(executed_parts["terms"]) > 0)

        # Check first term format
        first_term = executed_parts["terms"][0]
        self.assertTrue(isinstance(first_term, str))
        self.assertTrue(first_term.replace('*', '').replace('x', '').replace('-', '').replace('.', '').isdigit() or
                       first_term == 'x')

    def test_operator_generation(self):
        """Test generation of operators"""
        self.curriculum.set_attr_level("operators", 1)  # +, -
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms

        problem = self.exercise.generate(self.curriculum)
        executed_parts = problem["metadata"]["executed_parts"]

        # Check we have operators for n-1 terms
        self.assertEqual(len(executed_parts["operators"]), len(executed_parts["terms"]) - 1)

        # Check operator is valid
        if executed_parts["operators"]:
            self.assertIn(executed_parts["operators"][0], ["+", "-"])

class TestPolynomialEquationsComprehensive(unittest.TestCase):
    """Comprehensive tests for polynomial equations"""

    def setUp(self):
        self.curriculum = PolynomialEquationsCurriculum()
        self.exercise = PolynomialEquationsExercise()
        self.rng = random.Random(42)
        self.curriculum.rng = self.rng

    def test_variable_consistency(self):
        """Test that the same variable is used consistently throughout the equation"""
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            var_name = executed_parts["variable"]

            # Check variable appears in question
            self.assertIn(var_name, problem["question"])

            # Check variable is used consistently in terms
            for term in executed_parts["terms"]:
                if var_name in term:  # If term has a variable
                    self.assertIn(var_name, term)

    def test_coefficient_ranges(self):
        """Test that coefficients are within expected ranges"""
        self.curriculum.set_attr_level("coefficient_value", 0)  # 1-10
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            for term in executed_parts["terms"]:
                # Extract coefficient if term has one
                if '*' in term:
                    coeff = term.split('*')[0]
                    if coeff and coeff != '-':  # Skip if empty or just a minus sign
                        coeff = float(coeff)
                        self.assertLessEqual(abs(coeff), 10)
                        self.assertGreater(abs(coeff), 0)

    def test_degree_constraints(self):
        """Test that polynomial degrees respect the curriculum settings"""
        self.curriculum.set_attr_level("max_degree", 0)  # Level 0 means max degree 1
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            max_degree = 0
            for term in executed_parts["terms"]:
                if "**" in term:
                    degree = int(term.split("**")[1])
                    max_degree = max(max_degree, degree)
                elif executed_parts["variable"] in term:  # Variable without exponent means degree 1
                    max_degree = max(max_degree, 1)

            self.assertLessEqual(max_degree, 1)

    def test_solution_validity(self):
        """Test that generated solutions are valid"""
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            # Parse the answer string to get solutions
            solutions = eval(problem["answer"])  # Safe since we control the input

            if solutions:  # If there are real solutions
                # Verify each solution satisfies the equation
                var = Symbol(executed_parts["variable"])
                expr = executed_parts["terms"][0]

                # Reconstruct the expression
                for i, term in enumerate(executed_parts["terms"][1:], 1):
                    expr += f" {executed_parts['operators'][i-1]} {term}"

                # Verify each solution
                sympy_expr = parse_expr(expr)
                for sol in solutions:
                    result = abs(float(sympy_expr.subs(var, sol)))
                    self.assertAlmostEqual(result, 0, places=10)

    def test_comprehensive_random_evaluation(self):
        """Test 1000 random problems across all levels to verify correct generation and evaluation"""
        num_samples = 1000

        # Statistics tracking
        stats = {
            'operator_counts': {},      # Count of each operator used
            'degree_counts': {},        # Count of polynomial degrees
            'term_counts': {},          # Distribution of number of terms
            'variable_counts': {},      # Count of each variable used
            'coefficient_stats': {      # Track coefficient statistics
                'min': float('inf'),
                'max': float('-inf'),
                'total': 0,
                'count': 0,
                'unique': set()
            },
            'solution_stats': {         # Track solution statistics
                'no_solutions': 0,      # Count of equations with no real solutions
                'one_solution': 0,      # Count of equations with exactly one solution
                'two_solutions': 0,     # Count of equations with exactly two solutions
                'min': float('inf'),    # Minimum solution value
                'max': float('-inf'),   # Maximum solution value
            },
            'level_distribution': {     # Track curriculum level usage
                'max_degree': {},
                'num_terms': {},
                'coefficient_value': {},
                'operators': {},
                'sign': {},
                'var_name': {}
            }
        }

        for _ in range(num_samples):
            # Randomly set curriculum levels
            levels = {
                'max_degree': self.rng.randint(0, 2),
                'num_terms': self.rng.randint(0, 2),
                'coefficient_value': self.rng.randint(0, 2),
                'operators': self.rng.randint(0, 1),
                'sign': self.rng.randint(0, 1),
                'var_name': self.rng.randint(0, 2)
            }

            # Update level distribution stats
            for attr, level in levels.items():
                stats['level_distribution'][attr][level] = stats['level_distribution'][attr].get(level, 0) + 1

            # Set curriculum levels
            for attr, level in levels.items():
                self.curriculum.set_attr_level(attr, level)

            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            terms = executed_parts["terms"]
            operators = executed_parts["operators"]
            variable = executed_parts["variable"]

            # Update operator statistics
            for op in operators:
                stats['operator_counts'][op] = stats['operator_counts'].get(op, 0) + 1

            # Update term count statistics
            num_terms = len(terms)
            stats['term_counts'][num_terms] = stats['term_counts'].get(num_terms, 0) + 1

            # Update variable statistics
            stats['variable_counts'][variable] = stats['variable_counts'].get(variable, 0) + 1

            # Calculate and update degree statistics
            max_degree = 0
            for term in terms:
                if "**" in term:
                    degree = int(term.split("**")[1])
                    max_degree = max(max_degree, degree)
                elif variable in term:  # Variable without exponent means degree 1
                    max_degree = max(max_degree, 1)
            stats['degree_counts'][max_degree] = stats['degree_counts'].get(max_degree, 0) + 1

            # Update coefficient statistics
            for term in terms:
                if '*' in term:
                    coeff = term.split('*')[0]
                    if coeff and coeff not in ['-', '+']:
                        try:
                            value = abs(float(coeff))
                            stats['coefficient_stats']['min'] = min(stats['coefficient_stats']['min'], value)
                            stats['coefficient_stats']['max'] = max(stats['coefficient_stats']['max'], value)
                            stats['coefficient_stats']['total'] += value
                            stats['coefficient_stats']['count'] += 1
                            stats['coefficient_stats']['unique'].add(value)
                        except ValueError:
                            # Skip if coefficient is not a number (e.g., just a variable)
                            continue

            # Update solution statistics
            solutions = eval(problem["answer"])  # Safe since we control the input
            num_solutions = len(solutions)
            if num_solutions == 0:
                stats['solution_stats']['no_solutions'] += 1
            elif num_solutions == 1:
                stats['solution_stats']['one_solution'] += 1
                stats['solution_stats']['min'] = min(stats['solution_stats']['min'], solutions[0])
                stats['solution_stats']['max'] = max(stats['solution_stats']['max'], solutions[0])
            elif num_solutions == 2:
                stats['solution_stats']['two_solutions'] += 1
                stats['solution_stats']['min'] = min(stats['solution_stats']['min'], min(solutions))
                stats['solution_stats']['max'] = max(stats['solution_stats']['max'], max(solutions))

            # Verify solution correctness
            if solutions:  # If there are real solutions
                var = Symbol(variable)
                expr = terms[0]
                for i, term in enumerate(terms[1:], 1):
                    expr += f" {operators[i-1]} {term}"

                # Create local dict with the variable symbol
                local_dict = {variable: var}
                sympy_expr = parse_expr(expr, local_dict=local_dict)
                for sol in solutions:
                    result = abs(float(sympy_expr.subs(var, sol)))
                    self.assertAlmostEqual(result, 0, places=10)

        # Print comprehensive statistics
        print("\nComprehensive Random Evaluation Statistics:")
        print("-" * 50)

        print("\nOperator Distribution:")
        total_ops = sum(stats['operator_counts'].values())
        for op, count in sorted(stats['operator_counts'].items()):
            print(f"  {op}: {count} ({count/total_ops*100:.1f}%)")

        print("\nDegree Distribution:")
        total_eqs = num_samples
        for degree, count in sorted(stats['degree_counts'].items()):
            print(f"  Degree {degree}: {count} ({count/total_eqs*100:.1f}%)")

        print("\nTerm Count Distribution:")
        for terms, count in sorted(stats['term_counts'].items()):
            print(f"  {terms} terms: {count} ({count/total_eqs*100:.1f}%)")

        print("\nVariable Distribution:")
        total_vars = sum(stats['variable_counts'].values())
        for var, count in sorted(stats['variable_counts'].items()):
            print(f"  {var}: {count} ({count/total_vars*100:.1f}%)")

        print("\nCoefficient Statistics:")
        print(f"  Range: [{stats['coefficient_stats']['min']:.1f} to {stats['coefficient_stats']['max']:.1f}]")
        if stats['coefficient_stats']['count'] > 0:
            avg = stats['coefficient_stats']['total'] / stats['coefficient_stats']['count']
            print(f"  Average: {avg:.2f}")
            print(f"  Unique values: {len(stats['coefficient_stats']['unique'])}")

        print("\nSolution Statistics:")
        print(f"  No real solutions: {stats['solution_stats']['no_solutions']} ({stats['solution_stats']['no_solutions']/total_eqs*100:.1f}%)")
        print(f"  One solution: {stats['solution_stats']['one_solution']} ({stats['solution_stats']['one_solution']/total_eqs*100:.1f}%)")
        print(f"  Two solutions: {stats['solution_stats']['two_solutions']} ({stats['solution_stats']['two_solutions']/total_eqs*100:.1f}%)")
        if stats['solution_stats']['min'] != float('inf'):
            print(f"  Solution range: [{stats['solution_stats']['min']:.2f} to {stats['solution_stats']['max']:.2f}]")

        print("\nCurriculum Level Distribution:")
        for attr, levels in sorted(stats['level_distribution'].items()):
            print(f"\n  {attr}:")
            for level, count in sorted(levels.items()):
                print(f"    Level {level}: {count} ({count/total_eqs*100:.1f}%)")

        # Verify statistical properties
        # 1. Check we see all operators when using operator level 1
        if any(level == 1 for level in stats['level_distribution']['operators'].keys()):
            self.assertTrue(all(op in stats['operator_counts'] for op in ["+", "-"]),
                          "Not all operators were generated")

        # 2. Check degree distribution matches curriculum settings
        max_possible_degree = max(stats['degree_counts'].keys())
        self.assertLessEqual(max_possible_degree, 3, "Generated degree exceeds maximum allowed")

        # 3. Check term count constraints
        min_terms = min(stats['term_counts'].keys())
        max_terms = max(stats['term_counts'].keys())
        self.assertGreaterEqual(min_terms, 2, "Generated equations with too few terms")
        self.assertLessEqual(max_terms, 4, "Generated equations with too many terms")

        # 4. Check coefficient ranges
        if stats['coefficient_stats']['count'] > 0:
            self.assertGreater(len(stats['coefficient_stats']['unique']), 3,
                             "Too few unique coefficients generated")
            self.assertGreater(stats['coefficient_stats']['min'], 0,
                             "Generated zero or negative coefficients")
            self.assertLessEqual(stats['coefficient_stats']['max'], 100,
                               "Generated coefficients exceed maximum allowed")

        # 5. Check solution distribution
        total_with_solutions = stats['solution_stats']['one_solution'] + stats['solution_stats']['two_solutions']
        if total_with_solutions > 0:
            self.assertGreater(stats['solution_stats']['one_solution'], 0,
                             "No equations with exactly one solution generated")
            self.assertGreater(stats['solution_stats']['two_solutions'], 0,
                             "No equations with exactly two solutions generated")

class TestPolynomialEquationsGenerate(unittest.TestCase):
    """Test the generate function with different curriculum settings"""

    def setUp(self):
        self.curriculum = PolynomialEquationsCurriculum()
        self.exercise = PolynomialEquationsExercise()
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        self.curriculum.rng = self.rng

    def test_generate_basic_linear(self):
        """Test generation of basic linear equations"""
        # Configure curriculum for simple linear equations
        self.curriculum.set_attr_level("max_degree", 0)  # Linear equations
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms
        self.curriculum.set_attr_level("coefficient_value", 0)  # Small coefficients
        self.curriculum.set_attr_level("sign", 0)  # No signs
        self.curriculum.set_attr_level("operators", 0)  # Only +

        problem = self.exercise.generate(self.curriculum)

        # Verify structure
        self.assertIn("question", problem)
        self.assertIn("answer", problem)
        self.assertIn("metadata", problem)

        # Verify terms and operators
        executed_parts = problem["metadata"]["executed_parts"]
        self.assertTrue(len(executed_parts["terms"]) >= 2, "Not enough terms generated")
        self.assertTrue(len(executed_parts["operators"]) >= 1, "No operators generated")

        # Verify operator is addition
        self.assertEqual(executed_parts["operators"][0], "+")

        # Verify terms have correct degree
        for term in executed_parts["terms"]:
            self.assertNotIn("**", term, "Term should not have exponent > 1")

    def test_generate_with_signs(self):
        """Test generation with positive/negative signs"""
        self.curriculum.set_attr_level("operators", 0)  # Only +
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms
        self.curriculum.set_attr_level("sign", 1)  # Allow -
        self.curriculum.set_attr_level("max_degree", 0)  # Linear equations

        num_samples = 50
        terms_seen = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            terms_seen.extend(executed_parts["terms"])

        # Check we see both positive and negative terms
        has_negative = any(term.startswith('-') for term in terms_seen)
        has_positive = any(not term.startswith('-') for term in terms_seen)
        self.assertTrue(has_positive, "No positive terms generated")
        self.assertTrue(has_negative, "No negative terms generated")

    def test_term_count_distribution(self):
        """Test that term counts follow the correct distribution"""
        self.curriculum.set_attr_level("num_terms", 2)  # 2-4 terms
        num_samples = 100
        term_counts = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            term_count = len(executed_parts["terms"])
            term_counts.append(term_count)
            self.assertTrue(2 <= term_count <= 4, f"Term count {term_count} outside valid range [2,4]")

        # Verify we see different term counts
        unique_counts = set(term_counts)
        self.assertTrue(len(unique_counts) > 1, "Only one term count generated")

    def test_operator_distribution(self):
        """Test distribution of operators"""
        self.curriculum.set_attr_level("operators", 1)  # +, -
        num_samples = 100
        operators_seen = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            operators_seen.extend(executed_parts["operators"])

        # Check we see both operators
        has_plus = "+" in operators_seen
        has_minus = "-" in operators_seen
        self.assertTrue(has_plus, "No + operators generated")
        self.assertTrue(has_minus, "No - operators generated")

    def test_variable_distribution(self):
        """Test distribution of variable names"""
        self.curriculum.set_attr_level("var_name", 0)  # x, y, z
        num_samples = 100
        variables_seen = set()

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            variables_seen.add(executed_parts["variable"])

        # Check we see multiple variables
        self.assertTrue(len(variables_seen) > 1, "Only one variable name generated")
        self.assertTrue(all(var in "xyz" for var in variables_seen), 
                       f"Invalid variables generated: {variables_seen}")

    def test_coefficient_distribution(self):
        """Test distribution of coefficient values"""
        self.curriculum.set_attr_level("coefficient_value", 0)  # 1-10
        num_samples = 100
        coefficients = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            for term in executed_parts["terms"]:
                if '*' in term:
                    coeff = term.split('*')[0]
                    if coeff and coeff not in ['-', '+']:
                        coefficients.append(abs(float(coeff)))

        # Check coefficient range
        self.assertTrue(all(1 <= c <= 10 for c in coefficients), 
                       "Coefficients outside valid range [1,10]")
        # Check we see different values
        unique_coeffs = set(coefficients)
        self.assertTrue(len(unique_coeffs) > 3, 
                       f"Too few unique coefficients: {unique_coeffs}")

    def test_error_handling(self):
        """Test error handling in equation generation"""
        # Test with invalid attribute level
        with self.assertRaises(ValueError):
            self.curriculum.set_attr_level("max_degree", 999)

        # Test with invalid attribute name
        with self.assertRaises(KeyError):
            self.curriculum.set_attr_level("invalid_attr", 0)

if __name__ == '__main__':
    unittest.main()
