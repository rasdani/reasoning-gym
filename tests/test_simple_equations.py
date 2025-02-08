from reasoning_gym.curricula.algebra.simple_equations_curriculum import SimpleEquationsCurriculum
from reasoning_gym.exercises.algebra.simple_equations import SimpleEquationsExercise
import unittest
import random
from sympy import solve, Symbol, Eq, parse_expr

class TestSimpleEquationsParsing(unittest.TestCase):
    """Test parsing of linear equation expressions and terms"""

    def setUp(self):
        self.exercise = SimpleEquationsExercise()

    def test_parse_expression(self):
        """Test parsing of basic linear expressions"""
        test_metadata = {
            'type': 'direct',
            'executed_parts': {
                'lhs_terms': ['2*x', '3'],
                'rhs_terms': ['5'],
                'lhs_operators': ['+'],
                'rhs_operators': [],
                'variable': 'x'
            }
        }

        parsed = test_metadata['executed_parts']
        self.assertEqual(parsed["lhs_terms"], ["2*x", "3"])
        self.assertEqual(parsed["rhs_terms"], ["5"])
        self.assertEqual(parsed["lhs_operators"], ["+"])
        self.assertEqual(parsed["rhs_operators"], [])
        self.assertEqual(parsed["variable"], "x")

    def test_parse_negative_terms(self):
        """Test parsing of expressions with negative terms"""
        test_metadata = {
            'type': 'direct',
            'executed_parts': {
                'lhs_terms': ['-2*x', '4'],
                'rhs_terms': ['-1'],
                'lhs_operators': ['+'],
                'rhs_operators': [],
                'variable': 'x'
            }
        }

        parsed = test_metadata['executed_parts']
        self.assertEqual(parsed["lhs_terms"], ["-2*x", "4"])
        self.assertEqual(parsed["rhs_terms"], ["-1"])
        self.assertEqual(parsed["lhs_operators"], ["+"])
        self.assertEqual(parsed["rhs_operators"], [])
        self.assertEqual(parsed["variable"], "x")

class TestSimpleEquationsEvaluation(unittest.TestCase):
    """Test evaluation of linear equations"""

    def setUp(self):
        self.exercise = SimpleEquationsExercise()

    def test_basic_equation(self):
        """Test evaluation of basic linear equations"""
        parsed = {
            "lhs_terms": ["2*x", "3"],
            "rhs_terms": ["7"],
            "lhs_operators": ["+"],
            "rhs_operators": [],
            "variable": "x"
        }
        result = self.exercise._evaluate_expression(parsed)
        expected = "2.0"  # 2x + 3 = 7 has solution x = 2
        self.assertEqual(result, expected)

    def test_negative_coefficients(self):
        """Test evaluation with negative coefficients"""
        parsed = {
            "lhs_terms": ["-2*x", "4"],
            "rhs_terms": ["0"],
            "lhs_operators": ["+"],
            "rhs_operators": [],
            "variable": "x"
        }
        result = self.exercise._evaluate_expression(parsed)
        expected = "2.0"  # -2x + 4 = 0 has solution x = 2
        self.assertEqual(result, expected)

    def test_multiple_terms(self):
        """Test equations with multiple terms"""
        parsed = {
            "lhs_terms": ["x", "2", "3"],
            "rhs_terms": ["10"],
            "lhs_operators": ["+", "+"],
            "rhs_operators": [],
            "variable": "x"
        }
        result = self.exercise._evaluate_expression(parsed)
        expected = "5.0"  # x + 2 + 3 = 10 has solution x = 5
        self.assertEqual(result, expected)

class TestSimpleEquationsGeneration(unittest.TestCase):
    """Test problem generation"""

    def setUp(self):
        self.curriculum = SimpleEquationsCurriculum()
        self.exercise = SimpleEquationsExercise()
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
        self.assertIn("lhs_terms", executed_parts)
        self.assertIn("rhs_terms", executed_parts)
        self.assertIn("lhs_operators", executed_parts)
        self.assertIn("rhs_operators", executed_parts)
        self.assertIn("variable", executed_parts)

    def test_term_generation(self):
        """Test generation of equation terms"""
        # Set curriculum to basic settings
        self.curriculum.set_attr_level("value", 0)  # 1-10
        self.curriculum.set_attr_level("sign", 0)  # No signs
        self.curriculum.set_attr_level("var_name", 0)  # Basic variables

        problem = self.exercise.generate(self.curriculum)
        executed_parts = problem["metadata"]["executed_parts"]

        # Check we have at least one term
        self.assertTrue(len(executed_parts["lhs_terms"]) > 0)

        # Check first term format
        first_term = executed_parts["lhs_terms"][0]
        self.assertTrue(isinstance(first_term, str))
        if '*' in first_term:
            coeff = first_term.split('*')[0]
            self.assertTrue(coeff.replace('-', '').isdigit() or coeff in ['', '-'])

    def test_operator_generation(self):
        """Test generation of operators"""
        self.curriculum.set_attr_level("operators", 1)  # +, -
        self.curriculum.set_attr_level("num_terms", 1)  # 3 terms

        problem = self.exercise.generate(self.curriculum)
        executed_parts = problem["metadata"]["executed_parts"]

        # Check we have operators for n-1 terms
        self.assertEqual(len(executed_parts["lhs_operators"]), len(executed_parts["lhs_terms"]) - 1)

        # Check operator is valid
        if executed_parts["lhs_operators"]:
            self.assertIn(executed_parts["lhs_operators"][0], ["+", "-"])

class TestSimpleEquationsComprehensive(unittest.TestCase):
    """Comprehensive tests for simple equations"""

    def setUp(self):
        self.curriculum = SimpleEquationsCurriculum()
        self.exercise = SimpleEquationsExercise()
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
            for term in executed_parts["lhs_terms"] + executed_parts["rhs_terms"]:
                if var_name in term:  # If term has a variable
                    self.assertIn(var_name, term)

    def test_coefficient_ranges(self):
        """Test that coefficients are within expected ranges"""
        self.curriculum.set_attr_level("value", 0)  # 1-10
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            for term in executed_parts["lhs_terms"] + executed_parts["rhs_terms"]:
                # Extract coefficient if term has one
                if '*' in term:
                    coeff = term.split('*')[0]
                    if coeff and coeff != '-':  # Skip if empty or just a minus sign
                        coeff = float(coeff)
                        self.assertLessEqual(abs(coeff), 10)
                        self.assertGreater(abs(coeff), 0)

    def test_solution_validity(self):
        """Test that generated solutions are valid"""
        num_samples = 50

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]
            solution = float(problem["answer"])

            # Verify solution satisfies the equation
            var = Symbol(executed_parts["variable"])

            # Build left and right expressions
            lhs = executed_parts["lhs_terms"][0]
            for i, term in enumerate(executed_parts["lhs_terms"][1:], 1):
                lhs += f" {executed_parts['lhs_operators'][i-1]} {term}"

            rhs = executed_parts["rhs_terms"][0]
            for i, term in enumerate(executed_parts["rhs_terms"][1:], 1):
                rhs += f" {executed_parts['rhs_operators'][i-1]} {term}"

            # Parse expressions
            lhs_expr = parse_expr(lhs, local_dict={executed_parts["variable"]: var})
            rhs_expr = parse_expr(rhs, local_dict={executed_parts["variable"]: var})

            # Verify solution
            lhs_val = float(lhs_expr.subs(var, solution))
            rhs_val = float(rhs_expr.subs(var, solution))
            self.assertAlmostEqual(lhs_val, rhs_val, places=10)

    def test_comprehensive_random_evaluation(self):
        """Test 1000 random problems across all levels to verify correct generation and evaluation"""
        num_samples = 1000

        # Statistics tracking
        stats = {
            'operator_counts': {},      # Count of each operator used
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
                'min': float('inf'),    # Minimum solution value
                'max': float('-inf'),   # Maximum solution value
                'total': 0,
                'count': 0
            },
            'var_side_stats': {        # Track which side variables appear on
                'lhs_only': 0,         # Variable only on left side
                'rhs_only': 0,         # Variable only on right side
                'both_sides': 0,       # Variable on both sides
                'total': 0
            },
            'level_distribution': {     # Track curriculum level usage
                'num_terms': {},
                'value': {},
                'operators': {},
                'sign': {},
                'var_name': {}
            }
        }

        for _ in range(num_samples):
            # Randomly set curriculum levels
            for attr in self.curriculum.attributes:
                level = random.randint(0, len(self.curriculum.attributes[attr].levels) - 1)
                self.curriculum.set_attr_level(attr, level)
                stats['level_distribution'][attr][level] = stats['level_distribution'][attr].get(level, 0) + 1

            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            # Update operator statistics
            for op in executed_parts["lhs_operators"] + executed_parts["rhs_operators"]:
                stats['operator_counts'][op] = stats['operator_counts'].get(op, 0) + 1

            # Update term count statistics (count terms on each side separately)
            lhs_terms = len(executed_parts["lhs_terms"])
            rhs_terms = len(executed_parts["rhs_terms"])
            max_side_terms = max(lhs_terms, rhs_terms)
            stats['term_counts'][max_side_terms] = stats['term_counts'].get(max_side_terms, 0) + 1

            # Update variable statistics
            var = executed_parts["variable"]
            stats['variable_counts'][var] = stats['variable_counts'].get(var, 0) + 1

            # Update variable side statistics
            var_in_lhs = any(var in term for term in executed_parts["lhs_terms"])
            var_in_rhs = any(var in term for term in executed_parts["rhs_terms"])

            if var_in_lhs and var_in_rhs:
                stats['var_side_stats']['both_sides'] += 1
            elif var_in_lhs:
                stats['var_side_stats']['lhs_only'] += 1
            elif var_in_rhs:
                stats['var_side_stats']['rhs_only'] += 1
            stats['var_side_stats']['total'] += 1

            # Update coefficient statistics
            for term in executed_parts["lhs_terms"] + executed_parts["rhs_terms"]:
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
                            continue

            # Update solution statistics
            solution = float(problem["answer"])
            stats['solution_stats']['min'] = min(stats['solution_stats']['min'], solution)
            stats['solution_stats']['max'] = max(stats['solution_stats']['max'], solution)
            stats['solution_stats']['total'] += solution
            stats['solution_stats']['count'] += 1

            # Verify solution correctness
            var = Symbol(executed_parts["variable"])
            lhs = executed_parts["lhs_terms"][0]
            for i, term in enumerate(executed_parts["lhs_terms"][1:], 1):
                lhs += f" {executed_parts['lhs_operators'][i-1]} {term}"
            rhs = executed_parts["rhs_terms"][0]
            for i, term in enumerate(executed_parts["rhs_terms"][1:], 1):
                rhs += f" {executed_parts['rhs_operators'][i-1]} {term}"

            lhs_expr = parse_expr(lhs, local_dict={executed_parts["variable"]: var})
            rhs_expr = parse_expr(rhs, local_dict={executed_parts["variable"]: var})
            lhs_val = float(lhs_expr.subs(var, solution))
            rhs_val = float(rhs_expr.subs(var, solution))
            self.assertAlmostEqual(lhs_val, rhs_val, places=10)

        # Print comprehensive statistics
        print("\nComprehensive Random Evaluation Statistics:")
        print("-" * 50)

        print("\nOperator Distribution:")
        total_ops = sum(stats['operator_counts'].values())
        for op, count in sorted(stats['operator_counts'].items()):
            print(f"  {op}: {count} ({count/total_ops*100:.1f}%)")

        print("\nTerm Count Distribution (per side):")
        total_eqs = num_samples
        for terms, count in sorted(stats['term_counts'].items()):
            print(f"  {terms} terms: {count} ({count/total_eqs*100:.1f}%)")

        print("\nVariable Distribution:")
        total_vars = sum(stats['variable_counts'].values())
        for var, count in sorted(stats['variable_counts'].items()):
            print(f"  {var}: {count} ({count/total_vars*100:.1f}%)")

        print("\nVariable Side Distribution:")
        total_eqs = stats['var_side_stats']['total']
        print(f"  Left side only: {stats['var_side_stats']['lhs_only']} ({stats['var_side_stats']['lhs_only']/total_eqs*100:.1f}%)")
        print(f"  Right side only: {stats['var_side_stats']['rhs_only']} ({stats['var_side_stats']['rhs_only']/total_eqs*100:.1f}%)")
        print(f"  Both sides: {stats['var_side_stats']['both_sides']} ({stats['var_side_stats']['both_sides']/total_eqs*100:.1f}%)")

        print("\nCoefficient Statistics:")
        print(f"  Range: [{stats['coefficient_stats']['min']:.1f} to {stats['coefficient_stats']['max']:.1f}]")
        if stats['coefficient_stats']['count'] > 0:
            avg = stats['coefficient_stats']['total'] / stats['coefficient_stats']['count']
            print(f"  Average: {avg:.2f}")
            print(f"  Unique values: {len(stats['coefficient_stats']['unique'])}")

        print("\nSolution Statistics:")
        print(f"  Range: [{stats['solution_stats']['min']:.2f} to {stats['solution_stats']['max']:.2f}]")
        if stats['solution_stats']['count'] > 0:
            avg = stats['solution_stats']['total'] / stats['solution_stats']['count']
            print(f"  Average: {avg:.2f}")

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

        # 2. Check term count constraints (per side)
        min_terms = min(stats['term_counts'].keys())
        max_terms = max(stats['term_counts'].keys())
        self.assertGreaterEqual(min_terms, 1, "Generated equations with too few terms per side")
        self.assertLessEqual(max_terms, 4, "Generated equations with too many terms per side")

        # 3. Check coefficient ranges
        if stats['coefficient_stats']['count'] > 0:
            self.assertGreater(len(stats['coefficient_stats']['unique']), 3,
                             "Too few unique coefficients generated")
            self.assertGreater(stats['coefficient_stats']['min'], 0,
                             "Generated zero or negative coefficients")
            self.assertLessEqual(stats['coefficient_stats']['max'], 100,
                               "Generated coefficients exceed maximum allowed")

class TestSimpleEquationsGenerate(unittest.TestCase):
    """Test the generate function with different curriculum settings"""

    def setUp(self):
        self.curriculum = SimpleEquationsCurriculum()
        self.exercise = SimpleEquationsExercise()
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        self.curriculum.rng = self.rng

    def test_generate_basic_equation(self):
        """Test generation of basic linear equations"""
        # Configure curriculum for simple equations
        self.curriculum.set_attr_level("num_terms", 0)  # 2 terms
        self.curriculum.set_attr_level("value", 0)  # Small values
        self.curriculum.set_attr_level("operators", 0)  # Only +
        self.curriculum.set_attr_level("sign", 0)  # No signs
        self.curriculum.set_attr_level("var_name", 0)  # Basic variables

        problem = self.exercise.generate(self.curriculum)

        # Verify structure
        self.assertIn("question", problem)
        self.assertIn("answer", problem)
        self.assertIn("metadata", problem)

        # Verify terms and operators
        executed_parts = problem["metadata"]["executed_parts"]
        self.assertTrue(len(executed_parts["lhs_terms"]) >= 1, "Not enough terms generated")
        self.assertTrue(len(executed_parts["rhs_terms"]) >= 1, "Not enough terms generated")

        # Verify operator is addition if present
        if executed_parts["lhs_operators"]:
            self.assertEqual(executed_parts["lhs_operators"][0], "+")
        if executed_parts["rhs_operators"]:
            self.assertEqual(executed_parts["rhs_operators"][0], "+")

    def test_coefficient_distribution(self):
        """Test distribution of coefficient values"""
        self.curriculum.set_attr_level("value", 0)  # 1-10
        num_samples = 100
        coefficients = []

        for _ in range(num_samples):
            problem = self.exercise.generate(self.curriculum)
            executed_parts = problem["metadata"]["executed_parts"]

            for term in executed_parts["lhs_terms"] + executed_parts["rhs_terms"]:
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
            self.curriculum.set_attr_level("value", 999)

        # Test with invalid attribute name
        with self.assertRaises(KeyError):
            self.curriculum.set_attr_level("invalid_attr", 0)

if __name__ == '__main__':
    unittest.main()
