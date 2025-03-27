"""Tests for intermediate integration task generation"""

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr

from reasoning_gym.algebra.intermediate_integration import IntermediateIntegrationConfig, IntermediateIntegrationDataset


def test_intermediate_integration_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(linear_lower_bound=2, linear_upper_bound=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(linear_lower_bound=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_linear_degree=5, max_linear_degree=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_linear_degree=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(outer_constant_min=5, outer_constant_max=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(outer_constant_min=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_poly_degree=5, max_poly_degree=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_poly_degree=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(symbols=("x", "y"))
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(operators=("+", "-", "*", "/"))
        config.validate()


def test_intermediate_integration_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = IntermediateIntegrationConfig(seed=42, size=10)
    dataset1 = IntermediateIntegrationDataset(config)
    dataset2 = IntermediateIntegrationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_intermediate_integration_dataset_items():
    """Test that dataset items are valid"""
    config = IntermediateIntegrationConfig(seed=42, size=10)
    dataset = IntermediateIntegrationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert "integrand" in item["metadata"]
        assert "problem_type" in item["metadata"]
        assert "variable" in item["metadata"]
        # verify answer is mathematical expression
        answer = item["answer"]
        answer = answer.replace(" + C", "")
        assert isinstance(parse_expr(answer), sympy.Expr)


def test_verify_answer():
    config = IntermediateIntegrationConfig(seed=42)
    dataset = IntermediateIntegrationDataset(config)
    for i in range(len(dataset)):
        item = dataset[i]
        score = dataset.score_answer(answer=item["answer"], entry=item)
        assert score == 1.0


def test_score_answer_cases():
    """Test various answer scoring scenarios"""
    config = IntermediateIntegrationConfig(seed=42)
    dataset = IntermediateIntegrationDataset(config)
    x = sympy.Symbol("x")
    X = sympy.Symbol("X")

    # Test cases: (answer, metadata, expected_score)
    test_cases = [
        # Correct answers
        ("x**2 + C", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("X**3 - 5*X + C", {"variable": "X", "integrand": "3*X**2 - 5"}, 1.0),
        ("sin(x) + C", {"variable": "x", "integrand": "cos(x)"}, 1.0),
        # Correct without explicit constant
        ("x**2", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("log(x)", {"variable": "x", "integrand": "1/x"}, 1.0),
        # Incorrect but properly formatted
        ("x**3 + C", {"variable": "x", "integrand": "2*x"}, 0.0),
        ("cos(X)", {"variable": "X", "integrand": "sin(X)"}, 0.0),
        # Malformed expressions
        ("x**2 +", {"variable": "x", "integrand": "2*x"}, 0.0),
        ("sin(x", {"variable": "x", "integrand": "cos(x)"}, 0.0),
        # Empty answer
        ("", {"variable": "x", "integrand": "2*x"}, 0.0),
        # Case sensitivity
        ("x**2 + C", {"variable": "X", "integrand": "2*X"}, 0.0),
        ("X**2 + C", {"variable": "x", "integrand": "2*x"}, 0.0),
        # Alternative constant notation
        ("x**2 + K", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("sin(x) + D", {"variable": "x", "integrand": "cos(x)"}, 1.0),
        # Simplification required
        ("x**2 + C + 5 - 5", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("(x**3)/3 - 2*x + C", {"variable": "x", "integrand": "x**2 - 2"}, 1.0),
    ]

    for answer, metadata, expected in test_cases:
        dummy_entry = {"metadata": metadata}
        score = dataset.score_answer(answer, entry=dummy_entry)
        assert score == expected, f"Failed case: {answer} | Expected {expected}, got {score}"


def test_intermediate_integration_curriculum():
    """Test the IntermediateIntegrationCurriculum functionality"""
    from reasoning_gym.algebra.intermediate_integration import (
        IntermediateIntegrationConfig,
        IntermediateIntegrationCurriculum,
    )

    # Create a config for the curriculum
    config = IntermediateIntegrationConfig(
        size=150, seed=1, problem_type_weights=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    )

    curriculum = IntermediateIntegrationCurriculum()

    # Test initial configuration
    base_cfg = curriculum.generate_configuration({})
    assert base_cfg.problem_type_weights == [1, 0, 0, 0, 0, 0, 0, 0]  # Default level 0

    # Test incrementing problem_type_weights attribute
    curriculum.increment_attr_level("problem_type_weights")
    level1_cfg = curriculum.generate_configuration({})
    assert level1_cfg.problem_type_weights == [0, 1, 0, 0, 0, 0, 0, 0]  # Level 1

    # Test incrementing problem_type_weights attribute again
    curriculum.increment_attr_level("problem_type_weights")
    level2_cfg = curriculum.generate_configuration({})
    assert level2_cfg.problem_type_weights == [0, 0, 1, 0, 0, 0, 0, 0]  # Level 2

    # Test decrementing problem_type_weights attribute
    curriculum.decrement_attr_level("problem_type_weights")
    back_to_level1_cfg = curriculum.generate_configuration({})
    assert back_to_level1_cfg.problem_type_weights == [0, 1, 0, 0, 0, 0, 0, 0]  # Back to level 1

    # Test global level adjustments
    # Reset curriculum
    curriculum = IntermediateIntegrationCurriculum()
    assert curriculum.get_attr_level("problem_type_weights") == 0

    # Increase global level
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("problem_type_weights") == 1

    global_level_cfg = curriculum.generate_configuration({})
    assert global_level_cfg.problem_type_weights == [0, 1, 0, 0, 0, 0, 0, 0]

    # Increase global level again
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("problem_type_weights") == 2

    global_level_cfg_2 = curriculum.generate_configuration({})
    assert global_level_cfg_2.problem_type_weights == [0, 0, 1, 0, 0, 0, 0, 0]

    # Decrease global level
    curriculum.decrement_global_level()
    assert curriculum.get_attr_level("problem_type_weights") == 1

    global_level_cfg_3 = curriculum.generate_configuration({})
    assert global_level_cfg_3.problem_type_weights == [0, 1, 0, 0, 0, 0, 0, 0]

    # Test upper bound
    curriculum = IntermediateIntegrationCurriculum()  # Reset curriculum
    for _ in range(10):  # Try going beyond max level (7)
        curriculum.increment_attr_level("problem_type_weights")

    max_cfg = curriculum.generate_configuration({})
    assert max_cfg.problem_type_weights == [0, 0, 0, 0, 0, 0, 0, 1]  # Should be capped at level 7
