import string

import pytest
import sympy as sp

from reasoning_gym import create_dataset
from reasoning_gym.algebra.polynomial_multiplication import (
    PolynomialMultiplicationConfig,
    PolynomialMultiplicationCurriculum,
    PolynomialMultiplicationDataset,
)


def test_polynomial_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_terms=0).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_value=0).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_degree=-1, max_degree=3).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_degree=4, max_degree=3).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(operators=("^",)).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_polynomials=5, max_polynomials=2).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(variables="").validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(
            allow_cross_variable_product=False, allow_multivariate_polynomials=True
        ).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_polynomials=5, max_polynomials=2).validate()


def test_polynomial_multiplication_dataset_basic():
    """Test dataset creation and length"""
    dataset_size = 50
    config = PolynomialMultiplicationConfig(
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        min_polynomials=2,
        max_polynomials=3,
        variables=tuple(string.ascii_lowercase),
        allow_cross_variable_product=False,
        allow_multivariate_polynomials=False,
        seed=42,
        size=dataset_size,
    )

    dataset = PolynomialMultiplicationDataset(config)

    assert len(dataset) == dataset_size


def test_polynomial_equations_dataset_items():
    """Test that generated items have correct structure"""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        min_polynomials=2,
        max_polynomials=5,
        variables=tuple("xyz"),
        allow_cross_variable_product=False,
        allow_multivariate_polynomials=False,
        size=3,
        seed=100,
    )

    for item in ds:
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert isinstance(item["metadata"]["polynomial_expr"], str)
        assert isinstance(item["metadata"]["variables"], list)

        # Check polynomial_expr existence
        poly_str = item["metadata"]["polynomial_expr"]
        # Ensure it can parse with sympy
        sp.sympify(poly_str)


def test_cross_polynomial_equations_dataset_items():
    """Test that generated items have correct structure"""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        min_polynomials=2,
        max_polynomials=5,
        variables=tuple("xyz"),
        allow_cross_variable_product=True,
        allow_multivariate_polynomials=False,
        size=3,
        seed=100,
    )

    for item in ds:
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert isinstance(item["metadata"]["polynomial_expr"], str)
        assert isinstance(item["metadata"]["variables"], list)

        # Check polynomial_expr existence
        poly_str = item["metadata"]["polynomial_expr"]
        # Ensure it can parse with sympy
        sp.sympify(poly_str)


def test_multivariate_polynomial_equations_dataset_items():
    """Test that generated items have correct structure"""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        min_polynomials=2,
        max_polynomials=5,
        variables=tuple(["x", "y"]),
        allow_cross_variable_product=True,
        allow_multivariate_polynomials=True,
        size=3,
        seed=100,
    )

    for item in ds:
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert isinstance(item["metadata"]["polynomial_expr"], str)
        assert isinstance(item["metadata"]["variables"], list)

        # Check polynomial_expr existence
        poly_str = item["metadata"]["polynomial_expr"]
        # Ensure it can parse with sympy
        sp.sympify(poly_str)


def test_polynomial_equations_dataset_deterministic():
    """Test dataset reproducibility with fixed seed."""
    cfg = PolynomialMultiplicationConfig(seed=999, size=3)
    ds1 = PolynomialMultiplicationDataset(cfg)
    ds2 = PolynomialMultiplicationDataset(cfg)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i], "Polynomial datasets with same seed should match exactly."


def test_polynomial_solutions_evaluation():
    """Test that solution satisfy the polynomial multiplication."""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=4,
        min_value=1,
        max_value=10,
        min_degree=1,
        max_degree=3,
        min_polynomials=2,
        max_polynomials=5,
        variables=tuple(["x", "y"]),
        allow_cross_variable_product=True,
        allow_multivariate_polynomials=True,
        size=5,
        seed=42,
    )

    for item in ds:
        # Extract the polynomial expression
        poly_str = item["metadata"]["polynomial_expr"]
        # Get the polynomial product
        poly_expr = sp.expand(poly_str)

        # Verify that each solution satisfies the polynomial
        assert str(poly_expr) == item["answer"]


def test_score_function():
    """Test that solution satisfy the polynomial multiplication."""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=3,
        min_degree=1,
        max_degree=3,
        min_polynomials=3,
        max_polynomials=3,
        variables=tuple(["x", "y"]),
        allow_cross_variable_product=True,
        allow_multivariate_polynomials=True,
        size=3,
        seed=42,
    )

    for item in ds:
        poly_str = item["metadata"]["polynomial_expr"]
        assert ds.score_answer(poly_str, item) == 0.0

        poly_expr = str(sp.expand(poly_str))
        assert ds.score_answer(poly_expr, item) == 1.0

        assert ds.score_answer(None, item) == 0.0
        assert ds.score_answer("Not a polynomial", item) == 0.0
        assert ds.score_answer("x**4", item) == 0.0


def test_polynomial_multiplication_curriculum():
    """Test curriculum for polynomial multiplication."""
    curriculum = PolynomialMultiplicationCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: PolynomialMultiplicationCurriculum = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_terms == 2
    assert base_cfg.max_terms == 4
    assert base_cfg.min_value == 1
    assert base_cfg.max_value == 100
    assert base_cfg.min_degree == 0
    assert base_cfg.max_degree == 2
    assert base_cfg.min_polynomials == 2
    assert base_cfg.max_polynomials == 4

    # Increase and validate increase in level
    curriculum.increment_attr_level("min_terms")
    curriculum.increment_attr_level("max_terms")
    curriculum.increment_attr_level("min_value")
    curriculum.increment_attr_level("max_value")
    curriculum.increment_attr_level("min_degree")
    curriculum.increment_attr_level("max_degree")
    curriculum.increment_attr_level("min_polynomials")
    curriculum.increment_attr_level("max_polynomials")

    increased_cfg: PolynomialMultiplicationCurriculum = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_terms == 4
    assert increased_cfg.max_terms == 8
    assert increased_cfg.min_value == 10
    assert increased_cfg.max_value == 10000
    assert increased_cfg.min_degree == 1
    assert increased_cfg.max_degree == 4
    assert increased_cfg.min_polynomials == 3
    assert increased_cfg.max_polynomials == 6

    # Decrease and validate decrease in level
    curriculum.decrement_attr_level("min_terms")
    curriculum.decrement_attr_level("max_terms")
    curriculum.decrement_attr_level("min_value")
    curriculum.decrement_attr_level("max_value")
    curriculum.decrement_attr_level("min_degree")
    curriculum.decrement_attr_level("max_degree")
    curriculum.decrement_attr_level("min_polynomials")
    curriculum.decrement_attr_level("max_polynomials")

    decreased_cfg: PolynomialMultiplicationCurriculum = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_terms == 2
    assert decreased_cfg.max_terms == 4
    assert decreased_cfg.min_value == 1
    assert decreased_cfg.max_value == 100
    assert decreased_cfg.min_degree == 0
    assert decreased_cfg.max_degree == 2
    assert decreased_cfg.min_polynomials == 2
    assert decreased_cfg.max_polynomials == 4

    # Test upper bound boundary condition
    for _ in range(10):
        curriculum.increment_attr_level("min_terms")
        curriculum.increment_attr_level("max_terms")
        curriculum.increment_attr_level("min_value")
        curriculum.increment_attr_level("max_value")
        curriculum.increment_attr_level("min_degree")
        curriculum.increment_attr_level("max_degree")
        curriculum.increment_attr_level("min_polynomials")
        curriculum.increment_attr_level("max_polynomials")
    upper_bound_cfg: PolynomialMultiplicationCurriculum = curriculum.generate_configuration(base_value)
    assert upper_bound_cfg.min_terms == 8
    assert upper_bound_cfg.max_terms == 16
    assert upper_bound_cfg.min_value == 1000
    assert upper_bound_cfg.max_value == 100000000
    assert upper_bound_cfg.min_degree == 3
    assert upper_bound_cfg.max_degree == 10
    assert upper_bound_cfg.min_polynomials == 5
    assert upper_bound_cfg.max_polynomials == 10

    # Test lower bound boundary condition
    for _ in range(10):
        curriculum.decrement_attr_level("min_terms")
        curriculum.decrement_attr_level("max_terms")
        curriculum.decrement_attr_level("min_value")
        curriculum.decrement_attr_level("max_value")
        curriculum.decrement_attr_level("min_degree")
        curriculum.decrement_attr_level("max_degree")
        curriculum.decrement_attr_level("min_polynomials")
        curriculum.decrement_attr_level("max_polynomials")
    lower_bound_cfg: PolynomialMultiplicationCurriculum = curriculum.generate_configuration(base_value)
    assert lower_bound_cfg.min_terms == 2
    assert lower_bound_cfg.max_terms == 4
    assert lower_bound_cfg.min_value == 1
    assert lower_bound_cfg.max_value == 100
    assert lower_bound_cfg.min_degree == 0
    assert lower_bound_cfg.max_degree == 2
    assert lower_bound_cfg.min_polynomials == 2
    assert lower_bound_cfg.max_polynomials == 4
