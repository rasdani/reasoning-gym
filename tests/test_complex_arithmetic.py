import pytest

from reasoning_gym.algebra.complex_arithmetic import (
    ComplexArithmeticConfig,
    ComplexArithmeticCurriculum,
    ComplexArithmeticDataset,
)


def test_complex_arithmetic_basic():
    """Test basic functionality of complex arithmetic dataset."""
    config = ComplexArithmeticConfig(
        min_real=-5, max_real=5, min_imag=-5, max_imag=5, operations=("+", "-", "*", "/"), seed=42, size=10
    )
    dataset = ComplexArithmeticDataset(config)

    print(dataset)

    # Test dataset size
    assert len(dataset) == 10

    # Test a specific item
    item = dataset[0]
    assert "question" in item
    assert "answer" in item
    assert "metadata" in item

    # Add more detailed assertions
    assert isinstance(item["question"], str)
    assert isinstance(item["answer"], str)
    assert isinstance(item["metadata"], dict)

    # Check metadata structure
    assert "num1" in item["metadata"]
    assert "num2" in item["metadata"]
    assert "operation" in item["metadata"]
    assert "result" in item["metadata"]

    # Check data types in metadata
    assert isinstance(item["metadata"]["num1"], tuple)
    assert isinstance(item["metadata"]["num2"], tuple)
    assert len(item["metadata"]["num1"]) == 2  # Real and imaginary parts
    assert len(item["metadata"]["num2"]) == 2
    assert isinstance(item["metadata"]["operation"], str)
    assert isinstance(item["metadata"]["result"], tuple)

    # Make sure answer matches the result in metadata
    # results is a tuple of two floats (real, imag) and answer is a string
    # answer is formatted as "real + imagi"
    assert ComplexArithmeticDataset.parse_string_to_complex(item["answer"]) == complex(*item["metadata"]["result"])


def test_complex_arithmetic_scoring():
    """Test scoring function with various answer formats and accuracies."""
    config = ComplexArithmeticConfig(seed=42)
    dataset = ComplexArithmeticDataset(config)

    # Test case with answer 3 + 2i
    entry = {"metadata": {"result": (3.0, 2.0)}}

    # Test exact matches (should get score of 1.0)
    assert dataset.score_answer("3 + 2i", entry) == 1.0
    assert dataset.score_answer("3+2i", entry) == 1.0
    assert dataset.score_answer("3.0 + 2.0i", entry) == 1.0
    assert dataset.score_answer("((3.0 + 2.0i ) )", entry) == 1.0

    # Test answers with small errors (should get high but < 1.0 scores)
    print(dataset.score_answer("3.1 + 2i", entry))
    assert 0.9 < dataset.score_answer("3.1 + 2i", entry) < 1.0
    assert 0.9 < dataset.score_answer("3 + 2.1i", entry) < 1.0
    assert 0.7 < dataset.score_answer("3.1 + 2.1i", entry) < 0.95

    # Test answers with moderate errors (should get medium scores)
    assert 0.3 < dataset.score_answer("4 + 2i", entry) < 0.4
    assert 0.3 < dataset.score_answer("3 + 3i", entry) < 0.4

    # Test answers with large errors (should get very low scores)
    assert dataset.score_answer("10 + 10i", entry) < 0.01

    # Test invalid answers (should get 0.0)
    assert dataset.score_answer("invalid", entry) == 0.0
    assert dataset.score_answer(None, entry) == 0.0
    assert dataset.score_answer("inf + 2i", entry) == 0.0


def test_complex_arithmetic_division_by_zero():
    """Test that division by zero is handled properly."""
    config = ComplexArithmeticConfig(
        operations=("+", "-", "*", "/"), operations_weights=[0.0, 0.0, 0.0, 1.0], seed=42
    )  # Only test division
    dataset = ComplexArithmeticDataset(config)

    # Check multiple items to ensure no division by zero
    for i in range(10):
        item = dataset[i]
        num2 = complex(*item["metadata"]["num2"])
        assert num2 != 0


def test_parse_string_to_complex():
    """Test the parse_string_to_complex method with various input formats."""
    dataset = ComplexArithmeticDataset(ComplexArithmeticConfig())

    # Test pure real numbers
    assert dataset.parse_string_to_complex("5") == complex(5, 0)
    assert dataset.parse_string_to_complex("5.0") == complex(5, 0)
    assert dataset.parse_string_to_complex("-5") == complex(-5, 0)
    assert dataset.parse_string_to_complex("-5.5") == complex(-5.5, 0)

    # Test pure imaginary numbers
    assert dataset.parse_string_to_complex("i") == complex(0, 1)
    assert dataset.parse_string_to_complex("j") == complex(0, 1)
    assert dataset.parse_string_to_complex("7i") == complex(0, 7)
    assert dataset.parse_string_to_complex("7.0i") == complex(0, 7)
    assert dataset.parse_string_to_complex("-i") == complex(0, -1)
    assert dataset.parse_string_to_complex("-7i") == complex(0, -7)
    assert dataset.parse_string_to_complex("-7.5i") == complex(0, -7.5)

    # Test complex numbers with both parts
    assert dataset.parse_string_to_complex("3+2i") == complex(3, 2)
    assert dataset.parse_string_to_complex("3 + 2i") == complex(3, 2)
    assert dataset.parse_string_to_complex("3.5 + 2.5i") == complex(3.5, 2.5)
    assert dataset.parse_string_to_complex("-3 + 2i") == complex(-3, 2)
    assert dataset.parse_string_to_complex("3 - 2i") == complex(3, -2)
    assert dataset.parse_string_to_complex("-3 - 2i") == complex(-3, -2)

    # Test with j notation
    assert dataset.parse_string_to_complex("3+2j") == complex(3, 2)
    assert dataset.parse_string_to_complex("3 + 2j") == complex(3, 2)

    # Test with different spacing
    assert dataset.parse_string_to_complex("  3  +  2i  ") == complex(3, 2)
    assert dataset.parse_string_to_complex("3+i") == complex(3, 1)
    assert dataset.parse_string_to_complex("3-i") == complex(3, -1)

    # Test invalid inputs
    assert dataset.parse_string_to_complex("invalid") is None
    assert dataset.parse_string_to_complex("3 + i + 2") is None
    assert dataset.parse_string_to_complex("3 + 2x") is None


def test_complex_arithmetic_curriculum():
    """Test the curriculum for complex arithmetic."""
    curriculum = ComplexArithmeticCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: ComplexArithmeticCurriculum = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_real == base_cfg.min_imag == -10
    assert base_cfg.max_real == base_cfg.max_imag == 10
    assert base_cfg.operations_weights == [0.4, 0.4, 0.1, 0.1]

    # Increase and validate increase in level
    curriculum.increment_attr_level("min_real")
    curriculum.increment_attr_level("min_imag")
    curriculum.increment_attr_level("max_real")
    curriculum.increment_attr_level("max_imag")
    curriculum.increment_attr_level("operations_weights")

    increased_cfg: ComplexArithmeticCurriculum = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_real == increased_cfg.min_imag == -100
    assert increased_cfg.max_real == increased_cfg.max_imag == 100
    assert increased_cfg.operations_weights == [0.25, 0.25, 0.25, 0.25]

    # Decrease and validate decrease in level
    curriculum.decrement_attr_level("min_real")
    curriculum.decrement_attr_level("min_imag")
    curriculum.decrement_attr_level("max_real")
    curriculum.decrement_attr_level("max_imag")
    curriculum.decrement_attr_level("operations_weights")

    decreased_cfg: ComplexArithmeticCurriculum = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_real == decreased_cfg.min_imag == -10
    assert decreased_cfg.max_real == decreased_cfg.max_imag == 10
    assert decreased_cfg.operations_weights == [0.4, 0.4, 0.1, 0.1]

    # Test upper bound boundary condition
    for _ in range(10):
        curriculum.increment_attr_level("min_real")
        curriculum.increment_attr_level("min_imag")
        curriculum.increment_attr_level("max_real")
        curriculum.increment_attr_level("max_imag")
        curriculum.increment_attr_level("operations_weights")
    upper_bound_cfg: ComplexArithmeticCurriculum = curriculum.generate_configuration(base_value)
    assert upper_bound_cfg.min_real == upper_bound_cfg.min_imag == -100000000
    assert upper_bound_cfg.max_real == upper_bound_cfg.max_imag == 100000000
    assert upper_bound_cfg.operations_weights == [0.1, 0.1, 0.4, 0.4]

    # Test lower bound boundary condition
    for _ in range(10):
        curriculum.decrement_attr_level("min_real")
        curriculum.decrement_attr_level("min_imag")
        curriculum.decrement_attr_level("max_real")
        curriculum.decrement_attr_level("max_imag")
        curriculum.decrement_attr_level("operations_weights")
    lower_bound_cfg: ComplexArithmeticCurriculum = curriculum.generate_configuration(base_value)
    assert lower_bound_cfg.min_real == lower_bound_cfg.min_imag == -10
    assert lower_bound_cfg.max_real == lower_bound_cfg.max_imag == 10
    assert lower_bound_cfg.operations_weights == [0.4, 0.4, 0.1, 0.1]
