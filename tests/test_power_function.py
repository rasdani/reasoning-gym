"""Tests for Power Function questions generation"""

import pytest

from reasoning_gym.arithmetic import PowerFunctionConfig, PowerFunctionDataset


def test_power_function_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PowerFunctionConfig(seed=42, size=10)
    dataset1 = PowerFunctionDataset(config)
    dataset2 = PowerFunctionDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_power_function_dataset_items():
    """Test basic properties of generated items"""
    config = PowerFunctionConfig(min_base=-100, max_base=-100, min_exponent=-10, max_exponent=10, size=10, seed=42)
    dataset = PowerFunctionDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "base" in item["metadata"]
        assert "exponent" in item["metadata"]

        base = item["metadata"]["base"]
        exponent = item["metadata"]["exponent"]
        solution = item["metadata"]["solution"]

        # Verify values
        assert config.min_base <= base <= config.max_base
        assert config.min_exponent <= exponent <= config.max_exponent
        assert solution == pow(base, exponent)


def test_power_function_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PowerFunctionConfig(size=5, seed=42)
    dataset = PowerFunctionDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_power_function_score_function():
    """Test score function"""
    config = PowerFunctionConfig(seed=42)
    dataset = PowerFunctionDataset(config)

    for item in dataset:
        answer = item["answer"]
        assert dataset.score_answer(answer, item) == 1.0


def test_power_function_curriculum():
    """Test PowerFunctionCurriculum configuration generation and attribute manipulation"""
    from reasoning_gym.arithmetic import PowerFunctionCurriculum

    curriculum = PowerFunctionCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_exponent == 2 and base_cfg.max_exponent == 2

    # Test incrementing attribute levels for exponent & base attributes
    curriculum.increment_attr_level("exponent")

    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_exponent == 2 and increased_cfg.max_exponent == 4
