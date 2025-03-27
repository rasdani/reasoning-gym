"""Tests for number filtering task generation"""

import pytest

from reasoning_gym.algorithmic.number_filtering import (
    NumberFilteringConfig,
    NumberFilteringCurriculum,
    NumberFilteringDataset,
)


def test_number_filtering_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NumberFilteringConfig(min_numbers=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFilteringConfig(min_numbers=10, max_numbers=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFilteringConfig(min_decimals=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFilteringConfig(min_value=100, max_value=0)
        config.validate()


def test_number_filtering_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = NumberFilteringConfig(seed=42, size=10)
    dataset1 = NumberFilteringDataset(config)
    dataset2 = NumberFilteringDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_number_filtering_dataset_items():
    """Test basic properties of generated items"""
    config = NumberFilteringConfig(
        min_numbers=3, max_numbers=6, min_decimals=1, max_decimals=3, min_value=-10.0, max_value=10.0, size=10, seed=42
    )
    dataset = NumberFilteringDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "original_numbers" in item["metadata"]
        assert "filter_value" in item["metadata"]
        assert "operation" in item["metadata"]
        assert "result" in item["metadata"]

        # Verify number count constraints
        numbers = item["metadata"]["original_numbers"]
        assert len(numbers) >= config.min_numbers
        assert len(numbers) <= config.max_numbers

        # Verify decimal places
        for num in numbers:
            decimal_places = len(num.split(".")[-1]) if "." in num else 0
            assert decimal_places >= config.min_decimals
            assert decimal_places <= config.max_decimals

        # Verify value range
        for num in numbers:
            value = float(num)
            assert config.min_value <= value <= config.max_value

        # Verify filtering operation
        operation = item["metadata"]["operation"]
        filter_value = float(item["metadata"]["filter_value"])
        result = [float(x) for x in eval(item["answer"])] if item["answer"] != "[]" else []

        if operation == "keep_larger":
            assert all(x > filter_value for x in result)
        elif operation == "keep_smaller":
            assert all(x < filter_value for x in result)
        elif operation == "remove_larger":
            assert all(x <= filter_value for x in result)
        elif operation == "remove_smaller":
            assert all(x >= filter_value for x in result)


def test_number_filtering_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = NumberFilteringConfig(size=5, seed=42)
    dataset = NumberFilteringDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_number_filtering_precision():
    """Test that number formatting and precision handling works correctly"""
    config = NumberFilteringConfig(
        min_numbers=3,
        max_numbers=3,  # Fixed size for predictability
        min_decimals=2,
        max_decimals=2,  # Fixed decimals for predictability
        min_value=0.0,
        max_value=1.0,
        size=1,
        seed=42,
    )
    dataset = NumberFilteringDataset(config)
    item = dataset[0]

    # Check that string representations maintain precision
    for num in item["metadata"]["original_numbers"]:
        assert len(num.split(".")[-1]) == 2


def test_number_filtering_curriculum():
    curriculum = NumberFilteringCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: NumberFilteringConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_numbers == 10 and base_cfg.max_numbers == 100
    assert base_cfg.min_decimals == 0 and base_cfg.max_decimals == 2
    assert base_cfg.min_value == -10_000 and base_cfg.max_value == 10_000

    # test incrementing some attribute levels
    curriculum.increment_attr_level("numbers")
    curriculum.increment_attr_level("decimals")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_numbers == 10 and increased_cfg.max_numbers == 500
    assert increased_cfg.min_decimals == 0 and increased_cfg.max_decimals == 4
    assert increased_cfg.min_value == -10_000 and increased_cfg.max_value == 10_000

    # test decrementing attribute level for numbers again
    curriculum.decrement_attr_level("numbers")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_numbers == 10 and partially_decreased_cfg.max_numbers == 100
    assert partially_decreased_cfg.min_decimals == 0 and partially_decreased_cfg.max_decimals == 4
    assert partially_decreased_cfg.min_value == -10_000 and partially_decreased_cfg.max_value == 10_000
