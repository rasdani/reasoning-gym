"""Tests for Palindrome Partitioning questions generation"""

import json

from reasoning_gym.algorithmic.palindrome_partitioning import (
    PalindromePartitioningConfig,
    PalindromePartitioningCurriculum,
    PalindromePartitioningDataset,
)


def test_palindrome_partitioning_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PalindromePartitioningConfig(seed=42, size=10)
    dataset1 = PalindromePartitioningDataset(config)
    dataset2 = PalindromePartitioningDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_palindrome_partitioning_dataset_items():
    """Test basic properties of generated items"""
    config = PalindromePartitioningConfig(size=10, seed=42)
    dataset = PalindromePartitioningDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "string" in item["metadata"]
        assert "solution" in item["metadata"]
        string = item["metadata"]["string"]
        solution = item["metadata"]["solution"]

        # Verify string is not empty
        assert len(string) > 0

        # At least one partitioning exists (each letter is a palindrome)
        assert len(solution) >= 1

        # Verify each partitioning reconstructs the original string
        assert all(len(partitioning) > 0 for partitioning in solution)
        assert all("".join(partitioning) == string for partitioning in solution)


def test_palindrome_partitioning_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PalindromePartitioningConfig(size=5, seed=42)
    dataset = PalindromePartitioningDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_palindrome_partitioning_answer():
    """Test the _palindrome_partitioning method"""
    config = PalindromePartitioningConfig(seed=42)
    dataset = PalindromePartitioningDataset(config)

    # General use case
    word = "afternoon"
    correct = [
        ["a", "f", "t", "e", "r", "n", "o", "o", "n"],
        ["a", "f", "t", "e", "r", "n", "oo", "n"],
        ["a", "f", "t", "e", "r", "noon"],
    ]
    assert json.dumps(dataset._palindrome_partitioning(word)) == json.dumps(correct)

    # Single letter word
    word = "a"
    correct = [["a"]]
    assert json.dumps(dataset._palindrome_partitioning(word)) == json.dumps(correct)

    # Empty string
    word = ""
    correct = []
    assert json.dumps(dataset._palindrome_partitioning(word)) == json.dumps(correct)


def test_palindrome_partitioning_score_answer():
    """Test the score_answer method"""
    config = PalindromePartitioningConfig(seed=42)
    dataset = PalindromePartitioningDataset(config)

    # Verify the scoring function is permutation invariant
    answer = json.dumps([["n", "o", "o", "n"], ["no", "on"], ["noon"]])
    item = {"metadata": {"solution": [["no", "on"], ["noon"], ["n", "o", "o", "n"]]}}
    assert dataset.score_answer(answer, item) == 1

    # Verify the score is 0.0 when incorrect
    answer = json.dumps([["n", "o", "o", "n"], ["no", "on"]])
    item = {"metadata": {"solution": [["no", "on"], ["noon"], ["n", "o", "o", "n"]]}}
    assert dataset.score_answer(answer, item) == 0.0

    # Verify the score is 0.0 when answer is None
    answer = None
    item = {"metadata": {"solution": [["no", "on"], ["noon"], ["n", "o", "o", "n"]]}}
    assert dataset.score_answer(answer, item) == 0.0

    # Verify the score is 0.0 when answer is malformed JSON
    answer = '["n", "o", "o", "n"], ["no", "on"], ["noon"]'
    item = {"metadata": {"solution": [["no", "on"], ["noon"], ["n", "o", "o", "n"]]}}
    assert dataset.score_answer(answer, item) == 0.0


def test_palindrome_partitioning_curriculum():
    curriculum = PalindromePartitioningCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: PalindromePartitioningConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_string_len == 10 and base_cfg.max_string_len == 10
    assert base_cfg.min_substring_palindrome_len == 5 and base_cfg.max_substring_palindrome_len == 5

    # test incrementing attribute levels
    curriculum.increment_attr_level("string_len")
    curriculum.increment_attr_level("substring_palindrome_len")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_string_len == 10 and increased_cfg.max_string_len == 100
    assert increased_cfg.min_substring_palindrome_len == 5 and increased_cfg.max_substring_palindrome_len == 10

    # test decrementing attribute level for substring_palindrome_len again
    curriculum.decrement_attr_level("substring_palindrome_len")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_string_len == 10 and partially_decreased_cfg.max_string_len == 100
    assert (
        partially_decreased_cfg.min_substring_palindrome_len == 5
        and partially_decreased_cfg.max_substring_palindrome_len == 5
    )
