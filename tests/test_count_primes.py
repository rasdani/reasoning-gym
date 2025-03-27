"""Tests for Count Primes questions generation"""

import pytest

from reasoning_gym.algorithmic.count_primes import CountPrimesConfig, CountPrimesCurriculum, CountPrimesDataset


def test_count_primes_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CountPrimesConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CountPrimesConfig(max_n=0)  # Zero not allowed
        config.validate()


def test_count_primes_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CountPrimesConfig(seed=42, size=10)
    dataset1 = CountPrimesDataset(config)
    dataset2 = CountPrimesDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_count_primes_dataset_items():
    """Test basic properties of generated items"""
    config = CountPrimesConfig(max_n=10, size=10, seed=42)
    dataset = CountPrimesDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "start" in item["metadata"]
        assert "end" in item["metadata"]
        assert "primes" in item["metadata"]
        assert "solution" in item["metadata"]

        start = item["metadata"]["start"]
        end = item["metadata"]["end"]
        primes = item["metadata"]["primes"]

        assert start <= end
        assert len(primes) <= end - start + 1


def test_count_primes_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = CountPrimesConfig(size=5, seed=42)
    dataset = CountPrimesDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_count_primes_answer():
    """Test the _get_primes method"""
    config = CountPrimesConfig(seed=42)
    dataset = CountPrimesDataset(config)

    # Base cases
    assert dataset._get_primes(n=0) == []
    assert dataset._get_primes(n=1) == []
    assert dataset._get_primes(n=2) == [False, False]

    # Test primes up to 10
    primes = dataset._get_primes(n=11)
    assert primes[2] == True
    assert primes[3] == True
    assert primes[4] == False
    assert primes[5] == True
    assert primes[6] == False
    assert primes[7] == True
    assert primes[8] == False
    assert primes[9] == False
    assert primes[10] == False


def test_count_primes_list():
    """Test that list of primes was correctly generated"""
    config = CountPrimesConfig(max_n=100, size=100, seed=42)
    dataset = CountPrimesDataset(config)

    for item in dataset:
        start = item["metadata"]["start"]
        end = item["metadata"]["end"]
        primes = item["metadata"]["primes"]
        for p in primes:
            assert p >= start
            assert p <= end
            assert dataset.primes[p] == True


def test_shortest_path_curriculum():
    curriculum = CountPrimesCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: CountPrimesConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_n == 1000 and base_cfg.max_n == 1000

    # test incrementing attribute levels
    curriculum.increment_attr_level("n")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_n == 1000 and increased_cfg.max_n == 10000
