"""Tests for letter jumbling task generation"""

from random import Random

import pytest

from reasoning_gym.algorithmic.letter_jumble import LetterJumbleConfig, LetterJumbleCurriculum, LetterJumbleDataset


def test_letter_jumble_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(min_word_len=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(min_words=10, max_words=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(min_corruption_level=-0.1)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(max_corruption_level=1.1)
        config.validate()


def test_letter_jumble_deterministic():
    """Test that dataset generates same items with same seed"""
    config = LetterJumbleConfig(seed=42, size=10)
    dataset1 = LetterJumbleDataset(config)
    dataset2 = LetterJumbleDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_letter_jumble_scrambling():
    """Test the word scrambling logic"""
    config = LetterJumbleConfig(
        min_word_len=4,
        max_word_len=8,
        min_words=1,
        max_words=1,
        min_corruption_level=0.5,
        max_corruption_level=0.5,
        size=1,
        seed=42,
    )
    dataset = LetterJumbleDataset(config)

    # Test with known word
    word = "testing"
    rng = Random(42)
    scrambled = dataset._scramble_word(word, 0.5, rng)

    # Verify scrambled word:
    # - Has same length as original
    assert len(scrambled) == len(word)
    # - Contains same characters
    assert sorted(scrambled) == sorted(word)
    # - Is different from original (with high probability given 0.5 corruption)
    assert scrambled != word


def test_letter_jumble_dataset_items():
    """Test basic properties of generated items"""
    config = LetterJumbleConfig(
        min_word_len=4,
        max_word_len=8,
        min_words=3,
        max_words=5,
        min_corruption_level=0.1,
        max_corruption_level=0.3,
        size=50,
        seed=42,
    )
    dataset = LetterJumbleDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        metadata = item["metadata"]
        assert "num_words" in metadata
        assert "corruption_level" in metadata
        assert "scrambled_words" in metadata
        assert "original_words" in metadata

        # Verify word counts
        num_words = metadata["num_words"]
        assert config.min_words <= num_words <= config.max_words
        assert len(metadata["scrambled_words"]) == num_words
        assert len(metadata["original_words"]) == num_words

        # Verify corruption level
        assert config.min_corruption_level <= metadata["corruption_level"] <= config.max_corruption_level

        # Verify word properties
        for word in metadata["original_words"]:
            assert config.min_word_len <= len(word) <= config.max_word_len
            assert word.isalpha()

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        answera = item["answer"].split(" ")
        answera[0] = "flippityfloop"
        answera[1] = "doopadoopadoop"
        answerf = " ".join(answera)
        assert 0.01 <= dataset.score_answer(answer=answerf, entry=item) <= 1.0


def test_letter_jumble_iteration():
    """Test that iteration respects dataset size"""
    config = LetterJumbleConfig(size=5, seed=42)
    dataset = LetterJumbleDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_letter_jumble_curriculum():
    curriculum = LetterJumbleCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: LetterJumbleConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_word_len == 5 and base_cfg.max_word_len == 15
    assert base_cfg.min_words == 10 and base_cfg.max_words == 50
    assert base_cfg.min_corruption_level == 0.1 and base_cfg.max_corruption_level == 0.3

    # test incrementing attribute levels
    curriculum.increment_attr_level("word_len")
    curriculum.increment_attr_level("words")
    curriculum.increment_attr_level("corruption_level")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_word_len == 5 and increased_cfg.max_word_len == 30
    assert increased_cfg.min_words == 10 and increased_cfg.max_words == 100
    assert increased_cfg.min_corruption_level == 0.1 and increased_cfg.max_corruption_level == 0.6

    # test decrementing attribute level for words again
    curriculum.decrement_attr_level("words")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_word_len == 5 and partially_decreased_cfg.max_word_len == 30
    assert partially_decreased_cfg.min_words == 10 and partially_decreased_cfg.max_words == 50
    assert partially_decreased_cfg.min_corruption_level == 0.1 and partially_decreased_cfg.max_corruption_level == 0.6
