"""Tests for Ttsumego problem generation"""

import re

import pytest

from reasoning_gym.games.tsumego import TsumegoConfig, TsumegoDataset


def test_config_validation():
    # Valid configuration
    TsumegoConfig(min_board_size=9, max_board_size=13, max_stones=10, size=100, seed=42)

    # Invalid configurations
    with pytest.raises(ValueError):
        TsumegoConfig(min_board_size=4, max_board_size=13, max_stones=10)  # min_board_size too low
    with pytest.raises(ValueError):
        TsumegoConfig(min_board_size=9, max_board_size=20, max_stones=10)  # max_board_size too high
    with pytest.raises(ValueError):
        TsumegoConfig(min_board_size=13, max_board_size=9, max_stones=10)  # min_board_size > max_board_size
    with pytest.raises(ValueError):
        TsumegoConfig(min_board_size=9, max_board_size=13, max_stones=2)  # max_stones too low


def test_dataset_item_properties():
    config = TsumegoConfig(min_board_size=9, max_board_size=9, max_stones=15, size=100, seed=42)
    dataset = TsumegoDataset(config)
    item = dataset[0]
    # Check that item has the required keys
    for key in ["question", "answer", "metadata"]:
        assert key in item

    metadata = item["metadata"]
    for key in ["difficulty", "board"]:
        assert key in metadata

    board = metadata["board"]
    # Board size should be equal to the fixed min_board_size for this test
    assert len(board) == config.min_board_size
    assert all(len(row) == config.min_board_size for row in board)
    # Check stone count does not exceed max_stones + 7 (to account for extra fill in capture formation)
    stone_count = sum(cell in "XO" for row in board for cell in row)
    assert stone_count <= config.max_stones + 7


def test_deterministic_generation():
    config = TsumegoConfig(min_board_size=9, max_board_size=9, max_stones=10, seed=42)
    dataset1 = TsumegoDataset(config)
    dataset2 = TsumegoDataset(config)
    for i in range(3):
        item1 = dataset1[i]
        item2 = dataset2[i]
        assert item1["metadata"]["board"] == item2["metadata"]["board"]
        assert item1["answer"] == item2["answer"]


def test_liberties_and_move():
    # Use a small board for simplicity
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=10)
    dataset = TsumegoDataset(config)

    # Part 1: Liberty counting test
    board_liberties = [
        [".", "O", ".", ".", "."],
        ["O", "X", "O", ".", "."],
        [".", "O", ".", ".", "."],
        [".", ".", ".", ".", "."],
        [".", ".", ".", ".", "."],
    ]
    liberties = dataset._get_liberties(board_liberties, 1, 1)
    assert len(liberties) == 0
    liberties_edge = dataset._get_liberties(board_liberties, 0, 1)
    assert len(liberties_edge) == 2

    # Part 2: Test capturing move
    # Construct a board where an enemy stone at (2,2) is surrounded on three sides,
    # so that placing an "X" at (2,3) will remove its last liberty and capture it.
    board_capture = [["." for _ in range(5)] for _ in range(5)]
    board_capture[1][2] = "X"
    board_capture[2][1] = "X"
    board_capture[3][2] = "X"
    board_capture[2][2] = "O"
    # Now, (2,2) (enemy) has only one liberty at (2,3).
    # Placing "X" at (2,3) should capture the enemy stone.
    assert dataset._is_valid_move(board_capture, 2, 3, "X")
    dataset._make_move(board_capture, 2, 3, "X")
    # After move, captured_stones should be [(2,2)] and ko point set to (2,2).
    assert not dataset._is_valid_move(board_capture, 2, 2, "O"), "Ko move should be invalid"

    # Part 3: Test suicide move (without capture)
    board_move = [
        [".", "O", ".", ".", "."],
        ["O", ".", "O", ".", "."],
        [".", "O", ".", ".", "."],
        [".", ".", ".", ".", "."],
        [".", ".", ".", ".", "."],
    ]
    # Placing "X" at (1,1) would be suicide as all adjacent positions are occupied by "O".
    assert not dataset._is_valid_move(board_move, 1, 1, "X")


def test_score_answer():
    config = TsumegoConfig(min_board_size=9, max_board_size=9, max_stones=10, size=5, seed=123)
    dataset = TsumegoDataset(config)

    entry = dataset[0]

    assert dataset.score_answer(entry["answer"], entry) == 1.0

    # correct solution with additional text
    assert 0 < dataset.score_answer("move: " + entry["answer"], entry) < 1.0

    # Incorrect letter-number move, but correct letter-digit format
    assert dataset.score_answer("D4", entry) == 0.05

    # Invalid format
    assert dataset.score_answer("invalid", entry) == 0.01

    # Empty answer
    assert dataset.score_answer("", entry) == 0.0

    # None answer
    assert dataset.score_answer(None, entry) == 0.0

    # test optimal score for answers, patching each entry
    for x in dataset:
        assert len(x["metadata"]["board"]) == x["metadata"]["board_size"]
        assert dataset.score_answer(x["answer"], entry=x) == 1.0


# Additional tests for game logic edge cases


def test_get_group():
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    board = [
        ["X", "X", "."],
        [".", "X", "O"],
        [".", ".", "O"],
    ]
    group_X = dataset._get_group(board, 0, 0)
    expected_group_X = {(0, 0), (0, 1), (1, 1)}
    assert group_X == expected_group_X

    group_O = dataset._get_group(board, 1, 2)
    expected_group_O = {(1, 2), (2, 2)}
    assert group_O == expected_group_O


def test_count_liberties():
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    board = [
        ["X", "X", "."],
        [".", "X", "O"],
        [".", ".", "O"],
    ]
    group_X = {(0, 0), (0, 1), (1, 1)}
    liberties_X = dataset._count_liberties(board, group_X)
    # For (0,0): neighbor (1,0); (0,1): neighbor (0,2); (1,1): neighbors (1,0) and (2,1)
    # Combined unique liberties: {(1,0), (0,2), (2,1)} so count should be 3
    assert liberties_X == 3


def test_out_of_bounds_move():
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    board = [["." for _ in range(5)] for _ in range(5)]
    # Test moves that are out of bounds
    assert not dataset._is_valid_move(board, -1, 0, "X")
    assert not dataset._is_valid_move(board, 0, -1, "X")
    assert not dataset._is_valid_move(board, 5, 0, "X")
    assert not dataset._is_valid_move(board, 0, 5, "X")


def test_move_on_occupied_intersection():
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    board = [["." for _ in range(5)] for _ in range(5)]
    board[1][1] = "X"
    # Attempting to play on an occupied spot should be invalid
    assert not dataset._is_valid_move(board, 1, 1, "O")
    assert not dataset._is_valid_move(board, 1, 1, "X")


def test_valid_non_capturing_move():
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    board = [["." for _ in range(5)] for _ in range(5)]
    # A move on an empty board that doesn't result in capture or suicide should be valid
    assert dataset._is_valid_move(board, 0, 0, "X")
    move_result = dataset._make_move(board, 0, 0, "X")
    assert move_result
    assert board[0][0] == "X"


def test_multiple_capture():
    # Set up a board where a move will capture multiple opponent stones,
    # which should not trigger the ko rule (ko point remains None)
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    board = [
        [".", ".", ".", ".", "."],
        [".", "X", "X", "X", "."],
        ["X", "O", "O", ".", "."],
        [".", "X", "X", "X", "."],
        [".", ".", ".", ".", "."],
    ]
    # Move at (2,3) with 'X' should capture the opponent stones at (2,1) and (2,2)
    assert dataset._is_valid_move(board, 2, 3, "X")
    move_result = dataset._make_move(board, 2, 3, "X")
    assert move_result, "Move should be successfully made"
    assert board[2][1] == ".", "Stone at (2,1) should be captured"
    assert board[2][2] == ".", "Stone at (2,2) should be captured"
    assert dataset._ko_point is None, "Ko point should not be set for multiple captures"


def test_would_capture():
    config = TsumegoConfig(min_board_size=5, max_board_size=5, max_stones=10, size=1, seed=42)
    dataset = TsumegoDataset(config)
    # Create a scenario similar to the one in test_liberties_and_move for capturing
    board_capture = [["." for _ in range(5)] for _ in range(5)]
    board_capture[1][2] = "X"
    board_capture[2][1] = "X"
    board_capture[3][2] = "X"
    board_capture[2][2] = "O"
    # Placing 'X' at (2,3) should capture the stone at (2,2)
    assert dataset._would_capture(board_capture, 2, 3, "X")
    # In a scenario with no capture, the move should not be considered capturing
    board_no_capture = [["." for _ in range(5)] for _ in range(5)]
    board_no_capture[2][2] = "O"
    assert not dataset._would_capture(board_no_capture, 0, 0, "X")


def test_capture_verification():
    """Verifies that the solution move in a generated puzzle captures at least one opponent stone."""
    config = TsumegoConfig(min_board_size=9, max_board_size=9, max_stones=15, size=1, seed=10)
    dataset = TsumegoDataset(config)
    entry = dataset[0]
    board = entry["metadata"]["board"]

    # If solution is a letter+number string, convert it

    def convert_solution(sol, board_size):
        # sol is expected to be a string like 'E5'
        letter = sol[0].upper()
        number = int(sol[1:])
        return (board_size - number, ord(letter) - ord("A"))

    solution = entry["answer"]
    assert isinstance(solution, str)
    solution = convert_solution(solution, len(board))
    initial_white = sum(row.count("O") for row in board)

    # Make a deep copy of the board to simulate the move
    board_after = [row[:] for row in board]
    move_success = dataset._make_move(board_after, solution[0], solution[1], "X")
    assert move_success, "The solution move should be legal."

    final_white = sum(row.count("O") for row in board_after)
    assert final_white < initial_white, "The solution move should capture at least one opponent stone."


def test_tsumego_curriculum():
    """Test the TsumegoCurriculum functionality"""
    from reasoning_gym.games.tsumego import TsumegoCurriculum

    curriculum = TsumegoCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test initial configuration
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_board_size == 9 and base_cfg.max_board_size == 9
    assert base_cfg.max_stones == 15  # Default value from TsumegoConfig

    # Test incrementing attribute level
    curriculum.increment_attr_level("board_size")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_board_size == 9 and increased_cfg.max_board_size == 10
    assert increased_cfg.max_stones == 15  # Unchanged

    # Test incrementing attribute level again
    curriculum.increment_attr_level("board_size")
    increased_cfg_2 = curriculum.generate_configuration(base_value)
    assert increased_cfg_2.min_board_size == 9 and increased_cfg_2.max_board_size == 11
    assert increased_cfg_2.max_stones == 15  # Unchanged

    # Test decrementing attribute level
    curriculum.decrement_attr_level("board_size")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_board_size == 9 and decreased_cfg.max_board_size == 10
    assert decreased_cfg.max_stones == 15  # Unchanged

    # Test global level adjustments
    curriculum = TsumegoCurriculum()  # Reset curriculum
    assert curriculum.get_attr_level("board_size") == 0

    # Increase global level
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("board_size") == 1

    global_level_cfg = curriculum.generate_configuration(base_value)
    assert global_level_cfg.min_board_size == 9 and global_level_cfg.max_board_size == 10

    # Increase global level again
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("board_size") == 2

    global_level_cfg_2 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_2.min_board_size == 9 and global_level_cfg_2.max_board_size == 11

    # Decrease global level
    curriculum.decrement_global_level()
    assert curriculum.get_attr_level("board_size") == 1

    global_level_cfg_3 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_3.min_board_size == 9 and global_level_cfg_3.max_board_size == 10
