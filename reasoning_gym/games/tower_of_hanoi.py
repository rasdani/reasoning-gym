# reasoning_gym/games/tower_of_hanoi.py

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Solve the Tower of Hanoi problem with {num_disks} disks and {num_pegs} pegs.
Move all disks from {start_peg} to {target_peg} following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.

Provide the sequence of moves.

Formatting guidelines:
- Each instruction should be placed on a single line.
- Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
- Do not include any other text or formatting.
"""

DATASET_NAME = "tower_of_hanoi"


@dataclass
class HanoiConfig:
    """
    Configuration for the Tower of Hanoi task.

    - min_disks: Minimum number of disks in the puzzle.
    - max_disks: Maximum number of disks in the puzzle.
    - min_pegs: Minimum number of pegs (minimum 3).
    - max_pegs: Maximum number of pegs.
    - size: Number of problem instances in the dataset.
    - seed: Optional seed for reproducibility.
    - visualize: Whether to include a visualization of the initial state.
    """

    min_disks: int = 3
    max_disks: int = 7
    min_pegs: int = 3
    max_pegs: int = 4
    size: int = 500
    seed: Optional[int] = None
    visualize: bool = False  # New parameter

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_disks >= 1, "min_disks must be at least 1"
        assert self.max_disks >= self.min_disks, "max_disks must be >= min_disks"
        assert self.min_pegs >= 3, "min_pegs must be at least 3"
        assert self.max_pegs >= self.min_pegs, "max_pegs must be >= min_pegs"


class MoveGenerator:
    """
    Helper class to generate valid move sequences for Tower of Hanoi using the Frame-Stewart algorithm.
    It maintains the current state of all pegs to ensure move validity.
    """

    def __init__(self, num_disks: int, pegs: list[int], start: int, target: int):
        self.num_disks = num_disks
        self.pegs = pegs
        self.start = start
        self.target = target
        self.auxiliary_pegs = [peg for peg in pegs if peg not in (start, target)]
        self.pegs_state: dict[int, list[int]] = {peg: [] for peg in pegs}
        for disk in range(num_disks, 0, -1):  # Largest disk at the bottom
            self.pegs_state[start].append(disk)
        self.moves: list[str] = []
        self.memo: dict[tuple[int, int], int] = {}  # Memoization for T(n, k)

    def generate_moves(self) -> list[str]:
        self.move(n=self.num_disks, source=self.start, target=self.target, auxiliary_pegs=self.auxiliary_pegs)
        return self.moves

    def move(self, n: int, source: int, target: int, auxiliary_pegs: list[int]):
        if n == 0:
            return
        if n == 1:
            self._move_disk(source, target)
            return

        k = len(auxiliary_pegs) + 2  # Total number of pegs including source and target

        if k < 3:
            raise ValueError("At least 3 pegs are required.")

        if k == 3:
            # Classic Tower of Hanoi solution
            aux = auxiliary_pegs[0]
            self.move(n - 1, source, aux, [target])
            self._move_disk(source, target)
            self.move(n - 1, aux, target, [source])
            return

        # For k > 3, apply Frame-Stewart algorithm
        # Find m that minimizes 2*T(m, k) + T(n - m, k - 1)
        min_moves = math.inf
        best_m = 1
        for m in range(1, n):
            moves_m = self._compute_T(m, k)
            moves_n_minus_m = self._compute_T(n - m, k - 1)
            total_moves = 2 * moves_m + moves_n_minus_m
            if total_moves < min_moves:
                min_moves = total_moves
                best_m = m

        # Select a temporary peg to hold m disks
        temp_peg = auxiliary_pegs[0]
        new_auxiliary = [peg for peg in auxiliary_pegs if peg != temp_peg]

        # Step 1: Move top m disks to temp_peg using all pegs
        self.move(n=best_m, source=source, target=temp_peg, auxiliary_pegs=auxiliary_pegs[1:] + [target])

        # Step 2: Move remaining n - m disks to target using k - 1 pegs
        self.move(n=n - best_m, source=source, target=target, auxiliary_pegs=new_auxiliary)

        # Step 3: Move m disks from temp_peg to target using all pegs
        self.move(n=best_m, source=temp_peg, target=target, auxiliary_pegs=auxiliary_pegs[1:] + [source])

    def _move_disk(self, from_peg: int, to_peg: int):
        if not self.pegs_state[from_peg]:
            raise ValueError(f"No disks to move from Peg {from_peg}.")
        disk = self.pegs_state[from_peg][-1]
        self.pegs_state[from_peg].pop()
        self.pegs_state[to_peg].append(disk)
        self.moves.append(f"Move disk {disk} from Peg {from_peg} to Peg {to_peg}")

    def _compute_T(self, n: int, k: int) -> int:
        """
        Compute the minimal number of moves (T(n, k)) required to move n disks using k pegs.
        Utilizes memoization to store previously computed results.
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        if k == 3:
            return 2**n - 1
        if (n, k) in self.memo:
            return self.memo[(n, k)]

        min_moves = math.inf
        for m in range(1, n):
            moves = 2 * self._compute_T(m, k) + self._compute_T(n - m, k - 1)
            if moves < min_moves:
                min_moves = moves
        self.memo[(n, k)] = min_moves
        return min_moves


class HanoiDataset(ProceduralDataset):
    """
    Generates Tower of Hanoi problems with solutions.
    Supports variable number of pegs using the optimized Frame-Stewart algorithm with Peg State Tracking.
    """

    def __init__(self, config: HanoiConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.min_pegs = config.min_pegs
        self.max_pegs = config.max_pegs
        self.min_disks = config.min_disks
        self.max_disks = config.max_disks
        self.visualize = config.visualize  # Initialize the visualize attribute

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a Tower of Hanoi problem instance.

        Returns:
            dict with:
            - "question": Text describing the problem setup.
            - "answer": list of moves to solve the puzzle.
            - "metadata": Configuration and solution details.
            - "initial_state": (Optional) ASCII visualization of the initial pegs.
            - "states": (Optional) list of ASCII visualizations after each move.
        """
        rng = random.Random(self.seed + idx if self.seed is not None else None)

        # Randomly select number of disks and pegs within the specified ranges
        num_disks = rng.randint(self.min_disks, self.max_disks)
        num_pegs = rng.randint(self.min_pegs, self.max_pegs)

        # Assign unique peg identifiers (e.g., integers starting from 1)
        pegs = list(range(1, num_pegs + 1))

        """ #Debug: Print current instance configuration
        print(f"\n--- Generating Instance {idx} ---")
        print(f"Number of Disks: {num_disks}")
        print(f"Number of Pegs: {num_pegs}")
        print(f"Pegs: {pegs}")
        """

        # Randomly select start and target pegs
        start_peg, target_peg = rng.sample(pegs, 2)

        # Auxiliary pegs are the remaining pegs
        auxiliary_pegs = [peg for peg in pegs if peg not in (start_peg, target_peg)]

        """ # Debug: Print start, target, and auxiliary pegs
        print(f"Start Peg: {start_peg}")
        print(f"Target Peg: {target_peg}")
        print(f"Auxiliary Pegs: {auxiliary_pegs}")
        """

        # Initialize the MoveGenerator and generate moves
        move_gen = MoveGenerator(num_disks, pegs, start_peg, target_peg)
        try:
            solution = move_gen.generate_moves()
        except ValueError as ve:
            # print(f"Error during move generation: {ve}")
            raise ve

        """ # Debug: Print the solution moves
        print(f"Solution Length: {len(solution)}")
        print("Solution Moves:")
        for move_num, move in enumerate(solution, start=1):
            print(f"  Move {move_num}: {move}")
        """

        # Initialize pegs_state: all disks start on the start peg
        pegs_state = {peg: [] for peg in pegs}
        for disk in range(num_disks, 0, -1):  # Largest disk at the bottom
            pegs_state[start_peg].append(disk)

        # Generate initial state visualization if requested
        initial_state_str = None
        if self.visualize:
            initial_state_str = self._visualize_state(pegs_state)

        # Apply moves to track state changes
        states = []
        if self.visualize:
            states.append(initial_state_str)  # Initial state
            for move in solution:
                # Parse the move string using regex
                try:
                    disk, from_peg, to_peg = self._parse_move(move)
                except ValueError as ve:
                    # print(f"Error parsing move: {ve}")
                    raise ve

                # Validate the move
                if not self._validate_move(pegs_state, move):
                    # print(f"Invalid move detected: {move}")
                    # print(f"Current Pegs State: {pegs_state}")
                    raise ValueError(f"Invalid move detected: {move}")

                # Move the disk
                pegs_state[from_peg].pop()
                pegs_state[to_peg].append(disk)

                # Visualize the new state
                new_state_str = self._visualize_state(pegs_state)
                states.append(new_state_str)

        # Peg labels
        peg_labels = {peg: f"Peg {peg}" for peg in pegs}

        result = {
            "question": QUESTION_TEMPLATE.format(
                num_disks=num_disks,
                num_pegs=num_pegs,
                start_peg=peg_labels[start_peg],
                target_peg=peg_labels[target_peg],
            ),
            "answer": "\n".join(solution),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_disks": num_disks,
                "num_pegs": num_pegs,
                "start_peg": start_peg,
                "target_peg": target_peg,
                "auxiliary_pegs": auxiliary_pegs,
                "solution_length": len(solution),
                "difficulty": {
                    "num_disks": (self.min_disks, self.max_disks),
                },
            },
        }

        if self.visualize:
            result["initial_state"] = initial_state_str
            result["states"] = states  # list of all states including initial and after each move

        return result

    def _visualize_state(self, pegs_state: dict[int, list[int]]) -> str:
        """
        Create an ASCII visualization of the current state of the pegs.
        Adapts to variable number of pegs.

        Args:
            pegs_state (dict): Dictionary mapping peg numbers to lists of disks.

        Returns:
            str: ASCII art representing the pegs and disks.
        """
        # Determine the number of levels based on the maximum number of disks on any peg
        max_height = max(len(disks) for disks in pegs_state.values())
        pegs = sorted(pegs_state.keys())

        visualization = ""
        for level in range(max_height, 0, -1):
            for peg in pegs:
                if len(pegs_state[peg]) >= level:
                    disk_size = pegs_state[peg][level - 1]
                    disk_str = f"[{'*' * disk_size}]"
                else:
                    disk_str = "[ ]"
                visualization += disk_str.center(7)  # Adjust spacing as needed
            visualization += "\n"

        # Add the base and peg numbers
        visualization += "-" * (7 * len(pegs)) + "\n"
        for peg in pegs:
            peg_label = f"P{peg}".center(7)
            visualization += peg_label
        visualization += "\n"

        return visualization

    def _validate_move(self, pegs_state: dict[int, list[int]], move: str) -> bool:
        """
        Validate that a move adheres to the Tower of Hanoi rules.

        Args:
            pegs_state (dict): Current state of the pegs.
            move (str): Move instruction, e.g., "Move disk 2 from Peg 1 to Peg 3".

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        try:
            parts = move.split()
            if len(parts) != 9:
                # print(f"Unexpected move format: '{move}'")
                return False
            disk = int(parts[2])
            from_peg = int(parts[5])
            to_peg = int(parts[8])

            # Check if the disk to move is the top disk on the from_peg
            if not pegs_state[from_peg] or pegs_state[from_peg][-1] != disk:
                # print(f"Disk {disk} is not on top of Peg {from_peg}. Current state: {pegs_state[from_peg]}")
                return False

            # Check if placing the disk on the to_peg violates size constraints
            if pegs_state[to_peg] and pegs_state[to_peg][-1] < disk:
                # print(f"Cannot place disk {disk} on top of smaller disk {pegs_state[to_peg][-1]} on Peg {to_peg}.")
                return False

            return True
        except Exception as e:
            print(f"Error validating move '{move}': {e}")
            return False

    def _parse_move(self, move: str) -> tuple[int, int, int]:
        """
        Parse a move string and extract disk number, from peg, and to peg.

        Args:
            move (str): Move instruction, e.g., "Move disk 2 from Peg 1 to Peg 3".

        Returns:
            tuple: (disk, from_peg, to_peg)
        """
        pattern = r"Move disk (\d+) from Peg (\d+) to Peg (\d+)"
        match = re.search(pattern, move)
        if not match:
            raise ValueError(f"Unexpected move format: '{move}'")

        disk = int(match.group(1))
        from_peg = int(match.group(2))
        to_peg = int(match.group(3))
        return disk, from_peg, to_peg

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """
        Score the user's solution for the Tower of Hanoi puzzle.

        The answer is expected to be a newline-separated sequence of moves in the format:
        "Move disk X from Peg Y to Peg Z"

        Expected behavior:
            - Correct answer (i.e. equivalent in length, or better, than the one provided in the dataset item) gives 1.0.
            - A correct solution that is suboptimal length gives a proportional reward of optimal_move_count/user_move_count
            - An answer that is syntactically valid but does not solve the puzzle gives a partial reward (0.05).
            - A badly formatted or empty answer gives 0.0
        """
        if not isinstance(answer, str) or len(answer) == 0:
            return 0.0

        # Spilt answer string it into lines
        moves = [line.strip() for line in answer.strip().splitlines() if line.strip()]

        # Build the initial peg state from metadata.
        metadata = entry["metadata"]
        num_disks = metadata["num_disks"]
        num_pegs = metadata["num_pegs"]
        start_peg = metadata["start_peg"]
        target_peg = metadata["target_peg"]

        peg_state = {peg: [] for peg in range(1, num_pegs + 1)}
        for disk in range(num_disks, 0, -1):
            peg_state[start_peg].append(disk)

        # Process each move.
        for move in moves:
            try:
                disk, from_peg, to_peg = self._parse_move(move)
            except Exception:
                return 0.0  # Invalid move format

            # Validate the move using existing _validate_move method.
            if not self._validate_move(peg_state, move):
                return 0.0

            # Execute the move.
            peg_state[from_peg].pop()
            peg_state[to_peg].append(disk)

        # Check if the final state is solved: all disks on target peg in descending order.
        expected_final = list(range(num_disks, 0, -1))
        solved = peg_state[target_peg] == expected_final
        if not solved:
            return 0.05

        optimal_moves = metadata.get("solution_length", len(moves))
        user_moves = len(moves)
        if user_moves <= optimal_moves:
            return 1.0
        else:
            return optimal_moves / user_moves


class HanoiCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(HanoiCurriculum.__name__, HanoiConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_disks",
                levels=[3, 4, 5, 7],
                min_disks=3,
                lower_field_name="min_disks",
                upper_field_name="max_disks",
                description="Number of disks in the puzzle",
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, HanoiDataset, HanoiConfig, HanoiCurriculum)
