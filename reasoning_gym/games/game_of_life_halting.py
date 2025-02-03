from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Tuple

import cellpylib as cpl

from ..factory import ProceduralDataset, register_dataset


@dataclass
class GameOfLifeHaltingConfig:
    """Configuration for Game of Life halting problems generation"""

    grid_size_x: int = 24
    grid_size_y: int = 24
    max_difficulty: int = 1
    oscillators: int = 1
    max_simulation_steps: int = 1
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert 3 <= self.grid_size_x <= 999, "grid_size_x must be between 0 and 999"
        assert 3 <= self.grid_size_y <= 999, "grid_size_y must be between 0 and 999"
        assert self.simulation_steps >= 0, "simulation_steps must be gte 0"

class GameOfLifeHaltingDataset(ProceduralDataset):
    """Generates Game of Life games with configurable parameters"""

    # via this great wiki https://conwaylife.com/wiki/oscillator
    OSCILLATORS = [
        {
            "name": "blinker",
            "size_x": 3,
            "size_y": 3,
            "period": 2,
            'difficulty': 1,
            'cells': [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ]
        }
    ]

    def __init__(self, config: GameOfLifeHaltingConfig):
        self._prompt_templates = [
            "What will this Game of Life board look like after {simulation_steps} steps of simulation?\n\n{board}"
        ]

        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GameOfLife task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Make the board
        board = cpl.init_simple2d(self.config.grid_size_x, self.config.grid_size_y)
        # Reset the board to all 0s. [timestep, x, y] format
        board[:, :, :] = 0

        # Choose some oscillators at or below this difficulty

        # Add the oscillators with a 1-cell buffer to a random location where they fit and don't overlap any buffer

        # Evolve the solution
        evolved = cpl.evolve2d(
            board, timesteps=self.config.max_simulation_steps + 1, apply_rule=cpl.game_of_life_rule, memoize="recursive"
        )

        board_str = str(board[0])
        result_str = str(evolved[-1])

        return {
            "question":f "This is a 'Game of Life' grid. We consider a game halted if there are no cells alive. Will this game halt at or before {max_simulation_steps} steps?",
            "answer": result_str,
            "metadata": {
                "grid_size_x": self.config.grid_size_x,
                "grid_size_y": self.config.grid_size_y,
            },
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the GoL task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if answer.replace("\n", "") != entry["answer"].replace("\n", ""):
            return 0.01
        else:
            return 1.0  # Yay


register_dataset("game_of_life_halting", GameOfLifeHaltingDataset, GameOfLifeHaltingConfig)
