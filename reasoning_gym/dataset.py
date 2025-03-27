"""Base class for procedural dataset generators"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sized
from copy import deepcopy
from random import Random
from typing import Any, Iterator, Optional, Type, TypeVar


class ProceduralDataset(ABC, Sized, Iterable[dict[str, Any]]):
    """Abstract base class for procedural dataset generators"""

    def __init__(self, config: Any, seed: Optional[int] = None, size: int = 500):
        """Initialize the dataset with config, optional seed and size"""
        if hasattr(config, "validate") and callable(config.validate):
            config.validate()

        self.config = config
        self.size = size
        self.seed = seed if seed is not None else Random().randint(0, 2**32)

    @property
    def category(self) -> str:
        """Extract category from the module name."""
        module_name = self.__class__.__module__
        parts = module_name.split(".")
        if len(parts) >= 3:
            return parts[1]  # reasoning_gym.{category}.dataset_name
        return "other"

    def __len__(self) -> int:
        """Return the virtual size of the dataset"""
        return self.size

    def __iter__(self):
        """Make the dataset iterable"""
        self._current_idx = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Get next item in iteration"""
        if self._current_idx >= self.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single dataset item

        Args:
            idx: Index of the item to generate

        Returns:
            dict containing at least:
                - question: str
                - answer: str
                - metadata: dict
        """
        raise NotImplementedError

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        reward = 0.0
        if isinstance(answer, str) and len(answer) > 0:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = len(oracle_answer) / len(answer)
        return reward


T = TypeVar("T", bound="ProceduralDataset")


class ReseedingDataset(Iterable[dict[str, Any]]):
    """Wrapper that makes any ProceduralDataset infinite by reseeding when reaching the end"""

    def __init__(self, dataset: T, chunk_size: int = 500):
        """Initialize with dataset instance and chunk size

        Args:
            dataset: The ProceduralDataset instance to wrap
            chunk_size: Size of each generated chunk before reseeding
        """
        self.dataset = dataset
        self.dataset_cls: Type[T] = type(dataset)
        self.chunk_size = chunk_size

        # Start with chunk 0
        self._current_chunk = 0
        self._current_dataset = self._create_chunk(0)
        self._current_idx = 0

    def _create_chunk(self, chunk_num: int) -> T:
        """Create a new dataset chunk with unique seed"""
        # Create new config with modified seed
        new_config = deepcopy(self.dataset.config)
        if hasattr(new_config, "seed"):
            # Derive new seed from chunk number using dataset's seed, wrapping around at 2^32
            new_config.seed = (self.dataset.seed + chunk_num) % (2**32)

        # Create new dataset instance with chunk config
        return self.dataset_cls(new_config)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Make the dataset iterable"""
        self._current_chunk = 0
        self._current_dataset = self._create_chunk(0)
        self._current_idx = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Get next item, creating new chunk if needed"""
        if self._current_idx >= self.chunk_size:
            # Move to next chunk
            self._current_chunk += 1
            self._current_dataset = self._create_chunk(self._current_chunk)
            self._current_idx = 0

        item = self._current_dataset[self._current_idx]
        self._current_idx += 1
        return item

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Forward scoring to the wrapped dataset's implementation"""
        return self.dataset.score_answer(answer, entry)
