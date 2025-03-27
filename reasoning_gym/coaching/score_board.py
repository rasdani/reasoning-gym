"""Coaching module for difficulty adjustment and score tracking"""

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional, Union

from ..dataset import ProceduralDataset


@dataclass
class ScoreStats:
    """Container for score statistics with mean, std, min, max"""

    scores: OrderedDict[tuple[tuple[str, Any], ...], tuple[int, float, float, float, float]]

    def __str__(self) -> str:
        """Create a formatted report of the statistics

        Returns:
            Multi-line string with statistics for each group
        """
        if not self.scores:
            return "No scores recorded"

        lines = []

        for key, values in self.scores.items():
            params = ", ".join(f"{k}={v}" for k, v in key)
            lines.append(
                f"({params}): n={values[0]}, μ={values[1]:.3f}, σ={values[2]:.3f}, min={values[3]:.3f}, max={values[4]:.3f}"
            )

        return "\n".join(lines)


@dataclass
class GroupedScores:
    """Container for grouped scores with total count"""

    scores: OrderedDict[tuple[tuple[str, Any], ...], list[float]]
    total_scores: int

    def __str__(self) -> str:
        """Create a formatted report of the scores

        Returns:
            Multi-line string with score information for each difficulty group
        """
        if not self.scores:
            return "No scores recorded"

        lines = []
        lines.append(f"Total scores: {self.total_scores}")
        lines.append("")

        for key, values in self.scores.items():
            # Format the parameter combinations
            params = ", ".join(f"{k}={v}" for k, v in key)
            stats = (
                len(values),
                mean(values) if values else 0.0,
                stdev(values) if len(values) > 1 else 0.0,
                min(values) if values else 0.0,
                max(values) if values else 0.0,
            )
            lines.append(
                f"({params}): n={stats[0]}, μ={stats[1]:.3f}, σ={stats[2]:.3f}, min={stats[3]:.3f}, max={stats[4]:.3f}"
            )
            # Format score list, showing only last 100 if more
            score_strs = [f"{x:.2f}" for x in values[-100:]]
            if len(values) > 100:
                score_strs.insert(0, "..")
            lines.append(f"  Values: {', '.join(score_strs)}")

        return "\n".join(lines)

    def stats(self, ignore_empty: bool = True) -> ScoreStats:
        """Calculate statistics for each group of scores

        Args:
            ignore_empty: If True, skip empty score lists
                         If False, use NaN values for empty lists

        Returns:
            ScoreStats object containing statistics for each group
        """
        result = OrderedDict()

        for key, values in self.scores.items():
            if not values and ignore_empty:
                continue

            if not values:
                # Empty list and not ignoring - use NaN
                result[key] = (0, math.nan, math.nan, math.nan, math.nan)
            else:
                # Calculate stats as tuple
                result[key] = (
                    len(values),
                    mean(values),
                    stdev(values) if len(values) > 1 else 0.0,
                    min(values),
                    max(values),
                )

        return ScoreStats(scores=result)


@dataclass
class ScoreBoard:
    """Tracks scores and metadata for coaching sessions"""

    scores: list[float] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    conversations: list[Optional[list[dict]]] = field(default_factory=list)

    def add_score(self, score: float, metadata: dict[str, Any], conversation: Optional[list[dict]] = None) -> None:
        """Add a new score entry with associated metadata and optional conversation

        Args:
            score: The score achieved (typically 0.0-1.0)
            metadata: Dictionary of metadata about the task/attempt
            conversation: Optional list of conversation turns as dicts
        """
        self.scores.append(score)
        self.metadata.append(metadata)
        self.conversations.append(conversation)

    def clear(self) -> None:
        """Clear all stored scores, metadata and conversations"""
        self.scores.clear()
        self.metadata.clear()
        self.conversations.clear()

    def __len__(self) -> int:
        """Return the number of stored scores"""
        return len(self.scores)

    def _metadata_to_key(self, metadata: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        """Convert metadata dict to tuple of key-value pairs, sorted by key

        If source_dataset and source_index are present in metadata, they will be
        placed first in the tuple as ("source", dataset) and ("idx", index).
        """
        # Start with empty list
        key_items = [("source", metadata["source_dataset"]), ("idx", metadata["source_index"])]

        # Add difficulty parameters or other metadata
        if "difficulty" in metadata:
            # Use only difficulty parameters
            items = metadata["difficulty"].items()
        else:
            # Use all metadata except source info
            items = ((k, v) for k, v in metadata.items() if k not in ("source_dataset", "source_index"))

        # Add remaining items in sorted order
        key_items.extend(sorted((str(k), v) for k, v in items))

        return tuple(key_items)

    def aggregate(self, last_n: Optional[int] = None) -> GroupedScores:
        """Aggregate scores by difficulty parameters or full metadata if no difficulty present

        Args:
            last_n: Optional number of most recent entries to consider
                   If None, use all entries

        Returns:
            OrderedDict mapping difficulty parameter combinations to lists of scores
            Keys are tuples of (param_name, value) pairs, sorted by param_name
        """
        if not self.scores:
            return GroupedScores(scores=OrderedDict(), total_scores=0)

        # Determine start index for iteration
        start_idx = max(0, len(self.scores) - last_n) if last_n is not None else 0

        # Group scores by difficulty parameters without creating intermediate lists
        result = OrderedDict()
        for i in range(start_idx, len(self.scores)):
            key = self._metadata_to_key(self.metadata[i])
            if key not in result:
                result[key] = []
            result[key].append(self.scores[i])

        # Count total scores
        total_scores = sum(len(scores) for scores in result.values())

        return GroupedScores(scores=result, total_scores=total_scores)
