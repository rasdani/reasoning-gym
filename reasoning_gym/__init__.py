"""
Reasoning Gym - A library of procedural dataset generators for training reasoning models
"""

from .factory import create_dataset, register_dataset

__version__ = "0.1.1"
__all__ = [
    "create_dataset",
    "register_dataset",
]
