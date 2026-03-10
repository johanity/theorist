"""Theorist -- prediction-error learning for optimization.

Every experiment teaches you something. Theorist remembers.

Based on: https://github.com/johanity/epistemic-autoresearch
"""

from .brain import Brain
from .compare import compare
from .core import Theorist
from .experiment import experiment
from .types import Experiment, Prediction, Results

__version__ = "0.1.0"
__author__ = "Johan David Bonilla"
__url__ = "https://github.com/johanity/theorist"
__all__ = ["Brain", "Theorist", "compare", "experiment", "Experiment", "Prediction", "Results"]
