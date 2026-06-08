"""
deeplocres: A package for cryo-EM local resolution prediction.
"""

__version__ = "0.1.0"

# Optionally, expose key functions for easier access:
from .infer import main as infer_main
from .train import main as train_main

