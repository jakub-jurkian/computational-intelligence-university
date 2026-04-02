"""Experiment registry and experiment implementations."""

from src.experiments.linear_experiment import LinearClassifierExperiment
from src.experiments.mlp_dropout_experiment import MLPDropoutClassifierExperiment
from src.experiments.mlp_experiment import MLPClassifierExperiment
from src.experiments.registry import ExperimentFactory, register_experiment

__all__ = [
    "ExperimentFactory",
    "LinearClassifierExperiment",
    "MLPClassifierExperiment",
    "MLPDropoutClassifierExperiment",
    "register_experiment",
]
