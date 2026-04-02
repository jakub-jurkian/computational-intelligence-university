from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.mlp_dropout import IrisMLPDropout


@register_experiment("mlp_dropout_classifier")
class MLPDropoutClassifierExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "mlp_dropout_classifier"

    def build_model(self) -> IrisMLPDropout:
        hidden_dims = self.config.training.hidden_dims or [16, 16]
        return IrisMLPDropout(
            input_dim=len(self.config.data.feature_columns),
            hidden_dims=hidden_dims,
            num_classes=len(self.config.data.class_names),
        )
