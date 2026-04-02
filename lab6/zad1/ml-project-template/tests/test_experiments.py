import pytest

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig


def test_experiment_factory_lists_expected_experiments() -> None:
    available = ExperimentFactory.list()
    assert "mlp_classifier" in available
    assert "linear_classifier" in available
    assert "mlp_dropout_classifier" in available


def test_experiment_factory_rejects_unknown_experiment_name() -> None:
    config = ProjectConfig.from_yaml("configs/iris_mlp_zscore.yaml")
    with pytest.raises(ValueError, match="Nieznany eksperyment"):
        ExperimentFactory.build("unknown_experiment", config)
