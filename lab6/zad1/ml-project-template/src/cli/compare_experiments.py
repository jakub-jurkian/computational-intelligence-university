from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import ComparisonConfig, ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Porownaj wyniki wielu eksperymentow.")
    parser.add_argument(
        "--config", required=True, help="Sciezka do pliku YAML z konfiguracja porownania."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_config = ComparisonConfig.from_yaml(args.config)
    logger = get_logger("compare_experiments")

    rows: list[dict[str, str | float]] = []
    for experiment_config_path in compare_config.experiments:
        config = ProjectConfig.from_yaml(experiment_config_path)
        training_summary = read_json(config.paths.training_summary_json)
        evaluation = read_json(config.paths.evaluation_json)
        classification_report = evaluation["classification_report"]
        macro_f1 = float(classification_report["macro avg"]["f1-score"])
        rows.append(
            {
                "experiment_name": config.experiment_name,
                "classifier": config.model.name,
                "preprocessing": config.preprocessing.strategy,
                "best_validation_accuracy": float(training_summary["best_validation_accuracy"]),
                "test_accuracy": float(evaluation["accuracy"]),
                "macro_f1": macro_f1,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["test_accuracy", "best_validation_accuracy"],
        ascending=False,
    )
    ensure_parent(compare_config.comparison_csv)
    results.to_csv(compare_config.comparison_csv, index=False)
    write_json(
        compare_config.comparison_json,
        {"experiments": rows},
    )

    ensure_parent(compare_config.comparison_plot_png)
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    positions = range(len(results))
    width = 0.4
    axis.bar(
        [position - width / 2 for position in positions],
        results["test_accuracy"],
        width=width,
        label="test_accuracy",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        results["macro_f1"],
        width=width,
        label="macro_f1",
    )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(results["experiment_name"])
    axis.set_ylabel("Metric value")
    axis.set_title("Porownanie eksperymentow")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(compare_config.comparison_plot_png)
    plt.close(fig)

    logger.info("Zapisano porownanie eksperymentow do %s", compare_config.comparison_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
