from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wypisz dostepne konfiguracje eksperymentow.")
    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="Katalog z plikami YAML eksperymentow.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("list_configs")

    configs_dir = Path(args.configs_dir)
    config_paths = sorted(configs_dir.glob("*.yaml"))
    experiment_configs = [
        path
        for path in config_paths
        if path.name != "compare_experiments.yaml" and not path.name.startswith("invalid_")
    ]

    logger.info("Dostepne konfiguracje eksperymentow (%d):", len(experiment_configs))
    for path in experiment_configs:
        logger.info("- %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
