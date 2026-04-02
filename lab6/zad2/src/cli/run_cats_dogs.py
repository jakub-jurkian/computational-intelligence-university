from __future__ import annotations

import argparse
import json

from src.cats_dogs import CatsDogsConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cats-vs-dogs experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = CatsDogsConfig.from_yaml(args.config)
    summary = run_experiment(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
