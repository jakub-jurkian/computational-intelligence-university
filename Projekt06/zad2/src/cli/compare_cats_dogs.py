from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cats-vs-dogs experiment summaries.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of summary.json paths.")
    parser.add_argument("--out-dir", required=True, help="Output directory for comparison artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in args.inputs:
        with Path(item).open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))

    frame = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    csv_path = out_dir / "comparison.csv"
    frame.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.bar(frame["experiment_name"], frame["test_acc"])
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Experiment")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    plot_path = out_dir / "comparison_test_acc.png"
    fig.savefig(plot_path)
    plt.close(fig)

    best = frame.iloc[0].to_dict()
    result = {
        "best_experiment": best["experiment_name"],
        "best_test_acc": best["test_acc"],
        "comparison_csv": str(csv_path),
        "comparison_plot": str(plot_path),
    }
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
