import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class HyperparamConfig:
    width: int
    depth: int
    learning_rate: float
    label: str


class HPSearchError(ValueError):
    """Raised when hyperparameter search fails."""


def run_training(config: HyperparamConfig, dataset_path: Path, split_file: Path, epochs: int, device: str) -> Dict[str, Any]:
    output_dir = Path("artifacts/hp_search") / config.label
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "train_fno.py",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--split-file",
        str(split_file),
        "--train-ratio",
        "0.7",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.1",
        "--epochs",
        str(epochs),
        "--batch-size",
        "8",
        "--width",
        str(config.width),
        "--depth",
        str(config.depth),
        "--learning-rate",
        str(config.learning_rate),
        "--device",
        device,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode != 0:
        raise HPSearchError(f"Training failed for {config.label}")

    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        raise HPSearchError(f"Training summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return {"config": config.__dict__, "summary": summary, "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search over FNO hyperparameters.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/fno_dataset.npz"), help="Dataset NPZ path.")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("artifacts/splits/train_val_test_split.json"),
        help="Split file path.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--output-report", type=Path, default=Path("artifacts/hp_search_report.json"), help="Report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configs = [
        HyperparamConfig(width=16, depth=2, learning_rate=1e-3, label="narrow_shallow"),
        HyperparamConfig(width=32, depth=4, learning_rate=1e-3, label="medium_medium"),
        HyperparamConfig(width=64, depth=4, learning_rate=1e-3, label="wide_medium"),
        HyperparamConfig(width=32, depth=6, learning_rate=1e-3, label="medium_deep"),
        HyperparamConfig(width=32, depth=4, learning_rate=5e-4, label="medium_medium_low_lr"),
        HyperparamConfig(width=32, depth=4, learning_rate=5e-3, label="medium_medium_high_lr"),
    ]

    results = []
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Training: {config.label}")
        print(f"  width={config.width}, depth={config.depth}, lr={config.learning_rate}")
        print(f"{'='*70}")
        try:
            result = run_training(config, args.dataset, args.split_file, args.epochs, args.device)
            results.append(result)
            summary = result["summary"]
            print(f"Best Val MSE:  {summary.get('best_val_mse', 'N/A')}")
            print(f"Test MSE:      {summary.get('test_mse', 'N/A')}")
            print(f"Test MAE:      {summary.get('test_mae', 'N/A')}")
        except HPSearchError as e:
            print(f"ERROR: {e}")
            continue

    report = {
        "num_configs": len(results),
        "results": results,
        "best_config": None,
    }

    if results:
        best_result = min(results, key=lambda r: r["summary"].get("test_mse", float("inf")))
        report["best_config"] = {
            "label": best_result["config"]["label"],
            "config": best_result["config"],
            "test_mse": best_result["summary"].get("test_mse"),
            "test_mae": best_result["summary"].get("test_mae"),
        }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Report saved: {args.output_report}")
    if report["best_config"]:
        print(f"Best: {report['best_config']['label']}")
        print(f"  Test MSE: {report['best_config']['test_mse']}")
        print(f"  Test MAE: {report['best_config']['test_mae']}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
