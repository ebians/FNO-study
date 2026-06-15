import argparse
import json
from pathlib import Path
from typing import Any, Dict


class EvalError(ValueError):
    """Raised when evaluation inputs are invalid."""


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise EvalError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_baseline_report(training_summary: Dict[str, Any], history_path: Path) -> None:
    print("=" * 70)
    print("FNO Baseline Evaluation Report")
    print("=" * 70)

    config = training_summary.get("config", {})
    print("\n[Training Configuration]")
    print(f"  Dataset:         {config.get('dataset_path')}")
    print(f"  Checkpoint:      {training_summary.get('checkpoint_path')}")
    print(f"  Total Samples:   {training_summary.get('num_samples', 'N/A')}")
    print(f"  Train Samples:   {training_summary.get('num_train_samples', 'N/A')}")
    print(f"  Val Samples:     {training_summary.get('num_val_samples', 'N/A')}")
    print(f"  Test Samples:    {training_summary.get('num_test_samples', 'N/A')}")
    print(f"  Epochs:          {config.get('epochs')}")
    print(f"  Learning Rate:   {config.get('learning_rate')}")
    print(f"  Batch Size:      {config.get('batch_size')}")
    print(f"  Width:           {config.get('width')}")
    print(f"  Depth:           {config.get('depth')}")

    print("\n[Model Performance]")
    print(f"  Best Epoch:      {training_summary.get('best_epoch', 'N/A')}")
    print(f"  Best Val MSE:    {training_summary.get('best_val_mse', 'N/A'):.6e}")
    test_mse = training_summary.get("test_mse")
    test_mae = training_summary.get("test_mae")
    if not (isinstance(test_mse, float) and test_mse != test_mse):
        print(f"  Test MSE:        {test_mse:.6e}" if test_mse is not None else "  Test MSE:        N/A")
    if not (isinstance(test_mae, float) and test_mae != test_mae):
        print(f"  Test MAE:        {test_mae:.6e}" if test_mae is not None else "  Test MAE:        N/A")

    if history_path.exists():
        history = load_json(history_path)
        if history:
            first_mse = history[0].get("train_mse")
            last_entry = history[-1]
            final_mse = last_entry.get("train_mse")
            print(f"\n[Training Progress]")
            print(f"  Initial MSE:     {first_mse:.6e}" if first_mse else "  Initial MSE:     N/A")
            print(f"  Final MSE:       {final_mse:.6e}" if final_mse else "  Final MSE:       N/A")
            print(f"  Final Val MSE:   {last_entry.get('val_mse'):.6e}" if last_entry.get("val_mse") else "  Final Val MSE:   N/A")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display baseline evaluation from training summary.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/training/training_summary.json"),
        help="Training summary JSON path.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/training/training_history.json"),
        help="Training history JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = load_json(args.summary)
    print_baseline_report(summary, args.history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
