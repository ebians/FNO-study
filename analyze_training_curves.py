import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CurveAnalysisError(ValueError):
    """Raised when curve analysis fails."""


def load_history(path: Path) -> List[Dict[str, Any]]:
    """Load training history from JSON."""
    if not path.exists():
        raise CurveAnalysisError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_overfitting(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect overfitting by comparing train and val loss."""
    if not history:
        raise CurveAnalysisError("Empty history")

    train_losses = [h.get("train_mse", 0.0) for h in history]
    val_losses = [h.get("val_mse", 0.0) for h in history]

    if not train_losses or not val_losses:
        raise CurveAnalysisError("Missing train_mse or val_mse in history")

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    best_val_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_epoch]

    gap_at_end = val_losses[-1] - train_losses[-1]
    gap_at_best = val_losses[best_val_epoch] - train_losses[best_val_epoch]

    overfitting_ratio = (val_losses[-1] - train_losses[-1]) / (train_losses[-1] + 1e-8)

    return {
        "best_val_epoch": int(best_val_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "gap_at_best": float(gap_at_best),
        "gap_at_end": float(gap_at_end),
        "overfitting_ratio": float(overfitting_ratio),
        "early_stopping_recommended": bool(best_val_epoch < len(history) * 0.8),
    }


def analyze_convergence_rate(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how fast the model converges."""
    train_losses = np.array([h.get("train_mse", 0.0) for h in history])

    if len(train_losses) < 2:
        raise CurveAnalysisError("Need at least 2 epochs for convergence analysis")

    initial_loss = train_losses[0]
    final_loss = train_losses[-1]

    first_half_improvement = initial_loss - train_losses[len(train_losses) // 2]
    first_half_ratio = first_half_improvement / (initial_loss + 1e-8)

    convergence_threshold = initial_loss * 0.05
    epochs_to_threshold = None
    for i, loss in enumerate(train_losses):
        if loss < convergence_threshold:
            epochs_to_threshold = i
            break

    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "total_improvement": float(initial_loss - final_loss),
        "improvement_ratio": float((initial_loss - final_loss) / (initial_loss + 1e-8)),
        "first_half_improvement_ratio": float(first_half_ratio),
        "epochs_to_5pct_threshold": epochs_to_threshold,
        "convergence_speed": "fast" if first_half_ratio > 0.5 else "slow",
    }


def analyze_stability(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect unstable training (oscillations, spikes)."""
    val_losses = np.array([h.get("val_mse", 0.0) for h in history])

    if len(val_losses) < 2:
        return {"stability_score": 1.0, "issues": []}

    diffs = np.abs(np.diff(val_losses))
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    large_jumps = np.sum(diffs > mean_diff + 2 * std_diff)

    stability_score = max(0.0, 1.0 - (large_jumps / len(diffs)))

    issues = []
    if large_jumps > 0:
        issues.append(f"Detected {large_jumps} large loss spikes")
    if std_diff > mean_diff * 0.5:
        issues.append("High variance in loss changes")

    return {"stability_score": float(stability_score), "num_large_jumps": int(large_jumps), "issues": issues}


def analyze_learning_rate_effect(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate if learning rate is appropriate."""
    train_losses = np.array([h.get("train_mse", 0.0) for h in history])

    if len(train_losses) < 5:
        return {"lr_assessment": "insufficient_data"}

    early_phase = train_losses[:5]
    late_phase = train_losses[-5:]

    early_var = np.var(early_phase)
    late_var = np.var(late_phase)

    if np.min(train_losses) == train_losses[-1]:
        assessment = "LR may be too low (still improving at end)"
    elif early_var > late_var * 10:
        assessment = "LR might be too high (high early variance)"
    else:
        assessment = "LR seems appropriate"

    return {
        "lr_assessment": assessment,
        "early_phase_variance": float(early_var),
        "late_phase_variance": float(late_var),
    }


def generate_report(history_path: Path, summary_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate comprehensive training analysis report."""
    history = load_history(history_path)

    report = {
        "num_epochs": len(history),
        "overfitting_analysis": analyze_overfitting(history),
        "convergence_analysis": analyze_convergence_rate(history),
        "stability_analysis": analyze_stability(history),
        "learning_rate_analysis": analyze_learning_rate_effect(history),
    }

    return report


def plot_training_curves(history: List[Dict[str, Any]], output_path: Path, title: str = "Training Curves") -> None:
    """Plot training and validation curves."""
    if not HAS_MATPLOTLIB:
        raise CurveAnalysisError("matplotlib not installed")

    epochs = list(range(len(history)))
    train_losses = [h.get("train_mse", 0.0) for h in history]
    val_losses = [h.get("val_mse", 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, "b-o", label="Train MSE", markersize=3)
    axes[0].plot(epochs, val_losses, "r-s", label="Val MSE", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    gap = np.array(val_losses) - np.array(train_losses)
    axes[1].plot(epochs, gap, "g-^", label="Val - Train Gap", markersize=3)
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Gap")
    axes[1].set_title("Overfitting Indicator")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(history_list: List[List[Dict[str, Any]]], labels: List[str], output_path: Path) -> None:
    """Compare training curves from multiple experiments."""
    if not HAS_MATPLOTLIB:
        raise CurveAnalysisError("matplotlib not installed")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(history_list)))

    for i, (history, label, color) in enumerate(zip(history_list, labels, colors)):
        epochs = list(range(len(history)))
        val_losses = [h.get("val_mse", 0.0) for h in history]
        axes[0].plot(epochs, val_losses, marker="o", label=label, color=color, markersize=3, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val MSE")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    final_val_losses = [min([h.get("val_mse", float("inf")) for h in history]) for history in history_list]
    axes[1].bar(range(len(labels)), final_val_losses, color=colors)
    axes[1].set_xlabel("Experiment")
    axes[1].set_ylabel("Best Val MSE")
    axes[1].set_title("Best Validation Loss")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FNO training curves in detail.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze single training run")
    analyze_parser.add_argument("--history", type=Path, default=Path("artifacts/training/training_history.json"), help="History JSON")
    analyze_parser.add_argument("--output", type=Path, help="Save report to JSON")
    analyze_parser.add_argument("--plot", type=Path, help="Save plot to PNG")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple training runs")
    compare_parser.add_argument("history_files", nargs="+", type=Path, help="History JSON files")
    compare_parser.add_argument("--labels", type=str, nargs="+", help="Labels for each file")
    compare_parser.add_argument("--plot", type=Path, help="Save comparison plot")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "analyze":
        report = generate_report(args.history)
        print("=" * 70)
        print("TRAINING ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nEpochs: {report['num_epochs']}")

        print("\n[Overfitting Analysis]")
        ov = report["overfitting_analysis"]
        print(f"  Best Val Epoch:          {ov['best_val_epoch']}")
        print(f"  Best Val Loss:           {ov['best_val_loss']:.6e}")
        print(f"  Final Train Loss:        {ov['final_train_loss']:.6e}")
        print(f"  Final Val Loss:          {ov['final_val_loss']:.6e}")
        print(f"  Gap at Best:             {ov['gap_at_best']:.6e}")
        print(f"  Gap at End:              {ov['gap_at_end']:.6e}")
        print(f"  Overfitting Ratio:       {ov['overfitting_ratio']:.4f}")
        print(f"  Early Stopping Rec.:     {ov['early_stopping_recommended']}")

        print("\n[Convergence Analysis]")
        conv = report["convergence_analysis"]
        print(f"  Initial Loss:            {conv['initial_loss']:.6e}")
        print(f"  Final Loss:              {conv['final_loss']:.6e}")
        print(f"  Total Improvement:       {conv['total_improvement']:.6e}")
        print(f"  Improvement Ratio:       {conv['improvement_ratio']:.4f}")
        print(f"  First Half Improvement:  {conv['first_half_improvement_ratio']:.4f}")
        print(f"  Convergence Speed:       {conv['convergence_speed']}")
        if conv["epochs_to_5pct_threshold"] is not None:
            print(f"  Epochs to 5% Threshold:  {conv['epochs_to_5pct_threshold']}")

        print("\n[Stability Analysis]")
        stab = report["stability_analysis"]
        print(f"  Stability Score:         {stab['stability_score']:.4f}")
        print(f"  Large Jumps Detected:    {stab['num_large_jumps']}")
        if stab["issues"]:
            for issue in stab["issues"]:
                print(f"    - {issue}")

        print("\n[Learning Rate Assessment]")
        lr = report["learning_rate_analysis"]
        print(f"  LR Assessment:           {lr['lr_assessment']}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved: {args.output}")

        if args.plot and HAS_MATPLOTLIB:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            history = load_history(args.history)
            plot_training_curves(history, args.plot)
            print(f"Plot saved: {args.plot}")

        print("=" * 70)

    elif args.command == "compare":
        histories = [load_history(p) for p in args.history_files]
        labels = args.labels or [f"Exp {i}" for i in range(len(histories))]

        if args.plot and HAS_MATPLOTLIB:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            plot_comparison(histories, labels, args.plot)
            print(f"Comparison plot saved: {args.plot}")

        print("\nComparison Summary:")
        print("-" * 70)
        for label, history in zip(labels, histories):
            best_val = min([h.get("val_mse", float("inf")) for h in history])
            final_train = history[-1].get("train_mse", 0.0)
            print(f"{label:20s} | Best Val: {best_val:.6e} | Final Train: {final_train:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
