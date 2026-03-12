
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_from_summary(summary_file: Path, output_dir: Path):
    """Plot results from a summary file."""
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    metrics = summary.get("metrics", {})
    mode = summary.get("mode", "unknown")
    timestamp = summary.get("timestamp", "")
    
    if not metrics:
        print(f"No metrics found in {summary_file}")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    models = list(metrics.keys())
    
    # Overall accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracies = [metrics[m]["accuracy"] for m in models]
    bars = ax.bar(models, accuracies, color=sns.color_palette("husl", len(models)))
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title(f'Overall Accuracy by Model ({mode.replace("_", "-").title()})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"overall_accuracy_{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: overall_accuracy_{mode}_{timestamp}.png")
def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--summary",
        type=str,
        help="Path to summary JSON file to plot"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (default: results)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.summary:
        plot_from_summary(Path(args.summary), results_dir)
    else:
        print("Please provide --summary with the path to a summary JSON file.")
        print("\nExample usage:")
        print("  python plot_results.py --summary results/summary_openpose_6bin_20240101_120000.json")


if __name__ == "__main__":
    main()

