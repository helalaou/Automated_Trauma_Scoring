
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(results_file: Path) -> tuple:
    """Load results and summary from JSON files."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Try to find corresponding summary file
    summary_file = results_file.parent / results_file.name.replace("results_", "summary_")
    summary = None
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    return results, summary


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


def compare_summaries(zero_summary: Path, few_summary: Path, output_dir: Path):
    """Compare zero-shot and few-shot results."""
    with open(zero_summary, 'r') as f:
        zero_data = json.load(f)
    with open(few_summary, 'r') as f:
        few_data = json.load(f)
    
    zero_metrics = zero_data.get("metrics", {})
    few_metrics = few_data.get("metrics", {})
    
    models = list(set(list(zero_metrics.keys()) + list(few_metrics.keys())))
    if not models:
        print("No models found for comparison")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    timestamp = zero_data.get("timestamp", "")
    
    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    zero_acc = [zero_metrics.get(m, {}).get("accuracy", 0) for m in models]
    few_acc = [few_metrics.get(m, {}).get("accuracy", 0) for m in models]
    
    bars1 = ax.bar(x - width/2, zero_acc, width, label='Zero-Shot', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, few_acc, width, label='Few-Shot', alpha=0.8, color='#2ecc71')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Zero-Shot vs Few-Shot Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"comparison_zero_vs_few_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: comparison_zero_vs_few_{timestamp}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--summary",
        type=str,
        help="Path to summary JSON file to plot"
    )
    parser.add_argument(
        "--zero-summary",
        type=str,
        help="Path to zero-shot summary JSON file"
    )
    parser.add_argument(
        "--few-summary",
        type=str,
        help="Path to few-shot summary JSON file"
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
    elif args.zero_summary and args.few_summary:
        compare_summaries(Path(args.zero_summary), Path(args.few_summary), results_dir)
    else:
        print("Please provide either --summary or both --zero-summary and --few-summary")
        print("\nExample usage:")
        print("  python plot_results.py --summary results/summary_zero_shot_20240101_120000.json")
        print("  python plot_results.py --zero-summary results/summary_zero_shot_20240101_120000.json --few-summary results/summary_few_shot_20240101_120000.json")


if __name__ == "__main__":
    main()

