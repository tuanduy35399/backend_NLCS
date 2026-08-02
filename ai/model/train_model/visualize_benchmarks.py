"""Create charts from every benchmark CSV in a directory.

Examples (run from ``backend_NLCS``)::

    python ai/model/train_model/visualize_benchmarks.py
    python ai/model/train_model/visualize_benchmarks.py --input-dir ai/model/benchmark_results

The script discovers ``comparison_*.csv`` files dynamically; adding another
dataset benchmark does not require changing this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "benchmark_results"


def read_benchmarks(input_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(input_dir.glob("comparison_*.csv"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy comparison_*.csv trong {input_dir}")

    benchmarks = {}
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty or len(frame.columns) < 2:
            continue
        # train.py writes the algorithm name as the first CSV column.
        frame = frame.set_index(frame.columns[0])
        frame.index.name = "Algorithm"
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(how="all")
        if not frame.empty:
            dataset = path.stem.removeprefix("comparison_")
            benchmarks[dataset] = frame

    if not benchmarks:
        raise ValueError(f"Các file benchmark trong {input_dir} không có dữ liệu hợp lệ")
    return benchmarks


def metric_names(frame: pd.DataFrame) -> list[str]:
    return [column.removesuffix("_mean") for column in frame.columns if column.endswith("_mean")]


def plot_dataset(dataset: str, frame: pd.DataFrame, output_dir: Path) -> Path:
    metrics = metric_names(frame)
    if not metrics:
        raise ValueError(f"{dataset}: không có cột *_mean")

    sort_column = "accuracy_mean" if "accuracy_mean" in frame else f"{metrics[0]}_mean"
    frame = frame.sort_values(sort_column, ascending=False)
    algorithms = frame.index.astype(str).tolist()
    x = np.arange(len(algorithms))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(10, len(algorithms) * 1.7), 6))
    for index, metric in enumerate(metrics):
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        values = frame[mean_column].to_numpy(dtype=float)
        errors = frame[std_column].fillna(0).to_numpy(dtype=float) if std_column in frame else None
        bars = ax.bar(x + (index - (len(metrics) - 1) / 2) * width, values,
                      width, yerr=errors, capsize=3, label=metric.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_title(f"So sánh thuật toán - dataset {dataset}")
    ax.set_xlabel("Thuật toán")
    ax.set_ylabel("Điểm đánh giá")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=min(len(metrics), 4))
    fig.tight_layout()

    output = output_dir / f"comparison_{dataset}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_overview(benchmarks: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    rows = []
    for dataset, frame in benchmarks.items():
        for algorithm, values in frame.iterrows():
            rows.append({
                "Dataset": dataset,
                "Algorithm": algorithm,
                "Accuracy": values.get("accuracy_mean", np.nan),
                "F1": values.get("f1_mean", np.nan),
            })
    summary = pd.DataFrame(rows)
    datasets = summary["Dataset"].drop_duplicates().tolist()
    algorithms = summary["Algorithm"].drop_duplicates().tolist()
    x = np.arange(len(datasets))
    width = 0.8 / max(1, len(algorithms))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(datasets) * 3), 6), sharey=True)
    for ax, metric in zip(axes, ["Accuracy", "F1"]):
        for index, algorithm in enumerate(algorithms):
            values = summary[summary["Algorithm"] == algorithm].set_index("Dataset")[metric]
            plotted = [values.get(dataset, np.nan) for dataset in datasets]
            ax.bar(x + (index - (len(algorithms) - 1) / 2) * width, plotted,
                   width, label=algorithm)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylabel("Điểm trung bình")
    axes[0].legend(fontsize=8)
    fig.suptitle("Tổng hợp benchmark theo dataset")
    fig.tight_layout()

    output = output_dir / "benchmark_overview.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = read_benchmarks(args.input_dir)
    outputs = [plot_dataset(dataset, frame, output_dir) for dataset, frame in benchmarks.items()]
    outputs.append(plot_overview(benchmarks, output_dir))
    print("Đã tạo biểu đồ:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
