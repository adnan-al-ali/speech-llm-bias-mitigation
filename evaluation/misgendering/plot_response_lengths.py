from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import MODELS

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
VARIANT_SPECS = [
    ("base", lambda model_name: OUTPUT_DIR / f"{model_name}.csv"),
    ("GUM", lambda model_name: OUTPUT_DIR / f"best-checkpoint_{model_name}.csv"),
    ("distilled", lambda model_name: OUTPUT_DIR / f"best-lora_{model_name}.csv"),
]

SIFT_RESPONSE_LENGTH = 75.16

def get_model_display_name(model_name: str) -> str:
    short = model_name.split("/")[-1]
    if short.startswith("Qwen2"):
        return "Qwen2"
    if short.startswith("Qwen3"):
        return "Qwen3"
    if short.startswith("Nemotron"):
        return "Nemotron-3"
    return short


def average_response_length(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    return df["response"].astype(str).str.len().mean()

def sift50_custom_response_length() -> float:
    with open("distilling/dataset_stats.json", "r") as f:
        stats = json.load(f)
    return stats

def collect_lengths() -> list[dict]:
    records: list[dict] = []
    for model_idx, model_name in enumerate(MODELS):
        model_display = get_model_display_name(model_name)
        for variant_name, path_builder in VARIANT_SPECS:
            csv_path = path_builder(model_name)
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing length CSV for {model_name} ({variant_name}): {csv_path}")
            records.append(
                {
                    "model": model_name,
                    "model_display": model_display,
                    "variant": variant_name,
                    "length": average_response_length(csv_path),
                    "group_idx": model_idx,
                    "variant_idx": VariantOrder.index(variant_name),
                }
            )
    return records


VariantOrder = ["base", "GUM", "distilled"]


def main() -> None:
    records = collect_lengths()
    model_stride = len(VARIANT_SPECS) + 0.4
    x_positions: list[float] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    values: list[float] = []

    for model_idx, model_name in enumerate(MODELS):
        model_start = model_idx * model_stride
        for variant_idx, (variant_name, _) in enumerate(VARIANT_SPECS):
            x = model_start + variant_idx
            x_positions.append(x)
            tick_positions.append(x)
            tick_labels.append(variant_name)
            values.append(next(record["length"] for record in records if record["model"] == model_name and record["variant"] == variant_name))
    
    plt.style.use("seaborn-v0_8-dark")
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(4.2, 3), dpi=200)
    bars = ax.bar(x_positions, values, width=0.72, color="#4f80ff")

    # horizontal line for SIFT-50M custom response lengths, restricted to the model's group
    sift_custom_lengths = sift50_custom_response_length()
    model_group_x_ranges: list[tuple[str, float, float]] = []
    for model_idx, model_name in enumerate(MODELS):
        group_xs = [
            x_pos
            for x_pos, record in zip(x_positions, records)
            if record["model"] == model_name
        ]
        if not group_xs:
            continue
        model_group_x_ranges.append((model_name, min(group_xs), max(group_xs)))

    is_first = True
    custom_line_handles = []
    for model_name, x_start, x_end in model_group_x_ranges:
        custom_length = sift_custom_lengths.get(model_name)
        if custom_length is None:
            continue
        (line,) = ax.plot(
            [x_start-0.45, x_end+0.45],
            [custom_length, custom_length],
            color="#e89417",
            linestyle="dashed",
            linewidth=2,
            label=f'SIFT-50M$^{{\\mathcal{{M}}}}$' if is_first else None,
        )
        custom_line_handles.append(line)
        is_first = False

    # horizontal line for SIFT response length
    ax.axhline(SIFT_RESPONSE_LENGTH, color="#ee5252", linestyle='dashed', linewidth=2, label=f'SIFT-50M$^{{⚲}}$')
    ax.legend(loc="best", fontsize=9, handlelength=1, frameon=True, facecolor="white")
    y_max = max(values) * 1.05
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Average response length (characters)", fontsize=9)
    ax.grid(axis="y")#, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9, length=0)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    for model_idx, model_name in enumerate(MODELS):
        group_center = model_idx * model_stride + 1
        ax.text(
            group_center,
            1.02,
            get_model_display_name(model_name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(min(x_positions) - 0.6, max(x_positions) + 0.6)
    fig.subplots_adjust(bottom=0.16, top=0.9, left=0.14, right=0.96)

    fig.savefig(OUTPUT_DIR / "response_lengths.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "response_lengths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
