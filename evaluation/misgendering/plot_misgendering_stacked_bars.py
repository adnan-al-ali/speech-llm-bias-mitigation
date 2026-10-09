from pathlib import Path
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

OUTCOME_ORDER = ["correct", "misgendered", "they", "no_pronoun_or_tie"]
OUTCOME_LABELS = {
    "correct": "Correct",
    "misgendered": "Misgendered",
    "they": "They/Them",
    "no_pronoun_or_tie": "No pronoun / tie",
}
OUTCOME_COLORS = {
    "correct": "#62c662",
    "misgendered": "#ee5252",
    "they": "#3b82f6",
    "no_pronoun_or_tie": "#7f7f7f",
}
VARIANT_SPECS = [
    ("base", lambda model_name: Path(__file__).resolve().parent / "outputs" / f"{model_name}.csv"),
    ("GUM", lambda model_name: Path(__file__).resolve().parent / "outputs" / f"best-checkpoint_{model_name}.csv"),
    ("distilled", lambda model_name: Path(__file__).resolve().parent / "outputs" / f"best-lora_{model_name}.csv"),
]



def get_counts_from_table(csv_path: Path, accept_they_them: bool) -> dict[tuple[str, str], int]:
    df = pd.read_csv(csv_path)

    count_by_gender_and_outcome: dict[tuple[str, str], int] = {}
    for true_gender in ["female", "male"]:
        true_letter = true_gender[0].upper()
        other_letter = "M" if true_letter == "F" else "F"
        gender_df = df[df["ground_truth_gender"].str.lower() == true_gender]

        if accept_they_them:
            count_by_gender_and_outcome[(true_gender, "no_pronoun_or_tie")] = int((gender_df["dominant_pronoun"] == "_").sum())
            count_by_gender_and_outcome[(true_gender, "correct")] = int(gender_df["dominant_pronoun"].isin([true_letter, "X"]).sum())
            count_by_gender_and_outcome[(true_gender, "they")] = 0
            count_by_gender_and_outcome[(true_gender, "misgendered")] = int((gender_df["dominant_pronoun"] == other_letter).sum())
        else:
            count_by_gender_and_outcome[(true_gender, "no_pronoun_or_tie")] = int((gender_df["dominant_pronoun"].isin(["_"])).sum())
            count_by_gender_and_outcome[(true_gender, "correct")] = int((gender_df["dominant_pronoun"] == true_letter).sum())
            count_by_gender_and_outcome[(true_gender, "they")] = int((gender_df["dominant_pronoun"] == "X").sum())
            count_by_gender_and_outcome[(true_gender, "misgendered")] = int((gender_df["dominant_pronoun"] == other_letter).sum())
    return count_by_gender_and_outcome

def get_model_display_name(model_name: str) -> str:
    if "Instruct" in model_name:
        return model_name.split("-Instruct")[0]
    if "Reasoning" in model_name:
        return model_name.split("-Reasoning")[0]
    return model_name

def collect_bar_data(accept_they_them: bool) -> list[dict]:
    records: list[dict] = []
    for model_name in MODELS:
        model_display = get_model_display_name(model_name)
        for variant_idx, (variant_name, path_builder) in enumerate(VARIANT_SPECS):
            csv_path = path_builder(model_name)
            counts = get_counts_from_table(csv_path, accept_they_them=accept_they_them)
            for gender_idx, gender_name in enumerate(["female", "male"]):
                values = [counts[(gender_name, outcome)] for outcome in OUTCOME_ORDER]
                records.append(
                    {
                        "model": model_name,
                        "model_display": model_display,
                        "variant": variant_name,
                        "gender": gender_name,
                        "gender_short": "F" if gender_name == "female" else "M",
                        "variant_idx": variant_idx,
                        "gender_idx": gender_idx,
                        "values": values,
                        "total": sum(values),
                    }
                )
    return records


def build_positions(records: list[dict]) -> tuple[list[float], list[dict]]:
    positions: list[float] = []
    x_map: dict[tuple[str, str, str], float] = {}
    current = 0.0
    model_names = list(MODELS)

    for model_idx, model_name in enumerate(model_names):
        if model_idx > 0:
            current += 0.8

        for variant_idx, _ in enumerate(VARIANT_SPECS):
            for gender_idx, gender_name in enumerate(["female", "male"]):
                key = (model_name, VARIANT_SPECS[variant_idx][0], gender_name)
                x_map[key] = current
                positions.append(current)
                current += 1.0
            current += 0.2

    for record in records:
        key = (record["model"], record["variant"], record["gender"])
        record["xpos"] = x_map[key]

    return positions, records



def main(accept_they_them: bool) -> None:
    records = collect_bar_data(accept_they_them=accept_they_them)
    if not records:
        raise ValueError("No misgendering records were found for plotting.")

    _, records = build_positions(records)
    x_positions = [record["xpos"] for record in records]
    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=200)
    left = np.zeros(len(records), dtype=float)

    for outcome_idx, outcome in enumerate(OUTCOME_ORDER):
        heights = [record["values"][outcome_idx] for record in records]
        ax.bar(
            x_positions,
            heights,
            bottom=left,
            width=0.82,
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.5,
            label=OUTCOME_LABELS[outcome],
        )
        left += np.asarray(heights)

    max_total = max(record["total"] for record in records)
    ax.set_ylim(0, max_total * 1.1)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(OUTCOME_ORDER),
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        columnspacing=1.2,
    )
    legend._legend_box.align = "left"

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_frame_on(False)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    model_group_centers: list[tuple[str, float]] = []
    seen_models = set()
    for record in records:
        model_name = record["model"]
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        xs = [r["xpos"] for r in records if r["model"] == model_name]
        model_group_centers.append((record["model_display"], (min(xs) + max(xs)) / 2))

    for model_name, center_x in model_group_centers:
        ax.text(center_x, 0.92, model_name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=13, fontweight="bold")

    variant_midpoints: list[tuple[str, float]] = []
    seen_variant_pairs: set[tuple[str, str]] = set()
    for record in records:
        key = (record["model"], record["variant"])
        if key in seen_variant_pairs:
            continue
        seen_variant_pairs.add(key)
        xs = [r["xpos"] for r in records if r["model"] == record["model"] and r["variant"] == record["variant"]]
        variant_midpoints.append((record["variant"], (min(xs) + max(xs)) / 2))

    for record in records:
        ax.text(
            record["xpos"],
            -0.02,
            record["gender_short"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    for variant_name, center_x in variant_midpoints:
        ax.text(center_x, -0.08, variant_name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)

    ax.set_xlim(min(x_positions) - 0.6, max(x_positions) + 0.6)
    fig.subplots_adjust(bottom=0.12, top=0.9, left=0.04, right=0.99)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [("pdf", None), ("png", 300)]:
        output_path = output_dir / f"misgendering_stacked_bars.{suffix}"
        if dpi is None:
            fig.savefig(output_path, bbox_inches="tight")
        else:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(accept_they_them=False)
