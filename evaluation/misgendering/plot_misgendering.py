from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import MODELS

OUTCOME_ORDER = ["correct", "misgendered", "no_pronoun_or_tie"]
OUTCOME_COLORS = {
    "correct": "#62c662",
    "misgendered": "#ee5252",
    "no_pronoun_or_tie": "#7f7f7f",
}


def resolve_output_path(model_name: str) -> Path:
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    candidates = [
        outputs_dir / f"{model_name}.csv",
        outputs_dir / f"best-checkpoint_{model_name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find misgendering output CSV for {model_name}")


def get_counts_from_table(csv_path: Path) -> dict[tuple[str, str], int]:
    df = pd.read_csv(csv_path)

    gender_stats: dict[tuple[str, str], int] = {}
    for true_gender in ["female", "male"]:
        true_letter = true_gender[0].upper()
        other_letter = "M" if true_letter == "F" else "F"
        gender_df = df[df["ground_truth_gender"].str.lower() == true_gender]
        gender_stats[(true_gender, "no_pronoun_or_tie")] = int((gender_df["dominant_pronoun"] == "_").sum())
        gender_stats[(true_gender, "correct")] = int(gender_df["dominant_pronoun"].isin([true_letter, "X"]).sum())
        gender_stats[(true_gender, "misgendered")] = int((gender_df["dominant_pronoun"] == other_letter).sum())

    return gender_stats


variant_specs = [
    ("base", lambda model_name: Path(__file__).resolve().parent / "outputs" / f"{model_name}.csv"),
    ("GUM", lambda model_name: Path(__file__).resolve().parent / "outputs" / f"best-checkpoint_{model_name}.csv"),
    ("dist.", lambda model_name: Path(__file__).resolve().parent / "outputs" / f"best-lora_{model_name}.csv")
]

fig, ax = plt.subplots(len(MODELS) * len(variant_specs), 2, figsize=(4, 2.5 * len(MODELS) * len(variant_specs)))
if len(MODELS) * len(variant_specs) == 1:
    ax = [ax]
elif len(MODELS) * len(variant_specs) > 1:
    ax = ax.reshape(len(MODELS) * len(variant_specs), 2)

legend_handles = None

for model_idx, model_name in enumerate(MODELS):
    display_name = model_name.split("-")[0]
    for variant_idx, (variant_name, path_builder) in enumerate(variant_specs):
        row_idx = model_idx * len(variant_specs) + variant_idx
        csv_path = path_builder(model_name)
        counts = get_counts_from_table(csv_path)
        for gender_idx, true_gender in enumerate(["female", "male"]):
            axis = ax[row_idx][gender_idx]
            values = [counts[(true_gender, outcome)] for outcome in OUTCOME_ORDER]
            colors = [OUTCOME_COLORS[outcome] for outcome in OUTCOME_ORDER]
            total = sum(values)

            if total > 0:
                axis.pie(
                    values,
                    colors=colors,
                    startangle=90,
                    counterclock=False,
                    autopct=lambda pct: f"{pct:.0f}%" if pct > 1 else "",
                    pctdistance=0.72,
                    textprops={"fontsize": 8},
                )
            else:
                axis.text(0.5, 0.5, "No data", ha="center", va="center")

            axis.set_title(f"{display_name} - {variant_name} - {true_gender[0].upper()}")
            axis.set_aspect("equal")

            if legend_handles is None and total > 0:
                legend_handles = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=OUTCOME_COLORS[outcome], markersize=10)
                    for outcome in OUTCOME_ORDER
                ]

if legend_handles is None:
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OUTCOME_COLORS[outcome], markersize=10)
        for outcome in OUTCOME_ORDER
    ]

fig.legend(legend_handles, ["Correct", "Misgendered", "No pronoun / tie"], loc="upper center", ncol=3, frameon=False)
fig.tight_layout()

output_path = "evaluation/misgendering/outputs/misgendering_pie_charts.pdf"
fig.savefig(output_path, bbox_inches="tight")
plt.show()
