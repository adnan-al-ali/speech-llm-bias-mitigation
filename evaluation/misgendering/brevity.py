import json
from pathlib import Path

import pandas as pd

from const import MODELS

OUTPUT_DIR = Path("evaluation/misgendering/outputs")
VARIANT_SPECS = [
    ("base", lambda model_name: OUTPUT_DIR / f"{model_name}.csv"),
    ("GUM", lambda model_name: OUTPUT_DIR / f"best-checkpoint_{model_name}.csv"),
    ("distilled", lambda model_name: OUTPUT_DIR / f"best-lora_{model_name}.csv"),
]


def average_response_length(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if "response" not in df.columns:
        raise KeyError(f"Missing 'response' column in {csv_path}")
    return df["response"].astype(str).str.len().mean()


avg_lengths = {}
for model_name in MODELS:
    for variant_name, path_builder in VARIANT_SPECS:
        csv_path = path_builder(model_name)
        if not csv_path.exists():
            continue
        avg_lengths[f"{model_name}_{variant_name}"] = average_response_length(csv_path)

with open(OUTPUT_DIR / "avg_response_lengths.json", "w") as f:
    json.dump(avg_lengths, f, indent=4)
