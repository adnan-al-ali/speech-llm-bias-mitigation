import os
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import MODELS

model_len_stats = {}
for model_name in MODELS:
    lengths = []
    with open(os.path.join("distilling", "datasets", f"sift_50m_{model_name}.jsonl"), "r") as f:
        for line in f:
            record = json.loads(line)
            response = record[-1]["content"][0]["text"]
            lengths.append(len(response))
    avg_length = sum(lengths) / len(lengths)
    print(f"{model_name}: {avg_length:.2f}")
    model_len_stats[model_name] = avg_length

with open(os.path.join("distilling", "dataset_stats.json"), "w") as f:
    json.dump(model_len_stats, f, indent=4)