import json
import pandas as pd
import os
import re
from decimal import Decimal

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import MODELS

PITCH = 144

def match_filename(filename):
    if not filename.startswith("bs"):
        return None
    pattern = r".*gra=([^,]+),.*lr=([^,]+),.*llw=([^,]+),.*mn=([^,]+),.*checkpoint_(\d+)_(\d+).*"
    match = re.match(pattern, filename)
    if match:
        return {
            "Gradient reversal alpha": float(match.group(1)),
            "Learning rate": float(match.group(2)),
            "LLM loss weight": float(match.group(3)),
            "Model": match.group(4),
            "Checkpoint epoch": int(match.group(5)),
            "Checkpoint step": int(match.group(6)),
        }
    return None

def get_baseline_accuracy(model_name, pitch=None):
    with open(f"evaluation/gender_detection/outputs/{model_name}{'_normalized_' + str(pitch) if pitch is not None else ''}_accuracy.json", "r") as f:
        accuracy_data = json.load(f)
        accuracy_gender_detection = accuracy_data.get("accuracy", None)

    with open(f"evaluation/mmau_vllm/results/{model_name}{'_' + str(pitch) if pitch is not None else ''}_accuracy.txt", "r") as f:
        accuracy_mmau = float(f.read().strip())

    return accuracy_gender_detection, accuracy_mmau

def row_to_latex(row):
    converted = []
    row_length = len(row)
    for i, col in enumerate(row):
        # color last two columns
        if i >= row_length - 2:
            col = f"\\percentGrad{{{col*100:.1f}}}{{0}}"
        elif isinstance(col, float):
            if (col < 0.01 or col > 1000):
                col = f"{Decimal(col):.0e}"
            elif (col < 0.1):
                col = str(round(col, 3))
            else:
                col = str(round(col, 2))
        else:
            col = str(col)
        converted.append(col)
    return " & ".join(converted) + r" \\"

def to_latex(df):
    TABLE_HEADER = r"""\begin{table}[h]
\centering\footnotesize
\begin{tabularx}{\columnwidth}{C C C C | C C}
    \toprule
    $\alpha$ & LR & $\gamma$ & steps & \textbf{acc}\textsuperscript{$\male \female$} & \textbf{acc}\textsuperscript{\textsc{mmau}} \\"""
    # only include rows where gender acc is <0.6
    df_valid = df[df["accuracy_gender_detection"] < 0.6]
    table_parts = [TABLE_HEADER]
    for model in MODELS:
        table_parts.append("    \\midrule")
        model_print = model
        if "-Instruct" in model:
            model_print = model.split("-Instruct")[0]
        if "-Reasoning" in model:
            model_print = model.split("-Reasoning")[0]
        baseline_gender_detection, baseline_mmau = get_baseline_accuracy(model)
        normalized_gender_detection, normalized_mmau = get_baseline_accuracy(model, pitch=PITCH)
        table_parts.append(f"    \\multicolumn{{4}}{{c |}}{{\\textbf{{{model_print}}}}} & \\percentGrad{{{baseline_gender_detection*100:.1f}}}{{0}} & \\percentGrad{{{baseline_mmau*100:.1f}}}{{0}} \\\\")
        table_parts.append("    \\midrule")
        table_parts.append(f"    \\multicolumn{{4}}{{c |}}{{\\textit{{pitch normalisation to median}}}} & \\percentGrad{{{normalized_gender_detection*100:.1f}}}{{0}} & \\percentGrad{{{normalized_mmau*100:.1f}}}{{0}} \\\\")
        model_df = df_valid[df_valid["Model"] == model]
        model_df = model_df.sort_values(by="accuracy_mmau", ascending=False)
        top_5_df = model_df.head(3)
        for row in top_5_df[[
            "Gradient reversal alpha", 
            "Learning rate", 
            "LLM loss weight", 
            "Checkpoint step", 
            "accuracy_gender_detection", 
            "accuracy_mmau"]].itertuples(index=False):
            table_parts.append("    " + row_to_latex(row))
    table_parts.append(r"    \bottomrule")
    table_parts.append(r"\end{tabularx}")
    table_parts.append(f"\\caption{{Top three hyperparameter configurations for each model that achieve gender detection accuracy below 0.6. Pitch normalisation was applied to the original model; the median was {PITCH} Hz. Coloured cells are in percentage.}}")
    table_parts.append(r"\label{tab:top5_hyperparams}")
    table_parts.append(r"\end{table}")

    return "\n".join(table_parts)


def aggregate_results():
    gender_detection_entires = []
    gender_detection_result_files = os.listdir("evaluation/gender_detection/outputs")
    for result_file in gender_detection_result_files:
        if result_file.endswith(".json"):
            parameters = match_filename(result_file)
            if parameters:
                with open(os.path.join("evaluation/gender_detection/outputs", result_file), "r") as f:
                    accuracy_data = json.load(f)
                parameters.update(accuracy_data)
                gender_detection_entires.append(parameters)
    gender_detection_df = pd.DataFrame(gender_detection_entires)
    
    mmau_entires = []
    mmau_result_files = os.listdir("evaluation/mmau_vllm/results")
    for result_file in mmau_result_files:
        if result_file.endswith("_accuracy.txt"):
            parameters = match_filename(result_file)
            if parameters:
                with open(os.path.join("evaluation/mmau_vllm/results", result_file), "r") as f:
                    accuracy = float(f.read().strip())
                parameters["accuracy"] = accuracy
                mmau_entires.append(parameters)
    mmau_df = pd.DataFrame(mmau_entires)

    df_merged = pd.merge(gender_detection_df, mmau_df, on=["Gradient reversal alpha", "Learning rate", "LLM loss weight", "Model", "Checkpoint epoch", "Checkpoint step"], how="outer", suffixes=("_gender_detection", "_mmau"))
    
    df_merged.sort_values(by="accuracy_mmau", ascending=False, inplace=True)
    os.makedirs("evaluation/aggregated_results", exist_ok=True)
    df_merged.to_csv("evaluation/aggregated_results/aggregated_results.csv", index=False)

    latex_table = to_latex(df_merged)
    os.makedirs("evaluation/tex_tables", exist_ok=True)
    with open("evaluation/tex_tables/aggregated_results.tex", "w") as f:
        f.write(latex_table)

if __name__ == "__main__":
    aggregate_results()