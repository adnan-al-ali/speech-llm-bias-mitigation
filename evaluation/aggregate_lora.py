import re
import json
import pandas as pd
import os
from decimal import Decimal
import matplotlib.pyplot as plt

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import MODELS

# gender_bias_lora_bs=8,cs=80,ds=0.05,e=1,ebs=8,lr=1e-05,la=32.0,ld=0.05,lr=16,mn=Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16,ns=0.95,umn=best-checkpoint_Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16,wd=0.005_checkpoint_checkpoint_1_80


def extract_hyperparameters_from_filename(filename):
    pattern = r"gender_bias_lora_.*lr=(.*?),la=.*lr=(.*?),mn=(.*?),ns.*_([0-9]+).json"
    match = re.search(pattern, filename)
    if match:
        learning_rate = match.group(1)
        lora_rank = match.group(2)
        model_name = match.group(3)
        checkpoint_step = match.group(4)
        return learning_rate, lora_rank, model_name, checkpoint_step
    return None


def get_mmau_baseline_accuracy(model_name):
    baseline_file = os.path.join("evaluation", "mmau_vllm", "results", f"{model_name}_accuracy.txt")
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Baseline MMau accuracy missing for {model_name}: {baseline_file}")
    with open(baseline_file, "r") as f:
        return float(f.read().strip())


def get_baseline_accuracy(model_name):
    with open(f"evaluation/gender_detection/outputs/{model_name}_accuracy.json", "r") as f:
        accuracy_data = json.load(f)
        accuracy_gender_detection = accuracy_data.get("accuracy", None)

    with open(f"evaluation/mmau_vllm/results/{model_name}_accuracy.txt", "r") as f:
        accuracy_mmau = float(f.read().strip())

    return accuracy_gender_detection, accuracy_mmau

def get_baseline_bias(model_name):
    with open(f"evaluation/gender_bias/outputs/gender_bias_baseline_{model_name}.json", "r") as f:
        bias_data = json.load(f)
        log_odds_ratio = bias_data.get("log_odds_ratio", None)
    return log_odds_ratio

def row_to_latex(row):
    converted = []
    row_length = len(row)
    for idx, col in enumerate(row):
        if idx >= row_length - 3 and idx < row_length - 1:  # For the last three columns (accuracy and LOR)
            col = f"\\percentGrad{{{float(col) * 100:.1f}}}{{0}}"
        elif isinstance(col, float):
            if (col < 0.01 or col > 1000):
                col = f"{Decimal(str(col)):.0e}"
            elif (col < 0.1):
                col = str(round(col, 3))
            else:
                col = str(round(col, 2))
        else:
            col = str(col)
        converted.append(col)
    return " & ".join(converted) + r" \\"

def plot_histogram(df, baseline_lor, model_name):
    if "Instruct" in model_name:
        model_name = model_name.split("-Instruct")[0]
    if "Reasoning" in model_name:
        model_name = model_name.split("-Reasoning")[0]

    plt.style.use("seaborn-v0_8-dark")
    plt.style.use('seaborn-v0_8-talk')
    plt.figure(figsize=(5, 3))
    plt.hist(df['gender_bias_lor'].dropna(), bins=20, color='#4f80ff', histtype='stepfilled', ec="#3e64c5", linewidth=1.5)
    plt.axvline(baseline_lor, color='red', linestyle='dashed', linewidth=2, label=f'Baseline')
    plt.title(f'Checkpoints LOR distribution')
    plt.xlabel('Gender Bias LOR')
    plt.legend(handlelength=1, frameon=True, facecolor="white")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    os.makedirs("evaluation/plots", exist_ok=True)
    plot_file = os.path.join("evaluation", "plots", f"{model_name}_lor_histogram.pdf")
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved histogram plot to {plot_file}")

def to_latex(df):
    TABLE_HEADER = r"""\begin{table}[h]
\centering\footnotesize
\begin{tabularx}{\columnwidth}{C C C | C C C}
    \toprule
    LR & rank & step & \textbf{acc}\textsuperscript{$\male \female$} & \textbf{acc}\textsuperscript{\textsc{mmau}} & \textbf{LOR} \\
    \midrule"""
    table_parts = [TABLE_HEADER]

    for model_name in MODELS.keys():
        model_df = df[df["model_name"] == model_name]
        if model_df.empty:
            continue

        model_print = model_name
        if "-Instruct" in model_name:
            model_print = model_name.split("-Instruct")[0]
        if "-Reasoning" in model_name:
            model_print = model_name.split("-Reasoning")[0]

        baseline_gender_detection, baseline_mmau = get_baseline_accuracy(model_name)
        table_parts.append(
            f"    \\multicolumn{{3}}{{c |}}{{\\textbf{{{model_print}}}}} & "
            f"\\percentGrad{{{baseline_gender_detection * 100:.1f}}}{{0}} & "
            f"\\percentGrad{{{baseline_mmau * 100:.1f}}}{{0}} & {get_baseline_bias(model_name) :.2f} \\\\"
        )
        table_parts.append("    \\midrule")

        model_df = model_df.sort_values(
            by=["gender_bias_lor", "mmau_accuracy", "gender_detection_accuracy"],
            ascending=[True, False, False],
        ).head(3)

        for row in model_df[[
            "learning_rate",
            "lora_rank",
            "checkpoint_step",
            "gender_detection_accuracy",
            "mmau_accuracy",
            "gender_bias_lor",
        ]].itertuples(index=False):
            table_parts.append("    " + row_to_latex(row))

        table_parts.append(r"    \midrule")

    if table_parts[-1] == r"    \midrule":
        table_parts = table_parts[:-1]

    table_parts.extend([
        r"    \bottomrule",
        r"\end{tabularx}",
        r"\caption{Top three distilled \acrshort{lora} checkpoints per model that satisfy the following criteria: MMAU accuracy stays at least 97\% of the baseline and gender detection accuracy remains at least 70\%. Sorted by \acrshort{lor}, ascending. Coloured cells are in percentage.}",
        r"\label{tab:top3_lora_checkpoints}",
        r"\end{table}",
    ])
    return "\n".join(table_parts)


def main():
    results_dir = "evaluation/gender_bias/outputs"
    rows = []
    for filename in os.listdir(results_dir):
        hyperparams = extract_hyperparameters_from_filename(filename)
        if hyperparams:
            learning_rate, lora_rank, model_name, checkpoint_step = hyperparams
            name_root = filename[len("gender_bias_"):-len(".json")]

            mmau_filename = f"evaluation/mmau_vllm/results/{name_root}_accuracy.txt"
            mmau_accuracy = None
            if os.path.exists(mmau_filename):
                with open(mmau_filename, 'r') as f:
                    mmau_accuracy = float(f.read().strip())
            else:
                print(f"Warning: {mmau_filename} does not exist. Skipping mmau accuracy for {name_root}.")

            gender_detection_filename = f"evaluation/gender_detection/outputs/{name_root}_accuracy.json"
            gender_detection_accuracy = None
            if os.path.exists(gender_detection_filename):
                with open(gender_detection_filename, 'r') as f:
                    gender_detection_data = json.load(f)
                    gender_detection_accuracy = gender_detection_data.get("accuracy", None)
            else:
                print(f"Warning: {gender_detection_filename} does not exist. Skipping gender detection accuracy for {name_root}.")

            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                rows.append({
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                    "model_name": model_name,
                    "checkpoint_step": checkpoint_step,
                    "mmau_accuracy": mmau_accuracy,
                    "gender_detection_accuracy": gender_detection_accuracy,
                    "gender_bias_lor": data.get("log_odds_ratio", None)
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No LoRA results found in {results_dir}.")

    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
    df["lora_rank"] = pd.to_numeric(df["lora_rank"], errors="coerce")
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df["mmau_accuracy"] = pd.to_numeric(df["mmau_accuracy"], errors="coerce")
    df["gender_detection_accuracy"] = pd.to_numeric(df["gender_detection_accuracy"], errors="coerce")
    df["gender_bias_lor"] = pd.to_numeric(df["gender_bias_lor"], errors="coerce")

    # save one CSV per model, ordered by the lowest LOR first
    os.makedirs("evaluation/aggregated_results", exist_ok=True)
    for model_name, group in df.groupby("model_name"):
        plot_histogram(group, get_baseline_bias(model_name), model_name)
        group = group.sort_values(
            by=["gender_bias_lor", "mmau_accuracy", "gender_detection_accuracy"],
            ascending=[True, False, False],
        )
        output_file = os.path.join("evaluation", "aggregated_results", f"{model_name}_gender_bias_results.csv")
        group.to_csv(output_file, index=False)
        print(f"Saved results for {model_name} to {output_file}")

    eligible_rows = []
    for model_name, group in df.groupby("model_name"):
        baseline_mmau = get_mmau_baseline_accuracy(model_name)
        eligible = group[
            (group["mmau_accuracy"].notna())
            & (group["gender_detection_accuracy"].notna())
            & (group["gender_bias_lor"].notna())
            & (group["mmau_accuracy"] >= baseline_mmau * 0.97)
            & (group["gender_detection_accuracy"] >= 0.7)
        ]
        eligible_rows.append(eligible)

    eligible_df = pd.concat(eligible_rows, ignore_index=True) if eligible_rows else pd.DataFrame(columns=df.columns)
    if eligible_df.empty:
        raise ValueError("No LoRA checkpoints satisfy the quality gate (MMau >= 97% baseline and accuracy >= 0.7).")

    latex_table = to_latex(eligible_df)
    os.makedirs("evaluation/tex_tables", exist_ok=True)
    tex_file = os.path.join("evaluation", "tex_tables", "aggregated_lora_results.tex")
    with open(tex_file, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {tex_file}")


if __name__ == "__main__":
    main()