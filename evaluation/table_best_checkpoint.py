import pandas as pd
import json
import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from const import MODELS

def get_gender_detection_results(checkpoint_name):
    if "best-checkpoint" in checkpoint_name:
        with open(os.path.join("evaluation", "model_cache", checkpoint_name + ".json"), "r") as f:
            log = json.load(f)
            checkpoint_name = log.get("best_checkpoint")
    elif "best-lora" in checkpoint_name:
        with open(os.path.join("evaluation", "model_cache", checkpoint_name, "stats.json"), "r") as f:
            log = json.load(f)
            checkpoint_name = "lora_" + log.get("adapter_path").replace("/", "_")
    
    results_path = f"evaluation/gender_detection/outputs/{checkpoint_name}_accuracy.json"
    with open(results_path, "r") as f:
        results = json.load(f)

    raw_results_path = f"evaluation/gender_detection/outputs/{checkpoint_name}.csv"
    df = pd.read_csv(raw_results_path, index_col=False, names=["recording", "gender_predicted", "gender_true"])
    num_male_preds = (df["gender_predicted"] == "M").sum()
    num_female_preds = (df["gender_predicted"] == "F").sum()
    results["frac_male_preds"] = num_male_preds/len(df)
    results["frac_female_preds"] = num_female_preds/len(df)
    return results

def get_gender_bias_results(checkpoint_name):
    if "best-checkpoint" in checkpoint_name:
        path = f"evaluation/gender_bias/outputs/gender_bias_{checkpoint_name}.json"
    elif "best-lora" in checkpoint_name:
        with open(os.path.join("evaluation", "model_cache", checkpoint_name, "stats.json"), "r") as f:
            log = json.load(f)
            checkpoint_name = log.get("adapter_path").replace("/", "_")
        path = f"evaluation/gender_bias/outputs/gender_bias_lora_{checkpoint_name}.json"
    else:
        path = f"evaluation/gender_bias/outputs/gender_bias_baseline_{checkpoint_name}.json"
    with open(path, "r") as f:
        results = json.load(f)
    return results

def gender_detection_table():
    TABLE_HEADER = r"""\begin{table}[h]
    \centering\footnotesize
    \begin{tabularx}{\columnwidth}{C | C C C C C }
        \toprule
        \textbf{Model} & \textbf{acc}\textsubscript{abs} & \textbf{acc}\textsubscript{NR} & \textbf{refuse} & $\%\male$ & $\%\female$ \\"""

    table_parts = [TABLE_HEADER]
    for model_name in MODELS:
        results_gender_detection_baseline = get_gender_detection_results(model_name)
        results_gender_detection_best_checkpoint = get_gender_detection_results("best-checkpoint_" + model_name)
        results_gender_detection_distilled = get_gender_detection_results("best-lora_" + model_name)
        # results_gender_bias_baseline = get_gender_bias_results(model_name)
        # results_gender_bias_best_checkpoint = get_gender_bias_results("best-checkpoint_" + model_name)
        if "Instruct" in model_name:
            model_name = model_name.split("-Instruct")[0]
        elif "Reasoning" in model_name:
            model_name = model_name.split("-Reasoning")[0]
        table_parts.append(r"    \midrule")
        table_parts.append(f"    \\multicolumn{{6}}{{c}}{{\\textbf{{{model_name}}}}} \\\\")
        table_parts.append(r"    \midrule")
        table_parts.append(f"    base & \\percentGrad{{{results_gender_detection_baseline['accuracy']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_baseline['accuracy_on_non_refused']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_baseline['refusal_rate']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_baseline['frac_male_preds']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_baseline['frac_female_preds']*100:.1f}}}{{0}} \\\\")
        table_parts.append(f"    \\acrshort{{gum}} & \\percentGrad{{{results_gender_detection_best_checkpoint['accuracy']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_best_checkpoint['accuracy_on_non_refused']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_best_checkpoint['refusal_rate']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_best_checkpoint['frac_male_preds']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_best_checkpoint['frac_female_preds']*100:.1f}}}{{0}} \\\\")
        table_parts.append(f"    dist. & \\percentGrad{{{results_gender_detection_distilled['accuracy']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_distilled['accuracy_on_non_refused']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_distilled['refusal_rate']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_distilled['frac_male_preds']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_detection_distilled['frac_female_preds']*100:.1f}}}{{0}} \\\\")
    table_parts.append(r"    \bottomrule")
    table_parts.append(r"    \end{tabularx}")
    table_parts.append(r"    \caption{\textbf{Gender detection} evaluation results for the model variants. Key: acc\textsubscript{abs}: absolute gender detection accuracy, acc\textsubscript{NR}: accuracy on non-refused samples, refuse: refusal rate, $\%\male$/$\%\female$: fraction of samples predicted as male/female. Values in percentages.}")
    table_parts.append(r"    \label{tab:gender-detection-results}")
    table_parts.append(r"\end{table}")

    os.makedirs("evaluation/tex_tables", exist_ok=True)
    with open("evaluation/tex_tables/gender_detection_table.tex", "w") as f:
        f.write("\n".join(table_parts))

def gender_bias_table():
    TABLE_HEADER = r"""\begin{table}[h]
    \centering\footnotesize
    \begin{tabularx}{\columnwidth}{C | C C C}
        \toprule
        \textbf{Model} & \textbf{bias}\textsuperscript{$\male$} & \textbf{bias}\textsuperscript{$\female$} & \textbf{LOR} \\ 
    """

    table_parts = [TABLE_HEADER]
    for model_name in MODELS:
        results_gender_bias_baseline = get_gender_bias_results(model_name)
        results_gender_bias_best_checkpoint = get_gender_bias_results("best-checkpoint_" + model_name)
        results_gender_bias_lora = get_gender_bias_results("best-lora_" + model_name)
        if "Instruct" in model_name:
            model_name = model_name.split("-Instruct")[0]
        elif "Reasoning" in model_name:
            model_name = model_name.split("-Reasoning")[0]
        table_parts.append(r"    \midrule")
        table_parts.append(f"    \\multicolumn{{4}}{{c}}{{\\textbf{{{model_name}}}}} \\\\")
        table_parts.append(r"    \midrule")
        table_parts.append(f"    base & \\percentGrad{{{results_gender_bias_baseline['masculine_stereotype_percentage']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_bias_baseline['feminine_stereotype_percentage']*100:.1f}}}{{0}} & {results_gender_bias_baseline['log_odds_ratio']:.2f} \\\\")
        table_parts.append(f"    \\acrshort{{gum}} & \\percentGrad{{{results_gender_bias_best_checkpoint['masculine_stereotype_percentage']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_bias_best_checkpoint['feminine_stereotype_percentage']*100:.1f}}}{{0}} & {results_gender_bias_best_checkpoint['log_odds_ratio']:.2f} \\\\")
        table_parts.append(f"    dist. & \\percentGrad{{{results_gender_bias_lora['masculine_stereotype_percentage']*100:.1f}}}{{0}} & \\percentGrad{{{results_gender_bias_lora['feminine_stereotype_percentage']*100:.1f}}}{{0}} & {results_gender_bias_lora['log_odds_ratio']:.2f} \\\\")
    table_parts.append(r"    \bottomrule")
    table_parts.append(r"    \end{tabularx}")
    table_parts.append(r"    \caption{\textbf{Gender bias} evaluation results for the model variants. Key: bias\textsuperscript{$\male$}/bias\textsuperscript{$\female$}: percentage of samples where the model chose the stereotypical option for masculine/feminine voices (expected 50\% for uniformly random selection), LOR: log odds ratio. Coloured values are in percentages.}")
    table_parts.append(r"    \label{tab:gender-bias-results}")
    table_parts.append(r"\end{table}")

    os.makedirs("evaluation/tex_tables", exist_ok=True)
    with open("evaluation/tex_tables/gender_bias_table.tex", "w") as f:
        f.write("\n".join(table_parts))

if __name__ == "__main__":
    gender_detection_table()
    gender_bias_table()