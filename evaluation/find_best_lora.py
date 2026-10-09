import re
import json
import pandas as pd
import os
import shutil


def extract_hyperparameters_from_filename(filename, model_name):
    pattern = f"gender_bias_lora_.*lr=(.*?),la=.*lr=(.*?),mn={model_name},ns.*_([0-9]+).json"
    match = re.search(pattern, filename)
    if match:
        learning_rate = match.group(1)
        lora_rank = match.group(2)
        checkpoint_step = match.group(3)
        return learning_rate, lora_rank, model_name, checkpoint_step
    return None

def main(model_name):
    results_dir = "evaluation/gender_bias/outputs"
    rows = []
    for filename in os.listdir(results_dir):
        hyperparams = extract_hyperparameters_from_filename(filename, model_name)
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
                    "gender_bias_lor": data.get("log_odds_ratio", None),
                    "adapter_path": filename[len("gender_bias_lora_"):-len(".json")].replace("_checkpoint_checkpoint", "/checkpoint/checkpoint")
                })

    df = pd.DataFrame(rows)

    mmau_baseline_filename = f"evaluation/mmau_vllm/results/{model_name}_accuracy.txt"
    with open(mmau_baseline_filename, 'r') as f:
        mmau_baseline_accuracy = float(f.read().strip())

    satisfying_rows = df[(df["mmau_accuracy"] >= mmau_baseline_accuracy*0.97) & (df["gender_detection_accuracy"] >= 0.7)]
    best_lora = satisfying_rows.loc[satisfying_rows["gender_bias_lor"].idxmin()]

    shutil.copytree(
        os.path.join("distilling", "logs", best_lora["adapter_path"]),
        os.path.join("evaluation", "model_cache", f"best-lora_{model_name}")
    )

    with open(os.path.join("evaluation", "model_cache", f"best-lora_{model_name}", "stats.json"), "w") as f:
        json.dump({
            "learning_rate": best_lora["learning_rate"],
            "lora_rank": best_lora["lora_rank"],
            "model_name": best_lora["model_name"],
            "checkpoint_step": best_lora["checkpoint_step"],
            "mmau_accuracy": best_lora["mmau_accuracy"],
            "gender_detection_accuracy": best_lora["gender_detection_accuracy"],
            "gender_bias_lor": best_lora["gender_bias_lor"],
            "adapter_path": best_lora["adapter_path"]
        }, f, indent=4)

if __name__ == "__main__":
    main("Qwen3-Omni-30B-A3B-Instruct")