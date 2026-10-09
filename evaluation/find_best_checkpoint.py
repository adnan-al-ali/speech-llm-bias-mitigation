import re
import os
import json
from convert_to_hf import convert_model

def find_best_checkpoint(model_name, max_accepted_gender_accuracy=0.6):
    model_short = model_name.split("/")[-1]
    dataframe = dict()
    gender_detection_results = os.listdir("evaluation/gender_detection/outputs")
    for file in gender_detection_results:
        if re.match(f"bs.*mn={model_short}.*_accuracy.json", file):
            with open(os.path.join("evaluation/gender_detection/outputs", file), "r") as f:
                accuracy_data = json.load(f)
                accuracy = accuracy_data["accuracy"]
                if accuracy <= max_accepted_gender_accuracy:
                    dataframe[file[:-len("_accuracy.json")]] = {"gender_accuracy": accuracy}

    mmau_results = os.listdir("evaluation/mmau_vllm/results")
    for file in mmau_results:
        if re.match(f"bs.*mn={model_short}.*_accuracy.txt", file):
            with open(os.path.join("evaluation/mmau_vllm/results", file), "r") as f:
                accuracy = float(f.read().strip())
                if file[:-len("_accuracy.txt")] in dataframe:
                    dataframe[file[:-len("_accuracy.txt")]]["mmau_accuracy"] = accuracy

    assert dataframe, f"No checkpoints found for model {model_name} with gender accuracy <= {max_accepted_gender_accuracy}"

    best_checkpoint = max(dataframe.items(), key=lambda x: x[1].get("mmau_accuracy", 0))
    with open(os.path.join("evaluation", "model_cache", f"best-checkpoint_{model_short}.json"), "w") as f:
        json.dump({"model_name": model_name, "best_checkpoint": best_checkpoint[0], "gender_accuracy": best_checkpoint[1].get("gender_accuracy", 'N/A'), "mmau_accuracy": best_checkpoint[1].get("mmau_accuracy", 'N/A')}, f)
    print(f"Best checkpoint for model {model_name} with gender accuracy <= {max_accepted_gender_accuracy}: {best_checkpoint[0]} with gender accuracy {best_checkpoint[1].get('gender_accuracy', 'N/A')} and mmau accuracy {best_checkpoint[1].get('mmau_accuracy', 'N/A')}")
    name_match = re.search(r"(.*)_checkpoint_checkpoint_(\d+)_(\d+)", best_checkpoint[0])

    convert_model(
        os.path.join("adversarial", "logs", name_match.group(1), "checkpoint", f"checkpoint_{name_match.group(2)}_{name_match.group(3)}.pt"), 
        model_name, 
        custom_save_path=os.path.join("evaluation", "model_cache", f"best-checkpoint_{model_short}")
        )

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        find_best_checkpoint(snakemake.params.model_name)
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", type=str, help="Model name or path.")
        parser.add_argument("--max_accepted_gender_accuracy", default=0.6, type=float, help="Maximum accepted gender accuracy for selecting the best checkpoint.")
        args = parser.parse_args()
        find_best_checkpoint(args.model_name, args.max_accepted_gender_accuracy)