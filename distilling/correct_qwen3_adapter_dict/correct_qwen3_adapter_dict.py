from safetensors import safe_open
from safetensors.torch import save_file
import os
import shutil
import json

def correct_names(safetesors_dir):
    # create a backup of the original safetensors dir
    backup_dir = os.path.join(safetesors_dir, "backup")

    if os.path.exists(backup_dir):
        print(f"Backup directory {backup_dir} already exists, therefore the conversion has already been done. Skipping conversion, will only correct the adapter_config.json file.")
        with open(os.path.join(backup_dir, "adapter_config.json"), "r") as f:
            adapter_config_trained = json.load(f)
    else:
        shutil.copytree(safetesors_dir, backup_dir)
        tensors = {}
        with safe_open(os.path.join(safetesors_dir, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for name in f.keys():
                if "base_model.model.model" in name:
                    tensors[name.replace("base_model.model.model", "base_model.model.thinker.model")] = f.get_tensor(name)
        # remove the original safetensors file
        os.remove(os.path.join(safetesors_dir, "adapter_model.safetensors"))
        save_file(tensors, os.path.join(safetesors_dir, "adapter_model.safetensors"), metadata=metadata)

    reference_adapter_config_path = "distilling/correct_qwen3_adapter_dict/adapter_config.json"
    with open(reference_adapter_config_path, "r") as f:
        adapter_config_reference = json.load(f)

    attributes_to_fix = ["auto_mapping", "base_model_name_or_path", "exclude_modules", "task_type"]
    for attr in attributes_to_fix:
        if attr in adapter_config_reference:
            adapter_config_trained[attr] = adapter_config_reference[attr]
    
    with open(os.path.join(safetesors_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config_trained, f, indent=4)

if __name__ == "__main__":
    safetensors_dir = "distilling/logs/test/final_adapters/adapter/"
    correct_names(safetensors_dir)