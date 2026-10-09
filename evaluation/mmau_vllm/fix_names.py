import os
import re

# Qwen2-Audio-7B-Instruct_bs=8,cs=80,ds=0.05,e=1,ebs=8,lr=5e-05,la=32.0,ld=0.05,lr=16,mn=Qwen2-Audio-7B-Instruct,ns=0.95,umn=best-checkpoint_Qwen2-Audio-7B-Instruct,wd=0.005_checkpoint_checkpoint_1_560_accuracy.txt

def get_new_name(filename):
    pattern = r"(.*?)_.*lr=(.*?),la=.*lr=(.*?),mn=\1,ns.*_([0-9]+)_accuracy.txt"
    match = re.match(pattern, filename)
    if match:
        model_name = match.group(1)
        assert int(match.group(4)) % 80 == 0, "Checkpoint step should be a multiple of 80"
        new_filename = "lora" + filename[len(model_name):]
        return new_filename
    return None

def fix_names_in_directory(directory, dry_run=True):
    for filename in os.listdir(directory):
        if filename.endswith("_accuracy.txt"):
            new_name = get_new_name(filename)
            if new_name:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                if dry_run:
                    print(f"Would rename: {old_path} -> {new_path}")
                else:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    directory = "evaluation/mmau_vllm/results"
    fix_names_in_directory(directory, dry_run=False)