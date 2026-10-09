from transformers import Qwen2AudioForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration, VoxtralForConditionalGeneration, AutoProcessor
import torch
import argparse
import os
from pathlib import Path
import sys
import re
import time
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.convert_to_consolidated_mistral import write_consolidated_checkpoint
from adversarial.nemotron.modeling import NemotronH_Nano_Omni_Reasoning_V3
from const import MODELS

def convert_model(saved_encoder_path: str, model_name: str = None, max_concurrent_checkpoints: int = None, custom_save_path: str = None) -> str:
    if max_concurrent_checkpoints is not None:
        saved_checkpoints = list(Path(".").glob("evaluation/model_cache/bs*/checkpoint/checkpoint*"))
        while len(saved_checkpoints) >= max_concurrent_checkpoints:
            print(f"Waiting for available checkpoint slot. Currently {len(saved_checkpoints)} checkpoints in cache.")
            time.sleep(180)  # Wait for 3 minutes before checking again
            saved_checkpoints = list(Path(".").glob("evaluation/model_cache/bs*/checkpoint/checkpoint*"))


    if not saved_encoder_path.endswith(".pt"):
        saved_encoder_path += ".pt"
    if model_name is None:
        model_short = re.search(r"mn=([^,]+)", saved_encoder_path).group(1)
        model_name = MODELS.get(model_short)
    model_short = model_name.split("/")[-1]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code="nemotron" in model_name.lower())
    match model_short:
        case "Qwen2-Audio-7B-Instruct":
            model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            model.audio_tower.load_state_dict(torch.load(saved_encoder_path, map_location="cpu"))
        case "Qwen3-Omni-30B-A3B-Instruct":
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            model.thinker.audio_tower.load_state_dict(torch.load(saved_encoder_path, map_location="cpu"))
        case "Voxtral-Small-24B-2507":
            model = VoxtralForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            model.audio_tower.load_state_dict(torch.load(saved_encoder_path, map_location="cpu"))
        case "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16":
            model = NemotronH_Nano_Omni_Reasoning_V3.from_pretrained(
                model_name,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True
            )
            model.sound_encoder.load_state_dict(torch.load(saved_encoder_path, map_location="cpu"))
        case _:
            raise ValueError(f"Unsupported model: {model_short}")
    if saved_encoder_path.startswith("adversarial/logs/"):
        save_subpath = saved_encoder_path[len("adversarial/logs/"):-len(".pt")]
    else:
        save_subpath = saved_encoder_path[:-len(".pt")]

    if custom_save_path:
        save_path = custom_save_path
    else:
        save_path = "evaluation/model_cache/" + save_subpath
    model.save_pretrained(save_path, max_shard_size="5GB", save_original_format=(model_short != "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"))
    processor.save_pretrained(save_path)
    del(model)
    if model_short == "Voxtral-Small-24B-2507":
        write_consolidated_checkpoint(save_path, os.path.join(save_path, "consolidated.safetensors"))
        shutil.copy2("utils/params_voxtral.json", os.path.join(save_path, "params.json"))
    if "nemotron" in model_name.lower():
        # cp adversarial/nemotron/*.py $save_path
        for file in Path("adversarial/nemotron").glob("*.py"):
            shutil.copy2(file, save_path)
    return save_path

if __name__ == "__main__":
    if "snakemake" in globals():
            from snakemake.script import Snakemake
            snakemake: Snakemake
            convert_model(**snakemake.params)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", type=str, help="Model name or path.")
        parser.add_argument("--saved_encoder_path", default="adversarial/logs/bs=8,cs=125,ds=180,e=1,ebs=24,gra=0.1,kf=False,ls=0.05,lr=0.0001,llw=1.0,mn=Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16,ns=4000,wd=0.005/checkpoint/checkpoint_1_125.pt", type=str, help="Path to the saved encoder checkpoint.")
        parser.add_argument("--custom_save_path", default="evaluation/model_cache/nemotron-test1", type=str, help="Custom path to save the converted model.")
        args = parser.parse_args()
        convert_model(saved_encoder_path=args.saved_encoder_path, model_name=args.model_name, custom_save_path=args.custom_save_path)