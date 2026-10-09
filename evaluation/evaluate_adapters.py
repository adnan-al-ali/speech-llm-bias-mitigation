import argparse
import os
from types import SimpleNamespace
import shutil
import gc
import traceback
import torch

def find_adapter_paths(adapter_dir):
    adapter_paths = []
    for directory in os.listdir(adapter_dir):
        if os.path.isdir(os.path.join(adapter_dir, directory)):
            if "corrected" in os.listdir(os.path.join(adapter_dir, directory)):
                adapter_paths.append(os.path.join(adapter_dir, directory, "corrected"))
            else:
                adapter_paths.append(os.path.join(adapter_dir, directory))
    return adapter_paths

def main(args):
    if args.use_vllm:
        from gender_detection.inference_detection import evaluate_gender_detection
        from mmau_vllm.benchmark_inference import evaluate_mmau
    else: 
        from gender_detection.inference_detection_transformers import evaluate_gender_detection_transformers
    adapter_paths = find_adapter_paths(args.adapter_dir)
    print(f"Found {len(adapter_paths)} adapter paths: {adapter_paths}")
    if args.use_vllm:
        evaluate_gender_detection(checkpoint_path=args.model_name, evaluate=True, lora_checkpoints=adapter_paths)
    else:
        evaluate_gender_detection_transformers(checkpoint_path=args.model_name, evaluate=True, lora_checkpoints=adapter_paths)
    gc.collect()
    torch.cuda.empty_cache()
    if args.use_vllm:
        evaluate_mmau(model_name=args.model_name, lora_checkpoints=adapter_paths)
    else:
        raise NotImplementedError("MMAU evaluation is only implemented for vLLM for now.")

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        args = SimpleNamespace(**snakemake.params)
        main(args)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", type=str, help="Model name or path.")
        parser.add_argument("--adapter_dir", default="distilling/logs/test/checkpoint", type=str, help="Path to the saved encoder checkpoint.")
        parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for inference.")
        args = parser.parse_args()
        main(args)