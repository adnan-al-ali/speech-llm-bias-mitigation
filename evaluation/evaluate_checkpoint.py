import argparse
from types import SimpleNamespace
import shutil
import gc
import traceback
import torch
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.gender_detection.inference_detection import evaluate_gender_detection
from evaluation.mmau_vllm.benchmark_inference import evaluate_mmau
from evaluation.gender_bias.gender_bias_inference_adjectives import evaluate_gender_bias
from distilling.correct_qwen3_adapter_dict.correct_qwen3_adapter_dict import correct_names

def main(args):
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "Not set"))
    if hasattr(args, "lora_checkpoint_paths") and args.lora_checkpoint_paths and "qwen3" in args.checkpoint_path.lower():
        print(f"Correcting adapter names in LoRA checkpoints for {args.checkpoint_path}...")
        for lora_checkpoint in args.lora_checkpoint_paths:
            correct_names(lora_checkpoint)
    try:
        evaluate_mmau(model_name=args.checkpoint_path, lora_checkpoints=(args.lora_checkpoint_paths if hasattr(args, "lora_checkpoint_paths") else None), target_pitch=(args.target_pitch if hasattr(args, "target_pitch") else None))
        gc.collect()
        torch.cuda.empty_cache()
        evaluate_gender_detection(checkpoint_path=args.checkpoint_path, evaluate=True, lora_checkpoints=(args.lora_checkpoint_paths if hasattr(args, "lora_checkpoint_paths") else None), normalize_pitch=(args.target_pitch if hasattr(args, "target_pitch") else None))
        if hasattr(args, "evaluate_bias") and args.evaluate_bias:
            gc.collect()
            torch.cuda.empty_cache()
            evaluate_gender_bias(model_name=args.checkpoint_path, lora_checkpoints=(args.lora_checkpoint_paths if hasattr(args, "lora_checkpoint_paths") else None))
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        traceback.print_exc()
        raise
    finally:
        if hasattr(args, "cleanup") and args.cleanup:
            print(f"Cleaning up converted model at {args.checkpoint_path}...")
            shutil.rmtree(args.checkpoint_path, ignore_errors=True)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        args = SimpleNamespace(**snakemake.params)
        main(args)
    else:
        lora_checkpoint_dir = "distilling/logs/bs=8,cs=80,ds=0.05,e=1,ebs=8,lr=1e-05,la=32.0,ld=0.05,lr=16,mn=Qwen2-Audio-7B-Instruct,ns=0.95,umn=best-checkpoint_Qwen2-Audio-7B-Instruct,wd=0.005/checkpoint"
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint_path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", type=str, help="Path to the saved encoder checkpoint.")
        parser.add_argument("--lora_checkpoint_paths", nargs="+", default=["distilling/logs/bs=8,cs=80,ds=0.05,e=1,ebs=8,lr=1e-06,la=64.0,ld=0.05,lr=32,mn=Qwen3-Omni-30B-A3B-Instruct,ns=0.95,umn=best-checkpoint_Qwen3-Omni-30B-A3B-Instruct,wd=0.005/checkpoint/checkpoint_1_80"], type=str, help="Paths to the saved LoRA checkpoints.")
        # parser.add_argument("--lora_checkpoint_paths", nargs="+", default=[os.path.join(lora_checkpoint_dir, checkpoint) for checkpoint in os.listdir(lora_checkpoint_dir)], type=str, help="Paths to the saved LoRA checkpoints.")
        parser.add_argument("--cleanup", action="store_true", help="Cleanup converted model files after evaluation.")
        parser.add_argument("--evaluate_bias", action="store_true", help="Evaluate gender bias.")
        parser.add_argument("--target_pitch", type=float, default=None, help="Target pitch for pitch normalization. If not provided, no pitch normalization will be applied.")
        args = parser.parse_args()
        main(args)