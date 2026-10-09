from vllm import LLM, SamplingParams
from datasets import load_dataset
import random
import tempfile
import os
import torch
from pathlib import Path
import json
import argparse

from utils import convert_sift50m_sample_to_chat_format


def generate_chat_dataset(model_name: str, max_samples: int = 4000,seed: int = 67, temperature: float = 0.5):
    random.seed(seed)
    dataset = load_dataset("parquet", data_files="adversarial/downloads/sift_50m_vctk_en_train.parquet")["train"]
    dataset = dataset.shuffle(seed=seed).select(range(max_samples))["messages"]
    dataset = list(map(convert_sift50m_sample_to_chat_format, dataset))
    dataset = [sample[:-1] for sample in dataset]  # Remove the last message (assistant's response) from each sample

    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir
    llm = LLM(
        model=model_name,
        allowed_local_media_path=str(Path("adversarial/downloads").resolve()),
        max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        moe_backend="triton"
    )
    sampling_params = SamplingParams(max_tokens=4096, temperature=temperature)
    responses = llm.chat(dataset, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False} if "nemotron" in model_name.lower() else {})
    os.makedirs("distilling/datasets", exist_ok=True)
    with open(f"distilling/datasets/sift_50m_{model_name.split('/')[-1]}.jsonl", "w") as f:
        for response, sample in zip(responses, dataset):
            sample.append({"role": "assistant", "content": [{"type": "text", "text": response.outputs[0].text}]})
            f.write(json.dumps(sample, indent=None))
            f.write("\n")

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        generate_chat_dataset(
            model_name=snakemake.params.model_name,
            max_samples=snakemake.params.max_samples
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Name of the model to generate the dataset with.")
        parser.add_argument("--max_samples", type=int, default=4000, help="Maximum number of samples to generate.")
        parser.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility.")
        parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling.")
        args = parser.parse_args()

        generate_chat_dataset(args.model_name, args.max_samples, args.seed, args.temperature)