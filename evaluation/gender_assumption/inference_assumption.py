from pathlib import Path
import os
import sys
# os.environ["LD_LIBRARY_PATH"] = "/home/alali/personal_work_troja/miniconda/lib/"
# export LD_LIBRARY_PATH=/home/alali/personal_work_troja/miniconda/lib/

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers import pipeline
from tqdm import tqdm
import csv
import torch
import tempfile
import shutil
from typing import Union, Literal


AUDIO_PATH = "evaluation/gender_assumption/recordings"

def make_dataset(max_samples = None):
    prompts = []
    for audio_file in tqdm(list(Path(AUDIO_PATH).glob("*.wav")), desc="Creating dataset"):
        prompts.append({
            "audio_url": audio_file.resolve().as_uri(),
            "audio_path": str(audio_file.resolve()),
            "audio_file": audio_file.name,
        })
        if max_samples and len(prompts) >= max_samples:
            break
    return prompts

def make_prompt(audio_url):
    return [
    # {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ],
    }]

def make_prompt_hf(audio_path):
    return [
    # {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
        ],
    }]

def run_vllm(checkpoint_path = "google/gemma-4-12B-it", max_samples = None):

    ds = make_dataset(max_samples=max_samples)

    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir
    llm = LLM(
        model=checkpoint_path,
        allowed_local_media_path=str(Path.cwd().parent),
        # max_model_len=1024,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.8,
        # enforce_eager=True,
        # dtype="bfloat16",
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
        moe_backend="triton"
    )
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)
    
    prompts = [make_prompt(sample["audio_url"]) for sample in tqdm(ds, desc="Creating prompts")]

    outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False} if "nemotron" in checkpoint_path.lower() else {})

    os.makedirs("evaluation/gender_assumption/outputs", exist_ok=True)

    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))

    with open(f"evaluation/gender_assumption/outputs/{checkpoint_name}.csv", "w") as f:
        writer = csv.writer(f)
        for sample, o in zip(ds, outputs):
            generated_text = o.outputs[0].text
            writer.writerow([sample["audio_file"], generated_text])
    shutil.rmtree(cahce_dir)

@torch.no_grad()
def run_hf(checkpoint_path = "google/gemma-4-12B-it", max_samples = None):
    
    ds = make_dataset(max_samples=max_samples)

    pipe = pipeline(
        task="any-to-any",
        model=checkpoint_path,
        device_map="auto",
        dtype="auto",
        attention_implementation="flash_attention_2",
    )

    os.makedirs("evaluation/gender_assumption/outputs", exist_ok=True)
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))
    # clear file contents
    with open(f"evaluation/gender_assumption/outputs/{checkpoint_name}.csv", "w") as f:
        pass

    prompts = [make_prompt_hf(sample["audio_path"]) for sample in tqdm(ds, desc="Creating prompts")]
    with open(f"evaluation/gender_assumption/outputs/{checkpoint_name}.csv", "a") as f:
        writer = csv.writer(f)
        for sample, out in zip(ds, pipe(prompts, max_new_tokens=2048, temperature=0.0)):
            generated_text = out[0]["generated_text"]
            writer.writerow([sample["audio_file"], generated_text])

def run_gender_assumption(
    checkpoint_path = "google/gemma-4-12B-it",
    engine: Literal["hf", "vllm"] = "vllm",
    max_samples = None):

    match engine:
        case "hf":
            run_hf(checkpoint_path=checkpoint_path, max_samples=max_samples)
        case "vllm":
            run_vllm(checkpoint_path=checkpoint_path, max_samples=max_samples)
        case _:
            raise ValueError(f"Unsupported engine: {engine}")

    

if __name__ == "__main__":
    run_gender_assumption()