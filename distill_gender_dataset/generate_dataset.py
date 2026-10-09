from pathlib import Path
import os
import sys
# os.environ["LD_LIBRARY_PATH"] = "/home/alali/personal_work_troja/miniconda/lib/"
# export LD_LIBRARY_PATH=/home/alali/personal_work_troja/miniconda/lib/

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import pandas as pd
import csv
import json
import torch
import tempfile
from evaluation.pitch_mod import pitch_manipulation
from distill_gender_dataset.shared import PROMPTS
import argparse
import random


SOURCES = {
    "english_accents",
    # "stereoset_speech_gender",
    "british_dialects"
}

METADATA_PATH = "evaluation/gender_bias/full_diverse_voices_compare.csv"

VCTK_PATH = "adversarial/downloads/vctk-corpus/VCTK-Corpus/VCTK-Corpus"

def make_dataset(prompt_key, num_samples_per_prompt, target_pitch = None, max_samples = None):
    MAX_RECORDINGS_PER_SPEAKER = 100
    prompts = []
    metadata = pd.read_csv(os.path.join(VCTK_PATH, "speaker-info.txt"), sep=r"\s+", index_col=False).sample(frac=1, random_state=42)  # Shuffle the dataset
    # make gender balanced
    metadate_male = metadata[metadata["GENDER"] == "M"]
    metadata_female = metadata[metadata["GENDER"] == "F"]
    shorter_length = min(len(metadate_male), len(metadata_female))
    metadata = pd.concat([metadate_male.sample(n=shorter_length, random_state=42), metadata_female.sample(n=shorter_length, random_state=42)])
    all_recordings = []
    all_genders = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        speaker_id = row["ID"]
        speaker_recordings = os.listdir(os.path.join(VCTK_PATH, "wav48", "p" + str(speaker_id)))
        random.shuffle(speaker_recordings)
        all_recordings.extend([os.path.join(VCTK_PATH, "wav48", "p" + str(speaker_id), recording) for recording in speaker_recordings[:MAX_RECORDINGS_PER_SPEAKER]])
        all_genders.extend([row["GENDER"]] * len(speaker_recordings[:MAX_RECORDINGS_PER_SPEAKER]))

    rec_gender_pairs = list(zip(all_recordings, all_genders))
    random.shuffle(rec_gender_pairs)

    for audio_path, gender in list(rec_gender_pairs)[:num_samples_per_prompt]:
        if target_pitch is not None:
            audio_path = pitch_manipulation.change_pitch(audio_path, target_pitch, downsample_rate=22050)
        prompts.append({
            "prompt": PROMPTS[prompt_key],
            "audio_path": Path(audio_path).resolve().as_uri(),
            "gender": gender,
        })
    return prompts

def make_dataset_old(prompt_key, num_samples_per_prompt, gender_column = "gender_1", target_pitch = None, max_samples = None):
    prompts = []
    metadata = pd.read_csv(METADATA_PATH).sample(frac=1, random_state=42)  # Shuffle the dataset
    if max_samples is not None:
        metadata = metadata.head(max_samples)
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        if row["source_1"] in SOURCES:
            for _ in range(num_samples_per_prompt):
                audio_path = "evaluation/gender_bias/" + row["audio_file_1"][5:] # remove "data/"
                if target_pitch is not None:
                    audio_path = pitch_manipulation.change_pitch(audio_path, target_pitch, downsample_rate=22050)
                prompts.append({
                    "prompt": PROMPTS[prompt_key],
                    "audio_path": Path(audio_path).resolve().as_uri(),
                    "gender": row[gender_column] if gender_column else None,
                })
    return prompts

def make_prompt(prompt, audio_url):
    return [
    {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "text", "text": prompt},
        ],
    }]

def generate_dataset(model_name, num_samples_per_prompt, temperature, max_samples = None, adapter_path = None):
    random.seed(42)
    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir
    llm = LLM(
        model=model_name,
        allowed_local_media_path=str(Path.cwd().parent),
        max_model_len=2048,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.8,
        enable_lora=adapter_path is not None,
        max_lora_rank=64,
        trust_remote_code=True,
        moe_backend="triton",
        max_num_seqs=796
        # limit_mm_per_prompt={"image": 0, "video": 0},
    )
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature)
    
    os.makedirs("distill_gender_dataset/datasets", exist_ok=True)
    for prompt_key in PROMPTS.keys():
        checkpoint_name = model_name.split("/")[-1]
        output_csv_path = f"distill_gender_dataset/datasets/{checkpoint_name}{'_adapter' if adapter_path else ''}_{prompt_key}.csv"
        output_jsonl_path = f"distill_gender_dataset/datasets/{checkpoint_name}{'_adapter' if adapter_path else ''}_{prompt_key}.jsonl"
        if os.path.exists(output_csv_path):
            print(f"Output CSV for {checkpoint_name} and prompt {prompt_key} already exists. Skipping.")
            continue
        ds = make_dataset(prompt_key, num_samples_per_prompt, max_samples=max_samples)
        prompts = [make_prompt(sample["prompt"], sample["audio_path"]) for sample in tqdm(ds, desc="Creating prompts")]

        if adapter_path is None:
            outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False} if "nemotron" in model_name.lower() else {})
        else:
            lora_request = LoRARequest("adapter", 1, adapter_path)
            outputs = llm.chat(prompts, sampling_params=sampling_params, lora_request=lora_request, chat_template_kwargs={"enable_thinking": False} if "nemotron" in model_name.lower() else {})

        with open(output_csv_path, "a") as f:
            writer = csv.writer(f)
            for sample, o in zip(ds, outputs):
                generated_text = o.outputs[0].text
                
                writer.writerow([os.path.basename(sample["audio_path"]), generated_text, sample["gender"]])

        with open(output_jsonl_path, "a") as f:
            for prompt, o in zip(prompts, outputs):
                generated_text = o.outputs[0].text
                json.dump(prompt + [{
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": generated_text},
                        ],
                    }], f)
                f.write("\n")

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        generate_dataset(snakemake.params.model_name, snakemake.params.num_samples_per_prompt, snakemake.params.temperature, snakemake.params.max_samples)
    else:
        parser = argparse.ArgumentParser(description="Generate rejection emails based on audio recordings.")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name or path to the model checkpoint.") # "evaluation/model_cache/best_qwen3"
        parser.add_argument("--num_samples_per_prompt", type=int, default=240, help="Number of samples to generate per prompt.")
        parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the model.")
        parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process.")
        parser.add_argument("--adapter_path", type=str, default="evaluation/model_cache/best-lora_Qwen3-Omni-30B-A3B-Instruct", help="Path to the LoRA adapter checkpoint.") # "distilling/logs/test/final_adapters/adapter/corrected"
        args = parser.parse_args()
        generate_dataset(args.model_name, args.num_samples_per_prompt, args.temperature, args.max_samples, args.adapter_path)