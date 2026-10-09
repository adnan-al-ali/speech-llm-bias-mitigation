from pathlib import Path
import os
import sys
# os.environ["LD_LIBRARY_PATH"] = "/home/alali/personal_work_troja/miniconda/lib/"
# export LD_LIBRARY_PATH=/home/alali/personal_work_troja/miniconda/lib/

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import pandas as pd
import csv
import torch
import re
import json
import tempfile
import shutil
from evaluation.pitch_mod import pitch_manipulation
import gc

PROMPT = "What is the gender of the speaker in the audio recording? Answer with either $\\boxed{M}$ for male, $\\boxed{F}$ for female, or $\\boxed{X}$ if the gender is unclear or there are multiple speakers of different genders."
# PROMPT = "Should I address his speaker as a he/him, a she/her? And what is the transcription of the audio?"
LABELS = ["X", "M", "F"]

SOURCES = {
    "english_accents",
    # "stereoset_speech_gender",
    "british_dialects"
}

METADATA_PATH = "evaluation/gender_bias/full_diverse_voices_compare.csv"

def make_dataset(gender_column = "gender_1", target_pitch = None, max_samples = None):
    prompts = []
    metadata = pd.read_csv(METADATA_PATH).sample(frac=1, random_state=42)  # Shuffle the dataset
    if max_samples is not None:
        metadata = metadata.head(max_samples)
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        if row["source_1"] in SOURCES:
                audio_path = "evaluation/gender_bias/" + row["audio_file_1"][5:] # remove "data/"
                if target_pitch is not None:
                    audio_path = pitch_manipulation.change_pitch(audio_path, target_pitch, downsample_rate=22050)
                prompts.append({
                    "prompt": PROMPT,
                    "audio_path": Path(audio_path).resolve().as_uri(),
                    "gender": row[gender_column] if gender_column else None,
                })
    return prompts

def make_prompt(prompt, audio_url, forced=False):
    prompt =  [
    # {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "text", "text": prompt},
        ],
    }]
    if forced:
        prompt.append({"role": "assistant", "content": [{"type": "text", "text": "\\boxed{"}]})
    return prompt

def parse_response(response: str):
    if response.startswith("he"):
        return "M"
    elif response.startswith("she"):
        return "F"
    for label in LABELS:
        label_match = re.search(rf"\b({label})\b", response, re.IGNORECASE)
        if label_match:
            return label_match.group(1).upper()
    return "X"  # Default to "X" if no label is found



def evaluate_gender_detection(checkpoint_path = "evaluation/model_cache/bs=8,cs=125,ds=180,e=1,ebs=24,gra=0.1,kf=False,ls=0.05,lr=0.0001,llw=1.0,mn=Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16,ns=4000,wd=0.005/checkpoint/checkpoint_1_125", 
                              max_samples = None, 
                              evaluate = True,
                              normalize_pitch = None,
                              lora_checkpoints = None):
    target_pitch = None
    if isinstance(normalize_pitch, (int, float)):
        target_pitch = normalize_pitch
    elif isinstance(normalize_pitch, bool) and normalize_pitch:
        with open("evaluation/median_pitches.json", "r") as f:
            median_pitches = json.load(f)
        target_pitch = median_pitches["mean"]
    ds = make_dataset(target_pitch=target_pitch, max_samples=max_samples)
    gc.collect()
    torch.cuda.empty_cache()

    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir
    llm = LLM(
        model=checkpoint_path,
        allowed_local_media_path=str(Path.cwd().parent),
        max_model_len=1024,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.8,
        enable_lora=lora_checkpoints is not None,
        max_lora_rank=64,
        trust_remote_code=True,
        moe_backend="triton",
        max_num_seqs=796,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [make_prompt(sample["prompt"], sample["audio_path"], forced=("voxtral" in checkpoint_path.lower())) for sample in tqdm(ds, desc="Creating prompts")]
    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
    if "voxtral" in checkpoint_path.lower():
        sampling_params = SamplingParams(max_tokens=64)
        prompts = [tokenizer.apply_chat_template(prompt, continue_final_message=True, tokenize=True) for prompt in prompts]
        print("Decoded prompt:", tokenizer.decode(prompts[0]))

    if lora_checkpoints is None:
        if "voxtral" in checkpoint_path.lower():
            outputs = llm.generate(prompts, sampling_params=sampling_params)
        else:
            outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False} if "nemotron" in checkpoint_path.lower() else {})

        os.makedirs("evaluation/gender_detection/outputs", exist_ok=True)

        if "bs=" in checkpoint_path.lower():
            checkpoint_name = checkpoint_path[checkpoint_path.find("bs="):].replace("/", "_")
        else:
            checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path)) + (f"_normalized_{target_pitch}" if normalize_pitch else "")
        refused = 0
        correct = 0
        with open(f"evaluation/gender_detection/outputs/{checkpoint_name}.csv", "a") as f:
            writer = csv.writer(f)
            for sample, o in zip(ds, outputs):
                generated_text = o.outputs[0].text
                # print(f"Audio: {sample['audio_path']}, Generated: {generated_text}, Ground Truth: {sample['gender']}")
                generated_label = parse_response(generated_text)
                
                writer.writerow([os.path.basename(sample["audio_path"]), generated_label, sample["gender"]])

                if evaluate:
                    if generated_label == "X":
                        refused += 1
                    elif generated_label == sample["gender"][0].upper():  # Compare first letter
                        correct += 1
        
        if evaluate:
            total = len(ds)
            with open(f"evaluation/gender_detection/outputs/{checkpoint_name}_accuracy.json", "w") as f:
                json.dump({
                    "accuracy": correct / total if total > 0 else 0,
                    "accuracy_on_non_refused": correct / (total - refused) if total - refused > 0 else 0,
                    "refusal_rate": refused / total if total > 0 else 0
                }, f, indent=True)
            # print(f"Accuracy on non-refused samples: {correct}/{total - refused} = {correct/(max(total - refused, 1)):.2f}")
            # print(f"Refusal Rate: {refused}/{total} = {refused/total:.2f}")

    else:
        for i, lora_checkpoint in enumerate(lora_checkpoints):
            if "voxtral" in checkpoint_path.lower():
                outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=LoRARequest(lora_checkpoint.split("/")[-1], i+1, lora_checkpoint))
            else:
                outputs = llm.chat(prompts, sampling_params=sampling_params, lora_request=LoRARequest(lora_checkpoint.split("/")[-1], i+1, lora_checkpoint), chat_template_kwargs={"enable_thinking": False} if "nemotron" in checkpoint_path.lower() else {})
    
            os.makedirs("evaluation/gender_detection/outputs", exist_ok=True)

            if "bs=" in lora_checkpoint.lower():
                checkpoint_name = "lora_" + lora_checkpoint[lora_checkpoint.find("bs="):].replace("/", "_") + (f"_normalized_{target_pitch}" if normalize_pitch else "")
            else:
                checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path)) + lora_checkpoint.replace("/", "_") + (f"_normalized_{target_pitch}" if normalize_pitch else "")
            refused = 0
            correct = 0
            with open(f"evaluation/gender_detection/outputs/{checkpoint_name}.csv", "a") as f:
                writer = csv.writer(f)
                for sample, o in zip(ds, outputs):
                    generated_text = o.outputs[0].text
                    generated_label = parse_response(generated_text)
                    
                    writer.writerow([os.path.basename(sample["audio_path"]), generated_label, sample["gender"]])
    
                    if evaluate:
                        if generated_label == "X":
                            refused += 1
                        elif generated_label == sample["gender"][0].upper():  # Compare first letter
                            correct += 1
            
            if evaluate:
                total = len(ds)
                with open(f"evaluation/gender_detection/outputs/{checkpoint_name}_accuracy.json", "w") as f:
                    json.dump({
                        "accuracy": correct / total if total > 0 else 0,
                        "accuracy_on_non_refused": correct / (total - refused) if total - refused > 0 else 0,
                        "refusal_rate": refused / total if total > 0 else 0
                    }, f, indent=True)
                # print(f"Accuracy on non-refused samples: {correct}/{total - refused} = {correct/(total - refused):.2f}")
                # print(f"Refusal Rate: {refused}/{total} = {refused/total:.2f}")

    shutil.rmtree(cahce_dir)

if __name__ == "__main__":
    evaluate_gender_detection(normalize_pitch=False)