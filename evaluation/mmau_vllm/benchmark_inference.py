import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
import re
# import soundfile as sf
import json
from tqdm import tqdm
import argparse
import csv
import torch
import logging
import tempfile
import shutil
import torchaudio
from evaluation.pitch_mod import pitch_manipulation
import gc
# export LD_LIBRARY_PATH=${CONDA_PREFIX_1}/lib/


PROMPT = """Question: {question}

Options:
{options}

State the correct option in a box: $\\boxed{{(X) Example answer}}$."""

def contains_gendered(messages):
    gendered_terms = ["gender", "sex", "male", "female", "man", "men", "woman", "women", "boy", "girl", "wife", "husband", "father", "mother", "son", "daughter", "brother", "sister"]
    prompt = str(messages).lower()
    found_gendered =  any(re.search(rf"\b{re.escape(term)}s?\b", prompt) for term in gendered_terms)
    return found_gendered

def save_audio_to_local(audio_array, sampling_rate, sample_id, downsample_rate=16000, target_pitch=None):
    os.makedirs("evaluation/mmau_vllm/audio_files", exist_ok=True)

    file_path = f"evaluation/mmau_vllm/audio_files/{sample_id}.wav"
    if os.path.isfile(file_path) and target_pitch is None:
        return file_path
    audio_array = torch.from_numpy(audio_array)
    if downsample_rate is not None and sampling_rate != downsample_rate:
        audio_array = torchaudio.functional.resample(audio_array.cuda(), orig_freq=sampling_rate, new_freq=downsample_rate).cpu()
        sampling_rate = downsample_rate
    #sf.write(file_path, audio_array, sampling_rate)
    torchaudio.save(file_path, audio_array, sampling_rate)
    del(audio_array)
    gc.collect()
    torch.cuda.empty_cache()

    if target_pitch is None:
        return file_path
    else:
        return pitch_manipulation.change_pitch(file_path, target_pitch=target_pitch)

def make_prompt(question, options, audio_url):
    formatted_prompt = PROMPT.format(question=question, options="\n".join(options))
    return [{
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "text", "text": formatted_prompt},
        ],
    }]

def make_multimodal_prompt(data_sample: dict, target_pitch=None):
    instruction = data_sample["instruction"]
    options = data_sample["choices"]
    audio = data_sample["context"]
    sample_id = json.loads(data_sample["other_attributes"])["id"]
    audio_url = Path(save_audio_to_local(audio["array"], audio["sampling_rate"], sample_id, target_pitch=target_pitch)).resolve().as_uri()

    return make_prompt(instruction, options, audio_url)

def parse_response(response: str):
    if "$\\boxed{" in response:
        match = re.search(r"\\boxed{[^A-D]*([A-D])[^}]*}", response)
    else:
        match = re.search(r"\(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        return None


def evaluate_mmau(
        dataset_name = "gamma-lab-umd/MMAU-test-mini", 
        split = "test", 
        model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct", 
        target_pitch=None,
        lora_checkpoints = None):
    logger = logging.getLogger("mmau")
    logger.setLevel(logging.INFO)
    logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")

    ds = load_dataset(dataset_name, split=split).filter(lambda sample: not contains_gendered(sample["instruction"] + str(sample["choices"])), desc="Filtering gendered prompts")
    
    prompts = [make_multimodal_prompt(sample, target_pitch=target_pitch) for sample in tqdm(ds, desc="Creating prompts")]
    gc.collect()
    torch.cuda.empty_cache()

    cahce_dir = tempfile.mkdtemp()
    os.environ["VLLM_CACHE_ROOT"] = cahce_dir
    llm = LLM(
        model=model_name,
        allowed_local_media_path=str(Path.cwd()),
        max_model_len=1024,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.8,
        max_num_seqs=796,
        enable_lora=lora_checkpoints is not None,
        max_lora_rank=64,
        trust_remote_code=True,
        moe_backend="triton"
    )

    model_short = os.path.basename(os.path.normpath(model_name))
    if "bs=" in model_name.lower():
        model_short = model_name[model_name.find("bs="):].replace("/", "_")

    if lora_checkpoints is None:
        sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)
        outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False} if "nemotron" in model_name.lower() else {})

        os.makedirs("evaluation/mmau_vllm/results", exist_ok=True)
        with open(f"evaluation/mmau_vllm/results/{model_short}{'_' + str(target_pitch) if target_pitch is not None else ''}_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Sample ID", "Generated Option", "Ground Truth"])
            sum_correct = 0
            for sample, o in zip(ds, outputs):
                generated_text = o.outputs[0].text
                parsed_option = parse_response(generated_text)
                writer.writerow([json.loads(sample["other_attributes"])["id"], parsed_option, sample["answer"][1]])
                if parsed_option == sample["answer"][1]:  # Compare only the option letter
                    sum_correct += 1
        accuracy = sum_correct / len(ds)
        with open(f"evaluation/mmau_vllm/results/{model_short}{'_' + str(target_pitch) if target_pitch is not None else ''}_accuracy.txt", "w") as f:
            f.write(str(accuracy) + "\n")
    else:
        for i, lora_checkpoint in enumerate(lora_checkpoints):
            lora_save_checkpoint_name = lora_checkpoint[lora_checkpoint.find("bs="):].replace("/", "_")
            accuracy_filename = f"evaluation/mmau_vllm/results/lora_{lora_save_checkpoint_name}{'_' + str(target_pitch) if target_pitch is not None else ''}_accuracy.txt"
            if os.path.exists(accuracy_filename):
                logger.info(f"Accuracy file {accuracy_filename} already exists. Skipping evaluation for this LoRA checkpoint.")
                continue
            sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)
            outputs = llm.chat(prompts, sampling_params=sampling_params, lora_request=LoRARequest(lora_checkpoint.split("/")[-1], i + 1, lora_checkpoint), chat_template_kwargs={"enable_thinking": False} if "nemotron" in model_name.lower() else {})

            os.makedirs("evaluation/mmau_vllm/results", exist_ok=True)
            with open(f"evaluation/mmau_vllm/results/lora_{lora_save_checkpoint_name}{'_' + str(target_pitch) if target_pitch is not None else ''}_results.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Sample ID", "Generated Option", "Ground Truth"])
                sum_correct = 0
                for sample, o in zip(ds, outputs):
                    generated_text = o.outputs[0].text
                    parsed_option = parse_response(generated_text)
                    writer.writerow([json.loads(sample["other_attributes"])["id"], parsed_option, sample["answer"][1]])
                    if parsed_option == sample["answer"][1]:  # Compare only the option letter
                        sum_correct += 1
            accuracy = sum_correct / len(ds)
            with open(accuracy_filename, "w") as f:
                f.write(str(accuracy) + "\n")
    
    shutil.rmtree(cahce_dir)
        
if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        evaluate_mmau(
            **snakemake.params,
        )
    else:
        parser = argparse.ArgumentParser(description="Evaluate LLM on MMAU dataset")
        parser.add_argument("--dataset_name", type=str, default="gamma-lab-umd/MMAU-test-mini", help="Name of the dataset to evaluate on")
        parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
        parser.add_argument("--model_name", type=str, default="unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning", help="Name of the model to evaluate")
        parser.add_argument("--target_pitch", type=int, default=None, help="Target pitch in Hz for pitch manipulation")
        args = parser.parse_args()

        evaluate_mmau(dataset_name=args.dataset_name, split=args.split, model_name=args.model_name, target_pitch=args.target_pitch)