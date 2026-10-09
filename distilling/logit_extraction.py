from unittest import case

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration
import torch
import librosa
from pathlib import Path
from urllib.parse import unquote, urlparse
import sys
import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset, Column
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from adversarial.filter_gendered_prompts import find_audio_path
from adversarial.nemotron.modeling import NemotronH_Nano_Omni_Reasoning_V3
from adversarial.nemotron.processing import NemotronH_Nano_Omni_Reasoning_V3Processor
from utils import convert_sift50m_sample_to_chat_format
import argparse


def find_assistant_sequence_tokens(processor):
    sample = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "..."}]
        }
    ]

    try:
        chat_no_prompt = processor.apply_chat_template(
            sample, 
            add_generation_prompt=False, 
            tokenize=True, return_tensors="pt", 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {})).squeeze()
        chat_with_prompt = processor.apply_chat_template(
            sample, 
            add_generation_prompt=True, 
            tokenize=True, return_tensors="pt", 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {})).squeeze()
    except Exception:
        chat_no_prompt = processor.tokenizer.apply_chat_template(
            sample, 
            add_generation_prompt=False, 
            tokenize=True, 
            return_tensors="pt", 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {})).squeeze()
        chat_with_prompt = processor.tokenizer.apply_chat_template(
            sample, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt", 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {})).squeeze()
    assistant_sequence = chat_with_prompt[len(chat_no_prompt):]
    return assistant_sequence

def extract_audio_from_sample(sample: list[dict], processor):
    audios = []
    for i, message in enumerate(sample):
        assert i < 2, "Expected at most two messages per sample."
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("audio_url") is not None:
                    audio_url = item["audio_url"]["url"]
                    sr = int(processor.audio_sampling_rate if hasattr(processor, 'audio_sampling_rate') else processor.feature_extractor.sampling_rate)
                    audio_data, _ = librosa.load(unquote(urlparse(audio_url).path), sr=sr)
                    audios.append(audio_data)

                elif item.get("text") is not None:
                    item["type"] = "text"
    
    # assert len(audios) == 1, "Expected exactly one audio clip per sample."
    return audios
    
def prepare_sample_batch(samples: list[list[dict]], processor):
    if "nemotron" in type(processor).__name__.lower():
        samples = processor.preprocess_chat(samples)

    audios = []
    for s in samples:
        audios.extend(extract_audio_from_sample(s, processor))
    try:
        chat_applied = processor.apply_chat_template(
            samples, 
            add_generation_prompt=False, 
            tokenize=False, 
            continue_final_message=True, 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {}))
    except Exception:
        chat_applied = processor.tokenizer.apply_chat_template(
            samples, 
            add_generation_prompt=False, 
            tokenize=False, 
            continue_final_message=True, 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {}))

    inputs = processor(
        text=chat_applied,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,
        sampling_rate=processor.audio_sampling_rate if hasattr(processor, 'audio_sampling_rate') else processor.feature_extractor.sampling_rate
    )

    assistant_start_tokens = find_assistant_sequence_tokens(processor)
    assistant_tokens_masks = []
    for row_tokens, row_mask in zip(inputs["input_ids"], inputs["attention_mask"]):
        match_index = (row_tokens.unfold(0, len(assistant_start_tokens), 1) == assistant_start_tokens).all(dim=-1).int().argmax() - 1
        assistant_tokens_mask = torch.zeros_like(row_tokens, dtype=torch.bool)
        assistant_tokens_mask[match_index + len(assistant_start_tokens):] = True
        assistant_tokens_mask = assistant_tokens_mask[-row_mask.sum():]
        # print(f"Assistant tokens: `{processor.tokenizer.decode(row_tokens[match_index + len(assistant_start_tokens):], skip_special_tokens=False)}`")
        assistant_tokens_masks.append(assistant_tokens_mask)

    return {
        "inputs": inputs,
        "assistant_tokens_masks": assistant_tokens_masks
    }

@torch.no_grad()
def extract_logits(sample_batch: list[list[dict]], model, processor, target_directory, start_index):
    try:
        prepared_batch = prepare_sample_batch(sample_batch, processor)
        inputs = prepared_batch["inputs"] #.to(model.device, torch.bfloat16)
        batch_assistant_tokens_mask = prepared_batch["assistant_tokens_masks"]

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        
        if hasattr(model, "thinker"):
            logits = model.thinker(
                input_ids = inputs.input_ids.to(model.thinker.model.device),
                attention_mask = inputs.attention_mask.to(model.thinker.model.device),
                input_features = inputs.input_features.to(model.thinker.audio_tower.device, torch.bfloat16),
                feature_attention_mask = inputs.feature_attention_mask.to(model.thinker.audio_tower.device)
            ).logits
        else:
            logits = model(
                input_ids = inputs.input_ids.to(model.device),
                attention_mask = inputs.attention_mask.to(model.device),
                input_features = inputs.input_features.to(model.device) if hasattr(inputs, "input_features") else inputs.sound_clips,
                feature_attention_mask = inputs.feature_attention_mask.to(model.device) if hasattr(inputs, "feature_attention_mask") else None
            ).logits
        success = True
    except torch.OutOfMemoryError as e:
        if len(sample_batch) == 1:
            raise e
        print(f"OOM. Feeding the batch sample-by-sample to the model to avoid OOM.")
        success = False

    if not success:
        for sample in sample_batch:
            extract_logits([sample], model, processor, target_directory, start_index)
            start_index += 1
    else:
        os.makedirs(os.path.join(target_directory, "logits"), exist_ok=True)
        for i, (logit, assistant_mask, sample) in enumerate(zip(logits, batch_assistant_tokens_mask, sample_batch)):
            assistant_logits = logit[-assistant_mask.sum():]
            assistant_logits = assistant_logits.cpu()
            if torch.isnan(assistant_logits).any():
                print(f"Warning: NaN values found in assistant logits for sample {start_index + i}. Replacing NaNs with -inf.")
            # for debug
            # assistant_tokens = assistant_logits.argmax(dim=-1)
            # print("Assistant tokens:", processor.tokenizer.decode(assistant_tokens.cpu(), skip_special_tokens=False))

            # print("Saving tensor of shape", assistant_logits.shape, "of size", assistant_logits.element_size() * assistant_logits.nelement() / (1024 * 1024), "MB.") 
            torch.save(assistant_logits, os.path.join(target_directory, "logits", f"sample_{start_index + i}.pt"))

            with open(os.path.join(target_directory, "prompts.jsonl"), "a") as f:
                f.write(json.dumps(sample) + "\n")

        # logits_samples_assistant = [sample[-mask.sum():].cpu() for sample, mask in zip(logits, inputs.attention_mask)]
        # batch_assistant_tokens_mask = [mask_assistant[-mask.sum():].cpu() for mask_assistant, mask in zip(batch_assistant_tokens_mask, inputs.attention_mask)]
        # top_logits = logits_samples[0].argmax(dim=-1)
        # # print(f"Top logits: `{processor.tokenizer.decode(top_logits.cpu() * batch_assistant_tokens_mask[0], skip_special_tokens=False)}`")
        # # print(f"Targets: `{processor.tokenizer.decode(inputs.input_ids[0, -batch_assistant_tokens_mask[0].shape[0]:].cpu()  * batch_assistant_tokens_mask[0], skip_special_tokens=False)}`")

        # with open(target_filename, "wb") as f:
        #     pickle.dump([{"logits": logit, "mask": mask, "conversation": sample} for logit, mask, sample in zip(logits_samples, batch_assistant_tokens_mask, sample_batch)], f)

        # return logits_samples, batch_assistant_tokens_mask

def load_gendered_dataset(gendered_source_model, max_samples):
    samples = []
    files = os.listdir("distill_gender_dataset/datasets")
    for file in files:
        if file.startswith(gendered_source_model) and file.endswith(".jsonl"):
            dataset_path = os.path.join("distill_gender_dataset/datasets", file)
            with open(dataset_path, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    if len(sample[1]["content"][0]["text"]) > 3000:
                        continue # skip samples where the model overgenerated the assistant response
                    samples.append(sample)
    samples = random.sample(samples, min(max_samples, len(samples)))
    return samples

def load_sift50m_custom_dataset(model, max_samples):
    samples = []
    with open(f"distilling/datasets/sift_50m_{model}.jsonl", "r") as f:
        for line in f:
            sample = json.loads(line)
            if len(sample[1]["content"][0]["text"]) > 3000:
                continue # skip samples where the model overgenerated the assistant response
            samples.append(sample)
    samples = random.sample(samples, min(max_samples, len(samples)))
    return samples

def main(model_name, dataset_name, gendered_source_model=None, max_samples=8000, batch_size=6, seed=67):
    random.seed(seed)
    model_short = model_name.split("/")[-1]

    match dataset_name:
        case "sift-50m-custom":
            dataset = load_sift50m_custom_dataset(model_short, max_samples)
        case "sift-50m":
            dataset = load_dataset("parquet", data_files="adversarial/downloads/sift_50m_vctk_en_train.parquet")["train"]
            dataset: Column = dataset.shuffle(seed=seed).select(range(max_samples))["messages"]
            dataset = list(map(convert_sift50m_sample_to_chat_format, dataset))
        case "gendered":
            dataset = load_gendered_dataset(gendered_source_model, max_samples)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    directory = f"distilling/logits/{model_short}/{dataset_name}"
    print(f"Saving logits to: {directory}")
    os.makedirs(directory, exist_ok=True)
    # clear the jsonl
    with open(os.path.join(directory, "prompts.jsonl"), "w") as f:
        pass

    if "qwen2" in model_name.lower():
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, 
            device_map="auto", 
            attn_implementation="flash_attention_2", 
            dtype=torch.bfloat16)
    elif "qwen3" in model_name.lower():
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name, 
            device_map="auto", 
            attn_implementation="flash_attention_2", 
            dtype=torch.bfloat16)
    elif "nemotron" in model_name.lower():
        processor = NemotronH_Nano_Omni_Reasoning_V3Processor.from_pretrained(
            model_name,
            trust_remote_code=True)
        model = NemotronH_Nano_Omni_Reasoning_V3.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    processor.tokenizer.padding_side = "left"

    logits, masks = [], []
    for batch_number, batch_start in tqdm(enumerate(range(0, len(dataset), batch_size)), total=len(dataset)//batch_size, desc="Processing batches"):
        batch = dataset[batch_start:batch_start + batch_size]
        extract_logits(batch, model, processor, directory, start_index=batch_start)
        # batch_logits, batch_masks = extract_logits(batch, model, processor, f"{directory}/batch_{batch_number}.pkl")
        # logits.extend(batch_logits)
        # masks.extend(batch_masks)

    # os.makedirs(f"distilling/logits/{model_short}", exist_ok=True)
    # with open(f"distilling/logits/{model_short}/{dataset_name}.pkl", "wb") as f:
    #     pickle.dump([{"logits": logit, "mask": mask, "conversation": sample} for logit, mask, sample in zip(logits, masks, dataset)], f)
if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        main(
            model_name=snakemake.params.model_name,
            dataset_name=snakemake.params.dataset,
            gendered_source_model=snakemake.params.gendered_source_model,
            max_samples=snakemake.params.max_samples
        )
    else:
        parser = argparse.ArgumentParser(description="Extract logits from a sample conversation using a Qwen model.")
        parser.add_argument("--model_name", type=str, default="evaluation/model_cache/best-checkpoint_Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", help="Model name or path to the model checkpoint.") # Qwen/Qwen2-Audio-7B-Instruct Qwen/Qwen3-Omni-30B-A3B-Instruct evaluation/model_cache/best_qwen3
        parser.add_argument("--dataset", type=str, default="gendered", choices=["sift-50m", "sift-50m-custom", "gendered"], help="Dataset to use for extracting logits.")
        parser.add_argument("--gendered_source_model", type=str, default="best-checkpoint_Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", help="Source model for the gendered dataset logits extraction.")
        parser.add_argument("--max_samples", type=int, default=4_000, help="Maximum number of samples to process.")
        parser.add_argument("--batch_size", type=int, default=6, help="Batch size for processing samples.")
        parser.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility.")
        args = parser.parse_args()

        main(args.model_name, args.dataset, args.gendered_source_model, args.max_samples, args.batch_size, args.seed)