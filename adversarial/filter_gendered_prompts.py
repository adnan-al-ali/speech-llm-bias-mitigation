from datasets import load_dataset
from pathlib import Path
import os
from datasets import concatenate_datasets
import re

splits = ["train"]

def find_audio_path(dummy_path: str, as_uri: bool = True) -> str:
    filename = dummy_path.split("/")[-1]
    speaker_id = filename.split("_")[0]
    path = f"adversarial/downloads/vctk-corpus/VCTK-Corpus/VCTK-Corpus/wav48/{speaker_id}/{filename}"
    resolved_path = Path(path).resolve()
    if as_uri:
        return resolved_path.as_uri()
    return str(resolved_path)

def get_gender_dict():
    speaker_gender_dict = {}
    with open("adversarial/downloads/vctk-corpus/VCTK-Corpus/VCTK-Corpus/speaker-info.txt") as f:
        f.readline()
        for line in f:
            fields = line.strip().split()
            speaker_id = fields[0]
            gender  = fields[2]
            speaker_gender_dict[speaker_id] = gender
    return speaker_gender_dict

def get_speaker_gender(messages: list[dict], speaker_gender_dict):
    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["audio_path"] is not None:
                    # Fix the format

                    speaker = item["audio_path"].split("/")[-1].split("_")[0]
                    return speaker_gender_dict[speaker[1:]]

def contains_gendered(messages):
    gendered_terms = ["gender", "sex", "male", "female", "man", "men", "woman", "women", "boy", "girl", "wife", "husband", "father", "mother", "son", "daughter", "brother", "sister"]
    prompt = str(messages).lower()
    return any(re.search(rf"\b{re.escape(term)}s?\b", prompt) for term in gendered_terms)

def has_1_audio(messages):
    audio_count = 0
    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["audio_path"] is not None:
                    audio_count += 1
    return audio_count == 1

if __name__ == "__main__":
    for split in splits:
        if not os.path.isfile(f"adversarial/downloads/sift_50m_vctk_en_{split}_ungendered_filtered.parquet"):
            dataset = load_dataset("parquet", data_files=f"adversarial/downloads/sift_50m_vctk_en_{split}.parquet", split=split)
            ungendered_prompts = dataset.filter(lambda sample: not contains_gendered(sample["messages"]) and has_1_audio(sample["messages"]), desc="Filtering ungendered prompts")
            ungendered_prompts.shuffle(seed=42) 

            gender_dict = get_gender_dict()
            male_speakers_ds = ungendered_prompts.filter(lambda sample: get_speaker_gender(sample["messages"], gender_dict) == "M")
            female_speakers_ds = ungendered_prompts.filter(lambda sample: get_speaker_gender(sample["messages"], gender_dict) == "F")

            shorter_len = min(len(male_speakers_ds), len(female_speakers_ds))
            balanced_ungendered_prompts = concatenate_datasets([male_speakers_ds.select(range(shorter_len)), female_speakers_ds.select(range(shorter_len))]).shuffle(seed=42)


            balanced_ungendered_prompts.to_parquet(f"adversarial/downloads/sift_50m_vctk_en_{split}_ungendered_filtered.parquet")