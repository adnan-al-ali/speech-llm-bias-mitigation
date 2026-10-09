from pathlib import Path
import os
from tqdm import tqdm
import csv
import dotenv
import soundfile as sf
from openai import OpenAI
import base64

dotenv.load_dotenv()


AUDIO_PATH = "evaluation/gender_assumption/recordings"

def make_dataset(max_samples = None):
    prompts = []
    for audio_file in tqdm(list(Path(AUDIO_PATH).glob("*.wav")), desc="Creating dataset"):
        prompts.append({
            "audio_path": audio_file.resolve(),
            "audio_file": audio_file.name,
        })
        if max_samples and len(prompts) >= max_samples:
            break
    return prompts

def make_prompt(audio_url):
    with open(audio_url, "rb") as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    return [
    # {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "input_audio", 
                "input_audio": {
                  "data": audio_base64,
                  "format": "wav"
                }
            }
        ],
    }]

def run_gender_assumption(model_name = "openai/gpt-audio", 
                              max_samples = None):

    ds = make_dataset(max_samples=max_samples)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OR_API_KEY")
    )

    
    prompts = [make_prompt(sample["audio_path"]) for sample in tqdm(ds, desc="Creating prompts")]

    os.makedirs("evaluation/gender_assumption/outputs", exist_ok=True)
    checkpoint_name = model_name.split("/")[-1]
    with open(f"evaluation/gender_assumption/outputs/{checkpoint_name}.csv", "w") as f:
        writer = csv.writer(f)

        for sample, prompt in tqdm(zip(ds, prompts), desc="Prompting model", total=len(prompts)):
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                max_tokens=5000,
                temperature=0.0,
            )

            generated_text = completion.choices[0].message.content
            writer.writerow([sample["audio_file"], generated_text])

if __name__ == "__main__":
    run_gender_assumption()