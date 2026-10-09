import pandas as pd
import torchaudio
import torch
from tqdm import tqdm
import numpy as np
import os
import json

SOURCES = {
    "english_accents",
    # "stereoset_speech_gender",
    "british_dialects"
}

METADATA_PATH = "evaluation/gender_bias/full_diverse_voices_compare.csv"
OCTAVE_BINS = 12 # more bins = more precise pitch shifting, but more memory usage
FREQ_LOW = 65.41 # C2
FREQ_HIGH = 523.25 # C5 but woked best with C7

def find_median_pitch(gender_column = "gender_1"):
    metadata = pd.read_csv(METADATA_PATH).sample(frac=1, random_state=42)  # Shuffle the dataset
    pitches = {
           "female": [],
           "male": [],
    }
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        if row["source_1"] in SOURCES:
            waveform, sr = torchaudio.load("evaluation/gender_bias/" + row["audio_file_1"][5:])
            pitch_frames = torchaudio.functional.detect_pitch_frequency(waveform, 
                                                                      sample_rate=sr, 
                                                                      freq_low=FREQ_LOW,
                                                                      freq_high=FREQ_HIGH)
            avg_pitch = torch.median(pitch_frames[pitch_frames > 0])
            pitches[row[gender_column]].append(avg_pitch)
    return {
        "female": np.median(pitches["female"]).item(),
        "male": np.median(pitches["male"]).item()
    }

def semitone_difference(source_pitch: float, target_pitch: float) -> float:
    """
    Compute pitch shift (in semitones) required to transform source pitch
    to target pitch.

    from https://github.com/TAI-HAMBURG/SpeechAccess-Bias/blob/main/code/pitch_modifier.py
    """
    return round(OCTAVE_BINS * np.log2(target_pitch / source_pitch))

def change_pitch(audio_path, target_pitch=144, downsample_rate=None):
    name, ext = os.path.splitext(audio_path)
    new_audio_path = name + f"_shifted_{target_pitch}Hz" + ext
    if os.path.exists(new_audio_path):
        return new_audio_path
    
    waveform, sample_rate = torchaudio.load(audio_path)
    if downsample_rate is not None and sample_rate != downsample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=downsample_rate)(waveform)
        sample_rate = downsample_rate
    pitch_frames = torchaudio.functional.detect_pitch_frequency(waveform, 
                                                              sample_rate=sample_rate, 
                                                              freq_low=FREQ_LOW,
                                                              freq_high=FREQ_HIGH)
    avg_pitch = torch.median(pitch_frames[pitch_frames > 0])

    if torch.cuda.is_available():
        waveform = waveform.cuda()
        
    shifted_waveform = torchaudio.functional.pitch_shift(waveform, 
                                                         sample_rate, 
                                                         semitone_difference(avg_pitch.item(), target_pitch),
                                                         bins_per_octave=OCTAVE_BINS).cpu() 

    torchaudio.save(new_audio_path, shifted_waveform, sample_rate)

    return new_audio_path

if __name__ == "__main__":
    median_pitches = find_median_pitch()
    with open("evaluation/median_pitches.json", "w") as f:
        json.dump(median_pitches | {"mean": round((median_pitches['female'] + median_pitches['male']) / 2)}, f)