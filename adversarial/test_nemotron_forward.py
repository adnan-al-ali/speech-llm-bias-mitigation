import torch
import numpy as np
from pathlib import Path

from nemotron.modeling import NemotronH_Nano_Omni_Reasoning_V3
from nemotron.processing import NemotronH_Nano_Omni_Reasoning_V3Processor
from modeling import NemotronWithReversedGenderHead
from test_prompt import SAMPLE
import librosa


def decode_token_lists(token_lists, tokenizer):
    texts = [tokenizer.decode(t, skip_special_tokens=False) for t in token_lists]
    return texts


def prepare_sample(sample, processor):
    sr = int(processor.audio_sampling_rate if hasattr(processor, 'audio_sampling_rate') else processor.feature_extractor.sampling_rate)
    audios = []
    for message in sample:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("audio_url") is not None:
                    audio_url = item["audio_url"]["url"]
                    # Convert URI to local path if needed
                    if audio_url.startswith("file://"):
                        audio_path = Path(audio_url[len("file://"):]).resolve()
                    else:
                        audio_path = Path(audio_url).resolve()
                    audio_data, _ = librosa.load(str(audio_path), sr=sr)

                    # Load audio and resample
                    audios.append(audio_data)
        
        assert len(audios) == 1, "Expected exactly one audio clip per sample."
    
        try:
            chat_applied = processor.apply_chat_template(
                sample, 
                tokenize=False, 
                enable_thinking=False)
        except Exception:
            chat_applied = processor.tokenizer.apply_chat_template(
                sample, 
                tokenize=False, 
                enable_thinking=False)

    return chat_applied, audios, sr



def run_inference_single_and_batch(model_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model_name = model_name
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor.from_pretrained(full_model_name, trust_remote_code=True)
    base_model = NemotronH_Nano_Omni_Reasoning_V3.from_pretrained(full_model_name, trust_remote_code=True, device_map='auto')
    model = NemotronWithReversedGenderHead(base_model, gradient_reversal_alpha=1.0)
    model.eval()

    sample = SAMPLE

    # Single sample
    chat_applied, audios, sr = prepare_sample(sample, processor)
    inputs = processor(
        text=chat_applied,
        audio=audios[0],
        return_tensors="pt",
        padding=True,
        truncation=True,
        sampling_rate=sr
    )
    inputs = inputs.to(device, torch.bfloat16)

    # Autoregressive greedy generation for 32 tokens (no EOS checking)
    max_gen = 64
    input_ids = inputs.get('input_ids')
    attention_mask = inputs.get('attention_mask')
    feature_attention_mask = inputs.get('feature_attention_mask', None)
    input_features = inputs.sound_clips

    # ensure tensors on device
    if input_ids is not None:
        input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if input_features is not None and isinstance(input_features, torch.Tensor):
        input_features = input_features.to(device)

    generated_tokens = [[] for _ in range(input_ids.size(0))]
    for _step in range(max_gen):
        with torch.no_grad():
            logits, _ = model.forward(input_ids=input_ids, input_features=input_features, attention_mask=attention_mask, feature_attention_mask=feature_attention_mask)
        # logits shape returned as (batch, vocab, seq_len)
        # take last position logits
        last_logits = logits[:, :, -1]  # (batch, vocab)
        next_tokens = last_logits.argmax(dim=1)
        # append tokens to generated lists and to input_ids
        next_tokens = next_tokens.unsqueeze(1)
        for i in range(input_ids.size(0)):
            generated_tokens[i].append(int(next_tokens[i].item()))
        input_ids = torch.cat([input_ids, next_tokens.to(input_ids.device)], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

    texts = decode_token_lists(generated_tokens, processor.tokenizer)
    print('Single sample decoded (32 tokens):', texts)

    # Batch of two copies
    # Batch of two copies: prepare and duplicate tensors
    inputs2 = processor(
        text=[chat_applied] * 2,
        audio=audios * 2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        sampling_rate=sr
    ).to(device, torch.bfloat16)

    input_ids = inputs2.get('input_ids').to(device)
    attention_mask = inputs2.get('attention_mask').to(device)
    feature_attention_mask = inputs2.get('feature_attention_mask', None)
    input_features = inputs2.sound_clips
    if input_features is not None and isinstance(input_features, torch.Tensor):
        input_features = input_features.to(device)

    generated_tokens = [[] for _ in range(input_ids.size(0))]
    for _step in range(max_gen):
        with torch.no_grad():
            logits, _ = model.forward(input_ids=input_ids, input_features=input_features, attention_mask=attention_mask, feature_attention_mask=feature_attention_mask)
        last_logits = logits[:, :, -1]
        next_tokens = last_logits.argmax(dim=1).unsqueeze(1)
        for i in range(input_ids.size(0)):
            generated_tokens[i].append(int(next_tokens[i].item()))
        input_ids = torch.cat([input_ids, next_tokens.to(input_ids.device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

    texts2 = decode_token_lists(generated_tokens, processor.tokenizer)
    print('Batch decoded (2 copies, 32 tokens):', texts2)


if __name__ == '__main__':
    # Replace with the actual model name/path used during development
    MODEL_NAME = 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16'
    run_inference_single_and_batch(MODEL_NAME)
