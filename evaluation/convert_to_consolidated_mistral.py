# Inverse of the Voxtral (Mistral -> HF) conversion script https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/convert_voxtral_weights_to_hf.py
# Inverted by Claude
 
"""
Inverse of the Voxtral (Mistral -> HF) conversion script: takes a directory containing
HF-format model shards (config.json + one or more *.safetensors files, optionally with a
model.safetensors.index.json) and writes out a single Mistral-format `consolidated.safetensors`
file, with keys renamed and weights un-permuted back to Mistral's original layout.

Note: the tokenizer is intentionally not handled here.
"""
import argparse
import glob
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

from transformers import VoxtralConfig


# fmt: off
# Inverse of STATE_DICT_MAPPING from the forward conversion script.
# Maps HF key patterns -> original Mistral key patterns.
REVERSE_STATE_DICT_MAPPING = {
    # Text model keys
    r"^language_model.lm_head.weight":                                                  r"output.weight",
    r"^language_model.model.norm.weight":                                               r"norm.weight",
    r"^language_model.model.embed_tokens.weight":                                       r"tok_embeddings.weight",
    r"^language_model.model.layers.(\d+).input_layernorm.weight":                       r"layers.\1.attention_norm.weight",
    r"^language_model.model.layers.(\d+).post_attention_layernorm.weight":              r"layers.\1.ffn_norm.weight",
    r"^language_model.model.layers.(\d+).self_attn.(q|k|v|o)_proj.weight":              r"layers.\1.attention.w\2.weight",
    r"^language_model.model.layers.(\d+).mlp.gate_proj.weight":                         r"layers.\1.feed_forward.w1.weight",
    r"^language_model.model.layers.(\d+).mlp.down_proj.weight":                         r"layers.\1.feed_forward.w2.weight",
    r"^language_model.model.layers.(\d+).mlp.up_proj.weight":                           r"layers.\1.feed_forward.w3.weight",

    # audio model keys
    r"^audio_tower.conv1.(weight|bias)":                                                r"mm_whisper_embeddings.whisper_encoder.conv_layers.0.\1",
    r"^audio_tower.conv2.(weight|bias)":                                                r"mm_whisper_embeddings.whisper_encoder.conv_layers.1.\1",
    r"^audio_tower.layer_norm.(weight|bias)":                                           r"mm_whisper_embeddings.whisper_encoder.transformer.norm.\1",
    r"^audio_tower.layers.(\d+).self_attn.(q|k|v)_proj.(weight|bias)":                  r"mm_whisper_embeddings.whisper_encoder.transformer.layers.\1.attention.w\2.\3",
    r"^audio_tower.layers.(\d+).self_attn.out_proj.(weight|bias)":                      r"mm_whisper_embeddings.whisper_encoder.transformer.layers.\1.attention.wo.\2",
    r"^audio_tower.layers.(\d+).self_attn_layer_norm.(weight|bias)":                    r"mm_whisper_embeddings.whisper_encoder.transformer.layers.\1.attention_norm.\2",
    r"^audio_tower.layers.(\d+).fc1.(weight|bias)":                                     r"mm_whisper_embeddings.whisper_encoder.transformer.layers.\1.feed_forward.w1.\2",
    r"^audio_tower.layers.(\d+).fc2.(weight|bias)":                                     r"mm_whisper_embeddings.whisper_encoder.transformer.layers.\1.feed_forward.w2.\2",
    r"^audio_tower.layers.(\d+).final_layer_norm.(weight|bias)":                        r"mm_whisper_embeddings.whisper_encoder.transformer.layers.\1.ffn_norm.\2",
    r"^multi_modal_projector.linear_1.weight":                                          r"mm_whisper_embeddings.audio_language_projection.0.weight",
    r"^multi_modal_projector.linear_2.weight":                                          r"mm_whisper_embeddings.audio_language_projection.2.weight",
}
# fmt: on

# Keys that only exist on the HF side (computed at load time / registered as a persistent
# buffer) and were never part of the original Mistral consolidated checkpoint.
KEYS_TO_DROP = (
    "audio_tower.embed_positions.weight",
)

# Pattern identifying the language-model q/k projection weights, which need the RoPE
# permutation undone. Note this must NOT match `audio_tower...self_attn.q_proj...`, since
# Whisper's attention doesn't use RoPE and was never permuted in the forward conversion.
ROPE_PERMUTED_PATTERN = re.compile(r"^language_model\.model\.layers\.\d+\.self_attn\.(q|k)_proj\.weight$")


def map_new_key_to_old(new_key):
    """Map a key from the HF state dict back to its original Mistral key."""
    for pattern, replacement in REVERSE_STATE_DICT_MAPPING.items():
        old_key, n_replace = re.subn(pattern, replacement, new_key)
        if n_replace > 0:
            return old_key
    raise ValueError(f"Key: {new_key} could not be mapped (check REVERSE_STATE_DICT_MAPPING).")


def permute_for_rope(tensor, n_heads, dim1, dim2):
    """Permute the weights for the ROPE formulation. This is an involution: applying it
    twice returns the original tensor, so we reuse it to undo the forward conversion's
    permutation."""
    tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(dim1, dim2)
    return tensor


def load_sharded_state_dict(input_dir):
    """Load all tensors from an HF checkpoint directory, whether sharded or single-file."""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    state_dict = {}

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in shard_files:
            print(f"Loading shard {shard_file}...")
            state_dict.update(load_file(os.path.join(input_dir, shard_file)))
    else:
        shard_files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files found in {input_dir}")
        for shard_file in shard_files:
            print(f"Loading {shard_file}...")
            state_dict.update(load_file(shard_file))

    return state_dict


def convert_state_dict(hf_state_dict, text_config):
    """Convert an HF Voxtral state dict back into Mistral's original key names/layout."""
    num_attention_heads = text_config.num_attention_heads
    hidden_size = text_config.hidden_size
    head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)
    num_key_value_heads = text_config.num_key_value_heads
    key_value_dim = head_dim * num_key_value_heads
    query_dim = head_dim * num_attention_heads

    original_state_dict = {}

    for new_key, tensor in hf_state_dict.items():
        if new_key in KEYS_TO_DROP:
            continue

        if ROPE_PERMUTED_PATTERN.match(new_key):
            if ".q_proj." in new_key:
                tensor = permute_for_rope(tensor, num_attention_heads, query_dim, hidden_size)
            else:  # k_proj
                tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim, hidden_size)
            # the forward conversion's view(...).reshape(...) before the permute is a no-op
            # in terms of memory layout, so nothing else needs to be undone here.

        old_key = map_new_key_to_old(new_key)
        original_state_dict[old_key] = tensor.contiguous()

    return original_state_dict


def write_consolidated_checkpoint(input_dir, output_path):
    print(f"Loading config from {input_dir}...")
    config = VoxtralConfig.from_pretrained(input_dir)

    print("Loading HF state dict...")
    hf_state_dict = load_sharded_state_dict(input_dir)

    # Handle tied embeddings: if lm_head wasn't saved separately (tied to embed_tokens),
    # materialize it explicitly since Mistral's format stores `output.weight` on its own.
    lm_head_key = "language_model.lm_head.weight"
    embed_key = "language_model.model.embed_tokens.weight"
    if lm_head_key not in hf_state_dict and embed_key in hf_state_dict:
        print("lm_head.weight not found separately; tying it to embed_tokens.weight.")
        hf_state_dict[lm_head_key] = hf_state_dict[embed_key]

    print("Converting keys and un-permuting RoPE weights...")
    original_state_dict = convert_state_dict(hf_state_dict, config.text_config)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving consolidated checkpoint to {output_path}...")
    save_file(original_state_dict, output_path, metadata={"format": "pt"})
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an HF-format Voxtral checkpoint back to Mistral's consolidated.safetensors format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the HF model shards (and config.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the resulting consolidated.safetensors file",
    )
    args = parser.parse_args()

    write_consolidated_checkpoint(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()