from collections import OrderedDict

def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]

FULL_MODELS = [
    "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    # "mistralai/Voxtral-Small-24B-2507",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
]

MODELS = OrderedDict((short_name(m), m) for m in FULL_MODELS)
TARGET_PITCH = 144