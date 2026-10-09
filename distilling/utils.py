from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from adversarial.filter_gendered_prompts import find_audio_path

def convert_sift50m_sample_to_chat_format(sample):
    chat_format = []
    for message in sample:
        role = message["role"]
        content = message["content"]
        content_converted = []
        for item in content:
            if item.get("text") is not None:
                content_converted.append({"type": "text", "text": item["text"]})
            elif item.get("audio_path") is not None:
                content_converted.append({
                    "type": "audio_url",
                    "audio_url": {"url": find_audio_path(item["audio_path"], as_uri=True)},
                    "text": None
                })
        chat_format.append({"role": role, "content": content_converted})
    return chat_format