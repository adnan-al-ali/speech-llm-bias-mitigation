# with the help of the tutorial https://www.youtube.com/watch?v=Be8BM3wNNc0&

from transformers import Qwen2AudioForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration, VoxtralForConditionalGeneration, AutoProcessor
from nemotron.modeling import NemotronH_Nano_Omni_Reasoning_V3
from nemotron.processing import NemotronH_Nano_Omni_Reasoning_V3Processor
import torch
import librosa
from datasets import load_dataset, concatenate_datasets
import argparse
import numpy as np
import os
import datetime
import re
import torchmetrics
from collections import namedtuple
from types import SimpleNamespace


from filter_gendered_prompts import find_audio_path, get_gender_dict, get_speaker_gender
from modeling import QwenWithReversedGenderHead, Qwen3OmniWithReversedGenderHead, VoxtralWithReversedGenderHead, NemotronWithReversedGenderHead
# logging.set_verbosity_error()  # to suppress some of the warnings from the transformers library

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen/Qwen2-Audio-7B-Instruct", type=str, help="Model name or path.")
parser.add_argument("--checkpointing_steps", default=250, type=int, help="Save checkpoint every N steps.")
parser.add_argument("--num_samples", default=8000, type=int, help="Number of training samples.")
parser.add_argument("--dev_samples", default=180, type=int, help="Number of development samples.")
parser.add_argument("--gradient_reversal_alpha", default=1.0, type=float, help="Alpha for the gradient reversal layer.")
parser.add_argument("--llm_loss_weight", default=0.5, type=float, help="Weight for the LLM loss in the combined loss function.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--eval_batch_size", default=24, type=int, help="Batch size during evaluation.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0.05, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--keep_frozen", default=False, type=bool, help="Keep the encoder frozen?")
parser.add_argument("--weight_decay", default=5e-3, type=float, help="Weight decay in the optimizer")
args = parser.parse_args()



def process_sample(sample: list[dict], processor, speaker_gender_dict, apply_processing: bool = True, device = "cuda"):
    is_voxtral = "voxtral" in type(processor).__name__.lower()
    is_nemotron = "nemotron" in type(processor).__name__.lower()
    audios = []
    for message in sample:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["audio_path"] is not None:
                    # Fix the format
                    item["audio_url"] = {"url": find_audio_path(item["audio_path"])}
                    item["type"] = "audio_url"

                    # Load audio and resample
                    sr = int(processor.audio_sampling_rate if hasattr(processor, 'audio_sampling_rate') else processor.feature_extractor.sampling_rate)
                    audio_data, _ = librosa.load(find_audio_path(item["audio_path"], as_uri=False), sr=sr)
                    audios.append(audio_data)
                    speaker = item["audio_path"].split("/")[-1].split("_")[0]
                    gender = speaker_gender_dict[speaker[1:]] == "F"

                elif item["text"] is not None:
                    item["type"] = "text"
    
    assert len(audios) == 1, "Expected exactly one audio clip per sample."
    if is_voxtral:
        if not apply_processing:
            return sample, gender
        else:
            return processor.apply_chat_template(sample).to(device, torch.bfloat16), gender
    elif is_nemotron:
        sample = processor.preprocess_chat(sample)
    try:
        chat_applied = processor.apply_chat_template(
            sample, 
            tokenize=False, 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {}))
    except Exception:
        chat_applied = processor.tokenizer.apply_chat_template(
            sample, 
            tokenize=False, 
            **({"enable_thinking": False} if "nemotron" in type(processor).__name__.lower() else {}))

    if not apply_processing:
        return chat_applied, audios[0], gender

    inputs = processor(
        text=chat_applied,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,
        sampling_rate=sr
    ).to(device, torch.bfloat16)

    return inputs, gender

CombinedLabel = namedtuple("CombinedOutput", ["label_llm", "label_gender"])

class DataCollatorWithAudio:
    def __init__(self, processor, speaker_gender_dict, device = "cuda"):
        self.processor = processor
        self.speaker_gender_dict = speaker_gender_dict
        self.device = device
        self.assistant_start_tokens = self.__get_assistant_start_tokens()
        self.is_voxtral = "voxtral" in type(processor).__name__.lower()

    def __get_assistant_start_tokens(self):
        convo = [
            {"role": "user", "content": "XXX"},
            {"role": "assistant", "content": "XXX"},
        ]
        try:
            chat_applied = self.processor.apply_chat_template(convo, tokenize=False, **({"enable_thinking": False} if "nemotron" in type(self.processor).__name__.lower() else {}))
        except Exception:
            # Sometimes, the apply_chat_template method is directly on the processor
            chat_applied = self.processor.tokenizer.apply_chat_template(convo, tokenize=False, **({"enable_thinking": False} if "nemotron" in type(self.processor).__name__.lower() else {}))
        start_seq = chat_applied.split("XXX")[1]
        return self.processor.tokenizer.encode(start_seq, add_special_tokens=False, return_tensors="pt").reshape((-1,)).to(self.device)

    def __call__(self, batch):
        processed_batch = [process_sample(sample["messages"], self.processor, self.speaker_gender_dict, apply_processing=False, device=self.device) for sample in batch]

        if self.is_voxtral:
            texts, genders = zip(*processed_batch)
            inputs = self.processor.apply_chat_template(list(texts), **({"enable_thinking": False} if "nemotron" in type(self.processor).__name__.lower() else {})).to(self.device, torch.bfloat16)
            # print("Voxtral inputs:", self.processor.tokenizer.decode(inputs["input_ids"][0]))
        else:
            texts, audios, genders = zip(*processed_batch)
            
            inputs = self.processor(
                text=list(texts),
                audio=list(audios),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device, torch.bfloat16)

        labels_llm = inputs.input_ids.clone()
        labels_gender = torch.tensor(genders, dtype=torch.bfloat16, device=self.device).unsqueeze(1)

        # mask labels from the user
        # assistant_start_tokens = self.processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False, return_tensors="pt").reshape((-1,)).to(self.device)
        assistant_start_tokens = self.assistant_start_tokens
        for row in labels_llm:
            match_index = (row.unfold(0, len(assistant_start_tokens), 1) == assistant_start_tokens).all(dim=-1).int().argmax()
            row[:match_index + len(assistant_start_tokens)] = -100
        
        return ({
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features if hasattr(inputs, "input_features") else inputs.sound_clips,
            "feature_attention_mask": inputs.get("feature_attention_mask", None),
            # "labels_llm": labels_llm,
            # "labels_gender": labels_gender
        },
        CombinedLabel(label_llm=labels_llm, label_gender=labels_gender))

class CombinedLoss(torch.nn.modules.loss._Loss):
    def __init__(self, llm_loss_weight=1.0, gender_loss_weight=1.0, label_smoothing=0.0, ignore_index=-100):
        super().__init__()
        self.llm_loss_weight = llm_loss_weight
        self.gender_loss_weight = gender_loss_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.llm_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)
        self.gender_loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions, labels):
        preds_llm, preds_gender = predictions
        labels_llm, labels_gender = labels
        loss_llm = self.llm_loss_fn(preds_llm.to(labels_llm.device)[:, :, :-1], labels_llm[:, 1:])
        loss_gender = self.gender_loss_fn(preds_gender.to(labels_gender.device), labels_gender)
        return self.llm_loss_weight * loss_llm + self.gender_loss_weight * loss_gender

class GenderAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.metric = torchmetrics.Accuracy(task="binary").to("cuda")
    def update(self, preds, labels):
        self.metric.update(preds[1].to(labels[1].device), labels[1])
    def compute(self):
        return self.metric.compute()

class LLMLoss(torchmetrics.Metric):
    def __init__(self, label_smoothing=0.0, ignore_index=-100):
        super().__init__()
        self.metric = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)
        self.sum = 0.0
        self.count = 0
    def update(self, preds, labels):
        self.sum += self.metric(preds[0][:, :, :-1].to(labels[0].device), labels[0][:, 1:]).sum()
        self.count += preds[0].shape[0]
    def compute(self):
        avg_loss = self.sum / self.count if self.count > 0 else 0.0
        self.sum = 0.0
        self.count = 0
        return avg_loss

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    SEED = 41
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    print("Available CUDA devices:", torch.cuda.device_count())

    full_model_name = args.model_name
    args.model_name = full_model_name.split("/")[-1]
    # Create logdir name
    if "snakemake" not in globals():
        dir_name = "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        ).replace("/", r"_")
    else:
        dir_name = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()))).replace("/", r"_")
    args.logdir = os.path.join("adversarial", "logs", dir_name)
    print(f"Logging to {args.logdir}")

    if "nemotron" not in full_model_name.lower():
        processor = AutoProcessor.from_pretrained(
            full_model_name,
            mode="finetuning", # necessary for voxtral but not sure if it breaks anything for qwen2/qwen3
            )
    else:
        processor = NemotronH_Nano_Omni_Reasoning_V3Processor.from_pretrained(
            full_model_name,
            trust_remote_code=True,
            )

    gradient_accumulation_steps = 1
    if full_model_name == "Qwen/Qwen2-Audio-7B-Instruct":
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            full_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        model_with_head = QwenWithReversedGenderHead(model, gradient_reversal_alpha=args.gradient_reversal_alpha)
    elif full_model_name == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            full_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        model_with_head = Qwen3OmniWithReversedGenderHead(model.thinker, gradient_reversal_alpha=args.gradient_reversal_alpha)
    elif "voxtral" in full_model_name.lower():
        model = VoxtralForConditionalGeneration.from_pretrained(
            full_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        model_with_head = VoxtralWithReversedGenderHead(model, gradient_reversal_alpha=args.gradient_reversal_alpha)
        args.batch_size = args.batch_size // 2  # Voxtral is more memory intensive, so we reduce the batch size
        gradient_accumulation_steps = 2  # and increase the gradient accumulation steps to keep the effective batch size the same
    elif "nemotron" in full_model_name.lower():
        model = NemotronH_Nano_Omni_Reasoning_V3.from_pretrained(
            full_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model_with_head = NemotronWithReversedGenderHead(model, gradient_reversal_alpha=args.gradient_reversal_alpha)
        args.batch_size = args.batch_size // 2  # Nemotron is more memory intensive, so we reduce the batch size
        gradient_accumulation_steps = 2  # and increase the gradient accumulation steps to keep the effective batch size the same
    speaker_gender_dict = get_gender_dict()

    collator = DataCollatorWithAudio(processor, speaker_gender_dict, model.device)

    dataset = load_dataset("parquet", data_files="adversarial/downloads/sift_50m_vctk_en_train_ungendered_filtered.parquet")["train"].train_test_split(test_size=0.05, seed=SEED)

    train_f = dataset['train'].filter(lambda sample: get_speaker_gender(sample["messages"], speaker_gender_dict) == "F").select(range(min(args.num_samples // 2, len(dataset['train']) // 2)))
    train_m = dataset['train'].filter(lambda sample: get_speaker_gender(sample["messages"], speaker_gender_dict) == "M").select(range(min(args.num_samples // 2, len(dataset['train']) // 2)))

    # print avg response length
    shared_dataset = concatenate_datasets([train_f, train_m])
    lengths = [len(sample["messages"][-1]["content"][0]["text"]) for sample in shared_dataset]
    print(f"Average response length (train): {sum(lengths) / len(lengths):.2f}")

    train = torch.utils.data.DataLoader(shared_dataset.shuffle(seed=SEED), args.batch_size, shuffle=True, collate_fn=collator)

    dev_f = dataset['test'].filter(lambda sample: get_speaker_gender(sample["messages"], speaker_gender_dict) == "F").select(range(min(args.dev_samples // 2, len(dataset['test']) // 2)))
    dev_m = dataset['test'].filter(lambda sample: get_speaker_gender(sample["messages"], speaker_gender_dict) == "M").select(range(min(args.dev_samples // 2, len(dataset['test']) // 2)))
    dev = torch.utils.data.DataLoader(concatenate_datasets([dev_f, dev_m]).shuffle(seed=SEED), args.eval_batch_size, shuffle=False, collate_fn=collator)

    if not args.keep_frozen:
        model_with_head.unfreeze_encoder_weights()
    optimizer = torch.optim.AdamW(model_with_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-5)

    model_with_head.configure(
        optimizer=optimizer,
        loss=CombinedLoss(label_smoothing=args.label_smoothing, llm_loss_weight=args.llm_loss_weight),
        # schedule=schedule,
        metrics={
            "acc_gender": GenderAccuracy(),
            "loss_llm": LLMLoss(label_smoothing=args.label_smoothing)
        },
        logdir=args.logdir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_dataloader=train
    )
    try:
        model_with_head.fit(train, epochs=args.epochs, dev=dev, logging_steps=50, save_steps=args.checkpointing_steps)
    except KeyboardInterrupt:
        pass

    os.makedirs(os.path.join(args.logdir, "finetuned_encoder"), exist_ok=True)
    model_with_head.save_weights(os.path.join(args.logdir, "finetuned_encoder", "encoder.pt"))
    

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        args = SimpleNamespace(**snakemake.params)
        main(args)
    else:
        args = parser.parse_args([] if "__file__" not in globals() else None)
        main(args)