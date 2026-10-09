from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import argparse
import os
import datetime
import pickle
import re
import json
import numpy as np
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adversarial.trainable_module import TrainableModule
from distilling.logit_extraction import extract_audio_from_sample


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", type=str, help="Model name or path.") # Qwen/Qwen2-Audio-7B-Instruct
parser.add_argument("--ungendered_model_name", default="best_qwen3", type=str, help="Name of the model that was adversarially trained.")
parser.add_argument("--checkpointing_steps", default=250, type=int, help="Save checkpoint every N steps.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=5e-3, type=float, help="Weight decay in the optimizer")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--epochs_warmup", default=1, type=int, help="Number of warmup epochs.")
parser.add_argument("--lr_decay", default=True, type=bool, help="Whether to use learning rate decay.")
parser.add_argument("--lora_r", default=16, type=int, help="LoRA rank.")
parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA alpha.")
parser.add_argument("--lora_dropout", default=0.05, type=float, help="LoRA dropout.")
parser.add_argument("--num_samples", default=0.95, type=float, help="Number or proportion of training samples.")
parser.add_argument("--dev_samples", default=0.05, type=float, help="Number or proportion of development samples.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--eval_batch_size", default=24, type=int, help="Batch size during evaluation.")
args = parser.parse_args()


class TrainableAudioModel(TrainableModule):
    def __init__(self, model, device="cuda", lora_r = 16, lora_alpha = 32, lora_dropout = 0.05):
        super().__init__()
        self.model = model
        self.device = device
        self.peft_model = self.__get_peft_model(lora_r, lora_alpha, lora_dropout)

    def __get_peft_model(self, lora_r, lora_alpha, lora_dropout):
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            exclude_modules=".*audio_tower.*",
            task_type="CAUSAL_LM",
            bias="none",
        )

        for name, module in self.model.named_modules():
            if "lora" in name.lower() and "audio_tower" in name:
                raise AssertionError(f"LoRA modules should not be applied to audio_tower, but found in {name}")
            
        return get_peft_model(self.model, config)

    def forward(self,
            input_ids: torch.LongTensor | None = None,
            input_features: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            feature_attention_mask: torch.Tensor | None = None):
        return self.peft_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask
        )

    def save_weights(self, save_directory: str):
        if "." in save_directory:
            save_directory = save_directory.rsplit(".", 1)[0]
        self.peft_model.save_pretrained(save_directory)

class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, model_short: str, dataset_name: str):
        self.logits_dir = os.path.join("distilling", "logits", model_short, dataset_name, "logits")
        with open(os.path.join("distilling", "logits", model_short, dataset_name, "prompts.jsonl"), "r") as f:
            self.prompts = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int):
        return {"logits_path": os.path.join(self.logits_dir, f"sample_{idx}.pt"), "conversation": self.prompts[idx]}

    def concat(self, other_dataset):
        self.prompts.extend(other_dataset.prompts)
        return self

class DataCollatorWithAudio:
    def __init__(self, processor, device = "cuda"):
        self.processor = processor
        self.device = device
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, batch):
        audios = []
        for sample in batch:
            audios_sample = extract_audio_from_sample(sample["conversation"], self.processor)
            if audios_sample is not None:
                audios.extend(audios_sample)
        texts = [self.processor.apply_chat_template(sample["conversation"], add_generation_prompt=False, tokenize=False, continue_final_message=True) for sample in batch]
        
        inputs = self.processor(
            text=list(texts),
            audio=list(audios),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device, torch.bfloat16)

        assistant_logits = [torch.load(sample["logits_path"]) for sample in batch]
        assistant_masks = [torch.ones(logit.shape[:-1], dtype=bool) for logit in assistant_logits]

        # for setting the matching shape of the logits and masks
        dummy_row_logits = torch.zeros((inputs.input_ids.shape[1], assistant_logits[0].shape[-1]), dtype=torch.float32)
        dummy_row_mask = torch.zeros((inputs.input_ids.shape[1],), dtype=bool)

        target_logits = torch.nn.utils.rnn.pad_sequence(
            [dummy_row_logits] + assistant_logits, 
            batch_first=True, 
            padding_value=0.0, 
            padding_side="left"
            )[1:].to(device=self.device, dtype=torch.bfloat16)
        target_logits_debug = target_logits.to(device="cpu", dtype=torch.float32) # TODO: remove this line after debugging
        target_mask = torch.nn.utils.rnn.pad_sequence(
            [dummy_row_mask] + assistant_masks, 
            batch_first=True, 
            padding_value=False, 
            padding_side="left"
            )[1:].to(device=self.device)

        return ({
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
        },
        {
            "target_mask": target_mask,
            "target_logits": target_logits
        })

class MaskedKLLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        target_logprobs = torch.log_softmax(labels["target_logits"], dim=-1)
        target_logprobs_debug = target_logprobs.to(device="cpu", dtype=torch.float32) # TODO: remove this line after debugging
        target_mask = labels["target_mask"]
        predicted_logprobs = torch.log_softmax(predictions["logits"], dim=-1)
        predicted_logprobs_debug = predicted_logprobs.to(device="cpu", dtype=torch.float32) # TODO: remove this line after debugging

        raw_loss = F.kl_div(predicted_logprobs, target_logprobs, reduction='none', log_target=True)
        token_loss = raw_loss.sum(dim=-1)
        token_loss_debug = token_loss.to(device="cpu", dtype=torch.float32) # TODO: remove this line after debugging
        masked_loss = token_loss * target_mask
        masked_loss_debug = masked_loss.to(device="cpu", dtype=torch.float32) # TODO: remove this line after debugging
        final_loss = masked_loss.sum() / target_mask.sum()
        return final_loss

class KLMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.metric = MaskedKLLoss()
        self.sum = 0.
        self.count = 0
    def update(self, preds, labels):
        self.sum += self.metric(preds, labels)
        self.count += 1
    def compute(self):
        avg_loss = self.sum / self.count if self.count > 0 else 0.0
        self.sum = 0.
        self.count = 0
        return avg_loss

class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, args, train):
        self._warmup_steps = args.epochs_warmup * len(train)
        self._decay_steps = args.lr_decay and (args.epochs - args.epochs_warmup) * len(train)
        super().__init__(optimizer, self.current_learning_rate)

    def current_learning_rate(self, step):
        assert step >= 0 and (not self._decay_steps or step <= self._warmup_steps + self._decay_steps)
        return step / self._warmup_steps if step < self._warmup_steps else \
            0.5 * (1 + np.cos(np.pi * (step - self._warmup_steps) / self._decay_steps)) if self._decay_steps else 1

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    SEED = 41
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

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
    args.logdir = os.path.join("distill", "logs", dir_name)
    print(f"Logging to {args.logdir}")

    processor = AutoProcessor.from_pretrained(
        full_model_name,
        # device_map="auto",
        # attn_implementation="flash_attention_2",
        # dtype=torch.bfloat16,
        )

    if full_model_name == "Qwen/Qwen2-Audio-7B-Instruct":
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            full_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        peft_model = TrainableAudioModel(
            model, 
            device=model.device, 
            lora_r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout)
        lm_head = model.language_model.lm_head
    elif full_model_name == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            full_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).thinker
        peft_model = TrainableAudioModel(
            model, 
            device=model.device, 
            lora_r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout)
        lm_head = model.lm_head

    
    collator = DataCollatorWithAudio(processor, lm_head.weight.device)

    # with open("distilling/logits/sample_logits.pkl", "rb") as f:
    #     sample_logits = pickle.load(f)

    sift_dataset = DistillDataset(args.model_name.split("/")[-1], "sift-50m")
    gendered_dataset = DistillDataset(args.ungendered_model_name, "gendered")
    dataset = sift_dataset.concat(gendered_dataset)
    train_ds, dev_ds = torch.utils.data.random_split(dataset, [args.num_samples, args.dev_samples], generator=torch.Generator().manual_seed(SEED))
    # train_ds = DistillDataset(sample_logits)
    # dev_ds = DistillDataset(sample_logits)

    train = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collator)
    dev = torch.utils.data.DataLoader(dev_ds, args.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-5)
    schedule = CosineWithWarmup(optimizer, args, train)

    peft_model.configure(
        optimizer=optimizer,
        loss=MaskedKLLoss(),
        schedule=schedule,
        metrics={
            "loss": KLMetric(),
        },
        logdir=args.logdir,
    )
    try:
        peft_model.fit(train, epochs=args.epochs, dev=dev, logging_steps=50, save_steps=args.checkpointing_steps)
    except KeyboardInterrupt:
        pass

    os.makedirs(os.path.join(args.logdir, "finetuned_encoder"), exist_ok=True)
    peft_model.save_weights(os.path.join(args.logdir, "finetuned_encoder", "encoder.pt"))
    

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import Snakemake
        snakemake: Snakemake
        args = SimpleNamespace(**snakemake.params)
        main(args)
    else:
        args = parser.parse_args([] if "__file__" not in globals() else None)
        main(args)