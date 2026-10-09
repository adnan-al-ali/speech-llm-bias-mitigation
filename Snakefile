# Snakefile

from const import MODELS, FULL_MODELS, TARGET_PITCH
from distill_gender_dataset.shared import PROMPTS as distill_gender_prompts

MMAU_DATASET="gamma-lab-umd/MMAU-test-mini"
# MMAU_DATASET="gamma-lab-umd/MMAU-test"

CPU_PARTITION="cpu-troja"
GPU_PARTITION="gpu-ms,gpu-troja"
GPU_CONSTRAINT="gpuram40G|gpuram48G"
GPU_CONSTRAINT48="gpuram48G"
GPU_CONSTRAINT95="gpuram95G"


NUM_GPUS_INFERENCE = {
    "Qwen/Qwen2-Audio-7B-Instruct": 1,
    "mistralai/Voxtral-Small-24B-2507": 2,
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": 1,
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": 1
}

NUM_GPUS_TRAINING = {
    "Qwen/Qwen2-Audio-7B-Instruct": 2,
    "mistralai/Voxtral-Small-24B-2507": 2,
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": 2,
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": 2,
}

GPU_CONSTRAINT_INFERENCE = {
    "Qwen/Qwen2-Audio-7B-Instruct": GPU_CONSTRAINT48,
    "mistralai/Voxtral-Small-24B-2507": GPU_CONSTRAINT48,
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": GPU_CONSTRAINT95,
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": GPU_CONSTRAINT95
}

GPU_CONSTRAINT_TRAINING = {
    "Qwen/Qwen2-Audio-7B-Instruct": GPU_CONSTRAINT48,
    "mistralai/Voxtral-Small-24B-2507": GPU_CONSTRAINT95,
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": GPU_CONSTRAINT95,
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": GPU_CONSTRAINT95
}

GRADIENT_REVERSAL_ALPHA = [0.025, 0.05, 0.1, 0.2] # [0.05, 0.1, 0.2, 0.5, 1.0]
LLM_LOSS_WEIGHT = [0.5, 1.0, 2.0] # [0.2, 0.5, 1.0]
LEARNING_RATE = [5e-6, 1e-5, 5e-5, 1e-4] # [5e-5, 1e-4, 5e-4] # [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

NUM_SAMPLES = 4000
BATCH_SIZE = 8
EPOCHS = 1
CHECKPOINTING_STEPS = 125
WEIGHT_DECAY = 0.005

DISTILL_CHECKPOINTING_STEPS = 80
DISTILL_DEV_SAMPLES = 0.05
DISTILL_NUM_SAMPLES = 1-DISTILL_DEV_SAMPLES
DISTILL_TOTAL_SAMPLES = 7000
DISTILL_EVAL_BATCH_SIZE = 8
DISTILL_LEARNING_RATES = [1e-6, 5e-6, 1e-5, 5e-5]
DISTILL_LORA_RANK = [16, 32]
DISTILL_LORA_DROPOUT = 0.05

def gres_gpu(num):
    return f"--gres 'gpu:{num}'"


rule all:
    input:
        expand(
            "evaluation/gender_bias/outputs/gender_bias_baseline_{model_short_name}.json",
            model_short_name=MODELS.keys()
        ),
        expand(
            "evaluation/gender_bias/outputs/gender_bias_best-checkpoint_{model_short_name}.json",
            model_short_name=MODELS.keys()
        ),
        expand("evaluation/gender_detection/outputs/{model_short_name}_normalized_{target_pitch}_accuracy.json", model_short_name=MODELS.keys(), target_pitch=TARGET_PITCH),
        expand("evaluation/mmau_vllm/results/{model_short_name}_{target_pitch}_accuracy.txt", model_short_name=MODELS.keys(), target_pitch=TARGET_PITCH),
        # lora checkpoint evaluation outputs
        [
            (
            "evaluation/gender_detection/outputs/"
            f"lora_"
            f"bs={BATCH_SIZE},cs={DISTILL_CHECKPOINTING_STEPS},ds={DISTILL_DEV_SAMPLES},"
            f"e={EPOCHS},ebs={DISTILL_EVAL_BATCH_SIZE},"
            f"lr={learning_rate},la={float(lora_rank*2)},ld={DISTILL_LORA_DROPOUT},"
            f"lr={lora_rank},mn={model_short_name},"
            f"ns={DISTILL_NUM_SAMPLES},umn=best-checkpoint_{model_short_name},"
            f"wd={WEIGHT_DECAY}"
            f"_checkpoint_checkpoint_{epoch}_{step}_accuracy.json"
            )
            for lora_rank in DISTILL_LORA_RANK
            for learning_rate in DISTILL_LEARNING_RATES
            for model_short_name in MODELS.keys()
            for epoch in range(1, EPOCHS + 1)
            for step in range(DISTILL_CHECKPOINTING_STEPS, NUM_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        ],
        [
            (
            "evaluation/gender_bias/outputs/"
            f"gender_bias_lora_"
            f"bs={BATCH_SIZE},cs={DISTILL_CHECKPOINTING_STEPS},ds={DISTILL_DEV_SAMPLES},"
            f"e={EPOCHS},ebs={DISTILL_EVAL_BATCH_SIZE},"
            f"lr={learning_rate},la={float(lora_rank*2)},ld={DISTILL_LORA_DROPOUT},"
            f"lr={lora_rank},mn={model_short_name},"
            f"ns={DISTILL_NUM_SAMPLES},umn=best-checkpoint_{model_short_name},"
            f"wd={WEIGHT_DECAY}"
            f"_checkpoint_checkpoint_{epoch}_{step}.json"
            )
            for lora_rank in DISTILL_LORA_RANK
            for learning_rate in DISTILL_LEARNING_RATES
            for model_short_name in MODELS.keys()
            for epoch in range(1, EPOCHS + 1)
            for step in range(DISTILL_CHECKPOINTING_STEPS, NUM_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        ],
        [
            (
            "evaluation/mmau_vllm/results/"
            f"lora_"
            f"bs={BATCH_SIZE},cs={DISTILL_CHECKPOINTING_STEPS},ds={DISTILL_DEV_SAMPLES},"
            f"e={EPOCHS},ebs={DISTILL_EVAL_BATCH_SIZE},"
            f"lr={learning_rate},la={float(lora_rank*2)},ld={DISTILL_LORA_DROPOUT},"
            f"lr={lora_rank},mn={model_short_name},"
            f"ns={DISTILL_NUM_SAMPLES},umn=best-checkpoint_{model_short_name},"
            f"wd={WEIGHT_DECAY}"
            f"_checkpoint_checkpoint_{epoch}_{step}_accuracy.txt"
            )
            for lora_rank in DISTILL_LORA_RANK
            for learning_rate in DISTILL_LEARNING_RATES
            for model_short_name in MODELS.keys()
            for epoch in range(1, EPOCHS + 1)
            for step in range(DISTILL_CHECKPOINTING_STEPS, NUM_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        ],


rule evaluate_gender_bias:
    input:
        "evaluation/model_cache/best-checkpoint_{model_short_name}"
    output:
        "evaluation/gender_bias/outputs/gender_bias_best-checkpoint_{model_short_name}.json"
    conda:
        "envs/vllm.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    priority: 10
    params:
        model_name=lambda wildcards: f"evaluation/model_cache/best-checkpoint_{wildcards.model_short_name}",
    script:
        "evaluation/gender_bias/gender_bias_inference_adjectives.py"

rule evaluate_gender_bias_baseline:
    output:
        "evaluation/gender_bias/outputs/gender_bias_baseline_{model_short_name}.json"
    conda:
        "envs/vllm.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    priority: 10
    params:
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
    script:
        "evaluation/gender_bias/gender_bias_inference_adjectives.py"


rule evaluate_distill_checkpoints:
    input:
        checkpoint_paths=expand((
            "distilling/logs/"
            "bs={{batch_size}},"
            "cs={{checkpointing_steps}},"
            "ds={{dev_samples}},"
            "e={{epochs}},"
            "ebs={{eval_batch_size}},"
            "lr={{learning_rate}},"
            "la={{lora_alpha}},"
            "ld={{lora_dropout}},"
            "lr={{lora_r}},"
            "mn={{model_short_name}},"
            "ns={{num_samples}},"
            "umn=best-checkpoint_{{model_short_name}},"
            "wd={{weight_decay}}"
            "/checkpoint/checkpoint_{epoch}_{step}"),
            epoch=range(1, EPOCHS + 1),
            step=range(DISTILL_CHECKPOINTING_STEPS, DISTILL_TOTAL_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        )
    output:
        expand((
            "evaluation/gender_detection/outputs/"
            "lora_"
            "bs={{batch_size}},"
            "cs={{checkpointing_steps}},"
            "ds={{dev_samples}},"
            "e={{epochs}},"
            "ebs={{eval_batch_size}},"
            "lr={{learning_rate}},"
            "la={{lora_alpha}},"
            "ld={{lora_dropout}},"
            "lr={{lora_r}},"
            "mn={{model_short_name}},"
            "ns={{num_samples}},"
            "umn=best-checkpoint_{{model_short_name}},"
            "wd={{weight_decay}}"
            "_checkpoint_checkpoint_{epoch}_{step}_accuracy.json"),
            epoch=range(1, EPOCHS + 1),
            step=range(DISTILL_CHECKPOINTING_STEPS, DISTILL_TOTAL_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        ),
        expand((
            "evaluation/mmau_vllm/results/"
            "lora_"
            "bs={{batch_size}},"
            "cs={{checkpointing_steps}},"
            "ds={{dev_samples}},"
            "e={{epochs}},"
            "ebs={{eval_batch_size}},"
            "lr={{learning_rate}},"
            "la={{lora_alpha}},"
            "ld={{lora_dropout}},"
            "lr={{lora_r}},"
            "mn={{model_short_name}},"
            "ns={{num_samples}},"
            "umn=best-checkpoint_{{model_short_name}},"
            "wd={{weight_decay}}"
            "_checkpoint_checkpoint_{epoch}_{step}_accuracy.txt"),
            epoch=range(1, EPOCHS + 1),
            step=range(DISTILL_CHECKPOINTING_STEPS, DISTILL_TOTAL_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        ),
        expand((
            "evaluation/gender_bias/outputs/"
            "gender_bias_"
            "lora_"
            "bs={{batch_size}},"
            "cs={{checkpointing_steps}},"
            "ds={{dev_samples}},"
            "e={{epochs}},"
            "ebs={{eval_batch_size}},"
            "lr={{learning_rate}},"
            "la={{lora_alpha}},"
            "ld={{lora_dropout}},"
            "lr={{lora_r}},"
            "mn={{model_short_name}},"
            "ns={{num_samples}},"
            "umn=best-checkpoint_{{model_short_name}},"
            "wd={{weight_decay}}"
            "_checkpoint_checkpoint_{epoch}_{step}.json"),
            epoch=range(1, EPOCHS + 1),
            step=range(DISTILL_CHECKPOINTING_STEPS, DISTILL_TOTAL_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        )
    params:
        checkpoint_path = lambda wildcards: MODELS[wildcards.model_short_name],
        lora_checkpoint_paths = input.checkpoint_paths,
        evaluate_bias = True
    conda:
        "envs/vllm-latest.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    script:
        "evaluation/evaluate_checkpoint.py"

rule train_distill:
    input:
        "distilling/logits/best-checkpoint_{model_short_name}/gendered/logits",
        "distilling/logits/{model_short_name}/sift-50m-custom/logits"
    output:
        [directory(o) for o in expand((
            "distilling/logs/"
            "bs={{batch_size}},"
            "cs={{checkpointing_steps}},"
            "ds={{dev_samples}},"
            "e={{epochs}},"
            "ebs={{eval_batch_size}},"
            "lr={{learning_rate}},"
            "la={{lora_alpha}},"
            "ld={{lora_dropout}},"
            "lr={{lora_r}},"
            "mn={{model_short_name}},"
            "ns={{num_samples}},"
            "umn=best-checkpoint_{{model_short_name}},"
            "wd={{weight_decay}}"
            "/checkpoint/checkpoint_{epoch}_{step}"),
            epoch=range(1, EPOCHS + 1),
            step=range(DISTILL_CHECKPOINTING_STEPS, DISTILL_TOTAL_SAMPLES // BATCH_SIZE + 1, DISTILL_CHECKPOINTING_STEPS)
        )]
    conda:
        "envs/adversarial.yaml"
    priority: 5
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_TRAINING[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_TRAINING[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_TRAINING[MODELS[wildcards.model_short_name]]
    params:
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
        ungendered_model_name=lambda wildcards: f"best-checkpoint_{wildcards.model_short_name}",
        batch_size=lambda wildcards: int(wildcards.batch_size),
        checkpointing_steps=lambda wildcards: int(wildcards.checkpointing_steps),
        epochs=lambda wildcards: int(wildcards.epochs),
        eval_batch_size=lambda wildcards: int(wildcards.eval_batch_size),
        learning_rate=lambda wildcards: float(wildcards.learning_rate),
        lora_alpha=lambda wildcards: float(wildcards.lora_alpha),
        lora_dropout=lambda wildcards: float(wildcards.lora_dropout),
        lora_r=lambda wildcards: int(wildcards.lora_r),
        num_samples=lambda wildcards: float(wildcards.num_samples),
        dev_samples=lambda wildcards: float(wildcards.dev_samples),
        weight_decay=lambda wildcards: float(wildcards.weight_decay)
    script:
        "distilling/distill_fp8.py"

rule extract_logits_sift50m:
    input:
        "distilling/datasets/sift_50m_{model_short_name}.jsonl"
    output:
        directory("distilling/logits/{model_short_name}/sift-50m-custom/logits")
    conda:
        "envs/adversarial.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT48 if wildcards.model_short_name == "Qwen3-Omni-30B-A3B-Instruct" else GPU_CONSTRAINT,
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    params:
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
        dataset="sift-50m-custom",
        max_samples=8000,
        gendered_source_model=None
    script:
        "distilling/logit_extraction.py"


rule extract_logits_ungendered:
    input:
        expand("distill_gender_dataset/datasets/best-checkpoint_{{model_short_name}}_{prompt_key}.jsonl", prompt_key=distill_gender_prompts.keys())
    output:
        directory("distilling/logits/best-checkpoint_{model_short_name}/gendered/logits")
    conda:
        "envs/adversarial.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    params:
        model_name=lambda wildcards: f"evaluation/model_cache/best-checkpoint_{wildcards.model_short_name}",
        dataset="gendered",
        gendered_source_model=lambda wildcards: f"best-checkpoint_{wildcards.model_short_name}",
        max_samples=8000,
    script:
        "distilling/logit_extraction.py"


rule create_ungendered_dataset:
    input:
        "evaluation/model_cache/best-checkpoint_{model_short_name}"
    output:
        expand("distill_gender_dataset/datasets/best-checkpoint_{{model_short_name}}_{prompt_key}.jsonl", prompt_key=distill_gender_prompts.keys())
    conda:
        "envs/vllm.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    params:
        model_name=lambda wildcards: f"evaluation/model_cache/best-checkpoint_{wildcards.model_short_name}",
        num_samples_per_prompt=240,
        temperature=0.7,
        max_samples=8000
    script:
        "distill_gender_dataset/generate_dataset.py"

rule create_custom_sift50m:
    output:
        "distilling/datasets/sift_50m_{model_short_name}.jsonl"
    conda:
        "envs/vllm.yaml"
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    params:
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
        max_samples=8000
    script:
        "distilling/chat_dataset_generation.py"


rule find_best_ungendered_checkpoint:
    input:
        expand(
            (
                "evaluation/gender_detection/outputs/"
                f"bs={BATCH_SIZE},cs={CHECKPOINTING_STEPS},ds=180,e={EPOCHS},ebs=24,"
                "gra={gradient_reversal_alpha},"
                "kf=False,ls=0.05,"
                "lr={learning_rate},"
                "llw={llm_loss_weight},"
                "mn={{model_short_name}},"
                f"ns={NUM_SAMPLES},wd=0.005"
                "_checkpoint_checkpoint_{epoch}_{step}_accuracy.json"
            ),
            gradient_reversal_alpha=GRADIENT_REVERSAL_ALPHA,
            learning_rate=LEARNING_RATE,
            llm_loss_weight=LLM_LOSS_WEIGHT,
            epoch=range(1, EPOCHS + 1),
            step=range(CHECKPOINTING_STEPS, NUM_SAMPLES // BATCH_SIZE + 1, CHECKPOINTING_STEPS)
        )
    output:
        directory("evaluation/model_cache/best-checkpoint_{model_short_name}")
    priority: 100
    conda:
        "envs/adversarial.yaml"
    resources:
        slurm_partition=CPU_PARTITION,
        nodes=1,
        mem_mb=100000
    params:
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
    script:
        "evaluation/find_best_checkpoint.py"

rule evaluate_pitch_normalization:
    output:
        (
            "evaluation/gender_detection/outputs/"
            f"{{model_short_name}}_normalized_{TARGET_PITCH}_accuracy.json"
        ),
        (
            "evaluation/mmau_vllm/results/"
            f"{{model_short_name}}_{TARGET_PITCH}_accuracy.txt"
        ),
        
    params:
        checkpoint_path = lambda wildcards: MODELS[wildcards.model_short_name],
        target_pitch = TARGET_PITCH,
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    priority: 2
    conda:
        "envs/vllm.yaml"
    script:
        "evaluation/evaluate_checkpoint.py"

rule evaluate_checkpoint:
    input:
        checkpoint_path = (
            "evaluation/model_cache/"
            "bs={batch_size},"
            "cs={checkpointing_steps},"
            "ds={dev_samples},"
            "e={epochs},"
            "ebs={eval_batch_size},"
            "gra={gradient_reversal_alpha},"
            "kf={keep_frozen},"
            "ls={label_smoothing},"
            "lr={learning_rate},"
            "llw={llm_loss_weight},"
            "mn={model_short_name},"
            "ns={num_samples},"
            "wd={weight_decay}"
            "/checkpoint/checkpoint_{epoch}_{step}"
        )
    output:
        (
            "evaluation/gender_detection/outputs/"
            "bs={batch_size},"
            "cs={checkpointing_steps},"
            "ds={dev_samples},"
            "e={epochs},"
            "ebs={eval_batch_size},"
            "gra={gradient_reversal_alpha},"
            "kf={keep_frozen},"
            "ls={label_smoothing},"
            "lr={learning_rate},"
            "llw={llm_loss_weight},"
            "mn={model_short_name},"
            "ns={num_samples},"
            "wd={weight_decay}"
            "_checkpoint_checkpoint_{epoch}_{step}_accuracy.json"
        ),
        (
            "evaluation/mmau_vllm/results/"
            "bs={batch_size},"
            "cs={checkpointing_steps},"
            "ds={dev_samples},"
            "e={epochs},"
            "ebs={eval_batch_size},"
            "gra={gradient_reversal_alpha},"
            "kf={keep_frozen},"
            "ls={label_smoothing},"
            "lr={learning_rate},"
            "llw={llm_loss_weight},"
            "mn={model_short_name},"
            "ns={num_samples},"
            "wd={weight_decay}"
            "_checkpoint_checkpoint_{epoch}_{step}_accuracy.txt"
        ),
        
    params:
        checkpoint_path = input.checkpoint_path,
        cleanup = True
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_INFERENCE[MODELS[wildcards.model_short_name]]
    priority: 2
    conda:
        "envs/vllm.yaml"
    script:
        "evaluation/evaluate_checkpoint.py"

rule convert_checkpoint:
    input:
        saved_encoder_path = (
            "adversarial/logs/"
            "bs={batch_size},"
            "cs={checkpointing_steps},"
            "ds={dev_samples},"
            "e={epochs},"
            "ebs={eval_batch_size},"
            "gra={gradient_reversal_alpha},"
            "kf={keep_frozen},"
            "ls={label_smoothing},"
            "lr={learning_rate},"
            "llw={llm_loss_weight},"
            "mn={model_short_name},"
            "ns={num_samples},"
            "wd={weight_decay}"
            "/checkpoint/checkpoint_{epoch}_{step}.pt"
            )
    output:
        temp(directory(("evaluation/model_cache/"
            "bs={batch_size},"
            "cs={checkpointing_steps},"
            "ds={dev_samples},"
            "e={epochs},"
            "ebs={eval_batch_size},"
            "gra={gradient_reversal_alpha},"
            "kf={keep_frozen},"
            "ls={label_smoothing},"
            "lr={learning_rate},"
            "llw={llm_loss_weight},"
            "mn={model_short_name},"
            "ns={num_samples},"
            "wd={weight_decay}"
            "/checkpoint/checkpoint_{epoch}_{step}"
        )))
    params:
        saved_encoder_path=input.saved_encoder_path,
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
        max_concurrent_checkpoints=6
    conda:
        "envs/adversarial.yaml"
    priority: 1
    resources:
        slurm_partition=CPU_PARTITION,
        nodes=1,
        mem_mb=100000,
        checkpoint_converter_slots=1
    script:
        "evaluation/convert_to_hf.py"

rule train_adversarial:
    output:
        expand((
            "adversarial/logs/"
            "bs={{batch_size}},"
            "cs={{checkpointing_steps}},"
            "ds={{dev_samples}},"
            "e={{epochs}},"
            "ebs={{eval_batch_size}},"
            "gra={{gradient_reversal_alpha}},"
            "kf={{keep_frozen}},"
            "ls={{label_smoothing}},"
            "lr={{learning_rate}},"
            "llw={{llm_loss_weight}},"
            "mn={{model_short_name}},"
            "ns={{num_samples}},"
            "wd={{weight_decay}}"
            "/checkpoint/checkpoint_{epoch}_{step}.pt"),
            epoch=range(1, EPOCHS + 1),
            step=range(CHECKPOINTING_STEPS, NUM_SAMPLES // BATCH_SIZE + 1, CHECKPOINTING_STEPS)
        )
    params:
        model_name=lambda wildcards: MODELS[wildcards.model_short_name],
        batch_size=lambda wildcards: int(wildcards.batch_size),
        checkpointing_steps=lambda wildcards: int(wildcards.checkpointing_steps),
        epochs=lambda wildcards: int(wildcards.epochs),
        eval_batch_size=lambda wildcards: int(wildcards.eval_batch_size),
        gradient_reversal_alpha=lambda wildcards: float(wildcards.gradient_reversal_alpha),
        keep_frozen=lambda wildcards: wildcards.keep_frozen == "True",
        label_smoothing=lambda wildcards: float(wildcards.label_smoothing),
        learning_rate=lambda wildcards: float(wildcards.learning_rate),
        llm_loss_weight=lambda wildcards: float(wildcards.llm_loss_weight),
        num_samples=lambda wildcards: int(wildcards.num_samples),
        dev_samples=lambda wildcards: int(wildcards.dev_samples),
        weight_decay=lambda wildcards: float(wildcards.weight_decay)
    resources:
        slurm_partition=GPU_PARTITION,
        nodes=1,
        mem_mb=20000,
        constraint=lambda wildcards: GPU_CONSTRAINT_TRAINING[MODELS[wildcards.model_short_name]],
        gpu=lambda wildcards: NUM_GPUS_TRAINING[MODELS[wildcards.model_short_name]],
        gpu_use=lambda wildcards: NUM_GPUS_TRAINING[MODELS[wildcards.model_short_name]],
    priority: 0
    conda:
        "envs/adversarial.yaml"
    script:
        "adversarial/training.py"

# rule download_kaggle_vctk:
#     output:
#         ...
#     resources:
#         slurm_partition=CPU_PARTITION,
#         nodes=1,
#         mem_mb=50000
