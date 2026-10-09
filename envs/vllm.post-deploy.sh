uv pip install "vllm[audio]==0.21.0" --extra-index-url https://wheels.vllm.ai/0.21.0/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    "transformers>=5.0.1" \
    snakemake==9.20.0 \
    datasets==4.3.0 \
    torchcodec==0.16 \
    mistral_common[audio] \
    accelerate \
    numpy==2.2 \
    pandas \
    matplotlib \
    openai

# patch models to support LoRA loading
cp envs/qwen3_omni_moe_thinker_0.21.0.py ${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_omni_moe_thinker.py
cp envs/qwen2_audio_0.21.0.py ${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/model_executor/models/qwen2_audio.py
cp envs/nano_nemotron_vl_0.21.0.py ${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/model_executor/models/nano_nemotron_vl.py