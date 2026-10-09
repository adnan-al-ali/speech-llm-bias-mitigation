#!env bash

# Force the installation of transformers 5.0.0, which is required for the models, and is not officially supported by vLLM 0.19.0
python3 -m pip install "transformers==5.8.1"

# TODO: https://github.com/vllm-project/vllm/pull/42140/commits/b97be75455c48b039d59760e88477da0067dff8c