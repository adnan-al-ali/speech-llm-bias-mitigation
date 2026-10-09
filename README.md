# Bias Mitigation in Audio-Enabled LLMs through an Adversarial/Distillation Pipeline

## Setup

This project is managed using [Snakemake](https://snakemake.readthedocs.io/en/stable/). Prerequirements for running the Snakefile are:

- Filling out `adversarial/.env` using `adversarial/.env-template` and downloading the VCTK corpus using `aversarial/download_kaggle_vctk.sh`
- Download the audio files from [Google Drive](https://drive.google.com/drive/folders/1AwqwIt-28pXXApJ1eGRG43YC3OZUAE1F?usp=share_link) and extract them into `evaluation/gender_bias/audio_files`.
- Check out `Snakefile` and set variables on lines 9-13 to match your slurm cluster configuration.
- [Conda](https://www.anaconda.com/download/success) environment with `snakemake==9.20.0` an `snakemake-executor-plugin-slurm==2.6.1` installed.

After the setup, run `snakemake.sh` to execute all the experiments.