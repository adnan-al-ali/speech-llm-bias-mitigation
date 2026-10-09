#!/bin/bash
#SBATCH --mem=4G

set -e

python3 -m snakemake --executor slurm \
    --jobs 20 \
    --latency-wait 1 \
    --keep-incomplete \
    --use-conda \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD \
    --resources gpu_use=24 checkpoint_converter_slots=8 \
    --slurm-exclude-failed-nodes tdll-8gpu1,dll-3gpu5,dll-4gpu4 \
    --default-resources tasks=0 runtime=2400 \
    --keep-going \
    --retries 2 \
    # --dry-run 
    # --touch \
    # --slurm-exclude-failed-nodes dll-4gpu4,dll-4gpu3 \
    # --default-resources tasks=0 slurm_extra='-q low' \