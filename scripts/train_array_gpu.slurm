#!/bin/bash
#SBATCH --job-name=aurora_sac_sweep
#SBATCH --output=result_%A_%a.out
#SBATCH --error=result_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=GPUQ
#SBATCH --array=0-9

echo "Starting job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

module purge
module load Miniconda3/24.7.1-0

# Bypass conda activate (unreliable in non-interactive SLURM batch jobs).
# Use the full path to the target env's Python directly.
# NOTE: use python3.11 explicitly — the env's python symlink may resolve incorrectly.
PYTHON_ENV="/cluster/home/stephemo/.conda/envs/aurora-ferry-gpu/bin/python3.11"

if [ ! -f "$PYTHON_ENV" ]; then
    echo "FATAL: Python not found at expected path: $PYTHON_ENV"
    echo "Make sure aurora-ferry-gpu env is created and torch is installed."
    exit 2
fi

echo "Using Python: $PYTHON_ENV"
echo "Python version: $($PYTHON_ENV --version)"
nvidia-smi || true

echo "Running CUDA/PyTorch preflight check..."
$PYTHON_ENV - <<'PY'
import sys
import torch

print(f"torch.__version__={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print(
        "FATAL: CUDA not available to PyTorch. "
        "Most common cause is a CUDA version mismatch between torch build and cluster driver.",
        file=sys.stderr,
    )
    sys.exit(2)
PY

if [ $? -ne 0 ]; then
    echo "Aborting job because CUDA preflight failed."
    exit 2
fi

cd aurora_ferry

# Map array index to n_envs values to sweep.
# N_ENVS_LIST=(8 16 32)
# N_ENVS=${N_ENVS_LIST[$SLURM_ARRAY_TASK_ID]}

# Format: JSON string {"qf": [qf_neurons], "pi": [pi_neurons]} - Default is {"qf": [256, 256], "pi": [256, 256]}
NET_ARCH_LIST=(
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    '{"qf": [256, 256], "pi": [64, 32]}'
    )
NET_ARCH=${NET_ARCH_LIST[$SLURM_ARRAY_TASK_ID]}

if [ -z "$NET_ARCH" ]; then
    echo "FATAL: NET_ARCH is empty for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}."
    echo "Check #SBATCH --array range against NET_ARCH_LIST length."
    exit 2
fi

# Optional: tune timesteps and run naming from here.
N_ENVS=16
export N_ENVS
export ALG=sac
export RUN_PREFIX=aurora
export TOTAL_TIMESTEPS=3000000
export RL_DEVICE=cuda
export NET_ARCH

echo "Running with N_ENVS=${N_ENVS}, ALG=${ALG}, NET_ARCH=${NET_ARCH}, RL_DEVICE=${RL_DEVICE}"
$PYTHON_ENV src/rl/parallel_train.py
