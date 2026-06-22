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
#SBATCH --array=0-2

echo "Starting job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

module purge
module load Miniconda3/24.7.1-0
source $(conda info --base)/etc/profile.d/conda.sh
EXPECTED_ENV="aurora-ferry-gpu"
conda activate "$EXPECTED_ENV"

if [ "${CONDA_DEFAULT_ENV}" != "$EXPECTED_ENV" ]; then
    echo "FATAL: Expected conda env '$EXPECTED_ENV' but got '${CONDA_DEFAULT_ENV}'."
    exit 2
fi

PYTHON_ENV="${CONDA_PREFIX}/bin/python"
if [[ "$PYTHON_ENV" != *"/${EXPECTED_ENV}/bin/python" ]]; then
    echo "FATAL: Python path does not point to expected env: $PYTHON_ENV"
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
N_ENVS_LIST=(8 16 32)
N_ENVS=${N_ENVS_LIST[$SLURM_ARRAY_TASK_ID]}

# Optional: tune timesteps and run naming from here.
export N_ENVS
export ALG=sac
export RUN_PREFIX=aurora
export TOTAL_TIMESTEPS=5000000
export RL_DEVICE=cuda

echo "Running with N_ENVS=${N_ENVS}, ALG=${ALG}, TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}, RL_DEVICE=${RL_DEVICE}"
$PYTHON_ENV src/rl/parallel_train.py
