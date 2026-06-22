#!/bin/bash
#SBATCH --job-name=aurora_sac_sweep
#SBATCH --output=result_%A_%a.out
#SBATCH --error=result_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=CPUQ
#SBATCH --array=0-2

echo "Starting job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

module purge
module load Miniconda3/24.7.1-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aurora-ferry

PYTHON_ENV="/cluster/home/stephemo/.conda/envs/aurora-ferry/bin/python"

echo "Using Python: $PYTHON_ENV"
echo "Python version: $($PYTHON_ENV --version)"

cd aurora_ferry

# Map array index to n_envs values to sweep.
N_ENVS_LIST=(8 16 32)
N_ENVS=${N_ENVS_LIST[$SLURM_ARRAY_TASK_ID]}

# Optional: tune timesteps and run naming from here.
export N_ENVS
export ALG=sac
export RUN_PREFIX=aurora
export TOTAL_TIMESTEPS=5000000

echo "Running with N_ENVS=${N_ENVS}, ALG=${ALG}, TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
$PYTHON_ENV src/rl/parallel_train.py
