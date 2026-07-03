#!/bin/bash
#SBATCH --job-name=aurora_analysis      # Name of the job
#SBATCH --output=analysis_%A_%a.out     # Standard output file (%A jobId, %a array task id)
#SBATCH --error=analysis_%A_%a.err      # Standard error file
#SBATCH --ntasks=1                      # Number of processes
#SBATCH --cpus-per-task=8               # Cores per process
#SBATCH --mem=32G                       # Total memory requested
#SBATCH --time=05:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=CPUQ                # Partition/Queue to submit to
#SBATCH --array=0-4

set -euo pipefail

echo "Starting analysis job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

module purge
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aurora-ferry

PYTHON_ENV="/cluster/home/stephemo/.conda/envs/aurora-ferry/bin/python"

echo "Using Python: ${PYTHON_ENV}"
echo "Python version: $(${PYTHON_ENV} --version)"

# Change to project directory
cd aurora_ferry

EXPERIMENTS=(
    "dchi_2p5_width_60"
    "dchi_2p5_width_70"
    "dchi_2p5_width_80"
    "dchi_5_width_60"
    "dchi_5_width_70"
    "dchi_5_width_80"
    "dchi_7p5_width_60"
    "dchi_7p5_width_70"
    "dchi_7p5_width_80"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#EXPERIMENTS[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for EXPERIMENTS (${#EXPERIMENTS[@]} items)."
    exit 1
fi

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT_DIR="/cluster/home/stephemo/dev/aurora_ferry/sim_data/${EXPERIMENT}"

if [ ! -d "${EXPERIMENT_DIR}" ]; then
    echo "Error: experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

if [ ! -d "${EXPERIMENT_DIR}/simulations" ]; then
    echo "Error: simulations directory not found: ${EXPERIMENT_DIR}/simulations"
    exit 1
fi

echo "Running analysis for experiment: ${EXPERIMENT}"
"${PYTHON_ENV}" src/scenarios/sim_analyzer.py "${EXPERIMENT_DIR}"
