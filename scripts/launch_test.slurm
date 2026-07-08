#!/bin/bash
#SBATCH --job-name=aurora_traj    # Name of the job
#SBATCH --output=result_%j.out     # Standard output file (%j expands to jobId)
#SBATCH --error=result_%j.err      # Standard error file
#SBATCH --ntasks=1                 # Number of processes
#SBATCH --cpus-per-task=8          # Cores per process
#SBATCH --mem=8G                   # Total memory requested
#SBATCH --time=05:00:00            # Time limit (HH:MM:SS)
#SBATCH --partition=CPUQ          # Partition/Queue to submit to
#SBATCH --array=0-10

echo "Starting job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

module purge
module load Miniconda3/24.7.1-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aurora-ferry

# Force the correct Python by using the full path
PYTHON_ENV="/cluster/home/stephemo/.conda/envs/aurora-ferry/bin/python"

echo "Using Python: $PYTHON_ENV"
echo "Python version: $($PYTHON_ENV --version)"

# Change to project directory
cd aurora_ferry

EXPERIMENTS=(
    "use_ais_every_1_no_camera"
    "use_ais_every_2_no_camera"
    "use_ais_every_3_no_camera"
    "use_ais_every_4_no_camera"
    "use_ais_every_8_no_camera"
    "use_ais_every_16_no_camera"
    "use_ais_every_32_no_camera"
    "use_ais_every_64_no_camera"
    "use_ais_every_128_no_camera"
    "use_ais_every_256_no_camera"
    "use_ais_every_512_no_camera"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#EXPERIMENTS[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for EXPERIMENTS (${#EXPERIMENTS[@]} items)."
    exit 1
fi

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
PATH_TO_CONFIG="/cluster/home/stephemo/dev/aurora_ferry/sim_data/${EXPERIMENT}"

# Run the script with the correct Python
echo "Generating scenarios using generator.py..."
$PYTHON_ENV src/scenarios/generator.py "${PATH_TO_CONFIG}/${EXPERIMENT}.yaml"
echo "Running parallel_sim_launcher.py..."
$PYTHON_ENV src/scenarios/parallel_sim_launcher.py "${PATH_TO_CONFIG}/scenarios" --workers 8