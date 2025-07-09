#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --job-name="gp_training_25"
#SBATCH -p gpu 
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH -o output.%j
#SBATCH -e error.%j

echo "Job started on: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Activate env and set vars
source /home/m10937683/gp_ws/gpenv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun ./results/scripts/run_49_expt.bash

srun -A 

echo "Job complete"