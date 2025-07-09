#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --job-name="gp_training_25"
#SBATCH -p gpu 
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH -o output.%j
#SBATCH -e error.%j

echo "FullGP Job has started!"
source /home/m10937683/gp_ws/gpenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python3 fullGP_train.py
echo "FullGP Job Job done"