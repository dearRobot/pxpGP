#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=64 # number of tasks per node
#SBATCH --ntasks=64 # redundant; total number of tasks: ntasks = nodes * ntasks-per-node
#SBATCH --time=10:00:01 # time in HH:MM:SS
#SBATCH -o output.%j # standard print output labeled with SLURM job id %j
#SBATCH -e error.%j  # standard print error  labeled with SLURM job id %j

echo "Agent 64 Job has started!"
srun ./results/scripts/run_64_expt.bash
echo "Agent 64 Job done"