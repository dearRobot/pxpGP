#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=16 # number of tasks per node
#SBATCH --ntasks=16 # redundant; total number of tasks: ntasks = nodes * ntasks-per-node
#SBATCH --time=01:00:01 # time in HH:MM:SS
#SBATCH -o output.%j # standard print output labeled with SLURM job id %j
#SBATCH -e error.%j  # standard print error  labeled with SLURM job id %j

echo "Agent 16 Job has started!"
srun run_16_expt.bash
echo "Agent 16 Job done"