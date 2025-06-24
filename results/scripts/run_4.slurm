#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=4 # number of tasks per node
#SBATCH --ntasks=4 # redundant; total number of tasks: ntasks = nodes * ntasks-per-node
#SBATCH --time=01:00:01 # time in HH:MM:SS
#SBATCH -o output.%j # standard print output labeled with SLURM job id %j
#SBATCH -e error.%j  # standard print error  labeled with SLURM job id %j

echo "Agent 4 Job has started!"
srun ./results/scripts/run_4_expt.bash
echo "Agent 4 Job done"