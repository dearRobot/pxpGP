#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --job-name="full_gp_100"
#SBATCH -p gpu 
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=15
#SBATCH --time=12:00:00
#SBATCH -o output.%j
#SBATCH -e error.%j

source /home/m10937683/gp_ws/gpenv/bin/activate
echo "Job started on: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Activate env and set vars
source /home/m10937683/gp_ws/gpenv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


NUM_RUN=3
NUM_AGENTS=16
MAX_ATTEMPTS=3


for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpgp $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done


NUM_RUN=3
NUM_AGENTS=49
MAX_ATTEMPTS=3


for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpgp $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done


NUM_RUN=3
NUM_AGENTS=64
MAX_ATTEMPTS=3


for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpgp $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done


NUM_RUN=3
NUM_AGENTS=100
MAX_ATTEMPTS=3


for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpgp $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done


# for i in $(seq 1 $NUM_RUN); do
#     echo "Running fulGP $i"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         srun python3 fullGP_train.py 
#         # sbatch ./results/scripts/run_fullgp.slurm
#         if [ $? -eq 0 ]; then
#             echo "Run $i completed successfully"
#             break
#         else
#             echo "Run $i failed, retrying..."
#             sleep 2  # Optional: brief pause before retrying
#         fi
#     done
# done

# for i in $(seq 1 $NUM_RUN); do
#     echo "Running gapxgp $i"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
#         if [ $? -eq 0 ]; then
#             echo "Run $i completed successfully"
#             break
#         else
#             echo "Run $i failed, retrying..."
#             sleep 2  # Optional: brief pause before retrying
#         fi
#     done
# done

# for i in $(seq 1 $NUM_RUN); do
#     echo "Running apxgp $i"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
#         if [ $? -eq 0 ]; then
#             echo "Run $i completed successfully"
#             break
#         else
#             echo "Run $i failed, retrying..."
#             sleep 2  # Optional: brief pause before retrying
#         fi
#     done
# done

# for i in $(seq 1 $NUM_RUN); do
#     echo "Running cgp $i"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
#         if [ $? -eq 0 ]; then
#             echo "Run $i completed successfully"
#             break
#         else
#             echo "Run $i failed, retrying..."
#             sleep 2  # Optional: brief pause before retrying
#         fi
#     done
# done

echo "Job complete"