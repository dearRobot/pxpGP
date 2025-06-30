#!/bin/bash
#SBATCH -A 2506240925
#SBATCH --job-name="full_gp_16k"
#SBATCH -p gpu 
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=18
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

NUM_RUN=10
NUM_AGENTS=64
MAX_ATTEMPTS=3

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_gapxGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_apxGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_cGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

#--------------------------------------------------------------------------------------

NUM_RUN=10
NUM_AGENTS=100
MAX_ATTEMPTS=3

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_gapxGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_apxGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_cGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

#--------------------------------------------------------------------------------------

NUM_RUN=10
NUM_AGENTS=121
MAX_ATTEMPTS=3


for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_pxpGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_gapxGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_apxGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 dec_cGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done