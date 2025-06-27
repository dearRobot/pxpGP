#!/usr/bin/bash

source /home/m10937683/gp_ws/gpenv/bin/activate
echo "Job started on: $(hostname)"

NUM_RUN=10
NUM_AGENTS=16
MAX_ATTEMPTS=3

for i in $(seq 1 $NUM_RUN); do
    echo "Running fulGP $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun -A 2506240925 --job-name="full_gp_16" -p gpu --gres=gpu:v100:2 --cpus-per-task=6 --time=12:00:00 -o output.%j -e error.%j python3 fullGP_train.py 
        # sbatch ./results/scripts/run_fullgp.slurm
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
    echo "Running fulGP $i"
    
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

for i in $(seq 1 $NUM_RUN); do
    echo "Running fulGP $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
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
    echo "Running fulGP $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
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
    echo "Running fulGP $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
        srun torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
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
#     echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
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
#     echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
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
#     echo "Running apxGP $i with agents: $NUM_AGENTS"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
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
#     echo "Running cGP $i with agents: $NUM_AGENTS"
    
#     # Retry until success
#     for j in ${MAX_ATTEMPTS}; do
#         torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
#         if [ $? -eq 0 ]; then
#             echo "Run $i completed successfully"
#             break
#         else
#             echo "Run $i failed, retrying..."
#             sleep 2  # Optional: brief pause before retrying
#         fi
#     done
# done