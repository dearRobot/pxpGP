#!/usr/bin/bash

NUM_RUN=10
NUM_AGENTS=64
MAX_ATTEMPTS=3

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${seq 1 $MAX_ATTEMPTS}; do
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
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
    for j in ${seq 1 $MAX_ATTEMPTS}; do
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
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
    for j in ${seq 1 $MAX_ATTEMPTS}; do
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
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
    for j in ${seq 1 $MAX_ATTEMPTS}; do
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done