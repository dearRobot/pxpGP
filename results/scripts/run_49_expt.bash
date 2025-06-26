#!/usr/bin/bash

NUM_RUN=10
NUM_AGENTS=9
MAX_ATTEMPTS=3

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in $(seq 1 $MAX_ATTEMPTS); do
        export MASTER_PORT=$((12000 + RANDOM % 10000))
        surn torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=$MASTER_PORT pxpGP_train.py
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
    for j in $(seq 1 $MAX_ATTEMPTS); do
        export MASTER_PORT=$((12000 + RANDOM % 10000))
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=$MASTER_PORT gapxGP_train.py
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
    for j in $(seq 1 $MAX_ATTEMPTS); do
        export MASTER_PORT=$((12000 + RANDOM % 10000))
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=$MASTER_PORT apxGP_train.py
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
    for j in $(seq 1 $MAX_ATTEMPTS); do
        export MASTER_PORT=$((12000 + RANDOM % 10000))
        torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=$MASTER_PORT cGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done