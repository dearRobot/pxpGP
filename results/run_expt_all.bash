#!/usr/bin/bash

MAX_ATTEMPTS=5
NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running full GP $i"
    
    # Retry until success
    while true; do
        python3 fullGP_train.py
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            break
        else
            echo "Run $i failed, retrying..."
            sleep 2  # Optional: brief pause before retrying
        fi
    done
done

# ############################ Agents = 4 ############################

NUM_RUN=20
NUM_AGENTS=4

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

# ############################ Agents = 9 ############################

NUM_RUN=20
NUM_AGENTS=9

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

# ############################ Agents = 16 ############################

NUM_RUN=20
NUM_AGENTS=16

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

# ############################ Agents = 25 ############################

NUM_RUN=20
NUM_AGENTS=25

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

# ############################ Agents = 36 ############################

NUM_RUN=20
NUM_AGENTS=36

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

# ############################ Agents = 49 ############################

NUM_RUN=20
NUM_AGENTS=49

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

# ############################ Agents = 64 ############################

NUM_RUN=20
NUM_AGENTS=64

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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
    for j in ${MAX_ATTEMPTS}; do
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

