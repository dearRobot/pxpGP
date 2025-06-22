#!/usr/bin/bash

NUM_RUN=10
NUM_AGENTS=16

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done


for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

############################ Agents = 25 ############################

NUM_RUN=20
NUM_AGENTS=25

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

############################# Agents = 36 ############################

NUM_RUN=20
NUM_AGENTS=36

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

############################# Agents = 49 ############################

NUM_RUN=20
NUM_AGENTS=49

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

############################# Agents = 64 ############################

NUM_RUN=20
NUM_AGENTS=64

for i in $(seq 1 $NUM_RUN); do
    echo "Running pxpGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 pxpGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Running gapxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 gapxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running apxGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

for i in $(seq 1 $NUM_RUN); do
    echo "Running cGP $i with agents: $NUM_AGENTS"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 cGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done

############################# Agents = 100 ############################

NUM_RUN=20
NUM_AGENTS=100






