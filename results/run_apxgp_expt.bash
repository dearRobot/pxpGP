#!/usr/bin/bash

NUM_RUN=10
NUM_AGENTS=16

for i in $(seq 1 $NUM_RUN); do
    echo "Run $i"
    torchrun --nproc_per_node=$NUM_AGENTS --master_addr=localhost --master_port=12345 apxGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done