#!/usr/bin/bash

NUM_RUN=10

for i in $(seq 1 $NUM_RUN); do
    echo "Run $i"
    python3 fullGP_train.py
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i"
        exit 1
    fi
done