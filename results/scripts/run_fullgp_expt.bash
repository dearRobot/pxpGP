#!/usr/bin/bash

NUM_RUN=10
MAX_ATTEMPTS=3

for i in $(seq 1 $NUM_RUN); do
    echo "Running fulGP $i"
    
    # Retry until success
    for j in ${MAX_ATTEMPTS}; do
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

