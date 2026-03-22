#!/bin/bash

set -ueo pipefail

make residuum-tests
echo "Running residuum tests..."
srun -p gpu-short-teach -w ampere02 --gpus A100:1 ./build/residuum-tests
