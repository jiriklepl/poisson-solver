#!/bin/bash

set -ueo pipefail

make residuum-tests
echo "Running residuum tests..."
srun -p gpu-short-teach --gpus=1 ./build/residuum-tests
