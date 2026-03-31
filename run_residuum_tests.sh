#!/bin/bash

set -ueo pipefail

srun -p gpu-short-teach make residuum-tests

echo "Running residuum tests..."
srun -p gpu-short-teach --gpus=1 ./build/residuum-tests
