#!/bin/bash

set -ueo pipefail

# `gfortran` is not on head nodes, thus the need for `srun`
srun -p gpu-short-teach make fortran-residuum-tests

echo "Running residuum tests..."
srun -p gpu-short-teach --gpus=1 ./build/fortran-residuum-tests
