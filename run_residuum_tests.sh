#!/bin/bash

set -ueo pipefail

if [[ ! -f "build-debug/Makefile" ]]; then
    >&2 echo "Generate the 'Makefile' using 'cmake -B build-debug/' before running this script."
    exit 1
fi

srun -p gpu-short-teach -w volta01 make -C build-debug/ residuum-tests

echo "Running residuum tests..."
srun -p gpu-short-teach --gpus=V100:1 ./build-debug/tests/residuum/residuum-tests
