#!/bin/bash

set -ueo pipefail

make benchmarks
echo "Running benchmarks..."
# TODO: Parse inputs and put them in here
srun -p gpu-short-teach -w ampere02 --gpus A100:1 ./build/benchmarks $@


