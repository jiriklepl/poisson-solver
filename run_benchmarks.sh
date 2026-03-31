#!/bin/bash

set -ueo pipefail

make benchmarks
echo "Running benchmarks..."
srun -p gpu-short-teach -w bw01 --gpus BW6000:1 ./build/benchmarks $@

