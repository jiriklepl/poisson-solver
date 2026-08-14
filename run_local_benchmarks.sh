#!/bin/bash

set -ueo pipefail

NODE="volta05"

# Use 32 cores for parallel grid init
srun -p gpu-short-teach -c 32 -N 1 --ntasks-per-node 1 -w "$NODE" --gpus-per-task 1 -t 02:00:00 ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/poisson-solver && ./local_benchmarks.sh"
