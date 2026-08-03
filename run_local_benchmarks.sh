#!/bin/bash

set -ueo pipefail

NODE="volta02"

srun -p gpu-short-teach -N 1 --ntasks-per-node 1 -w "$NODE" --gpus-per-task 1 ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/poisson-solver && ./local_benchmarks.sh"
