#!/bin/bash

set -ueo pipefail

if [[ ! -f "build-release/Makefile" ]]; then
    >&2 echo "Generate the 'Makefile' using 'cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=No -B build-release/' before running this script."
    exit 1
fi

srun -p gpu-short-teach make -C build-release/ benchmarks
echo "Running benchmarks..."
srun -p gpu-short-teach -w bw01 --gpus BW6000:1 ./build-release/tests/benchmarks/benchmarks $@

