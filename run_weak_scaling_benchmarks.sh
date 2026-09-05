#!/bin/bash

set -ueo pipefail

# Format: NODES, TASKS NPS_Y, NPS_Z, NODES_NAMES, GRID_SIZE, NODES_SATURATED
# We scale the global GRID_SIZE so that the local grid elements per NODE remains constant.
CONFIGS=(
    "1 1 1 1 volta05 161 0.25"
    "1 2 1 2 volta05 203 0.5"
    "1 4 2 2 volta05 256 1"
    "2 6 2 3 volta[02,05] 323 2"
    "3 8 2 4 volta[02,03,05] 369 3"
)

# Use 8 cores for parallel grid init
# Only allocate and run weak_scaling_benchmarks locally,
# that script will then call the benchmarks using srun
for cfg in "${CONFIGS[@]}"; do
    CFG_ARR=(${cfg})
    NUM_NODES=${CFG_ARR[0]}
    NTASKS=${CFG_ARR[1]}
    NPS_Y=${CFG_ARR[2]}
    NPS_Z=${CFG_ARR[3]}
    NODES=${CFG_ARR[4]}
    GRID_SIZE=${CFG_ARR[5]}

    # Currently unused
    NODES_SATURATED=${CFG_ARR[6]}

    salloc -p gpu-ffa-gpulab -c 8 -N $NUM_NODES -n $NTASKS -w "$NODES" --gpus $NTASKS -t 12:00:00 ./mapped/poisson-solver/weak_scaling_benchmarks.sh $NUM_NODES $NTASKS $NPS_Y $NPS_Z $GRID_SIZE
done
