#!/bin/bash

set -ueo pipefail

NUM_NODES=$1
NTASKS_PER_NODE=$2
NPS_Y=$3
NPS_Z=$4

EXEC_DIR="./build/tests/benchmarks"
RESULT_DIR="${EXEC_DIR}/results/strong_scaling/"

EXEC="${EXEC_DIR}/benchmarks"

# Benchmark parameters
ITERS=30
GRID_START=512
GRID_END=512
# Not used, start == end
GRID_STEP=64

BCS_3D=(
    "P P P P P P"
    "D D D D D D"
    "N N N N N N"
    "DS DS DS DS DS DS"
    "NS NS NS NS NS NS"
    "P P P P NS NS"
    "P P P P DS DS"
    "P P P P DS NS"
    "P P P P NS DS"
    "P P P P D N"
    "P P P P N D"
    "P P NS NS NS NS"
)

mkdir -p "mapped/poisson-solver/$RESULT_DIR"

# Make UCX not use ptrace
# Otherwise many errors get printed before falling back to a different method
export UCX_POSIX_USE_PROC_LINK=n

for prec in "float" "double"; do
    for gpu_val in "no" "yes"; do
        gpu_str=$([ "$gpu_val" = "yes" ] && echo "gpu" || echo "nogpu")
        dim=3
        bcs=("${BCS_3D[@]}")
        nonuniform_vals=("no" "yes")

        for nonunif_val in "${nonuniform_vals[@]}"; do
            nonunif_str=$([ "$nonunif_val" = "yes" ] && echo "nonuniform-z" || echo "uniform")

            # nonuniform-z solver has a special set of allowed bcs
            if [ "$nonunif_val" = "yes" ]; then
                current_bcs=("P P P P NS NS" "NS NS NS NS NS NS")
            else
                current_bcs=("${bcs[@]}")
            fi

            for bc_args in "${current_bcs[@]}"; do
                # Strip spaces from BCs for the filename
                bc_name=$(echo "$bc_args" | tr -d ' ')

                OUT_FILE="${RESULT_DIR}/poisson-solver_distributed_${prec}_dim${dim}_${gpu_str}_${nonunif_str}_${bc_name}_N${NUM_NODES}_T${NTASKS_PER_NODE}.csv"

                if [ -f "mapped/poisson-solver/${OUT_FILE}" ]; then
                    echo "Skipping $OUT_FILE (Already completed)"
                    continue
                fi

                echo "=========================================================="
                echo "Running... Precision: $prec | Dim: $dim | GPU: $gpu_val | NonuniformZ: $nonunif_val | BCs: $bc_args"

                srun --mpi=pmix bash -c "sleep \$((SLURM_PROCID % 2)); exec ch-run -b mapped:/opt/build imgdir -- /bin/bash -c \"cd /opt/build/poisson-solver && $EXEC $prec $dim $gpu_val $nonunif_val $ITERS $GRID_START $GRID_END $GRID_STEP $OUT_FILE $NPS_Y $NPS_Z $bc_args\""

                echo "Finished, result in $OUT_FILE"
            done
        done
    done
done

echo "All benchmarks completed."
