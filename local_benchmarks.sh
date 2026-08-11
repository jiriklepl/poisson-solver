#!/bin/bash

set -ueo pipefail

EXEC_DIR="./build/tests/benchmarks"
RESULT_DIR="${EXEC_DIR}/results"

EXEC="${EXEC_DIR}/benchmarks"

# Benchmark parameters
ITERS=20
GRID_START=64
GRID_END=512
GRID_STEP=64

BCS_1D=(
    "P P"
    "D D"
    "N N"
    "DS DS"
    "NS NS"
    "NS DS"
    "DS NS"
    "D N"
    "N D"
)

BCS_2D=(
    "P P P P"
    "D D D D"
    "N N N N"
    "DS DS DS DS"
    "NS NS NS NS"
)

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
    "NS NS P P P P"
    "P P NS NS P P"
    "P P NS NS NS NS"
    "DS DS DS DS P P"
    "DS DS DS DS NS NS"
    "DS DS DS DS NS DS"
    "D D D D N D"
    "NS DS NS NS NS NS"
    "N D N N N N"
)

mkdir -p "$RESULT_DIR"

for prec in "float" "double"; do
    for gpu_val in "no" "yes"; do
        gpu_str=$([ "$gpu_val" = "yes" ] && echo "gpu" || echo "nogpu")

        for dim in 1 2 3; do
            if [ "$dim" -eq 1 ]; then
                bcs=("${BCS_1D[@]}")
            elif [ "$dim" -eq 2 ]; then
                bcs=("${BCS_2D[@]}")
            elif [ "$dim" -eq 3 ]; then
                bcs=("${BCS_3D[@]}")
            fi

            if [ "$dim" -eq 3 ]; then
                nonuniform_vals=("no" "yes")
            else
                # nonuniform-z only supported in 3D
                nonuniform_vals=("no")
            fi

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

                    OUT_FILE="${RESULT_DIR}/poisson-solver_local_${prec}_dim${dim}_${gpu_str}_${nonunif_str}_${bc_name}.csv"

                    if [ -f "$OUT_FILE" ]; then
                        echo "Skipping $OUT_FILE (Already completed)"
                        continue
                    fi

                    echo "=========================================================="
                    echo "Running... Precision: $prec | Dim: $dim | GPU: $gpu_val | NonuniformZ: $nonunif_val | BCs: $bc_args"

                    $EXEC $prec $dim $gpu_val $nonunif_val $ITERS $GRID_START $GRID_END $GRID_STEP "$OUT_FILE" $bc_args

                    echo "Finished, result in $OUT_FILE"
                done
            done
        done
    done
done

echo "All benchmarks completed."
