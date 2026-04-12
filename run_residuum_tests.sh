#!/bin/bash

set -ueo pipefail

if [[ ! -f "build-debug/Makefile" ]]; then
    >&2 echo "Generate the 'Makefile' using 'cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=Yes -B build-debug/' before running this script."
    exit 1
fi

make -C build-debug/ residuum-tests

echo "Running residuum tests..."
./build-debug/tests/residuum/residuum-tests
