#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "poisson-solver/cuch.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace poisson_solver {
void RunVectorAdd();
}  // namespace poisson_solver

#endif  // KERNEL_CUH
