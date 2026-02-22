#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/axis_policies.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/cuch.hpp"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver {
// Forward declarations
struct CudaFreeDeleter;

// Usings
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaFreeDeleter>;

template <BCPair BCs, std::floating_point T>
void CalculateEigenvalues(T len, std::size_t n, const IApproximation<T> &approx,
                          std::vector<T> &eigenvals) {
    using Policy = AxisPolicy<BCs, T>;

    eigenvals.resize(n);

    const T grid_dx = Policy::GetGridDx(len, n);
    const T dkx = std::numbers::pi_v<T> * grid_dx / len;

    const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

    for (std::size_t i = 0; i < n; ++i) {
        eigenvals[i] = approx(Policy::GetApproximationInput(i, n, dkx)) *
                       inv_grid_dx_squared;
    }
}

struct CudaFreeDeleter {
    void operator()(void *ptr) { cudaFree(ptr); }
};

// Creates a unique pointer to memory such that it can hold `size` elements of
// type `T`
template <typename T>
CudaUniquePtr<T> make_cuda_unique(std::size_t size) {
    T *ptr;
    CUCH(cudaMalloc(&ptr, size * sizeof(T)));
    return CudaUniquePtr<T>(ptr);
}
}  // namespace poisson_solver

#endif  // HELPERS_HPP
