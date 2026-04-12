#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/approximations.hpp"
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

template <typename T>
inline constexpr bool always_false = false;

template <std::size_t Dim>
constexpr std::size_t FlatIndex(std::array<std::size_t, Dim> indices,
                                std::array<std::size_t, Dim> ns) {
    std::size_t idx = 0;
    std::size_t stride = 1;

    for (std::size_t i = Dim; i > 0; --i) {
        idx += indices[i - 1] * stride;
        stride *= ns[i - 1];
    }

    return idx;
}

struct CudaFreeDeleter {
    void operator()(void *ptr) {
        cudaFree(ptr);
    }
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
