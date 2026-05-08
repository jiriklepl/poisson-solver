#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/cuch.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver::detail {
// === Variables ===
template <typename T>
inline constexpr bool always_false = false;

// === Structs ===
struct CudaFreeDeleter {
    void operator()(void *ptr) {
        cudaFree(ptr);
    }
};

// This type helps to decide the complex type based on the real type
template <std::floating_point T>
struct CuComplexTraitImpl;

template <>
struct CuComplexTraitImpl<float> {
    using Type = cuComplex;
};

template <>
struct CuComplexTraitImpl<double> {
    using Type = cuDoubleComplex;
};

// === Aliases ===
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaFreeDeleter>;

template <std::floating_point T>
using CuComplexTrait = typename CuComplexTraitImpl<T>::Type;

template <std::size_t Dim>
using MultidimIdx = std::array<std::size_t, Dim>;

template <std::floating_point T, std::size_t Dim>
using DenomConstPtrs = std::array<const T *, Dim>;

// === Functions ===
template <std::size_t Dim>
constexpr std::size_t FlatIndex(MultidimIdx<Dim> midx, GridSizes<Dim> ns) {
    std::size_t idx = 0;
    std::size_t stride = 1;

    for (std::size_t i = Dim; i > 0; --i) {
        idx += midx[i - 1] * stride;
        stride *= ns[i - 1];
    }

    return idx;
}

// Creates a unique pointer to memory such that it can hold `size` elements of
// type `T`
template <typename T>
CudaUniquePtr<T> MakeCudaUnique(std::size_t size) {
    T *ptr;
    CUCH(cudaMalloc(&ptr, size * sizeof(T)));
    return CudaUniquePtr<T>(ptr);
}
}  // namespace poisson_solver::detail

#endif  // HELPERS_HPP
