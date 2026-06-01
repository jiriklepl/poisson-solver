#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>

namespace poisson_solver::detail {
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
constexpr auto FlatIndex(MultidimIdx<Dim> midx,
                         GridSizes<Dim> ns) -> std::size_t {
    std::size_t idx = 0;
    std::size_t stride = 1;

    for (std::size_t i = 0; i < Dim; ++i) {
        idx += midx[i] * stride;
        stride *= ns[i];
    }

    return idx;
}

// Creates a unique pointer to memory such that it can hold `size` elements of
// type `T`
template <typename T>
auto MakeCudaUnique(std::size_t size) -> CudaUniquePtr<T> {
    T *ptr;
    CUCH(cudaMalloc(&ptr, size * sizeof(T)));
    return CudaUniquePtr<T>(ptr);
}

// Transfers the centered active grid between host and device.
template <typename T, std::size_t Dim>
void MemcpyActiveGrid(T *host_ptr,
                      T *dev_ptr,
                      GridSizes<Dim> active_ns,
                      GridSizes<Dim> ngs,
                      cudaMemcpyKind kind) {
    cudaMemcpy3DParms params{};

    // At multiple places in this function, we compute
    // sizes for all three dims but only the first (X)
    // size is multiplied by `sizeof(T)`.
    // This is because the Y and Z sizes are then expressed
    // in terms of the X size which is already in bytes.

    GridSizes<Dim> full_ns;
    for (std::size_t d = 0; d < Dim; ++d) {
        full_ns[d] = active_ns[d] + 2 * ngs[d];
    }

    // Extent of the active grid (the sub-volume to copy)
    params.extent = make_cudaExtent(active_ns[0] * sizeof(T),
                                    (Dim >= 2) ? active_ns[1] : 1,
                                    (Dim >= 3) ? active_ns[2] : 1);

    // Host side (padded)
    cudaPitchedPtr host_pitched =
        make_cudaPitchedPtr(host_ptr,
                            full_ns[0] * sizeof(T),
                            full_ns[0],
                            (Dim >= 2) ? full_ns[1] : 1);

    // Device side (contiguous)
    cudaPitchedPtr dev_pitched =
        make_cudaPitchedPtr(dev_ptr,
                            active_ns[0] * sizeof(T),
                            active_ns[0],
                            (Dim >= 2) ? active_ns[1] : 1);

    std::size_t offset_x = ngs[0];
    std::size_t offset_y = (Dim >= 2) ? ngs[1] : 0;
    std::size_t offset_z = (Dim >= 3) ? ngs[2] : 0;

    cudaPos host_pos = make_cudaPos(offset_x * sizeof(T), offset_y, offset_z);
    cudaPos dev_pos = make_cudaPos(0, 0, 0);

    if (kind == cudaMemcpyHostToDevice) {
        params.srcPtr = host_pitched;
        params.srcPos = host_pos;
        params.dstPtr = dev_pitched;
        params.dstPos = dev_pos;
    } else {
        params.srcPtr = dev_pitched;
        params.srcPos = dev_pos;
        params.dstPtr = host_pitched;
        params.dstPos = host_pos;
    }

    params.kind = kind;
    CUCH(cudaMemcpy3D(&params));
}
}  // namespace poisson_solver::detail

#endif  // HELPERS_HPP
