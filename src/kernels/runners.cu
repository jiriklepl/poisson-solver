#include "poisson-solver/kernel.cuh"
#include "poisson-solver/runners.hpp"

#include <concepts>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunInitCmplxWorkspace(CuComplexTrait_t<T> *cmplx_workspace,
                           const T *rhs,
                           std::array<std::size_t, Dim> ns,
                           dim3 grid_size,
                           dim3 block_size) {
    InitCmplxWorkspace<<<grid_size, block_size>>>(cmplx_workspace, rhs, ns);
}

template <std::floating_point T, std::size_t Dim>
void RunDivideCmplxWorkspaceByDenoms(CuComplexTrait_t<T> *cmplx_workspace,
                                     std::array<std::size_t, Dim> ns,
                                     std::array<const T *, Dim> denoms,
                                     dim3 grid_size,
                                     dim3 block_size) {
    DivideCmplxWorkspaceByDenoms<<<grid_size, block_size>>>(
        cmplx_workspace, ns, denoms);
}

template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const CuComplexTrait_t<T> *cmplx_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size) {
    NormalizeCmplxWorkspaceToPhi<<<grid_size, block_size>>>(
        phi, cmplx_workspace, ns, norm_factor);
}

// Explicit instantiations
// Doubles
template void RunInitCmplxWorkspace<double, 1>(
    CuComplexTrait_t<double> *cmplx_workspace,
    const double *rhs,
    std::array<std::size_t, 1> ns,
    dim3 grid_size,
    dim3 block_size);
template void RunInitCmplxWorkspace<double, 2>(
    CuComplexTrait_t<double> *cmplx_workspace,
    const double *rhs,
    std::array<std::size_t, 2> ns,
    dim3 grid_size,
    dim3 block_size);
template void RunInitCmplxWorkspace<double, 3>(
    CuComplexTrait_t<double> *cmplx_workspace,
    const double *rhs,
    std::array<std::size_t, 3> ns,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<double, 1>(
    CuComplexTrait_t<double> *cmplx_workspace,
    std::array<std::size_t, 1> ns,
    std::array<const double *, 1> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<double, 2>(
    CuComplexTrait_t<double> *cmplx_workspace,
    std::array<std::size_t, 2> ns,
    std::array<const double *, 2> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<double, 3>(
    CuComplexTrait_t<double> *cmplx_workspace,
    std::array<std::size_t, 3> ns,
    std::array<const double *, 3> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<double, 1>(
    double *phi,
    const CuComplexTrait_t<double> *cmplx_workspace,
    std::array<std::size_t, 1> ns,
    double norm_factor,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<double, 2>(
    double *phi,
    const CuComplexTrait_t<double> *cmplx_workspace,
    std::array<std::size_t, 2> ns,
    double norm_factor,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<double, 3>(
    double *phi,
    const CuComplexTrait_t<double> *cmplx_workspace,
    std::array<std::size_t, 3> ns,
    double norm_factor,
    dim3 grid_size,
    dim3 block_size);

// Floats
template void RunInitCmplxWorkspace<float, 1>(
    CuComplexTrait_t<float> *cmplx_workspace,
    const float *rhs,
    std::array<std::size_t, 1> ns,
    dim3 grid_size,
    dim3 block_size);
template void RunInitCmplxWorkspace<float, 2>(
    CuComplexTrait_t<float> *cmplx_workspace,
    const float *rhs,
    std::array<std::size_t, 2> ns,
    dim3 grid_size,
    dim3 block_size);
template void RunInitCmplxWorkspace<float, 3>(
    CuComplexTrait_t<float> *cmplx_workspace,
    const float *rhs,
    std::array<std::size_t, 3> ns,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<float, 1>(
    CuComplexTrait_t<float> *cmplx_workspace,
    std::array<std::size_t, 1> ns,
    std::array<const float *, 1> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<float, 2>(
    CuComplexTrait_t<float> *cmplx_workspace,
    std::array<std::size_t, 2> ns,
    std::array<const float *, 2> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<float, 3>(
    CuComplexTrait_t<float> *cmplx_workspace,
    std::array<std::size_t, 3> ns,
    std::array<const float *, 3> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<float, 1>(
    float *phi,
    const CuComplexTrait_t<float> *cmplx_workspace,
    std::array<std::size_t, 1> ns,
    float norm_factor,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<float, 2>(
    float *phi,
    const CuComplexTrait_t<float> *cmplx_workspace,
    std::array<std::size_t, 2> ns,
    float norm_factor,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<float, 3>(
    float *phi,
    const CuComplexTrait_t<float> *cmplx_workspace,
    std::array<std::size_t, 3> ns,
    float norm_factor,
    dim3 grid_size,
    dim3 block_size);
}  // namespace poisson_solver
