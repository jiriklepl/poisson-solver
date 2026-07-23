#ifndef SOLVER_CORE_HPP
#define SOLVER_CORE_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/detail/axis_strategies.hpp"
#include "poisson-solver/detail/fft_policies.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/reshaper.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#ifdef USE_MPI
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/mpi_check.hpp"

#include <mpi.h>
#endif  // USE_MPI

namespace poisson_solver::detail {
/**
 * @brief Class encapsulating core solver data.
 * Every solver is composed (at least) of this core.
 *
 * @tparam T Floating point type (only `float` and `double` supported).
 * @tparam Dim Number of grid dimensions.
 * @tparam FFTDim Number of grid dimensions transformed via FFT.
 */
template <std::floating_point T, std::size_t Dim, std::size_t FFTDim = Dim>
    requires(FFTDim <= Dim)
struct SolverCore {
public:
    using HostOneAxisDenoms = std::vector<T>;
    using HostDenoms = std::array<HostOneAxisDenoms, FFTDim>;
    using CudaUniquePtrT = detail::CudaUniquePtr<T>;
    using CudaUniquePtrCmplxT =
        detail::CudaUniquePtr<detail::CuComplexTrait<T>>;
    using DenomConstPtrs = detail::DenomConstPtrs<T, FFTDim>;

    // FFT needs to know the dimensionality of the whole grid.
    // Transformation of only some dimensions are handled using
    // the 1D and 2D methods of FFTPolicy.
    using CoreFFTPolicy = FFTPolicy<T, Dim>;

    using FFTPlan = typename CoreFFTPolicy::Plan;

    // === Plans ===
    FFTPlan fft_plan;

    // === Grid ===
    GridSizes<Dim> ns;
    std::size_t cnt;
    std::unique_ptr<IReshaper<T, Dim>> reshaper{};

    // === Physics and Math ===
    std::array<CudaUniquePtrT, FFTDim> cu_denoms;

    // These will be passed into the kernels.
    DenomConstPtrs cu_denom_observers;

    T norm_factor;

    // === Workspaces ===
    // We need two of each to account for out-of-place
    // reshaping in distributed solvers.
    // We "ping-pong" between the two (i.e., use wspace1, reshape into
    // wspace2, use wspace2, etc.).

    // Only some solvers need complex workspace.
    CudaUniquePtrCmplxT cu_cmplx_workspace1{};
    CudaUniquePtrCmplxT cu_cmplx_workspace2{};

    // This is always used - either straight for computation
    // or to initialize complex workspace from it.
    CudaUniquePtrT cu_real_workspace1{};
    CudaUniquePtrT cu_real_workspace2{};

    // === Device Settings ===
    dim3 cu_block_size;

    /**
     * @brief Constructs core for local solver.
     */
    SolverCore(GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        : SolverCore(CoreFFTPolicy::Init(ns, sp.fft_layout),
                     DummyReshapePipeline<T, Dim>::From(ns),
                     ns,
                     lens,
                     ns,
                     GridOffsets<Dim>{},  // Zero-initialize all elements
                     approx,
                     cuda_block_size,
                     sp) {}

#ifdef USE_MPI
    /**
     * @brief Constructs core for distributed solver.
     */
    SolverCore(GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               GridSizes<Dim> gns,
               GridOffsets<Dim> offs,
               MPI_Comm comm,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        // The GPU has to be assigned before subsequent CUDA calls
        : SolverCore(
              CoreFFTPolicy::Init(ns,
                                  offs,
                                  AssignGPUAndReturnComm(std::move(comm)),
                                  sp.fft_layout),
              ReshapePipeline<T, Dim>::From(ns, offs, comm),
              ns,
              lens,
              gns,
              offs,
              approx,
              cuda_block_size,
              sp) {}
#endif  // USE_MPI

    /**
     * @note The dimensionality of the result is determined by `FFTDim`.
     * The computations in dimensions not transformed via FFT
     * (e.g. Z in `NonuniformZSolver`) contain dependencies in that dimension.
     * This forces the GPU threads to parallelize only on the remaining dims
     * (e.g. X and Y in `NonuniformZSolver`).
     */
    auto ComputeCUDAGridSize(GridSizes<Dim> ns) -> dim3 {
        if constexpr (FFTDim == 1) {
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                    cu_block_size.x};
        } else if constexpr (FFTDim == 2) {
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                        cu_block_size.x,
                    (static_cast<unsigned>(ns[1]) + cu_block_size.y - 1) /
                        cu_block_size.y};
        } else {
            // FFTDim == 3
            return {(static_cast<unsigned>(ns[0]) + cu_block_size.x - 1) /
                        cu_block_size.x,
                    (static_cast<unsigned>(ns[1]) + cu_block_size.y - 1) /
                        cu_block_size.y,
                    (static_cast<unsigned>(ns[2]) + cu_block_size.z - 1) /
                        cu_block_size.z};
        }
    }

private:
    /**
     * @brief Master constructor.
     * Initializes everything common for local and distributed solver.
     *
     * @note: \p fft_plan should be initialized by the calling ctor and
     * passed here.
     */
    SolverCore(FFTPlan &&fft_plan,
               std::unique_ptr<IReshapePipeline<T, Dim>> &&rp,
               GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               GridSizes<Dim> gns,
               GridOffsets<Dim> offs,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        : fft_plan(std::move(fft_plan)),
          ns(ns),
          cnt(ComputeLocalGridSize(ns)),
          reshaper(BuildReshaper(std::move(rp), sp)),
          norm_factor(ComputeNormFactor(gns, sp)),
          cu_block_size(cuda_block_size) {
        AllocDeviceBuffers(gns);

        // Denoms are constant for the solver lifetime.
        // They can be copied just once, here.
        const auto denoms = GetEigenvalues(lens, gns, approx, sp);
        CopyDenomsToDevice(denoms, gns);
        cu_denom_observers = CreateDenomObservers();
    }

    static auto ComputeLocalGridSize(GridSizes<Dim> ns) -> std::size_t {
        return std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>{});
    }

    static auto BuildReshaper(std::unique_ptr<IReshapePipeline<T, Dim>> &&rp,
                              const ISolverPlan<T, FFTDim> &sp)
        -> std::unique_ptr<IReshaper<T, Dim>> {
        using enum ReshapeType;

        std::size_t pencil_dir{};
        std::array<std::size_t, 2> slab_dirs{};

        std::visit(
            Overloaded{[&](FullLayout fl) { /* nop */ },
                       [&](SplitLayout sl) {
                           pencil_dir = sl.isolated_dim_idx;
                           slab_dirs[0] = (sl.isolated_dim_idx + 1) % 3;
                           slab_dirs[1] = (sl.isolated_dim_idx + 2) % 3;
                       },
                       [](std::monostate) -> Plan {
                           throw std::runtime_error("Visited a monostate.");
                       }},
            sp.fft_layout);

        for (const auto &reshape_type : sp.reshape_types) {
            switch (reshape_type) {
                case ToPencils:
                    rp->ToPencils(pencil_dir);
                    break;
                case ToSlabs:
                    rp->ToSlabs(slab_dirs[0], slab_dirs[1]);
                    break;
                default:
                    throw std::runtime_error("Unknown reshape type.");
            }
        }

        return rp->Build();
    }

    auto ComputeNormFactor(GridSizes<Dim> gns,
                           const ISolverPlan<T, FFTDim> &sp) const -> T {
        std::size_t factor = 1;
        for (std::size_t d = 0; d < FFTDim; ++d) {
            const auto &strategy = sp.axis_strategies[d];
            factor *= strategy->ComputeNormalizationFactor(
                gns[d], std::holds_alternative<FullLayout>(sp.fft_layout));
        }
        return static_cast<T>(factor);
    }

    void AllocDeviceWorkspaces() {
        using detail::CuComplexTrait;
        using detail::MakeCudaUnique;

        // TODO - No cmplx wspace is required for the following solvers:
        // Uniform - FullDirichlet, FullNeumann, 2Real1Real, 1Real2Real and
        // NonuniformZ - FullNeumann.
        // For those allocate only real.
        // TODO - If we aren't using SplitLayout then the pointers should
        // be aliased so that no copying takes place.
        cu_cmplx_workspace1 =
            MakeCudaUnique<CuComplexTrait<T>>(reshaper->MaxLocalElements());
        cu_cmplx_workspace2 =
            MakeCudaUnique<CuComplexTrait<T>>(reshaper->MaxLocalElements());
        cu_real_workspace1 = MakeCudaUnique<T>(reshaper->MaxLocalElements());
        cu_real_workspace2 = MakeCudaUnique<T>(reshaper->MaxLocalElements());
    }

    void AllocDeviceDenoms(GridSizes<FFTDim> gns) {
        using detail::MakeCudaUnique;
        for (std::size_t d = 0; d < FFTDim; ++d) {
            cu_denoms[d] = MakeCudaUnique<T>(gns[d]);
        }
    }

    void AllocDeviceBuffers(GridSizes<FFTDim> gns) {
        AllocDeviceWorkspaces();
        AllocDeviceDenoms(gns);
    }

    auto GetOneAxisEigenvalues(T len,
                               std::size_t gn,
                               const IApproximation<T> &approx,
                               const IAxisStrategy<T> &axis_strategy) const
        -> HostOneAxisDenoms {
        HostOneAxisDenoms eigenvals(gn);

        const T grid_dx = axis_strategy.GetGridDx(gn, len);
        const T dkx = std::numbers::pi_v<T> * grid_dx / len;
        const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

        for (std::size_t i = 0; i < gn; ++i) {
            eigenvals[i] =
                approx(axis_strategy.GetApproximationInput(i, gn, dkx)) *
                inv_grid_dx_squared;
        }

        return eigenvals;
    }

    auto GetEigenvalues(std::array<T, FFTDim> lens,
                        GridSizes<Dim> gns,
                        const IApproximation<T> &approx,
                        const ISolverPlan<T, FFTDim> &sp) const -> HostDenoms {
        HostDenoms denoms{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            denoms[d] = GetOneAxisEigenvalues(
                lens[d], gns[d], approx, *sp.axis_strategies[d]);
        }
        return denoms;
    }

    /**
     * @note Device `denom` buffers have to be already allocated.
     */
    void CopyDenomsToDevice(const HostDenoms &denoms, GridSizes<FFTDim> gns) {
        for (std::size_t d = 0; d < FFTDim; ++d) {
            CUCH(cudaMemcpy(cu_denoms[d].get(),
                            denoms[d].data(),
                            gns[d] * sizeof(T),
                            cudaMemcpyHostToDevice));
        }
    }

    /**
     * @note Device `denom` buffers have to be already allocated.
     */
    auto CreateDenomObservers() -> DenomConstPtrs {
        DenomConstPtrs observers{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            observers[d] = cu_denoms[d].get();
        }
        return observers;
    }
#ifdef USE_MPI
    /**
     * @brief Assigns the correct GPU in case there are multiple on a single
     * node.
     *
     * @param comm World communicator.
     * @return Returns \p comm back.
     */
    static auto AssignGPUAndReturnComm(MPI_Comm &&comm) -> MPI_Comm {
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(
            comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));

        int local_rank;
        MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));

        int num_gpus{};
        CUCH(cudaGetDeviceCount(&num_gpus));

        CUCH(cudaSetDevice(local_rank % num_gpus));

        MPI_CHECK(MPI_Comm_free(&local_comm));

        return std::move(comm);
    }
#endif  // USE_MPI
};
}  // namespace poisson_solver::detail

#endif  // SOLVER_CORE_HPP
