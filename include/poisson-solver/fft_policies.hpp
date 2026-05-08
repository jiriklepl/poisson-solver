#ifndef FFT_POLICIES_HPP
#define FFT_POLICIES_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

#ifdef USE_MPI
#include <heffte.h>
#include <mpi.h>
#else
#include "poisson-solver/detail/cuch.hpp"

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vkFFT.h>
#endif  // USE_MPI

namespace poisson_solver {
// Restricts workspace to be a real or complex workspace
template <typename Workspace, typename T>
concept ValidWorkspace = std::floating_point<T> &&
                         (std::is_same_v<Workspace, T> ||
                          std::is_same_v<Workspace, detail::CuComplexTrait<T>>);

#ifdef USE_MPI

// TODO: NOT TESTED!
template <std::floating_point T, std::size_t Dim>
struct HeFFTePolicy {
    // This needs to be a ptr because `heffte::fft3d` doesn't have a default
    // ctor and we defer initialization to a solver ctor body
    using Plan = std::unique_ptr<heffte::fft3d<heffte::backend::cufft>>;

    static void Init(Plan &plan,
                     GridSizes<Dim> ns,
                     // TODO: Currently unused but will be needed to determine
                     // the FFT plan type.
                     std::array<BCPair, Dim> bcs) {
        int nx = ns[0];
        int ny = (Dim >= 2) ? ns[1] : 1;
        int nz = (Dim >= 3) ? ns[2] : 1;
        // TODO: Order is not used right now, but may improve performance in
        // distributed environmnet i->slowest, j->middle, k->fastest growing
        // indices `std::array<int, 3> order = {2, 1, 0};`

        // Apparently `box3d`s first index is the fastest changing, second is
        // middle and last is slowest changing.
        // Also, the order apparently doesn't affect this and serves only as an
        // optimization parameter for better data spliting for the MPI ranks.
        heffte::box3d<> grid_box({0, 0, 0}, {nz - 1, ny - 1, nx - 1});
        // TODO: Accept custom comm through param
        plan = std::make_unique<heffte::fft3d<heffte::backend::cufft>>(
            grid_box, grid_box, MPI_COMM_SELF);
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward(Workspace *dev_buffer, Plan &plan) {
        plan->forward(dev_buffer, dev_buffer, heffte::scale::none);
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        plan->backward(dev_buffer, dev_buffer, heffte::scale::none);
    }
};

#else

struct VkFFTApplicationDeleter {
    void operator()(VkFFTApplication *app) const {
        // It's possible that `app` wasn't even initialized, calling
        // `deleteVkFFT` on that could be dangerous.
        if (app) {
            deleteVkFFT(app);
            delete app;
        }
    }
};

struct VkFFTPlan {
    CUdevice cu_device{};
    VkFFTConfiguration config{};
    std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> app;
};

template <std::floating_point T, std::size_t Dim>
struct VkFFTPolicy {
    using Plan = VkFFTPlan;
    static constexpr int ForwardDirection = -1;
    static constexpr int BackwardDirection = 1;

    static void Init(Plan &plan,
                     GridSizes<Dim> ns,
                     std::array<BCPair, Dim> bcs) {
        // Set dim and axes sizes
        plan.config.FFTdim = Dim;
        for (std::size_t d = 0; d < Dim; ++d) {
            // VkFFT has fastests changing index on 0 and slowest on Dim - 1
            // and we have it the exact opposite.
            plan.config.size[Dim - d - 1] = ns[d];
        }

        // 0 for float, 1 for doubles.
        plan.config.doublePrecision = !std::is_same_v<T, float>;

        GetCudaDevice(&plan.cu_device);
        plan.config.device = &plan.cu_device;
        SetFFTType(plan, bcs);

        // Need to use `reset` because `make_unique` doesn't work with custom
        // deleters.
        plan.app.reset(new VkFFTApplication{});

        VkFFTResult res = initializeVkFFT(plan.app.get(), plan.config);
        if (res != VKFFT_SUCCESS)
            throw std::runtime_error(
                "Failed to initialize VkFFT, err code is " +
                std::to_string(res) + ".");
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward(Workspace *dev_buffer, Plan &plan) {
        VkFFTLaunchParams launch_params{};
        launch_params.buffer = (void **)&dev_buffer;

        VkFFTResult res =
            VkFFTAppend(plan.app.get(), ForwardDirection, &launch_params);
        if (res != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT forward failed, err code is " +
                                     std::to_string(res) + ".");
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        VkFFTLaunchParams launch_params{};
        launch_params.buffer = (void **)&dev_buffer;

        VkFFTResult res =
            VkFFTAppend(plan.app.get(), BackwardDirection, &launch_params);
        if (res != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT backward failed, err code is " +
                                     std::to_string(res) + ".");
    }

private:
    static void GetCudaDevice(CUdevice *device) {
        // Ensure CUDA runtime is initialized.
        // This is essentailly a nop but if runtime isn't init, then it does it.
        cudaFree(0);

        int ordinal;
        CUCH(cudaGetDevice(&ordinal));
        cuDeviceGet(device, ordinal);
    }

    // Sets the FFT type for `bcs` that are not all equal.
    static void SetFFTType1D(Plan &plan, std::array<BCPair, 1> bcs) {
        using BC = BoundaryCondition;
        constexpr BC D = BC::kDirichlet;
        constexpr BC N = BC::kNeumann;
        constexpr BC DS = BC::kDirichletStag;
        constexpr BC NS = BC::kNeumannStag;

        BCPair bcp = bcs[0];
        if (bcp.first == NS && bcp.second == DS) {
            // RE11
            plan.config.performDCT = 4;
        } else if (bcp.first == DS && bcp.second == NS) {
            // RO11
            plan.config.performDST = 4;
        } else if (bcp.first == N && bcp.second == D) {
            // RE01
            plan.config.performDCT = 3;
        } else if (bcp.first == D && bcp.second == N) {
            // RO01
            plan.config.performDST = 3;
        } else {
            throw std::invalid_argument("Invalid combination of 1D BCs.");
        }
    }

    // Sets the FFT type for `bcs` that are not all equal.
    static void SetFFTType3D(Plan &plan, std::array<BCPair, 3> bcs) {
        // TODO: Finish implementation of all possible types
        throw std::runtime_error("Not implemented.");
    }

    // Sets the FFT type if all the `bcs` are equal as this behaviour is
    // shared in all dimensions. Returns whether all `bcs` are equal. If not,
    // further processing and FFT type setting should be done.
    static bool TrySetFullBCsFFTType(Plan &plan, std::array<BCPair, Dim> bcs) {
        using namespace std::ranges;

        if (all_of(bcs, IsPeriodic)) {
            plan.config.performDCT = 0;
            plan.config.performDST = 0;
        } else if (all_of(bcs, IsDirichlet)) {
            // RO00
            plan.config.performDST = 1;
        } else if (all_of(bcs, IsDirichletStag)) {
            // RO10
            plan.config.performDST = 2;
        } else if (all_of(bcs, IsNeumann)) {
            // RE00
            plan.config.performDCT = 1;
        } else if (all_of(bcs, IsNeumannStag)) {
            // RE10
            plan.config.performDCT = 2;
        } else {
            return false;
        }
        return true;
    }

    // Sets the FFT type of the plan.
    static void SetFFTType(Plan &plan, std::array<BCPair, Dim> bcs) {
        if (TrySetFullBCsFFTType(plan, bcs))
            return;

        if constexpr (Dim == 1) {
            SetFFTType1D(plan, bcs);
        } else if constexpr (Dim == 2) {
            throw std::invalid_argument("Invalid combination of 2D BCs.");
        } else if constexpr (Dim == 3) {
            SetFFTType3D(plan, bcs);
        }
    }
};

#endif  // USE_MPI
}  // namespace poisson_solver

#endif  // FFT_POLICIES_HPP
