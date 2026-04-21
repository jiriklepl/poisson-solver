#ifndef FFT_POLICIES_HPP
#define FFT_POLICIES_HPP

#include "poisson-solver/helpers.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

#ifdef USE_MPI
#include <heffte.h>
#include <mpi.h>
#else
#include "poisson-solver/cuch.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vkFFT.h>
#endif  // USE_MPI

namespace poisson_solver {
#ifdef USE_MPI

// TODO: NOT TESTED!
template <std::floating_point T>
struct HeFFTePolicy {
    // This needs to be a ptr because `heffte::fft3d` doesn't have a default
    // ctor and we defer initialization to a solver ctor body
    using Plan = std::unique_ptr<heffte::fft3d<heffte::backend::cufft>>;

    template <std::size_t Dim>
    static void Init(Plan &plan, std::array<std::size_t, Dim> ns) {
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

    static void Forward(CuComplexTrait_t<T> *dev_buffer, Plan &plan) {
        plan->forward(dev_buffer, dev_buffer, heffte::scale::none);
    }

    static void Backward(CuComplexTrait_t<T> *dev_buffer, Plan &plan) {
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

template <std::floating_point T>
struct VkFFTPolicy {
    using Plan = VkFFTPlan;
    static constexpr int ForwardDirection = -1;
    static constexpr int BackwardDirection = 1;

    template <std::size_t Dim>
    static void Init(Plan &plan, std::array<std::size_t, Dim> ns) {
        // Set dim and axes sizes
        plan.config.FFTdim = Dim;
        for (std::size_t d = 0; d < Dim; ++d) {
            // VkFFT has fastests changing index on 0 and slowest on Dim - 1
            // and we have it the exact opposite.
            plan.config.size[Dim - d - 1] = ns[d];
        }

        // 0 for float, 1 for doubles.
        plan.config.doublePrecision = !std::is_same_v<T, float>;

        // Need to use `reset` because `make_unique` doesn't work with custom
        // deleters.
        plan.app.reset(new VkFFTApplication{});

        GetCudaDevice(&plan.cu_device);
        plan.config.device = &plan.cu_device;

        VkFFTResult res = initializeVkFFT(plan.app.get(), plan.config);
        if (res != VKFFT_SUCCESS)
            throw std::runtime_error(
                "Failed to initialize VkFFT, err code is " +
                std::to_string(res) + ".");
    }

    static void Forward(CuComplexTrait_t<T> *dev_buffer, Plan &plan) {
        VkFFTLaunchParams launch_params{};
        launch_params.buffer = (void **)&dev_buffer;

        VkFFTResult res =
            VkFFTAppend(plan.app.get(), ForwardDirection, &launch_params);
        if (res != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT forward failed, err code is " +
                                     std::to_string(res) + ".");
    }

    static void Backward(CuComplexTrait_t<T> *dev_buffer, Plan &plan) {
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
};

#endif  // USE_MPI
}  // namespace poisson_solver

#endif  // FFT_POLICIES_HPP
