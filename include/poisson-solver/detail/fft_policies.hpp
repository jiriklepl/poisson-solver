#ifndef FFT_POLICIES_HPP
#define FFT_POLICIES_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cstddef>
#include <memory>

#ifdef USE_MPI
#include <heffte.h>
#include <mpi.h>
#else
#include "poisson-solver/detail/cuch.hpp"

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vkFFT.h>
#endif  // USE_MPI

namespace poisson_solver::detail {
// === Concepts ===
// Restricts workspace to be a real or complex workspace
template <typename Workspace, typename T>
concept ValidWorkspace = std::floating_point<T> &&
                         (std::is_same_v<Workspace, T> ||
                          std::is_same_v<Workspace, detail::CuComplexTrait<T>>);

// === Enums ===
enum class FFTTransformType {
    Complex,

    DCT1,
    DCT2,
    DCT3,
    DCT4,

    DST1,
    DST2,
    DST3,
    DST4
};

// === Layout Structs ===
// Different layouts of FFTs are used based on BCs.
// TODO: Only VkFFT works currenlty
// and that handles the backward type
// based on the forward type
// so no need to save that.

struct FullLayout {
    FFTTransformType forward_type{};
};

struct SplitLayout {
    FFTTransformType forward_type_1d{};
    FFTTransformType forward_type_2d{};

    // 0 = X, 1 = Y, 2 = Z
    std::size_t isolated_dim_idx{};
};

// === Aliases ===
using VariantLayout = std::variant<std::monostate, FullLayout, SplitLayout>;

#ifdef USE_MPI

// TODO: NOT TESTED!
template <std::floating_point T, std::size_t Dim>
struct HeFFTePolicy {
    // This needs to be a ptr because `heffte::fft3d` doesn't have a default
    // ctor and we defer initialization to a solver ctor body
    using Plan = std::unique_ptr<heffte::fft3d<heffte::backend::cufft>>;

    static Plan Init(GridSizes<Dim> ns,
                     // TODO: Currently unused but will be needed to determine
                     // the FFT plan type.
                     BCs<Dim> bcs) {
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
        return std::make_unique<heffte::fft3d<heffte::backend::cufft>>(
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

template <std::floating_point T, std::size_t Dim>
using FFTPolicy = HefftePolicy<T, Dim>;

#else

// === Forward Declarations ===
struct FullHandle;

struct SplitHandle;

template <std::floating_point T, std::size_t Dim>
struct VkFFTPolicy;

// === Aliases ===
using VariantHandle = std::variant<std::monostate, FullHandle, SplitHandle>;

template <std::floating_point T, std::size_t Dim>
using FFTPolicy = VkFFTPolicy<T, Dim>;

// === Structs ===

// Helper struct for `std::variant` visiting.
template <typename... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};

struct VkFFTApplicationDeleter {
    void operator()(VkFFTApplication *app) const {
        // It's possible that `app` wasn't even initialized, calling
        // `deleteVkFFT` on that could be dangerous.
        if (app != nullptr) {
            deleteVkFFT(app);
            delete app;
        }
    }
};

struct VkFFTHandle {
    VkFFTConfiguration config{};
    std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> app{};
};

struct FullHandle {
    VkFFTHandle handle{};
};

struct SplitHandle {
    VkFFTHandle handle_1d{};
    VkFFTHandle handle_2d{};
};

struct VkFFTPlan {
    std::unique_ptr<CUdevice> cu_device{};
    VariantHandle handle{};
};

// Holds values of `performDCT` and `performDST`
// used in VkFFT configuration.
struct VkFFTTypeToPerform {
    std::size_t dct{};
    std::size_t dst{};
};

inline void _VkFFTCheck(VkFFTResult res,
                        int line,
                        const char *src_file,
                        const char *err_msg = nullptr) {
    if (res != VKFFT_SUCCESS) {
        std::stringstream err_stream{};
        err_stream << "VkFFT Error (" << res << ")\n"
                   << "at " << src_file << "[" << line << "]:" << err_msg;
        throw std::runtime_error(err_stream.str());
    }
}

#define VKFFT_CHECK(res) \
    ::poisson_solver::detail::_VkFFTCheck(res, __LINE__, __FILE__, #res)

template <std::floating_point T, std::size_t Dim>
struct VkFFTPolicy {
public:
    using Plan = VkFFTPlan;
    static constexpr int ForwardDirection = -1;
    static constexpr int BackwardDirection = 1;

    static auto Init(GridSizes<Dim> ns, VariantLayout layout) -> Plan {
        Plan plan{};

        // Shared among all layouts.
        plan.cu_device = GetCudaDevice();

        std::visit(Overloaded{[&plan, ns](FullLayout l) {
                                  plan.handle = GetFullHandle(
                                      l, ns, plan.cu_device.get());
                              },
                              [&plan, ns](SplitLayout l) {
                                  plan.handle = GetSplitHandle(
                                      l, ns, plan.cu_device.get());
                              },
                              [](std::monostate) {
                                  assert(false && "Visited a monostate.");
                              }},
                   layout);

        return plan;
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<FullHandle>(plan.handle));
        auto &handle = std::get<FullHandle>(plan.handle).handle;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), ForwardDirection, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<FullHandle>(plan.handle));
        auto &handle = std::get<FullHandle>(plan.handle).handle;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), BackwardDirection, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward1D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<SplitHandle>(plan.handle));
        auto &handle = std::get<SplitHandle>(plan.handle).handle_1d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), ForwardDirection, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward1D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<SplitHandle>(plan.handle));
        auto &handle = std::get<SplitHandle>(plan.handle).handle_1d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), BackwardDirection, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward2D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<SplitHandle>(plan.handle));
        auto &handle = std::get<SplitHandle>(plan.handle).handle_2d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), ForwardDirection, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward2D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<SplitHandle>(plan.handle));
        auto &handle = std::get<SplitHandle>(plan.handle).handle_2d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), BackwardDirection, &launch_params));
    }

private:
    static auto GetCudaDevice() -> std::unique_ptr<CUdevice> {
        // Ensure CUDA runtime is initialized.
        // This is essentailly a nop but if runtime isn't init, then it does it.
        cudaFree(nullptr);

        int ordinal{};
        CUCH(cudaGetDevice(&ordinal));

        auto device = std::make_unique<CUdevice>();
        cuDeviceGet(device.get(), ordinal);

        return device;
    }

    // Returns initialized full handle.
    static auto GetFullHandle(FullLayout layout,
                              GridSizes<Dim> ns,
                              CUdevice *device_ptr) -> FullHandle {
        FullHandle h{};
        auto &config = h.handle.config;

        config.device = device_ptr;

        // 0 for float, 1 for doubles.
        config.doublePrecision = !std::is_same_v<T, float>;

        config.FFTdim = Dim;
        for (std::size_t d = 0; d < Dim; ++d) {
            config.size[d] = ns[d];
        }
        auto fft_type = GetVkFFTTransformType(layout.forward_type);
        config.performDCT = fft_type.dct;
        config.performDST = fft_type.dst;

        auto &app = h.handle.app;

        // Need to use `reset` because `make_unique`
        // doesn't work with custom deleters.
        app.reset(new VkFFTApplication{});

        VKFFT_CHECK(initializeVkFFT(app.get(), config));

        return h;
    };

    // Returns initialized split handle.
    static auto GetSplitHandle(SplitLayout layout,
                               GridSizes<Dim> ns,
                               CUdevice *device_ptr) -> SplitHandle {
        SplitHandle h{};
        auto &config_1d = h.handle_1d.config;
        auto &config_2d = h.handle_2d.config;

        config_1d.device = device_ptr;
        config_2d.device = device_ptr;

        // 0 for float, 1 for doubles.
        bool double_prec_used = !std::is_same_v<T, float>;
        config_1d.doublePrecision = static_cast<pfUINT>(double_prec_used);
        config_2d.doublePrecision = static_cast<pfUINT>(double_prec_used);

        // Both configs will know about the
        // grid being 3D but will omit the
        // appropriate dimensions.
        config_1d.FFTdim = Dim;
        config_2d.FFTdim = Dim;
        for (std::size_t d = 0; d < Dim; ++d) {
            config_1d.size[d] = ns[d];
            config_2d.size[d] = ns[d];
        }

        switch (layout.isolated_dim_idx) {
            case 0: {
                config_1d.omitDimension[1] = 1;
                config_1d.omitDimension[2] = 1;
                config_2d.omitDimension[0] = 1;
                break;
            }
            case 1: {
                config_1d.omitDimension[0] = 1;
                config_1d.omitDimension[2] = 1;
                config_2d.omitDimension[1] = 1;
                break;
            }
            case 2: {
                config_1d.omitDimension[0] = 1;
                config_1d.omitDimension[1] = 1;
                config_2d.omitDimension[2] = 1;
                break;
            }
            default: {
                assert(false && "Invalid index of an isolated axis.");
            }
        }

        auto fft_type_1d = GetVkFFTTransformType(layout.forward_type_1d);
        auto fft_type_2d = GetVkFFTTransformType(layout.forward_type_2d);

        config_1d.performDCT = fft_type_1d.dct;
        config_1d.performDST = fft_type_1d.dst;

        config_2d.performDCT = fft_type_2d.dct;
        config_2d.performDST = fft_type_2d.dst;

        auto &app_1d = h.handle_1d.app;
        auto &app_2d = h.handle_2d.app;

        // Need to use `reset` because `make_unique`
        // doesn't work with custom deleters.
        app_1d.reset(new VkFFTApplication{});
        app_2d.reset(new VkFFTApplication{});

        VKFFT_CHECK(initializeVkFFT(app_1d.get(), config_1d));
        VKFFT_CHECK(initializeVkFFT(app_2d.get(), config_2d));

        return h;
    }

    // Maps values of the `FFTTransformType` enum
    // to the `int` values used by VkFFT.
    static auto GetVkFFTTransformType(FFTTransformType t)
        -> VkFFTTypeToPerform {
        using enum FFTTransformType;

        switch (t) {
            case Complex:
                return {.dct = 0, .dst = 0};
            case DCT1:
                return {.dct = 1, .dst = 0};
            case DCT2:
                return {.dct = 2, .dst = 0};
            case DCT3:
                return {.dct = 3, .dst = 0};
            case DCT4:
                return {.dct = 4, .dst = 0};
            case DST1:
                return {.dct = 0, .dst = 1};
            case DST2:
                return {.dct = 0, .dst = 2};
            case DST3:
                return {.dct = 0, .dst = 3};
            case DST4:
                return {.dct = 0, .dst = 4};
            default:
                assert(false && "Unhandled VkFFT transform type mapping.");
        }
    }
};

#endif  // USE_MPI
}  // namespace poisson_solver::detail

#endif  // FFT_POLICIES_HPP
