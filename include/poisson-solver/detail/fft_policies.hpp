#ifndef FFT_POLICIES_HPP
#define FFT_POLICIES_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vkFFT.h>

#ifdef USE_MPI
#include <heffte.h>
#include <mpi.h>
#include <utility>
#endif  // USE_MPI

namespace poisson_solver::detail {
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

// === Forward Declarations ===
struct FullLayout;
struct SplitLayout;

struct VkFFTFullHandle;
struct VkFFTSplitHandle;

// === Aliases ===
using VariantLayout = std::variant<std::monostate, FullLayout, SplitLayout>;

using VkFFTVariantHandle =
    std::variant<std::monostate, VkFFTFullHandle, VkFFTSplitHandle>;

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

struct VkFFTFullHandle {
    VkFFTHandle handle{};
};

struct VkFFTSplitHandle {
    VkFFTHandle handle_1d{};
    VkFFTHandle handle_2d{};
};

struct VkFFTPlan {
    std::unique_ptr<CUdevice> cu_device{};
    VkFFTVariantHandle handle{};
};

// Holds values of `performDCT` and `performDST`
// used in VkFFT configuration.
struct VkFFTTypeToPerform {
    std::size_t dct{};
    std::size_t dst{};
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

// === Functions and Macros ===
inline void VkFFTCheck(VkFFTResult res,
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
    ::poisson_solver::detail::VkFFTCheck(res, __LINE__, __FILE__, #res)

// === Classes ===
template <std::floating_point T, std::size_t Dim>
struct VkFFTPolicy {
public:
    using Plan = VkFFTPlan;

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
        assert(std::holds_alternative<VkFFTFullHandle>(plan.handle));
        auto &handle = std::get<VkFFTFullHandle>(plan.handle).handle;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), forward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTFullHandle>(plan.handle));
        auto &handle = std::get<VkFFTFullHandle>(plan.handle).handle;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), backward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward1D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_1d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), forward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward1D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_1d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), backward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward2D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_2d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), forward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward2D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_2d;

        VkFFTLaunchParams launch_params{};
        launch_params.buffer = reinterpret_cast<void **>(&dev_buffer);

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), backward_direction_, &launch_params));
    }

private:
    static constexpr int forward_direction_ = -1;
    static constexpr int backward_direction_ = 1;

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
                              CUdevice *device_ptr) -> VkFFTFullHandle {
        VkFFTFullHandle h{};
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
                               CUdevice *device_ptr) -> VkFFTSplitHandle {
        VkFFTSplitHandle h{};
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
                return {0, 0};
        }
    }
};

#ifdef USE_MPI
// == Forward Declarations ===
template <std::floating_point T>
struct IMpiFftHandle;

// === Structs ===
// We have to use the type erasure idiom with heFFTe to
// transform the compile-time tags into run-time virtual calls.

template <std::floating_point T>
class IMpiFftHandle {
public:
    virtual ~IMpiFftHandle() = default;

    // Provide default asserts to catch invalid calls
    // (e.g., calling Forward1D on a Full plan)
    virtual void Forward(void *dev_buffer) const {
        assert(false && "Not a full plan");
    }

    virtual void Backward(void *dev_buffer) const {
        assert(false && "Not a full plan");
    }

    virtual void Forward1D(void *dev_buffer) const {
        assert(false && "Not a split plan");
    }

    virtual void Backward1D(void *dev_buffer) const {
        assert(false && "Not a split plan");
    }

    virtual void Forward2D(void *dev_buffer) const {
        assert(false && "Not a split plan");
    }

    virtual void Backward2D(void *dev_buffer) const {
        assert(false && "Not a split plan");
    }

protected:
    IMpiFftHandle(std::size_t workspace_size_bytes)
        : workspace_{MakeCudaUnique<std::byte>(workspace_size_bytes)} {}

    // For FullHandle: If we didn't choose to provide a workspace, heFFTe
    // would allocate and deallocate it itself in every FFT call.
    //
    // For SplitHandle: TODO - Why does heffte reshape API need it
    // Is std::byte because for split handle will be used both for
    // complex and real numbers.
    CudaUniquePtr<std::byte> workspace_{};
};

template <std::floating_point T, typename TransformTag>
struct MpiFftFullHandle : public IMpiFftHandle<T> {
public:
    // If using C2C transform, workspace is complex, otherwise it's real
    using Workspace =
        std::conditional_t<std::is_same_v<heffte::backend::cufft, TransformTag>,
                           CuComplexTrait<T>,
                           T>;

    MpiFftFullHandle(heffte::fft3d<TransformTag> &&plan)
        : IMpiFftHandle<T>{plan.size_workspace() * sizeof(Workspace)},
          plan_{std::move(plan)} {}

    void Forward(void *dev_buffer) const override {
        plan_.forward(reinterpret_cast<Workspace *>(dev_buffer),
                      reinterpret_cast<Workspace *>(dev_buffer),
                      reinterpret_cast<Workspace *>(this->workspace_.get()));
    }

    void Backward(void *dev_buffer) const override {
        plan_.backward(reinterpret_cast<Workspace *>(dev_buffer),
                       reinterpret_cast<Workspace *>(dev_buffer),
                       reinterpret_cast<Workspace *>(this->workspace_.get()));
    }

private:
    heffte::fft3d<TransformTag> plan_;
};

// TODO - My concept of how heFFTe works with split solvers is incorrect
// This whole class needs reworking
template <std::floating_point T>
struct MpiFftSplitHandle : public IMpiFftHandle<T> {
public:
    using LocalFFTPolicy = VkFFTPolicy<T, 3>;
    using LocalFFTPlan = typename LocalFFTPolicy::Plan;

    MpiFftSplitHandle(LocalFFTPlan &&fft_plan)
        : IMpiFftHandle<T>{0/* TODO - How big wspace required for heffte reshape API */},
          fft_plan_(std::move(fft_plan)) {}

    void Forward1D(void *dev_buffer) const override {
        // TODO
    }

    void Backward1D(void *dev_buffer) const override {
        // TODO
    }

    void Forward2D(void *dev_buffer) const override {
        // TODO
    }

    void Backward2D(void *dev_buffer) const override {
        // TODO
    }

private:
    LocalFFTPlan fft_plan_;
};

template <std::floating_point T>
struct HeFFTePlan {
public:
    std::unique_ptr<IMpiFftHandle<T>> handle{};
};

template <std::floating_point T, std::size_t Dim>
struct MpiFftPolicy {
public:
    using Plan = HeFFTePlan<T>;

    static auto Init(GridSizes<Dim> ns,
                     GridOffsets<Dim> offs,
                     MPI_Comm comm,
                     VariantLayout layout) -> Plan {
        return std::visit(
            Overloaded{[ns, offs, comm](FullLayout l) {
                           return GetFullPlan(l, ns, offs, comm);
                       },
                       [ns, offs, comm](SplitLayout l) -> Plan {
                           // This shouldn't be reachable with Dim != 3
                           // but if not disposed of for Dim != 3, the
                           // code won't compile.
                           if constexpr(Dim == 3) {
                               return GetSplitPlan(l, ns, offs, comm);
                           } else {
                               throw std::runtime_error("Getting a split plan when Dim != 3");
                           }
                       },
                       [](std::monostate) -> Plan {
                           throw std::runtime_error("Visited a monostate.");
                       }},
            layout);
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Forward(reinterpret_cast<void *>(dev_buffer));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Backward(reinterpret_cast<void *>(dev_buffer));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward1D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Forward1D(reinterpret_cast<void *>(dev_buffer));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward1D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Backward1D(reinterpret_cast<void *>(dev_buffer));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward2D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Forward2D(reinterpret_cast<void *>(dev_buffer));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward2D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Backward2D(reinterpret_cast<void *>(dev_buffer));
    }

private:
    static auto GetGridBox(GridSizes<Dim> ns, GridOffsets<Dim> offs) {
        auto begin_x = offs[0];
        auto begin_y = (Dim >= 2) ? offs[1] : 0;
        auto begin_z = (Dim >= 3) ? offs[2] : 0;

        auto end_x = begin_x + ns[0] - 1;
        auto end_y = (Dim >= 2) ? (begin_y + ns[1] - 1) : 0;
        auto end_z = (Dim >= 3) ? (begin_z + ns[2] - 1) : 0;

        std::array<int, 3> low = {static_cast<int>(begin_x), static_cast<int>(begin_y), static_cast<int>(begin_z)};
        std::array<int, 3> high = {static_cast<int>(end_x), static_cast<int>(end_y), static_cast<int>(end_z)};

        return heffte::box3d<>{low, high};
    }

    static auto GetFullPlan(FullLayout l,
                            GridSizes<Dim> ns,
                            GridOffsets<Dim> offs,
                            MPI_Comm comm) -> Plan {
        using enum FFTTransformType;

        auto grid_box = GetGridBox(ns, offs);

        switch (l.forward_type) {
            case Complex: {
                using TransformTag = heffte::backend::cufft;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DCT1: {
                using TransformTag = heffte::backend::cufft_cos1;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DCT2: {
                using TransformTag = heffte::backend::cufft_cos;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DCT3:
                // There is never a situation where full solver has DCT3 set as
                // a forward FFT type.
                throw std::runtime_error(
                    "Internal error. Full solver shouldn't have DCT3 as "
                    "forward FFT.");
            case DCT4: {
                using TransformTag = heffte::backend::cufft_cos4;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DST1: {
                using TransformTag = heffte::backend::cufft_sin1;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DST2: {
                using TransformTag = heffte::backend::cufft_sin;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DST3:
                // There is never a situation where full solver has DST3 set as
                // a forward FFT type.
                throw std::runtime_error(
                    "Internal error. Full solver shouldn't have DST3 as "
                    "forward FFT.");
            case DST4: {
                using TransformTag = heffte::backend::cufft_sin4;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            default:
                throw std::runtime_error("Invalid FFT forward type.");
        }
    }

    static auto GetSplitPlan(SplitLayout l,
                             GridSizes<3> ns,
                             GridOffsets<3> offs,
                             MPI_Comm comm) -> Plan {
        /* TODO return */ Plan{std::make_unique<MpiFftSplitHandle<T>>(
            MpiFftSplitHandle<T>::LocalFFTPolicy::Init(ns, l))};
        // TODO - Add more args to ctor if required by heffte reshape API usage
        throw std::runtime_error("Not implemented");
    }
};

template <std::floating_point T, std::size_t Dim>
using FFTPolicy = MpiFftPolicy<T, Dim>;

#else

template <std::floating_point T, std::size_t Dim>
using FFTPolicy = VkFFTPolicy<T, Dim>;

#endif  // USE_MPI
}  // namespace poisson_solver::detail

#endif  // FFT_POLICIES_HPP
