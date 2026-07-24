#ifndef RESHAPER_HPP
#define RESHAPER_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

#ifdef USE_MPI
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/mpi_utils.hpp"

#include <complex>
#include <mpi.h>
#include <vector>
#endif  // USE_MPI

// TODO - Docs (for the whole file as well as for individual classes/fctions)
// Mention the guarantee that reshape to pencils and slabs will be there
// exactly once.

namespace poisson_solver::detail {
// === Enums ===
enum class ReshapeType { ToPencils, ToSlabs };

// === Structs ===
template <std::size_t Dim>
struct Layout {
    GridSizes<Dim> ns{};
    GridOffsets<Dim> offs{};
};

// === Interfaces ===
template <std::floating_point T, std::size_t Dim>
class IReshaper {
public:
    virtual ~IReshaper() = default;
    virtual auto ReshapeReal(T *dest, const T *src) -> Layout<Dim> = 0;
    virtual auto ReshapeCmplx(CuComplexTrait<T> *dest,
                              const CuComplexTrait<T> *src) -> Layout<Dim> = 0;
    virtual auto MaxLocalElements() const -> std::size_t = 0;
    virtual auto PencilLayout() const -> Layout<Dim> = 0;
    virtual auto SlabLayout() const -> Layout<Dim> = 0;
};

template <std::floating_point T, std::size_t Dim>
class IReshapePipeline {
public:
    virtual ~IReshapePipeline() = default;
    virtual auto ToPencils(std::size_t dimension) -> IReshapePipeline & = 0;
    virtual auto ToSlabs(std::size_t dimension1,
                         std::size_t dimension2) -> IReshapePipeline & = 0;
    virtual auto Build() -> std::unique_ptr<IReshaper<T, Dim>> = 0;
};

// === Local Forward Declarations ===
template <std::floating_point T, std::size_t Dim>
class DummyReshaper;

// === Local Implementations ===
/**
 * @brief Acts as a sink of build operatins (i.e., does no-op) so the
 * source code of distributed vs local solver doesn't need to change
 * but the local solver buids no reshape pipeline and reshapes nothing.
 */
template <std::floating_point T, std::size_t Dim>
class DummyReshapePipeline : public IReshapePipeline<T, Dim> {
private:
    /**
     * struct ConstructorKey - Key to the ctor.
     */
    struct ConstructorKey {};

public:
    friend class DummyReshaper<T, Dim>;

    DummyReshapePipeline(ConstructorKey /*unused*/, GridSizes<Dim> ns, GridOffsets<Dim> offs)
        : ns_{ns},
          offs_{offs} {}

    static auto From(GridSizes<Dim> ns)
        -> std::unique_ptr<IReshapePipeline<T, Dim>> {
        return std::make_unique<DummyReshapePipeline<T, Dim>>(ConstructorKey{}, ns, {0, 0, 0});
    }

    static auto From(GridSizes<Dim> ns, GridOffsets<Dim> offs)
        -> std::unique_ptr<IReshapePipeline<T, Dim>> {
        return std::make_unique<DummyReshapePipeline<T, Dim>>(ConstructorKey{}, ns, offs);
    }

    auto ToPencils([[maybe_unused]] std::size_t dimension)
        -> DummyReshapePipeline & override {
        if (contains_reshape_to_pencils_) {
            throw std::runtime_error(
                "Cannot reshape to pencils more than once.");
        }
        contains_reshape_to_pencils_ = true;
        // nop

        return *this;
    }

    auto ToSlabs([[maybe_unused]] std::size_t dimension1,
                 [[maybe_unused]] std::size_t dimension2)
        -> DummyReshapePipeline & override {
        if (contains_reshape_to_slabs_) {
            throw std::runtime_error("Cannot reshape to slabs more than once.");
        }
        contains_reshape_to_slabs_ = true;
        // nop

        return *this;
    }

    auto Build() -> std::unique_ptr<IReshaper<T, Dim>> override {
        if (!(contains_reshape_to_pencils_ && contains_reshape_to_slabs_)) {
            throw std::runtime_error(
                "Not all reshapes were specified in the pipeline (pencil, slab "
                "or both missing).");
        }
        return std::make_unique<DummyReshaper<T, Dim>>(std::move(*this));
    }

private:
    GridSizes<Dim> ns_;
    GridOffsets<Dim> offs_{};

    bool contains_reshape_to_pencils_{false};
    bool contains_reshape_to_slabs_{false};
};

template <std::floating_point T, std::size_t Dim>
class DummyReshaper : public IReshaper<T, Dim> {
public:
    explicit DummyReshaper(DummyReshapePipeline<T, Dim> &&rp)
        : ns_{std::move(rp.ns_)},
          offs_{std::move(rp.offs_)} {};

    auto ReshapeReal([[maybe_unused]] T *dest,
                     [[maybe_unused]] const T *src) -> Layout<Dim> override {
        // nop if aliased
        if (src != dest) {
            CUCH(cudaMemcpy(dest,
                            src,
                            MaxLocalElements() * sizeof(T),
                            cudaMemcpyDeviceToDevice));
        }
        return {ns_, offs_};
    }

    auto ReshapeCmplx([[maybe_unused]] CuComplexTrait<T> *dest,
                      [[maybe_unused]] const CuComplexTrait<T> *src)
        -> Layout<Dim> override {
        // nop if aliased
        if (src != dest) {
            CUCH(cudaMemcpy(dest,
                            src,
                            MaxLocalElements() * sizeof(CuComplexTrait<T>),
                            cudaMemcpyDeviceToDevice));
        }
        return {ns_, offs_};
    }

    auto MaxLocalElements() const -> std::size_t override {
        return std::accumulate(
            ns_.begin(), ns_.end(), std::size_t{1}, std::multiplies<>());
    }

    auto PencilLayout() const -> Layout<Dim> override {
        return Layout<Dim>{};
    }

    auto SlabLayout() const -> Layout<Dim> override {
        return Layout<Dim>{};
    }

private:
    GridSizes<Dim> ns_{};
    GridOffsets<Dim> offs_{};
};

#ifdef USE_MPI
// === Distributed Forward Declarations ===
template <std::floating_point T>
class Reshaper;

// === Distributed Helper Structs ===
template <typename Workspace>
struct HeFFTeWorkspaceTraitImpl;

template <>
struct HeFFTeWorkspaceTraitImpl<float> {
    using Type = float;
};

template <>
struct HeFFTeWorkspaceTraitImpl<CuComplexTrait<float>> {
    using Type = std::complex<float>;
};

template <>
struct HeFFTeWorkspaceTraitImpl<double> {
    using Type = double;
};

template <>
struct HeFFTeWorkspaceTraitImpl<CuComplexTrait<double>> {
    using Type = std::complex<double>;
};

// === Aliases ===
template <typename Workspace>
using HeFFTeWorkspaceTrait = HeFFTeWorkspaceTraitImpl<Workspace>::Type;

// === Distributed Implementations ===
template <std::floating_point T, std::size_t Dim>
class ReshapePipeline : public IReshapePipeline<T, Dim> {
private:
    /**
     * struct ConstructorKey - Key to the ctor.
     */
    struct ConstructorKey {};

    using Decomposition = std::vector<heffte::box3d<>>;

public:
    friend class Reshaper<T>;

    /**
     * @brief Publicly exposed private ctor.
     * Only callable from inside of this class using ConstructorKey.
     */
    ReshapePipeline(ConstructorKey, Decomposition &&original_boxes,
                    mpi_utils::ProcsPerDim<2> nps_yz,
                    MPI_Comm comm)
        : world_(heffte::find_world(original_boxes)),
          nps_yz_{nps_yz},
          comm_{comm} {
        decompositions_.emplace_back(std::move(original_boxes));
    }

    static auto From(GridSizes<Dim> ns,
                     GridOffsets<Dim> offs,
                     MPI_Comm cart_comm)
        -> std::unique_ptr<IReshapePipeline<T, Dim>> {
        if constexpr (Dim == 1 || Dim == 2) {
            return DummyReshapePipeline<T, Dim>::From(ns, offs);
        } else {
            heffte::box3d<> my_box = GetGridBox(ns, offs);
            auto all_boxes =
                heffte::mpi::gather_boxes(my_box, my_box, cart_comm).in;

            auto nps_xyz = mpi_utils::GetProcGrid(cart_comm);
            mpi_utils::ProcsPerDim<2> nps_yz = {nps_xyz[1], nps_xyz[2]};

            return std::make_unique<ReshapePipeline<T, 3>>(
                ConstructorKey{}, std::move(all_boxes), nps_yz, cart_comm);
        }
    }

    auto ToPencils(std::size_t dimension) -> ReshapePipeline & override {
        // TODO - Currently pretty much unused because we preserve
        // the original order. This probably runs slower in case
        // the FFT is in non-contiguous dimension but idk.
        // Later get rid of this or make it work with reversing the order.
        // auto pencil_order = GetOrder(sl.isolated_dim_idx);
        // auto slab_order = GetOrder(pencil_order[1]);

        // Resulting decomp by reshaping last decomp into pencils
        // in the direction of `dimension`
        if (contains_reshape_to_pencils_) {
            throw std::runtime_error(
                "Cannot reshape to pencils more than once.");
        }
        contains_reshape_to_pencils_ = true;

        decompositions_.emplace_back(heffte::make_pencils(
            world_, nps_yz_, dimension, decompositions_.back(), {0, 1, 2}));

        auto my_pencil = decompositions_.back()[rank_];
        auto my_pencil_sizes = my_pencil.size;
        auto my_pencil_offs = my_pencil.low;
        pencil_sizes_ = {static_cast<std::size_t>(my_pencil_sizes[0]),
                         static_cast<std::size_t>(my_pencil_sizes[1]),
                         static_cast<std::size_t>(my_pencil_sizes[2])};
        pencil_offs_ = {static_cast<std::size_t>(my_pencil_offs[0]),
                        static_cast<std::size_t>(my_pencil_offs[1]),
                        static_cast<std::size_t>(my_pencil_offs[2])};

        return *this;
    }

    auto ToSlabs(std::size_t dimension1,
                 std::size_t dimension2) -> ReshapePipeline & override {
        if (contains_reshape_to_slabs_) {
            throw std::runtime_error("Cannot reshape to slabs more than once.");
        }
        contains_reshape_to_slabs_ = true;

        auto num_procs = nps_yz_[0] * nps_yz_[1];

        // Resulting decomp by reshaping last decomp into slabs
        // that span the whole `dimension1` x `dimension2`
        decompositions_.emplace_back(heffte::make_slabs(world_,
                                                        num_procs,
                                                        dimension1,
                                                        dimension2,
                                                        decompositions_.back(),
                                                        {0, 1, 2},
                                                        heffte::rank_remap{}));

        auto my_slab = decompositions_.back()[rank_];
        auto my_slab_sizes = my_slab.size;
        auto my_slab_offs = my_slab.low;
        slab_sizes_ = {static_cast<std::size_t>(my_slab_sizes[0]),
                       static_cast<std::size_t>(my_slab_sizes[1]),
                       static_cast<std::size_t>(my_slab_sizes[2])};
        slab_offs_ = {static_cast<std::size_t>(my_slab_offs[0]),
                      static_cast<std::size_t>(my_slab_offs[1]),
                      static_cast<std::size_t>(my_slab_offs[2])};

        return *this;
    }

    auto Build() -> std::unique_ptr<IReshaper<T, Dim>> override {
        if (!(contains_reshape_to_pencils_ && contains_reshape_to_slabs_)) {
            throw std::runtime_error(
                "Not all reshapes were specified in the pipeline (pencil, slab "
                "or both missing).");
        }
        return std::make_unique<Reshaper<T>>(std::move(*this));
    }

private:
    bool contains_reshape_to_pencils_{false};
    bool contains_reshape_to_slabs_{false};

    GridSizes<Dim> pencil_sizes_{};
    GridOffsets<Dim> pencil_offs_{};
    GridSizes<Dim> slab_sizes_{};
    GridOffsets<Dim> slab_offs_{};

    heffte::box3d<> world_;
    mpi_utils::ProcsPerDim<2> nps_yz_{};

    MPI_Comm comm_;
    int rank_{-1};

    std::vector<Decomposition> decompositions_{};
};

template <std::floating_point T>
class Reshaper : public IReshaper<T, 3> {
public:
    Reshaper(ReshapePipeline<T, 3> &&rp)
        : rp_{std::move(rp)} {
        AssignReshapes(rp_.decompositions_);

        // TODO - Max possible sizeof is used here
        // But in the future it would be cool to allocate
        // only the amount we need
        this->workspace_ = MakeCudaUnique<std::byte>(
            MaxWorkspaceSize(reshapes_) * sizeof(std::complex<double>));
    };

    auto ReshapeReal(T *dest, const T *src) -> Layout<3> override {
        ApplyReshape(dest, src);

        auto &my_new_box =
            rp_.decompositions_[next_reshape_box_idx_][rp_.rank_];

        UpdateIdxs();

        GridSizes<3> new_ns = {static_cast<std::size_t>(my_new_box.size[0]),
                               static_cast<std::size_t>(my_new_box.size[1]),
                               static_cast<std::size_t>(my_new_box.size[2])};
        GridOffsets<3> new_offs = {static_cast<std::size_t>(my_new_box.low[0]),
                                   static_cast<std::size_t>(my_new_box.low[1]),
                                   static_cast<std::size_t>(my_new_box.low[2])};
        return Layout{new_ns, new_offs};
    }

    auto ReshapeCmplx(CuComplexTrait<T> *dest,
                      const CuComplexTrait<T> *src) -> Layout<3> override {
        ApplyReshape(dest, src);

        auto &my_new_box =
            rp_.decompositions_[next_reshape_box_idx_][rp_.rank_];

        UpdateIdxs();

        GridSizes<3> new_ns = {static_cast<std::size_t>(my_new_box.size[0]),
                               static_cast<std::size_t>(my_new_box.size[1]),
                               static_cast<std::size_t>(my_new_box.size[2])};
        GridOffsets<3> new_offs = {static_cast<std::size_t>(my_new_box.low[0]),
                                   static_cast<std::size_t>(my_new_box.low[1]),
                                   static_cast<std::size_t>(my_new_box.low[2])};
        return Layout{new_ns, new_offs};
    }

    auto MaxLocalElements() const -> std::size_t override {
        std::size_t max{0};

        for (const auto &decomp : rp_.decompositions_) {
            max = std::max(max,
                           static_cast<std::size_t>(decomp[rp_.rank_].count()));
        }

        return max;
    }

    auto PencilLayout() const -> Layout<3> override {
        return {rp_.pencil_sizes_, rp_.pencil_offs_};
    }

    auto SlabLayout() const -> Layout<3> override {
        return {rp_.slab_sizes_, rp_.slab_offs_};
    }

private:
    using Reshapes = std::vector<std::unique_ptr<heffte::reshape3d_base<int>>>;

    ReshapePipeline<T, 3> rp_;
    Reshapes reshapes_{};
    std::size_t next_reshape_idx_{0};
    std::size_t next_reshape_box_idx_{1};

    CudaUniquePtr<std::byte> workspace_{};

    void AssignReshapes(
        const std::vector<typename ReshapePipeline<T, 3>::Decomposition>
            &decomps) {
        using BackendTag = heffte::backend::cufft;

        // For both forward reshapes and backward reshapes
        reshapes_.resize(2 * (decomps.size() - 1));

        auto options = heffte::default_options<BackendTag>();

        for (std::size_t i{0}; i < decomps.size() - 1; ++i) {
            auto decomp_from = decomps[i];
            auto decomp_to = decomps[i + 1];

            // Forward reshape
            reshapes_[i] = heffte::make_reshape3d<BackendTag>(
                nullptr, decomp_from, decomp_to, rp_.comm_, options);

            // Backward reshape
            reshapes_[reshapes_.size() - 1 - i] =
                heffte::make_reshape3d<BackendTag>(
                    nullptr, decomp_to, decomp_from, rp_.comm_, options);
        }
    }

    auto MaxWorkspaceSize(const Reshapes &reshapes) const -> std::size_t {
        std::size_t max{0};

        for (const auto &reshape : reshapes) {
            max = std::max(max, reshape->size_workspace());
        }

        return max;
    }

    template <ValidWorkspace<T> Workspace>
    void ApplyReshape(Workspace *dest, const Workspace *src) {
        auto &next_reshape = reshapes_[next_reshape_idx_];
        if (next_reshape) {
            next_reshape->apply(
                1,
                reinterpret_cast<const HeFFTeWorkspaceTrait<Workspace> *>(src),
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(dest),
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(
                    workspace_.get()));
        }
    }

    void UpdateIdxs() {
        bool next_reshape_is_forward =
            next_reshape_idx_ < (reshapes_.size() / 2) - 1;
        bool next_reshape_is_the_last =
            next_reshape_idx_ == reshapes_.size() - 1;

        if (next_reshape_is_forward || next_reshape_is_the_last) {
            next_reshape_box_idx_++;
        } else {
            next_reshape_box_idx_--;
        }
        next_reshape_idx_++;
        next_reshape_idx_ %= reshapes_.size();
    }
};
}  // namespace poisson_solver::detail
#endif  // USE_MPI
#endif  // RESHAPER_HPP
