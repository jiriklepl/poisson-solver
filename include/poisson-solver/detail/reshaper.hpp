#ifndef RESHAPER_HPP
#define RESHAPER_HPP
#ifdef USE_MPI

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/mpi_utils.hpp"

#include <concepts>
#include <cstddef>
#include <memory>
#include <mpi.h>
#include <vector>

// TODO - See what is not templated and move it to src file
// TODO - Docs

namespace poisson_solver::detail {
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

class ReshapePipeline {
public:
    friend class Reshaper;

    static auto From(GridSizes<3> ns, MPI_Comm cart_comm) -> ReshapePipeline;
    auto ToPencils(std::size_t dimension) -> ReshapePipeline &;
    auto ToSlabs(std::size_t dimension1,
                 std::size_t dimension2) -> ReshapePipeline &;
    auto MaxLocalElements() const -> std::size_t;

private:
    using Decomposition = std::vector<heffte::box3d<>>;

    heffte::box3d<> world_;
    mpi_utils::ProcsPerDim<2> nps_yz_{};

    MPI_Comm comm_;
    int rank_{-1};

    std::vector<Decomposition> decompositions_{};

    ReshapePipeline(Decomposition &&original_boxes,
                    mpi_utils::ProcsPerDim<2> &&nps_yz,
                    MPI_Comm comm);

    static auto GetGridBox(GridSizes<Dim> ns,
                           GridOffsets<Dim> offs) -> heffte::box3d<>;
};

class Reshaper {
public:
    Reshaper(ReshapePipeline &&rp);

    template <std::floating_point T, ValidWorkspace<T> Workspace>
    auto Reshape(const Workspace *src, Workspace *dest) const -> GridSizes<3> {
        ApplyReshape(src, dest);
        auto &my_new_box =
            rp_.decompositions_[next_reshape_box_idx_][rp_.rank_].size;
        UpdateIdxs();
        return {static_cast<std::size_t>(my_new_box[0]),
                static_cast<std::size_t>(my_new_box[1]),
                static_cast<std::size_t>(my_new_box[2])};
    }

private:
    using Reshapes = std::vector<std::unique_ptr<heffte::reshape3d_base<int>>>;

    ReshapePipeline rp_;
    Reshapes reshapes_{};
    std::size_t next_reshape_idx_{0};
    std::size_t next_reshape_box_idx_{1};

    CudaUniquePtr<std::byte> workspace_{};

    void AssignReshapes(
        const std::vector<ReshapePipeline::Decomposition> &decomps);

    static auto MaxWorkspaceSize(Reshapes reshapes) -> std::size_t;

    template <std::floating_point T, ValidWorkspace<T> Workspace>
    void ApplyReshape(const Workspace *src, Workspace *dest) {
        auto &next_reshape = reshapes_[next_reshape_idx_];
        if (next_reshape) {
            next_reshape->apply(
                1,
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(src),
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(dest),
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(
                    workspace_.get()));
        }
    }

    void UpdateIdxs();
};
}  // namespace poisson_solver::detail

#else

namespace poisson_solver::detail {
// TODO - Dummy classes and methods for non-mpi reshaper
}

#endif  // USE_MPI
#endif  // RESHAPER_HPP
