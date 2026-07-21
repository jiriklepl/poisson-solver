#ifdef USE_MPI
#include "poisson-solver/detail/reshaper.hpp"

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/mpi_check.hpp"
#include "poisson-solver/mpi_utils.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <heffte.h>
#include <mpi.h>
#include <utility>
#include <vector>

namespace poisson_solver::detail {
auto ReshapePipeline::From(GridSizes<3> ns,
                           GridOffsets<3> offs,
                           MPI_Comm cart_comm) -> ReshapePipeline {
    MPI_CHECK(MPI_Comm_rank(comm_, &rank_));

    heffte::box3d<> my_box = GetGridBox(ns, offs);
    auto all_boxes = heffte::mpi::gather_boxes(my_box, my_box, cart_comm);

    auto nps_xyz = mpi_utils::GetProcGrid(cart_comm);
    mpi_utils::ProcsPerDim<2> nps_yz = {nps_xyz[1], nps_xyz[2]};

    return ReshapePipeline{all_boxes, nps_yz, cart_comm};
}

auto ReshapePipeline::ToPencils(std::size_t dimension) -> ReshapePipeline {
    auto num_procs = nps_yz_[0] * nps_yz_[1];

    // TODO - Currently pretty much unused because we preserve
    // the original order. This probably runs slower in case
    // the FFT is in non-contiguous dimension but idk.
    // Later get rid of this or make it work with reversing the order.
    // auto pencil_order = GetOrder(sl.isolated_dim_idx);
    // auto slab_order = GetOrder(pencil_order[1]);

    // Resulting decomp by reshaping last decomp into pencils
    // in the direction of `dimension`
    decompositions_.emplace_back(heffte::make_pencils(
        world_, nps_yz_, dimension, decompositions_.back(), {0, 1, 2}));

    return *this;
}

auto ReshapePipeline::ToSlabs(std::size_t dimension1,
                              std::size_t dimension2) -> ReshapePipeline {
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
    return *this;
}

auto ReshapePipeline::MaxLocalElements() const -> std::size_t {
    std::size_t max{0};

    for (const auto &decomp : decompositions_) {
        max = std::max(max, static_cast<std::size_t>(decomp[rank_].count()));
    }

    return max;
}

ReshapePipeline::ReshapePipeline(Decomposition &&original_boxes,
                                 mpi_utils::ProcsPerDim<2> &&nps_yz,
                                 MPI_Comm comm)
    : world_(heffte::find_world(original_boxes.in)),
      nps_yz_{std::move(nps_yz)},
      comm_{comm} {
    decompositions_.emplace_back(std::move(original_boxes));
}

auto ReshapePipeline::GetGridBox(GridSizes<3> ns,
                                 GridOffsets<3> offs) -> heffte::box3d<> {
    auto begin_x = offs[0];
    auto begin_y = offs[1];
    auto begin_z = offs[2];

    auto end_x = begin_x + ns[0] - 1;
    auto end_y = (begin_y + ns[1] - 1);
    auto end_z = (begin_z + ns[2] - 1);

    std::array<int, 3> low = {static_cast<int>(begin_x),
                              static_cast<int>(begin_y),
                              static_cast<int>(begin_z)};
    std::array<int, 3> high = {static_cast<int>(end_x),
                               static_cast<int>(end_y),
                               static_cast<int>(end_z)};

    return heffte::box3d<>{low, high};
}

Reshaper::Reshaper(ReshapePipeline &&rp)
    : rp_{std::move(rp)} {
    AssignReshapes(rp_.decompositions_);

    // TODO - jakou size pouzit tady? Nemame forward typy
    // Pozadovat nejakej layout jako param?
    this->workspace_ = MakeCudaUnique<std::byte>(MaxWorkspaceSize(reshapes_) *
                                                 sizeof(std::complex<double>));
};

void Reshaper::AssignReshapes(
    const std::vector<ReshapePipeline::Decomposition> &decomps) {
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

auto Reshaper::MaxWorkspaceSize(const Reshapes &reshapes) -> std::size_t {
    std::size_t max{0};

    for (const auto &reshape : reshapes) {
        max = std::max(max, reshape->size_workspace());
    }

    return max;
}

void Reshaper::UpdateIdxs() {
    bool next_reshape_is_forward =
        next_reshape_idx_ < (reshapes_.size() / 2) - 1;
    bool next_reshape_is_the_last = next_reshape_idx_ == reshapes_.size() - 1;

    if (next_reshape_is_forward || next_reshape_is_the_last) {
        next_reshape_box_idx_++;
    } else {
        next_reshape_box_idx_--;
    }
    next_reshape_idx_++;
    next_reshape_idx_ %= reshapes_.size();
}
}  // namespace poisson_solver::detail
#endif  // USE_MPI
