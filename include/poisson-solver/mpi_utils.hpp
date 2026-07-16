#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP
#ifdef USE_MPI

#include "poisson-solver/aliases.hpp"

#include <array>
#include <cstddef>
#include <heffte.h>
#include <mpi.h>

namespace poisson_solver::mpi_utils {
// === Aliases ===
template <std::size_t Dim>
using ProcsPerDim = std::array<int, Dim>;

// === Structs ===
/**
 * struct LocalGridInfo - Stores information about local grid.
 * This includes its size and its offset from the origin.
 */
struct LocalGridInfo {
    GridSizes<3> ns{};
    GridOffsets<3> offs{};

    /**
     * @brief Factory method to convert heffte::box3d<> to LocalGridInfo.
     */
    static auto ConvertToLocalGridInfo(const heffte::box3d<> &box)
        -> LocalGridInfo;
};

// === Functions ===
/**
 * @brief Initializes MPI cartesian grid
 * with at most two axes of decomposition (the X axis is local).
 *
 * @param comm_old The old communicator.
 * @param nps Desired number of processes per dimension.
 * nps[0] corresponds to the Y axis and nps[1] to the Z axis.
 * For 1D decomposition, leave the non-decomposed axis as 1.
 * @return Cartesian grid with the specified decomposition.
 */
auto InitMPIGrid(MPI_Comm comm_old, ProcsPerDim<2> nps_yz) -> MPI_Comm;

/**
 * @brief Retrieves information about the local grid.
 *
 * @param gns Global sizes of the grid in each dimension.
 * @param cart_comm Communicator created via a call to InitMPIGrid.
 * @return Information about the local grid.
 */
auto GetLocalGridInfo(GridSizes<3> gns, MPI_Comm cart_comm) -> LocalGridInfo;

}  // namespace poisson_solver::mpi_utils

#endif  // USE_MPI
#endif  // MPI_UTILS_HPP
