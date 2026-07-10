#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP
#ifdef USE_MPI

#include "poisson-solver/aliases.hpp"

#include <array>
#include <cstddef>
#include <mpi.h>

namespace poisson_solver::mpi_utils {
// === Aliases ===
template <std::size_t Dim>
using ProcsPerDim = std::array<int, Dim>;

// === Structs ===
// TODO - Struct to return from LocalGridSize
template <std::size_t Dim>
struct LocalGridInfo {
    GridSizes<Dim> ns{};
    GridOffsets<Dim> offs{};

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
auto InitMPIGrid(MPI_Comm comm_old, ProcsPerDim<2> nps) -> MPI_Comm;

// TODO - The communicator has to be created via InitMPIGrid
auto GetLocalGridInfo(GridSizes<3> ngs, MPI_Comm cart_comm) -> LocalGridInfo<3> {
}

}  // namespace poisson_solver::mpi_utils

#endif  // USE_MPI
#endif  // MPI_UTILS_HPP
