#ifdef USE_MPI
#include "poisson-solver/mpi_utils.hpp"

#include "poisson-solver/detail/mpi_check.hpp"

#include <cstddef>
#include <functional>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>

namespace poisson_solver::mpi_utils {
void CheckCartCommCanBeCreated(MPI_Comm comm, ProcsPerDim<2> nps) {
    auto size{0};
    MPI_CHECK(MPI_Comm_size(comm, &size));
    if (std::accumulate(
            nps.begin(), nps.end(), std::size_t{1}, std::multiplies<>()) !=
        static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            "Communicator size differs from the product of all processed "
            "into which the communicator should be decomposed");
    }
}

auto InitMPIGrid(MPI_Comm comm_old, ProcsPerDim<2> nps) -> MPI_Comm {
    CheckCartCommCanBeCreated(comm_old, nps);

    MPI_Comm cart_comm{};
    ProcsPerDim<3> nps_3d{1, nps[0], nps[1]};
    std::array<int, 3> periods{false, false, false};

    MPI_CHECK(MPI_Cart_create(
        comm_old, 3, nps_3d.data(), periods.data(), false, &cart_comm));

    return cart_comm;
}
}  // namespace poisson_solver::mpi_utils

#endif  // USE_MPI
