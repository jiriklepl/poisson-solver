#include "poisson-solver/detail/c_binding_for_fortran.h"

extern "C" {
void PoissonSolverNewForFortran(PoissonSolver *solver_out,
                                int dims,
                                const int *ns,
                                const double *lens,
                                const BoundaryCondition *bcs,
                                Approximation approx,
                                const int *gns,
                                const int *offs,
#ifdef USE_MPI
                                const MPI_Fint *mpi_comm
#else
                                const void *mpi_comm
#endif  // USE_MPI
) {
#ifdef USE_MPI
    MPI_Comm c_comm;
    const void *c_comm_ptr = nullptr;
    if (mpi_comm) {
        c_comm = MPI_Comm_f2c(*mpi_comm);
        c_comm_ptr = (void*)&c_comm;
    }
#else
    const void *c_comm_ptr = mpi_comm;
#endif  // USE_MPI
    PoissonSolverNew(
        solver_out, dims, ns, lens, bcs, approx, gns, offs, c_comm_ptr);
}

void PoissonSolverNewFloatForFortran(PoissonSolver *solver_out,
                                     int dims,
                                     const int *ns,
                                     const float *lens,
                                     const BoundaryCondition *bcs,
                                     Approximation approx,
                                     const int *gns,
                                     const int *offs,
#ifdef USE_MPI
                                     const MPI_Fint *mpi_comm
#else
                                     const void *mpi_comm
#endif  // USE_MPI
) {
#ifdef USE_MPI
    MPI_Comm c_comm;
    const void *c_comm_ptr = nullptr;
    if (mpi_comm) {
        c_comm = MPI_Comm_f2c(*mpi_comm);
        c_comm_ptr = (void*)&c_comm;
    }
#else
    const void *c_comm_ptr = mpi_comm;
#endif  // USE_MPI
    PoissonSolverNewFloat(
        solver_out, dims, ns, lens, bcs, approx, gns, offs, c_comm_ptr);
}
}
