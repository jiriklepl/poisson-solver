#ifndef C_BINDING_H
#define C_BINDING_H

// Turn off C++ modernization rules for this pure C header
// NOLINTBEGIN(modernize-*)

#include <stddef.h>
#include <stdint.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct PoissonSolver {
    void *handle;
    size_t grid_size;
    size_t dim;
    size_t nx, ny, nz;
} PoissonSolver;

typedef enum {
    Periodic,
    Dirichlet,
    Neumann,
    DirichletStag,
    NeumannStag
} BoundaryCondition;

typedef enum { Spectral = 0, FD2 = 2, FD4 = 4 } Approximation;

// === Uniform Solvers ===
void PoissonSolverNew(PoissonSolver *solver_out,
                      int dims,
                      const int *ns,
                      const double *lens,
                      const BoundaryCondition *bcs,
                      Approximation approx,
                      const int *gns,
                      const int *offs,
                      const void *mpi_comm);

void PoissonSolverExecute(PoissonSolver *solver,
                          double *phi,
                          const double *rhs,
                          const int *ngs_phi,
                          const int *ngs_rhs);

void PoissonSolverExecuteGPU(PoissonSolver *solver,
                             double *d_phi,
                             const double *d_rhs);

void PoissonSolverFinalize(PoissonSolver *solver);

void PoissonSolverNewFloat(PoissonSolver *solver_out,
                           int dims,
                           const int *ns,
                           const float *lens,
                           const BoundaryCondition *bcs,
                           Approximation approx,
                           const int *gns,
                           const int *offs,
                           const void *mpi_comm);

void PoissonSolverExecuteFloat(PoissonSolver *solver,
                               float *phi,
                               const float *rhs,
                               const int *ngs_phi,
                               const int *ngs_rhs);

void PoissonSolverExecuteGPUFloat(PoissonSolver *solver,
                                  float *d_phi,
                                  const float *d_rhs);

void PoissonSolverFinalizeFloat(PoissonSolver *solver);

// === Nonuniform Solvers ===
// `z` has to be of length `ns[2]` and `z_u` of `ns[2] + 1`
// `lens` should contain only the lengths in X and Y dimension
void PoissonSolverNonuniformZNew(PoissonSolver *solver_out,
                                 const int ns[3],
                                 const double lens[2],
                                 const double *z,
                                 const double *z_u,
                                 const BoundaryCondition bcs[6],
                                 Approximation approx,
                                 const int *gns,
                                 const int *offs,
                                 const void *mpi_comm);

void PoissonSolverNonuniformZExecute(PoissonSolver *solver,
                                     double *phi,
                                     const double *rhs,
                                     const int *ngs_phi,
                                     const int *ngs_rhs);

void PoissonSolverNonuniformZExecuteGPU(PoissonSolver *solver,
                                        double *d_phi,
                                        const double *d_rhs);

void PoissonSolverNonuniformZFinalize(PoissonSolver *solver);

void PoissonSolverNonuniformZNewFloat(PoissonSolver *solver_out,
                                      const int *ns,
                                      const float *lens,
                                      const float *z,
                                      const float *z_u,
                                      const BoundaryCondition *bcs,
                                      Approximation approx,
                                      const int *gns,
                                      const int *offs,
                                      const void *mpi_comm);

void PoissonSolverNonuniformZExecuteFloat(PoissonSolver *solver,
                                          float *phi,
                                          const float *rhs,
                                          const int *ngs_phi,
                                          const int *ngs_rhs);

void PoissonSolverNonuniformZExecuteGPUFloat(PoissonSolver *solver,
                                             float *d_phi,
                                             const float *d_rhs);

void PoissonSolverNonuniformZFinalizeFloat(PoissonSolver *solver);

#ifdef USE_MPI
void PoissonSolverInitMPIGrid(MPI_Comm comm_old,
                              const int *nps,
                              MPI_Comm *comm_out);

void PoissonSolverGetLocalGridInfo(const int *gns,
                                   MPI_Comm cart_comm,
                                   int *ns,
                                   int *offs);
#endif

#ifdef __cplusplus
}
#endif

// NOLINTEND(modernize-*)

#endif  // C_BINDING_H
