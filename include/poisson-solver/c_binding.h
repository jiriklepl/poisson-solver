#ifndef C_BINDING_H
#define C_BINDING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct PoissonSolver {
    void *handle;
    size_t grid_size;
    size_t nx, ny, nz;
} PoissonSolver;

typedef enum {
    kPeriodic,
    kDirichlet,
    kNeumann,
    kDirichletStag,
    kNeumannStag
} BoundaryCondition;

typedef enum { kSpectral = 0, kFD2 = 2, kFD4 = 4 } Approximation;

void PoissonSolverNew(PoissonSolver *solver_out,
                      int dims,
                      const int *ns,
                      const double *lens,
                      const BoundaryCondition *bcs,
                      Approximation approx);

void PoissonSolverExecute(PoissonSolver *solver,
                          double *phi,
                          const double *rhs);

void PoissonSolverFinalize(PoissonSolver *solver);

void PoissonSolverNewFloat(PoissonSolver *solver_out,
                           int dims,
                           const int *ns,
                           const float *lens,
                           const BoundaryCondition *bcs,
                           Approximation approx);

void PoissonSolverExecuteFloat(PoissonSolver *solver,
                               float *phi,
                               const float *rhs);

void PoissonSolverFinalizeFloat(PoissonSolver *solver);
#ifdef __cplusplus
}
#endif

#endif  // C_BINDING_H
