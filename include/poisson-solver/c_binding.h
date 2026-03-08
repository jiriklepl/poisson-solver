#ifndef C_BINDING_H
#define C_BINDING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct PoissonSolver {
    void *handle;
    uint64_t grid_size;
    unsigned dim;
} PoissonSolver;

typedef enum {
    kPeriodic,
    kDirichlet,
    kNeumann,
    kDirichletStag,
    kNeumannStag
} BoundaryCondition;

typedef enum { kSpectral, kFD2, kFD4 } Approximation;

PoissonSolver PoissonSolverNew(int dims, const int *ns, const double *lens,
                               const BoundaryCondition *bcs,
                               Approximation approx);

void PoissonSolverExecute(PoissonSolver solver, double *phi, const double *rhs);

void PoissonSolverFinalize(PoissonSolver solver);
#ifdef __cplusplus
}
#endif

#endif  // C_BINDING_H
