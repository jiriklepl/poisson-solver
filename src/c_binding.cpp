#include "poisson-solver/c_binding.h"

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/c_solver_wrapper.hpp"

#include <array>
#include <memory>
#include <span>
#include <stdexcept>

// Here is the mapping between the C interface and C++ implementation.
// Most of it comes down to translating compile time C++ errors into runtime
// errors. This implementation is not really pretty but serves its purpose for
// now.

extern "C" {
namespace ps = poisson_solver;
using BC = ps::BoundaryCondition;

constexpr BC kP = BC::kPeriodic;
constexpr BC kD = BC::kDirichlet;
constexpr BC kN = BC::kNeumann;
constexpr BC kDS = BC::kDirichletStag;
constexpr BC kNS = BC::kNeumannStag;

constexpr ps::BCPair kPP = {kP, kP};

std::unique_ptr<ps::IApproximation<double>> GetApprox(Approximation approx) {
    switch (approx) {
        case kSpectral:
            return std::make_unique<ps::SpectralApproximation<double>>();
        case kFD2:
            return std::make_unique<
                ps::FiniteDifference2Approximation<double>>();
        case kFD4:
            throw std::logic_error("Not yet implemented");
            // return std::make_unique<
            //     ps::FiniteDifference4Approximation<double>>();
        default:
            throw std::invalid_argument("Invalid approximation.");
    }
}

void PoissonSolver1DNew(PoissonSolver* solver_out, const int ns[1],
                        const double lens[1], const BoundaryCondition bcs[2],
                        const ps::IApproximation<double>& approx) {
    ps::ISolver* solver_handle = nullptr;
    if (bcs[0] == kPeriodic && bcs[1] == kPeriodic) {
        solver_handle = new ps::SolverWrapper<1, {kPP}>(
            {static_cast<std::size_t>(ns[0])}, {lens[0]}, approx);
    } else {
        throw std::invalid_argument("Invalid boundary conditions.");
    }
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize()};
}

void PoissonSolver2DNew(PoissonSolver* solver_out, const int ns[2],
                        const double lens[2], const BoundaryCondition bcs[4],
                        const ps::IApproximation<double>& approx) {
    ps::ISolver* solver_handle = nullptr;
    if (bcs[0] == kPeriodic && bcs[1] == kPeriodic && bcs[2] == kPeriodic &&
        bcs[3] == kPeriodic) {
        solver_handle = new ps::SolverWrapper<2, {kPP, kPP}>(
            {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
            {lens[0], lens[1]}, approx);
    } else {
        throw std::invalid_argument("Invalid boundary conditions.");
    }
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize()};
}

void PoissonSolver3DNew(PoissonSolver* solver_out, const int ns[3],
                        const double lens[3], const BoundaryCondition bcs[6],
                        const ps::IApproximation<double>& approx) {
    ps::ISolver* solver_handle = nullptr;
    if (bcs[0] == kPeriodic && bcs[1] == kPeriodic && bcs[2] == kPeriodic &&
        bcs[3] == kPeriodic && bcs[4] == kPeriodic && bcs[5] == kPeriodic) {
        solver_handle = new ps::SolverWrapper<3, {kPP, kPP, kPP}>(
            {static_cast<std::size_t>(ns[0]),
             static_cast<std::size_t>(ns[1]),
             static_cast<std::size_t>(ns[2])},
             {lens[0], lens[1], lens[2]},
            approx);
    } else {
        throw std::invalid_argument("Invalid boundary conditions.");
    }
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize()};
}

// TODO: Doesn't support all BCs
void PoissonSolverNew(PoissonSolver* solver_out, int dims, const int* ns,
                      const double* lens, const BoundaryCondition* bcs,
                      Approximation approx) {
    std::unique_ptr<ps::IApproximation<double>> ps_approx = GetApprox(approx);

    switch (dims) {
        case 1:
            PoissonSolver1DNew(solver_out, ns, lens, bcs, *ps_approx);
            break;
        case 2:
            PoissonSolver2DNew(solver_out, ns, lens, bcs, *ps_approx);
            break;
        case 3:
            PoissonSolver3DNew(solver_out, ns, lens, bcs, *ps_approx);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions.");
    }
}

void PoissonSolverExecute(PoissonSolver* solver, double* phi,
                          const double* rhs) {
    static_cast<ps::ISolver*>(solver->handle)
        ->Execute(std::span{phi, solver->grid_size},
                  std::span{rhs, solver->grid_size});
}

void PoissonSolverFinalize(PoissonSolver* solver) {
    delete static_cast<ps::ISolver*>(solver->handle);
    solver->handle = nullptr;
    solver->grid_size = 0;
}
}
