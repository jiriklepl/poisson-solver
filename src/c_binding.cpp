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

PoissonSolver PoissonSolver1DNew(const int ns[1], const double lens[1],
                                 const BoundaryCondition bcs[2],
                                 const ps::IApproximation<double> &approx) {
    ps::ISolver *solver = nullptr;
    if (bcs[0] == kPeriodic && bcs[1] == kPeriodic) {
        solver = new ps::SolverWrapper<1, {kPP}>(
            {lens[0]}, {static_cast<std::size_t>(ns[0])}, approx);
    } else {
        throw std::invalid_argument("Invalid boundary conditions.");
    }
    return {
        .handle = solver, .grid_size = solver->GetTotalGridSize(), .dim = 1};
}

PoissonSolver PoissonSolver2DNew(const int ns[2], const double lens[2],
                                 const BoundaryCondition bcs[4],
                                 const ps::IApproximation<double> &approx) {
    ps::ISolver *solver = nullptr;
    if (bcs[0] == kPeriodic && bcs[1] == kPeriodic && bcs[2] == kPeriodic &&
        bcs[3] == kPeriodic) {
        solver = new ps::SolverWrapper<2, {kPP, kPP}>(
            {lens[0], lens[1]},
            {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
            approx);
    } else {
        throw std::invalid_argument("Invalid boundary conditions.");
    }
    return {
        .handle = solver, .grid_size = solver->GetTotalGridSize(), .dim = 2};
}

PoissonSolver PoissonSolver3DNew(const int ns[3], const double lens[3],
                                 const BoundaryCondition bcs[6],
                                 const ps::IApproximation<double> &approx) {
    ps::ISolver *solver = nullptr;
    if (bcs[0] == kPeriodic && bcs[1] == kPeriodic && bcs[2] == kPeriodic &&
        bcs[3] == kPeriodic && bcs[4] == kPeriodic && bcs[5] == kPeriodic) {
        solver = new ps::SolverWrapper<3, {kPP, kPP, kPP}>(
            {lens[0], lens[1], lens[2]},
            {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1]),
             static_cast<std::size_t>(ns[2])},
            approx);
    } else {
        throw std::invalid_argument("Invalid boundary conditions.");
    }
    return {
        .handle = solver, .grid_size = solver->GetTotalGridSize(), .dim = 3};
}

// TODO: Doesn't support all BCs
PoissonSolver PoissonSolverNew(int dims, const int *ns, const double *lens,
                               const BoundaryCondition *bcs,
                               Approximation approx) {
    std::unique_ptr<ps::IApproximation<double>> ps_approx = GetApprox(approx);

    switch (dims) {
        case 1:
            return PoissonSolver1DNew(ns, lens, bcs, *ps_approx);
        case 2:
            return PoissonSolver2DNew(ns, lens, bcs, *ps_approx);
        case 3:
            return PoissonSolver3DNew(ns, lens, bcs, *ps_approx);
        default:
            throw std::invalid_argument("Invalid amount of dimensions.");
    }
}

void PoissonSolverExecute(PoissonSolver solver, double *phi,
                          const double *rhs) {
    static_cast<ps::ISolver *>(solver.handle)
        ->Execute(std::span{phi, solver.grid_size},
                  std::span{rhs, solver.grid_size});
}

void PoissonSolverFinalize(PoissonSolver solver) {
    delete static_cast<ps::ISolver *>(solver.handle);
}
}
