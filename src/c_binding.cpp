#include "poisson-solver/c_binding.h"

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/c_solver_wrapper.hpp"

#include <array>
#include <concepts>
#include <memory>
#include <span>
#include <stdexcept>

// Here is the mapping between the C interface and C++ implementation.

namespace ps = poisson_solver;

namespace {
template <std::floating_point T>
std::unique_ptr<ps::IApproximation<T>> GetApprox(Approximation approx) {
    switch (approx) {
        case kSpectral:
            return std::make_unique<ps::SpectralApproximation<T>>();
        case kFD2:
            return std::make_unique<ps::FiniteDifference2Approximation<T>>();
        case kFD4:
            return std::make_unique<ps::FiniteDifference4Approximation<T>>();
        default:
            throw std::invalid_argument("Invalid approximation.");
    }
}
}  // namespace

extern "C" {
ps::BoundaryCondition ToCxxBC(BoundaryCondition c_bc) {
    using CxxBC = ps::BoundaryCondition;

    constexpr auto cxxP = CxxBC::kPeriodic;
    constexpr auto cxxD = CxxBC::kDirichlet;
    constexpr auto cxxN = CxxBC::kNeumann;
    constexpr auto cxxDS = CxxBC::kDirichletStag;
    constexpr auto cxxNS = CxxBC::kNeumannStag;

    switch (c_bc) {
        case kPeriodic:
            return cxxP;
        case kDirichlet:
            return cxxD;
        case kNeumann:
            return cxxN;
        case kDirichletStag:
            return cxxDS;
        case kNeumannStag:
            return cxxNS;
        default:
            throw std::invalid_argument("Unknown boundary condition.");
    }
}

ps::BCPair ToCxxBCPair(BoundaryCondition c_bc1, BoundaryCondition c_bc2) {
    return ps::BCPair{ToCxxBC(c_bc1), ToCxxBC(c_bc2)};
}

// === DOUBLE VERSIONS ===

void PoissonSolver1DNew(PoissonSolver *solver_out,
                        const int ns[1],
                        const double lens[1],
                        const BoundaryCondition bcs[2],
                        const ps::IApproximation<double> &approx) {
    ps::BCPair bcp = ToCxxBCPair(bcs[0], bcs[1]);
    ps::ISolver<double> *solver_handle = new ps::SolverWrapper<double, 1>(
        {static_cast<std::size_t>(ns[0])}, {lens[0]}, {bcp}, approx);
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = 1,
                   .nz = 1};
}

void PoissonSolver2DNew(PoissonSolver *solver_out,
                        const int ns[2],
                        const double lens[2],
                        const BoundaryCondition bcs[4],
                        const ps::IApproximation<double> &approx) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::ISolver<double> *solver_handle = new ps::SolverWrapper<double, 2>(
        {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
        {lens[0], lens[1]},
        {bcp1, bcp2},
        approx);
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = 1};
}

void PoissonSolver3DNew(PoissonSolver *solver_out,
                        const int ns[3],
                        const double lens[3],
                        const BoundaryCondition bcs[6],
                        const ps::IApproximation<double> &approx) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::BCPair bcp3 = ToCxxBCPair(bcs[4], bcs[5]);
    ps::ISolver<double> *solver_handle =
        new ps::SolverWrapper<double, 3>({static_cast<std::size_t>(ns[0]),
                                          static_cast<std::size_t>(ns[1]),
                                          static_cast<std::size_t>(ns[2])},
                                         {lens[0], lens[1], lens[2]},
                                         {bcp1, bcp2, bcp3},
                                         approx);
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = static_cast<std::size_t>(ns[2])};
}

void PoissonSolverNew(PoissonSolver *solver_out,
                      int dims,
                      const int *ns,
                      const double *lens,
                      const BoundaryCondition *bcs,
                      Approximation approx) {
    std::unique_ptr<ps::IApproximation<double>> ps_approx =
        GetApprox<double>(approx);

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

void PoissonSolverExecute(PoissonSolver *solver,
                          double *phi,
                          const double *rhs) {
    static_cast<ps::ISolver<double> *>(solver->handle)
        ->Execute(std::span{phi, solver->grid_size},
                  std::span{rhs, solver->grid_size});
}

void PoissonSolverFinalize(PoissonSolver *solver) {
    delete static_cast<ps::ISolver<double> *>(solver->handle);
    solver->handle = nullptr;
    solver->grid_size = 0;
}

// === FLOAT VERSIONS ===

void PoissonSolver1DNewFloat(PoissonSolver *solver_out,
                             const int ns[1],
                             const float lens[1],
                             const BoundaryCondition bcs[2],
                             const ps::IApproximation<float> &approx) {
    ps::BCPair bcp = ToCxxBCPair(bcs[0], bcs[1]);
    ps::ISolver<float> *solver_handle = new ps::SolverWrapper<float, 1>(
        {static_cast<std::size_t>(ns[0])}, {lens[0]}, {bcp}, approx);
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = 1,
                   .nz = 1};
}

void PoissonSolver2DNewFloat(PoissonSolver *solver_out,
                             const int ns[2],
                             const float lens[2],
                             const BoundaryCondition bcs[4],
                             const ps::IApproximation<float> &approx) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::ISolver<float> *solver_handle = new ps::SolverWrapper<float, 2>(
        {static_cast<std::size_t>(ns[0]), static_cast<std::size_t>(ns[1])},
        {lens[0], lens[1]},
        {bcp1, bcp2},
        approx);
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = 1};
}

void PoissonSolver3DNewFloat(PoissonSolver *solver_out,
                             const int ns[3],
                             const float lens[3],
                             const BoundaryCondition bcs[6],
                             const ps::IApproximation<float> &approx) {
    ps::BCPair bcp1 = ToCxxBCPair(bcs[0], bcs[1]);
    ps::BCPair bcp2 = ToCxxBCPair(bcs[2], bcs[3]);
    ps::BCPair bcp3 = ToCxxBCPair(bcs[4], bcs[5]);
    ps::ISolver<float> *solver_handle =
        new ps::SolverWrapper<float, 3>({static_cast<std::size_t>(ns[0]),
                                         static_cast<std::size_t>(ns[1]),
                                         static_cast<std::size_t>(ns[2])},
                                        {lens[0], lens[1], lens[2]},
                                        {bcp1, bcp2, bcp3},
                                        approx);
    *solver_out = {.handle = solver_handle,
                   .grid_size = solver_handle->GetTotalGridSize(),
                   .nx = static_cast<std::size_t>(ns[0]),
                   .ny = static_cast<std::size_t>(ns[1]),
                   .nz = static_cast<std::size_t>(ns[2])};
}

void PoissonSolverNewFloat(PoissonSolver *solver_out,
                           int dims,
                           const int *ns,
                           const float *lens,
                           const BoundaryCondition *bcs,
                           Approximation approx) {
    std::unique_ptr<ps::IApproximation<float>> ps_approx =
        GetApprox<float>(approx);

    switch (dims) {
        case 1:
            PoissonSolver1DNewFloat(solver_out, ns, lens, bcs, *ps_approx);
            break;
        case 2:
            PoissonSolver2DNewFloat(solver_out, ns, lens, bcs, *ps_approx);
            break;
        case 3:
            PoissonSolver3DNewFloat(solver_out, ns, lens, bcs, *ps_approx);
            break;
        default:
            throw std::invalid_argument("Invalid amount of dimensions.");
    }
}

void PoissonSolverExecuteFloat(PoissonSolver *solver,
                               float *phi,
                               const float *rhs) {
    static_cast<ps::ISolver<float> *>(solver->handle)
        ->Execute(std::span{phi, solver->grid_size},
                  std::span{rhs, solver->grid_size});
}

void PoissonSolverFinalizeFloat(PoissonSolver *solver) {
    delete static_cast<ps::ISolver<float> *>(solver->handle);
    solver->handle = nullptr;
    solver->grid_size = 0;
}
}
