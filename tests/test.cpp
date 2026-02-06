#include "poisson-solver/cuch.hpp"
#include "poisson-solver/solver.hpp"
#include "poisson-solver/workspace.hpp"

#include <exception>
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "Creating a solver\n";

    std::pair<std::size_t, std::size_t> grid_n{100, 200};

    poisson_solver::Solver2D solver(
        1.23f, 3.3f, grid_n.first, grid_n.second,
        {{poisson_solver::BoundaryCondition::kPeriodic,
          poisson_solver::BoundaryCondition::kPeriodic},
         {poisson_solver::BoundaryCondition::kPeriodic,
          poisson_solver::BoundaryCondition::kPeriodic}});

    poisson_solver::Workspace2D<float> phi{grid_n.first, grid_n.second};
    poisson_solver::Workspace2D<float> rhs{grid_n.first, grid_n.second};

    try {
        solver.Execute(phi, rhs);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::cout << "Solver creation ... OK!\n";
    return 0;
}
