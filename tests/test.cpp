#include "poisson-solver/solver.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "Creating a solver\n";

    Solver2D solver(
        1.23f, 3.3f, 100, 200,
        {BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic,
         BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic});

    solver.Solve();

    std::cout << "Solver creation ... OK!\n";
    return 0;
}
