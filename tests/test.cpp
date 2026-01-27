#include "poisson-solver/solver.hpp"
#include "poisson-solver/cuch.hpp"

#include <iostream>
#include <exception>

int main(int argc, char *argv[]) {
    std::cout << "Creating a solver\n";

    Solver2D solver(
        1.23f, 3.3f, 100, 200,
        {BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic,
         BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic});

    try {
        solver.Solve();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::cout << "Solver creation ... OK!\n";
    return 0;
}
