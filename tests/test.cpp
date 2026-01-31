#include "poisson-solver/cuch.hpp"
#include "poisson-solver/solver.hpp"

#include <exception>
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "Creating a solver\n";

    Solver2D solver(1.23f, 3.3f, 100, 200,
                    {BoundaryCondition::kNeumann, BoundaryCondition::kNeumann,
                     BoundaryCondition::kNeumann, BoundaryCondition::kNeumann});

    try {
        solver.Execute();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::cout << "Solver creation ... OK!\n";
    return 0;
}
