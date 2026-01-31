#include "poisson-solver/helpers.hpp"

float eig_fn_spectral(float x) { return -4.0 * x * x; }

void CalculateEigenvalues(float grid_len, std::size_t grid_n,
                          std::vector<float> &eigenvalues) {
    eigenvalues.resize(grid_n);

    float grid_dx = grid_len / grid_n;
    float dkx = std::numbers::pi * grid_dx / grid_len;

    float inv_grid_dx_squared = 1.0f / (grid_dx * grid_dx);

    for (std::size_t i = 0; i < grid_n; ++i) {
        if (i < grid_n / 2) {
            eigenvalues[i] = eig_fn_spectral(i * dkx);
        } else {
            eigenvalues[i] = eig_fn_spectral((grid_n - i) * dkx);
        }

        eigenvalues[i] *= inv_grid_dx_squared;
    }
}
