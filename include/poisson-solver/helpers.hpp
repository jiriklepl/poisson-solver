#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cstddef>
#include <numbers>
#include <vector>

float eig_fn_spectral(float x);

// TODO: Incomplete, doesn't work for all possible scenarios
void CalculateEigenvalues(float grid_len, std::size_t grid_n,
                          std::vector<float> &eigenvalues);

#endif  // HELPERS_HPP
