#include <concepts>
#include <random>
#include <span>
#include <stdexcept>

namespace tests {
template <std::floating_point T>
void Initialize(std::span<T> phi, std::span<T> rhs, unsigned seed) {
    if (rhs.size() != phi.size()) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }

    const std::size_t size = rhs.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    T rhs_sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        phi[i] = dist(rng);
        T rhs_val = dist(rng);
        rhs[i] = rhs_val;
        rhs_sum += rhs_val;
    }

    T avg = rhs_sum / (size);
    for (std::size_t i = 0; i < size; ++i) {
        rhs[i] -= avg;
    }
}
}  // namespace tests
