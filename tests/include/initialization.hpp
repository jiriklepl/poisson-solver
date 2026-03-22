#include <random>
#include <span>
#include <stdexcept>

namespace tests {
void Initialize(std::span<double> phi, std::span<double> rhs, unsigned seed) {
    if (rhs.size() != phi.size()) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }

    const std::size_t size = rhs.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double rhs_sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        phi[i] = dist(rng);
        double rhs_val = dist(rng);
        rhs[i] = rhs_val;
        rhs_sum += rhs_val;
    }

    double avg = rhs_sum / (size);
    for (std::size_t i = 0; i < size; ++i) {
        rhs[i] -= avg;
    }
}
}  // namespace tests
