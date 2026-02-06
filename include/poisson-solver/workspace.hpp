#ifndef WORKSPACE_HPP
#define WORKSPACE_HPP

#include <cstddef>
#include <vector>

namespace poisson_solver {
template <typename T>
class Workspace2D {
private:
    std::size_t rows;
    std::size_t cols;
    std::vector<T> data;

public:
    Workspace2D(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c) {}

    T& operator()(std::size_t row, std::size_t col) {
        return data[row * cols + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const {
        return data[row * cols + col];
    }

    T* ptr() { return data.data(); }

    const T* ptr() const { return data.data(); }

    std::size_t get_rows() const { return rows; }
    std::size_t get_cols() const { return cols; }
    std::size_t size() const { return data.size(); }
};
}  // namespace poisson_solver

#endif  // WORKSPACE_HPP
