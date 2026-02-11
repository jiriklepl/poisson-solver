#ifndef WORKSPACE_HPP
#define WORKSPACE_HPP

#include <cstddef>
#include <vector>

namespace poisson_solver {
template <typename T>
class Workspace2D {
public:
    Workspace2D(std::size_t r, std::size_t c)
        : rows_(r), cols_(c), data_(r * c) {}

    // Initializes the workspace with value `value` for all elements
    Workspace2D(std::size_t r, std::size_t c, const T& value)
        : rows_(r), cols_(c), data_(r * c, value) {}

    T& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    const T& operator()(std::size_t r, std::size_t c) const {
        return data_[r * cols_ + c];
    }

    T* Ptr() { return data_.data(); }
    const T* Ptr() const { return data_.data(); }

    std::size_t Rows() const { return rows_; }
    std::size_t Cols() const { return cols_; }

    std::size_t Size() const { return data_.size(); }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};
}  // namespace poisson_solver

#endif  // WORKSPACE_HPP
