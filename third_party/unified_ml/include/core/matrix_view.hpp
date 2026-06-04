#pragma once

#include <cstddef>
#include <stdexcept>

namespace core {

struct MatrixView {
    const double* data{nullptr};
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t row_stride{0};

    MatrixView() = default;
    MatrixView(const double* ptr, std::size_t r, std::size_t c, std::size_t stride)
        : data(ptr), rows(r), cols(c), row_stride(stride) {}

    double operator()(std::size_t r, std::size_t c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("MatrixView: index out of range");
        return data[r * row_stride + c];
    }
};

struct MutableMatrixView {
    double* data{nullptr};
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t row_stride{0};

    MutableMatrixView() = default;
    MutableMatrixView(double* ptr, std::size_t r, std::size_t c, std::size_t stride)
        : data(ptr), rows(r), cols(c), row_stride(stride) {}

    double& operator()(std::size_t r, std::size_t c) {
        if (r >= rows || c >= cols) throw std::out_of_range("MutableMatrixView: index out of range");
        return data[r * row_stride + c];
    }
};

using Tensor2DView = MatrixView;

}  // namespace core
