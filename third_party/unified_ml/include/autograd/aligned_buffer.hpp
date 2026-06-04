#pragma once

#include "core/linalg.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace autograd {

class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t n, double value = 0.0)
        : storage_(n, value) {}

    explicit AlignedBuffer(const std::vector<double>& src)
        : storage_(src.size()) {
        std::copy(src.begin(), src.end(), storage_.data());
    }

    explicit AlignedBuffer(std::vector<double>&& src)
        : storage_(src.size()) {
        std::copy(src.begin(), src.end(), storage_.data());
    }

    std::size_t size() const noexcept { return storage_.size(); }
    bool empty() const noexcept { return size() == 0; }

    double* data() noexcept { return storage_.data(); }
    const double* data() const noexcept { return storage_.data(); }

    double& operator[](std::size_t i) noexcept { return storage_[i]; }
    const double& operator[](std::size_t i) const noexcept { return storage_[i]; }

    std::vector<double> to_std_vector() const {
        return std::vector<double>(storage_.data(), storage_.data() + storage_.size());
    }

private:
    core::AlignedVec storage_;
};

using SharedAlignedBuffer = std::shared_ptr<AlignedBuffer>;

class VectorCompatProxy {
public:
    VectorCompatProxy() = default;
    explicit VectorCompatProxy(std::shared_ptr<AlignedBuffer> storage)
        : storage_(std::move(storage)) {}

    std::size_t size() const noexcept { return storage_ ? storage_->size() : 0; }
    bool empty() const noexcept { return size() == 0; }

    double* data() noexcept { return storage_ ? storage_->data() : nullptr; }
    const double* data() const noexcept { return storage_ ? storage_->data() : nullptr; }

    double& operator[](std::size_t i) {
        if (!storage_) throw std::runtime_error("VectorCompatProxy: no storage");
        return (*storage_)[i];
    }
    const double& operator[](std::size_t i) const {
        if (!storage_) throw std::runtime_error("VectorCompatProxy: no storage");
        return (*storage_)[i];
    }

    operator std::vector<double>() const {
        if (!storage_) return {};
        return storage_->to_std_vector();
    }

    const double* begin() const noexcept { return data(); }
    const double* end() const noexcept { return data() ? data() + size() : nullptr; }

private:
    std::shared_ptr<AlignedBuffer> storage_;
};

}  // namespace autograd
