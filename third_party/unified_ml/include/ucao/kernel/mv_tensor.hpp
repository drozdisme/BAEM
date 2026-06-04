#pragma once

#include "ucao/kernel/sign_table.hpp"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace ucao::kernel {

template <std::size_t Align = 64>
struct AlignedAlloc {
    [[nodiscard]] static void* alloc(std::size_t bytes) noexcept {
        if (bytes == 0) {
            return nullptr;
        }
        const std::size_t rounded = ((bytes + Align - 1) / Align) * Align;
        return std::aligned_alloc(Align, rounded);
    }

    static void free(void* p) noexcept {
        std::free(p);
    }
};

template <int D, typename Scalar = float>
class MVTensor {
public:
    enum class Layout { AoS, SoA };

    explicit MVTensor(int batch, Layout layout = Layout::SoA)
        : data_(static_cast<Scalar*>(AlignedAlloc<64>::alloc(sizeof(Scalar) * static_cast<std::size_t>(batch) * D))),
          batch_(batch),
          layout_(layout) {
        if (batch_ < 0 || data_ == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(data_, 0, sizeof(Scalar) * static_cast<std::size_t>(batch_) * D);
    }

    ~MVTensor() {
        AlignedAlloc<64>::free(data_);
    }

    MVTensor(MVTensor&& other) noexcept : data_(other.data_), batch_(other.batch_), layout_(other.layout_) {
        other.data_ = nullptr;
        other.batch_ = 0;
    }

    MVTensor(const MVTensor&) = delete;
    MVTensor& operator=(const MVTensor&) = delete;

    [[nodiscard]] __attribute__((always_inline)) Scalar& at(int b, int d) noexcept {
#ifndef NDEBUG
        assert(b >= 0 && b < batch_ && d >= 0 && d < D);
#endif
        return layout_ == Layout::AoS ? data_[static_cast<std::size_t>(b) * D + d]
                                      : data_[static_cast<std::size_t>(d) * batch_ + b];
    }

    [[nodiscard]] __attribute__((always_inline)) const Scalar& at(int b, int d) const noexcept {
#ifndef NDEBUG
        assert(b >= 0 && b < batch_ && d >= 0 && d < D);
#endif
        return layout_ == Layout::AoS ? data_[static_cast<std::size_t>(b) * D + d]
                                      : data_[static_cast<std::size_t>(d) * batch_ + b];
    }

    [[nodiscard]] Scalar* soa_row(int d) noexcept {
#ifndef NDEBUG
        assert(layout_ == Layout::SoA && d >= 0 && d < D);
#endif
        return data_ + static_cast<std::size_t>(d) * batch_;
    }

    [[nodiscard]] const Scalar* soa_row(int d) const noexcept {
#ifndef NDEBUG
        assert(layout_ == Layout::SoA && d >= 0 && d < D);
#endif
        return data_ + static_cast<std::size_t>(d) * batch_;
    }

    void transpose_to(Layout target) {
        if (target == layout_) {
            return;
        }
        Scalar* tmp = static_cast<Scalar*>(AlignedAlloc<64>::alloc(sizeof(Scalar) * static_cast<std::size_t>(batch_) * D));
        if (tmp == nullptr) {
            throw std::bad_alloc();
        }
        constexpr int TILE = 16;
        if (layout_ == Layout::AoS) {
            for (int b0 = 0; b0 < batch_; b0 += TILE) {
                for (int d0 = 0; d0 < D; d0 += TILE) {
                    const int b_max = (b0 + TILE < batch_) ? (b0 + TILE) : batch_;
                    const int d_max = (d0 + TILE < D) ? (d0 + TILE) : D;
                    for (int b = b0; b < b_max; ++b) {
                        for (int d = d0; d < d_max; ++d) {
                            tmp[static_cast<std::size_t>(d) * batch_ + b] = data_[static_cast<std::size_t>(b) * D + d];
                        }
                    }
                }
            }
        } else {
            for (int d0 = 0; d0 < D; d0 += TILE) {
                for (int b0 = 0; b0 < batch_; b0 += TILE) {
                    const int d_max = (d0 + TILE < D) ? (d0 + TILE) : D;
                    const int b_max = (b0 + TILE < batch_) ? (b0 + TILE) : batch_;
                    for (int d = d0; d < d_max; ++d) {
                        for (int b = b0; b < b_max; ++b) {
                            tmp[static_cast<std::size_t>(b) * D + d] = data_[static_cast<std::size_t>(d) * batch_ + b];
                        }
                    }
                }
            }
        }
        AlignedAlloc<64>::free(data_);
        data_ = tmp;
        layout_ = target;
    }

    [[nodiscard]] int batch() const noexcept { return batch_; }
    [[nodiscard]] Layout layout() const noexcept { return layout_; }
    [[nodiscard]] Scalar* raw() noexcept { return data_; }
    [[nodiscard]] const Scalar* raw() const noexcept { return data_; }

private:
    Scalar* data_ = nullptr;
    int batch_ = 0;
    Layout layout_ = Layout::SoA;
};

template <int N, int P, int Q>
void gp_batched(MVTensor<(1 << N)>& C,
                const MVTensor<(1 << N)>& A,
                const MVTensor<(1 << N)>& B) noexcept {
    static constexpr int D = 1 << N;
    const int batch = C.batch();
    std::memset(C.raw(), 0, static_cast<std::size_t>(batch) * D * sizeof(float));
#ifndef NDEBUG
    assert(C.layout() == MVTensor<D>::Layout::SoA);
    assert(A.layout() == MVTensor<D>::Layout::SoA);
    assert(B.layout() == MVTensor<D>::Layout::SoA);
    assert(A.batch() == batch && B.batch() == batch);
#endif
    for (int c = 0; c < D; ++c) {
        float* row_c = C.soa_row(c);
        const auto& entries = kGPTable<N, P, Q>.sparse_[c];
        const int nnz = kGPTable<N, P, Q>.nnz_[c];
        for (int idx = 0; idx < nnz; ++idx) {
            const auto e = entries[idx];
            const float* row_a = A.soa_row(e.a);
            const float* row_b = B.soa_row(e.a ^ c);
            int bi = 0;
#ifdef __AVX512F__
            const __m512 sign = _mm512_set1_ps(static_cast<float>(e.sign));
            for (; bi + 16 <= batch; bi += 16) {
                const __m512 va = _mm512_loadu_ps(row_a + bi);
                const __m512 vb = _mm512_loadu_ps(row_b + bi);
                const __m512 vc = _mm512_loadu_ps(row_c + bi);
                const __m512 prod = _mm512_mul_ps(_mm512_mul_ps(va, vb), sign);
                _mm512_storeu_ps(row_c + bi, _mm512_add_ps(vc, prod));
            }
#endif
            for (; bi < batch; ++bi) {
                row_c[bi] += static_cast<float>(e.sign) * row_a[bi] * row_b[bi];
            }
        }
    }
}

template <int N, int P, int Q>
void gp_dispatch_layout(MVTensor<(1 << N)>& C,
                        MVTensor<(1 << N)>& A,
                        MVTensor<(1 << N)>& B) {
    static constexpr int D = 1 << N;
    if (C.batch() < 16) {
        if (A.layout() != MVTensor<D>::Layout::AoS) A.transpose_to(MVTensor<D>::Layout::AoS);
        if (B.layout() != MVTensor<D>::Layout::AoS) B.transpose_to(MVTensor<D>::Layout::AoS);
        if (C.layout() != MVTensor<D>::Layout::AoS) C.transpose_to(MVTensor<D>::Layout::AoS);
        alignas(64) float lhs[D]{};
        alignas(64) float rhs[D]{};
        alignas(64) float out[D]{};
        for (int b = 0; b < C.batch(); ++b) {
            for (int i = 0; i < D; ++i) {
                lhs[i] = A.at(b, i);
                rhs[i] = B.at(b, i);
            }
            gp_single<N, P, Q>(out, lhs, rhs);
            for (int i = 0; i < D; ++i) {
                C.at(b, i) = out[i];
            }
        }
        return;
    }
    if (A.layout() != MVTensor<D>::Layout::SoA) A.transpose_to(MVTensor<D>::Layout::SoA);
    if (B.layout() != MVTensor<D>::Layout::SoA) B.transpose_to(MVTensor<D>::Layout::SoA);
    if (C.layout() != MVTensor<D>::Layout::SoA) C.transpose_to(MVTensor<D>::Layout::SoA);
    gp_batched<N, P, Q>(C, A, B);
}

} // namespace ucao::kernel
