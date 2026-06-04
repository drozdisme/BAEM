#pragma once

#include <algorithm>
#include "core/compat.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

//   SIMD / alignment helpers                          
// All hot-path buffers must be 64-byte aligned (AVX-512 requirement).
// Use AlignedVec<T> everywhere instead of std::vector<T> for performance.
#ifndef CACHE_LINE
#  define CACHE_LINE 64
#endif

// L1 tile size for cache-blocked matrix multiply (doubles).
// 64 × 64 × 8 = 32768 bytes  → fits in 32 KB L1 data cache with room to spare.
#ifndef TILE_K
#  define TILE_K 64
#  define TILE_N 64
#  define TILE_M 32
#endif

namespace core {

//   AlignedVec                                 
// Drop-in replacement for std::vector<double> that guarantees 64-byte alignment.
struct AlignedVec {
    double* ptr  = nullptr;
    std::size_t n = 0;

    AlignedVec() = default;
    explicit AlignedVec(std::size_t size, double val = 0.0)
        : n(size)
    {
        if (n == 0) return;
        if (n > (std::numeric_limits<std::size_t>::max() - (CACHE_LINE - 1)) / sizeof(double)) {
            throw std::length_error("AlignedVec: requested size is too large");
        }
        std::size_t bytes = ((n * sizeof(double) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
        ptr = static_cast<double*>(hpc_aligned_alloc(CACHE_LINE, bytes));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        std::fill_n(ptr, n, val);
    }
    AlignedVec(const AlignedVec& o) : n(o.n) {
        if (n == 0) { ptr = nullptr; return; }
        if (n > (std::numeric_limits<std::size_t>::max() - (CACHE_LINE - 1)) / sizeof(double)) {
            throw std::length_error("AlignedVec: requested size is too large");
        }
        std::size_t bytes = ((n * sizeof(double) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
        ptr = static_cast<double*>(hpc_aligned_alloc(CACHE_LINE, bytes));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        std::copy(o.ptr, o.ptr + n, ptr);
    }
    AlignedVec(AlignedVec&& o) noexcept : ptr(o.ptr), n(o.n) { o.ptr = nullptr; o.n = 0; }
    AlignedVec& operator=(const AlignedVec& o) {
        if (this == &o) return *this;
        AlignedVec tmp(o); std::swap(ptr, tmp.ptr); std::swap(n, tmp.n); return *this;
    }
    AlignedVec& operator=(AlignedVec&& o) noexcept {
        hpc_aligned_free(ptr); ptr = o.ptr; n = o.n; o.ptr = nullptr; o.n = 0; return *this;
    }
    ~AlignedVec() { hpc_aligned_free(ptr); }

    double&       operator[](std::size_t i)       { return ptr[i]; }
    const double& operator[](std::size_t i) const { return ptr[i]; }
    std::size_t size() const noexcept { return n; }
    double* data()             noexcept { return ptr; }
    const double* data() const noexcept { return ptr; }
    void zero() noexcept { if (ptr) std::memset(ptr, 0, n * sizeof(double)); }
};

//   Low-level tiled matmul kernel                        
// C[M×N] += A[M×K] × B[K×N]   (accumulate into C, caller zeros C first)
// All pointers must be 64-byte aligned; strides are the logical column counts.
void matmul_tiled(const double* HPC_RESTRICT A, const double* HPC_RESTRICT B,
                  double* HPC_RESTRICT C,
                  std::size_t M, std::size_t K, std::size_t N,
                  std::size_t lda, std::size_t ldb, std::size_t ldc) noexcept;

// ============================================================================
// Vector
// ============================================================================
class Vector {
public:
    Vector() = default;
    explicit Vector(std::size_t n, double val = 0.0) : data_(n, val) {}
    Vector(std::initializer_list<double> il)
        : data_(il.size()) {
        std::size_t i = 0;
        for (double v : il) data_[i++] = v;
    }

    std::size_t size()  const noexcept { return data_.size(); }
    bool        empty() const noexcept { return data_.size() == 0; }

    double&       operator[](std::size_t i)       { return data_[i]; }
    const double& operator[](std::size_t i) const { return data_[i]; }
    double&       at(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("Vector::at");
        return data_[i];
    }
    const double& at(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("Vector::at");
        return data_[i];
    }

    Vector  operator+(const Vector& rhs) const;
    Vector  operator-(const Vector& rhs) const;
    Vector  operator*(double s)          const;
    Vector  operator/(double s)          const;
    Vector  operator-()                  const;
    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);
    Vector& operator*=(double s);

    double dot(const Vector& rhs) const;
    double norm_sq() const { return dot(*this); }
    double norm()    const { return std::sqrt(norm_sq()); }
    Vector normalized() const;

    // Access underlying aligned buffer
    double*       data()       noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    // Compatibility with code that calls .data() returning std::vector
    // (returns a copy — used only in non-hot paths)
    std::vector<double> to_std_vector() const {
        return std::vector<double>(data_.ptr, data_.ptr + data_.n);
    }

    [[deprecated("Use format() and let the application own output handling.")]]
    void print(const std::string& label = "", int precision = 6) const;
    std::string format(const std::string& label = "", int precision = 6) const;

private:
    AlignedVec data_;

    void check_same(const Vector& o, const char* op) const {
        if (data_.size() != o.data_.size())
            throw std::invalid_argument(std::string("Vector::") + op + ": size mismatch");
    }
};

inline Vector operator*(double s, const Vector& v) { return v * s; }

// ============================================================================
// Matrix  (row-major, 64-byte aligned storage)
// ============================================================================
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rows, std::size_t cols, double val = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, val) {}

    static Matrix identity(std::size_t n);
    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix from_rows(const std::vector<Vector>& rows);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    bool        empty() const noexcept { return data_.size() == 0; }

    double& operator()(std::size_t r, std::size_t c)       { return at(r, c); }
    const double& operator()(std::size_t r, std::size_t c) const { return at(r, c); }
    double&       at(std::size_t r, std::size_t c);
    const double& at(std::size_t r, std::size_t c) const;

    Vector row(std::size_t r) const;
    Vector col(std::size_t c) const;
    void set_row(std::size_t r, const Vector& v);
    void set_col(std::size_t c, const Vector& v);

    // Arithmetic — use tiled kernel internally for M*V and M*M
    Matrix  operator+(const Matrix& rhs) const;
    Matrix  operator-(const Matrix& rhs) const;
    Matrix  operator*(const Matrix& rhs) const;   // tiled matmul
    Vector  operator*(const Vector& v)   const;   // tiled matvec
    Matrix  operator*(double s)          const;
    Matrix  operator/(double s)          const;
    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);

    Matrix transpose() const;
    Matrix submatrix(std::size_t r0, std::size_t c0,
                     std::size_t r1, std::size_t c1) const;

    Vector solve(const Vector& b) const;

    // Raw storage access (64-byte aligned)
    double*       storage()       noexcept { return data_.data(); }
    const double* storage() const noexcept { return data_.data(); }

    // Legacy compatibility
    std::vector<double> to_std_vector() const {
        return std::vector<double>(data_.ptr, data_.ptr + data_.n);
    }

    [[deprecated("Use format() and let the application own output handling.")]]
    void print(const std::string& label = "", int precision = 6) const;
    std::string format(const std::string& label = "", int precision = 6) const;

private:
    std::size_t rows_, cols_;
    AlignedVec  data_;    // row-major, 64-byte aligned
};

inline Matrix operator*(double s, const Matrix& m) { return m * s; }

// ============================================================================
// Jacobi eigendecomposition (real symmetric matrices)
// ============================================================================
struct EigenResult {
    Vector eigenvalues;   // length n, descending order
    Matrix eigenvectors;  // n×n, columns are eigenvectors
};

EigenResult jacobi_eigen(const Matrix& A,
                         int    max_sweeps = 200,
                         double tol        = 1e-12);

void sort_eigen(Vector& eigenvalues, Matrix& eigenvectors);

} // namespace core
