#pragma once
#include "core/linalg.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kriging {

struct Point2D {
    double x = 0.0, y = 0.0;
    Point2D() = default;
    Point2D(double x_, double y_) : x(x_), y(y_) {}
};

struct Point3D {
    double x = 0.0, y = 0.0, z = 0.0;
    Point3D() = default;
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

inline double euclidean_distance(const Point2D& a, const Point2D& b) noexcept {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}
inline double euclidean_distance(const Point3D& a, const Point3D& b) noexcept {
    const double dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

template <typename PointT>
inline std::vector<std::vector<double>>
build_distance_matrix(const std::vector<PointT>& pts) {
    const std::size_t n = pts.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i+1; j < n; ++j) {
            double d = euclidean_distance(pts[i], pts[j]);
            D[i][j] = d; D[j][i] = d;
        }
    return D;
}

// ============================================================
// Variogram models
// ============================================================
class VariogramModel {
public:
    VariogramModel(double nugget, double sill, double range)
        : nugget_(nugget), sill_(sill), range_(range) {
        if (nugget < 0.0) throw std::invalid_argument("Variogram: nugget must be >= 0");
        if (sill <= nugget) throw std::invalid_argument("Variogram: sill must be > nugget");
        if (range <= 0.0)  throw std::invalid_argument("Variogram: range must be > 0");
    }
    virtual ~VariogramModel() = default;
    VariogramModel(const VariogramModel&)            = delete;
    VariogramModel& operator=(const VariogramModel&) = delete;

    virtual double semivariance(double h) const = 0;
    double covariance(double h) const { return sill_ - semivariance(h); }
    virtual std::string name() const = 0;
    double nugget() const noexcept { return nugget_; }
    double sill()   const noexcept { return sill_;   }
    double range()  const noexcept { return range_;  }
protected:
    double nugget_, sill_, range_;
};

class SphericalVariogram : public VariogramModel {
public:
    SphericalVariogram(double nugget, double sill, double range)
        : VariogramModel(nugget, sill, range) {}
    double semivariance(double h) const override {
        if (h <= 0.0) return 0.0;
        if (h >= range_) return sill_;
        const double hr = h / range_;
        return nugget_ + (sill_-nugget_) * (1.5*hr - 0.5*hr*hr*hr);
    }
    std::string name() const override { return "Spherical"; }
};

class ExponentialVariogram : public VariogramModel {
public:
    ExponentialVariogram(double nugget, double sill, double range)
        : VariogramModel(nugget, sill, range) {}
    double semivariance(double h) const override {
        if (h <= 0.0) return 0.0;
        return nugget_ + (sill_-nugget_) * (1.0 - std::exp(-h / (range_/3.0)));
    }
    std::string name() const override { return "Exponential"; }
};

class GaussianVariogram : public VariogramModel {
public:
    GaussianVariogram(double nugget, double sill, double range)
        : VariogramModel(nugget, sill, range) {}
    double semivariance(double h) const override {
        if (h <= 0.0) return 0.0;
        const double hr = h / range_;
        return nugget_ + (sill_-nugget_) * (1.0 - std::exp(-3.0*hr*hr));
    }
    std::string name() const override { return "Gaussian"; }
};

enum class VariogramType { Spherical, Exponential, Gaussian };

inline std::unique_ptr<VariogramModel>
make_variogram(VariogramType type, double nugget, double sill, double range) {
    switch (type) {
        case VariogramType::Spherical:   return std::make_unique<SphericalVariogram>(nugget, sill, range);
        case VariogramType::Exponential: return std::make_unique<ExponentialVariogram>(nugget, sill, range);
        case VariogramType::Gaussian:    return std::make_unique<GaussianVariogram>(nugget, sill, range);
    }
    throw std::invalid_argument("make_variogram: unknown VariogramType");
}

// ============================================================
// OrdinaryKriging — Best Linear Unbiased Predictor
// ============================================================
struct PredictionResult { double value = 0.0, variance = 0.0; };

class OrdinaryKriging {
public:
    explicit OrdinaryKriging(std::shared_ptr<VariogramModel> variogram)
        : variogram_(std::move(variogram)) {
        if (!variogram_) throw std::invalid_argument("OrdinaryKriging: variogram must not be null");
    }
    OrdinaryKriging(VariogramType type, double nugget, double sill, double range)
        : OrdinaryKriging(make_variogram(type, nugget, sill, range)) {}

    //   fit                                  
    // Builds the (n+1)×(n+1) kriging matrix K, then precomputes K⁻¹ via
    // Gauss-Jordan elimination.  Cost: O(n³) once at fit time.
    // Each subsequent predict() is then O(n²) instead of O(n³).
    void fit(const std::vector<Point2D>& points,
             const std::vector<double>&  values)
    {
        if (points.size() != values.size())
            throw std::invalid_argument("OrdinaryKriging::fit: points and values must be same size");
        if (points.size() < 2)
            throw std::invalid_argument("OrdinaryKriging::fit: need at least 2 training points");

        points_ = points;
        values_ = values;
        K_      = _build_kriging_matrix();
        sz_     = K_.rows();      // n + 1

        // Precompute inverse: K_inv_data_ [sz × sz], flat row-major
        _precompute_inverse();
        fitted_ = true;
    }

    //   predict                                
    double predict(const Point2D& query) const {
        return predict_with_variance(query).value;
    }

    // predict_with_variance: O(n²) — matrix-vector multiply with cached K⁻¹
    PredictionResult predict_with_variance(const Point2D& query) const {
        _require_fitted("predict_with_variance");

        const std::size_t n  = points_.size();
        const std::size_t sz = sz_;

        // Build RHS vector k(x): [cov(x, x_i), ..., 1]
        // All distances computed with SIMD-friendly loop below
        std::vector<double> rhs(sz);
        {
            const double qx = query.x, qy = query.y;
            for (std::size_t i = 0; i < n; ++i) {
                const double dx = qx - points_[i].x;
                const double dy = qy - points_[i].y;
                rhs[i] = variogram_->covariance(std::sqrt(dx*dx + dy*dy));
            }
            rhs[n] = 1.0;
        }

        // sol = K⁻¹ @ rhs  — O(n²) SIMD-vectorized matvec
        std::vector<double> sol(sz);
        const double* Kinv = K_inv_data_.data();
        for (std::size_t i = 0; i < sz; ++i) {
            double acc = 0.0;
            const double* row = Kinv + i * sz;
            const double* rp  = rhs.data();
#pragma omp simd reduction(+:acc)
            for (std::size_t j = 0; j < sz; ++j)
                acc += row[j] * rp[j];
            sol[i] = acc;
        }

        // prediction = w^T z
        double prediction = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            prediction += sol[i] * values_[i];

        // kriging variance = C(0) - w^T k - mu
        double c00  = variogram_->covariance(0.0);
        double wTk  = 0.0;
        for (std::size_t i = 0; i < sz; ++i) wTk += sol[i] * rhs[i];
        double mu       = sol[n];
        double variance = c00 - wTk - mu;
        if (variance < 0.0) variance = 0.0;

        return { prediction, variance };
    }

    bool        is_fitted() const noexcept { return fitted_; }
    std::size_t n_samples() const noexcept { return values_.size(); }
    const VariogramModel& variogram() const { return *variogram_; }

private:
    //   Build (n+1)×(n+1) kriging matrix                   
    core::Matrix _build_kriging_matrix() const {
        const std::size_t n  = points_.size();
        const std::size_t sz = n + 1;
        core::Matrix K(sz, sz, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double h = euclidean_distance(points_[i], points_[j]);
                K(i, j)  = variogram_->covariance(h);
            }
            K(i, n) = 1.0;
            K(n, i) = 1.0;
        }
        // K(n,n) = 0 already (default-constructed to 0)
        return K;
    }

    //   Gauss-Jordan inversion of K_ → K_inv_data_              
    //
    // Operates on augmented matrix [K | I] of size sz × 2·sz in flat storage.
    // After row reduction: [I | K⁻¹].  Partial pivoting for numerical stability.
    //
    // Complexity: O(sz³) = O(n³), executed ONCE in fit().
    void _precompute_inverse() {
        const std::size_t sz  = sz_;
        const std::size_t sz2 = sz * 2;

        // Build augmented [K | I]
        std::vector<double> aug(sz * sz2, 0.0);
        for (std::size_t r = 0; r < sz; ++r) {
            for (std::size_t c = 0; c < sz; ++c)
                aug[r * sz2 + c] = K_(r, c);
            aug[r * sz2 + sz + r] = 1.0;   // identity block
        }

        // Gauss-Jordan with partial pivoting
        for (std::size_t col = 0; col < sz; ++col) {
            // Find pivot row (max |value| in column)
            std::size_t pivot = col;
            double max_val = std::abs(aug[col * sz2 + col]);
            for (std::size_t r = col+1; r < sz; ++r) {
                double v = std::abs(aug[r * sz2 + col]);
                if (v > max_val) { max_val = v; pivot = r; }
            }
            if (max_val < 1e-13)
                throw std::runtime_error(
                    "OrdinaryKriging: kriging matrix is singular or near-singular "
                    "(check for duplicate points or degenerate variogram parameters)");

            // Swap rows
            if (pivot != col) {
                double* row_col   = aug.data() + col   * sz2;
                double* row_pivot = aug.data() + pivot * sz2;
                for (std::size_t c = 0; c < sz2; ++c)
                    std::swap(row_col[c], row_pivot[c]);
            }

            // Scale pivot row so pivot element → 1
            double inv_diag = 1.0 / aug[col * sz2 + col];
            {
                double* row_col = aug.data() + col * sz2;
#pragma omp simd
                for (std::size_t c = 0; c < sz2; ++c)
                    row_col[c] *= inv_diag;
            }

            // Eliminate all other rows (full Gauss-Jordan, not just forward)
            const double* pivot_row = aug.data() + col * sz2;
            for (std::size_t r = 0; r < sz; ++r) {
                if (r == col) continue;
                const double factor = aug[r * sz2 + col];
                if (factor == 0.0) continue;
                double* this_row = aug.data() + r * sz2;
#pragma omp simd
                for (std::size_t c = 0; c < sz2; ++c)
                    this_row[c] -= factor * pivot_row[c];
            }
        }

        // Extract right half [I | K⁻¹] → K_inv_data_  (flat row-major, sz×sz)
        K_inv_data_.resize(sz * sz);
        for (std::size_t r = 0; r < sz; ++r)
            for (std::size_t c = 0; c < sz; ++c)
                K_inv_data_[r * sz + c] = aug[r * sz2 + sz + c];
    }

    void _require_fitted(const char* method) const {
        if (!fitted_)
            throw std::logic_error(
                std::string("OrdinaryKriging::") + method +
                ": model has not been fitted — call fit() first");
    }

    //   State                                 
    std::shared_ptr<VariogramModel> variogram_;
    std::vector<Point2D>            points_;
    std::vector<double>             values_;
    core::Matrix                    K_;
    std::vector<double>             K_inv_data_;  // sz×sz, flat row-major
    std::size_t                     sz_    = 0;
    bool                            fitted_= false;
};

} // namespace kriging
