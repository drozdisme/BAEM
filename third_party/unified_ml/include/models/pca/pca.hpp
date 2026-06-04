#pragma once
#include "core/linalg.hpp"
#include "core/matrix_view.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pca {

class PCA {
public:
    explicit PCA(std::size_t n_components = 0);

    void           fit(const core::Matrix& X);
    void           fit(const core::MatrixView& X);
    core::Matrix   transform(const core::Matrix& X, std::size_t n_components = 0) const;
    core::Matrix   transform(const core::MatrixView& X, std::size_t n_components = 0) const;
    core::Matrix   fit_transform(const core::Matrix& X, std::size_t n_components = 0);
    core::Matrix   inverse_transform(const core::Matrix& Z) const;

    const core::Matrix& components()         const;
    const core::Vector& explained_variance() const;
    core::Vector        explained_variance_ratio() const;
    const core::Vector& mean()               const;

    void save(const std::string& filepath) const;
    static PCA load(const std::string& filepath);

    std::size_t n_components() const noexcept { return n_components_; }
    std::size_t n_features()   const noexcept { return n_features_;   }
    bool        is_fitted()    const noexcept { return fitted_;        }

private:
    void check_fitted() const;
    core::Matrix center(const core::Matrix& X) const;

    std::size_t  n_components_ = 0;
    std::size_t  n_features_   = 0;
    core::Vector mean_;
    core::Matrix components_;
    core::Vector eigenvalues_;
    double       total_variance_ = 0.0;
    bool         fitted_         = false;
};

} // namespace pca
