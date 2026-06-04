#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/utils/logger.hpp"
#include <numeric>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace xgb {

std::shared_ptr<DMatrix> DMatrix::from_dense(
    const std::vector<std::vector<bst_float>>& X,
    const std::vector<bst_float>& y)
{
    if (X.empty()) throw std::runtime_error("DMatrix: empty X");
    auto dm = std::make_shared<DMatrix>();
    dm->page_.n_rows = static_cast<bst_uint>(X.size());
    dm->page_.n_cols = static_cast<bst_uint>(X[0].size());
    dm->page_.data.reserve(dm->page_.n_rows * dm->page_.n_cols);
    for (const auto& row : X)
        for (bst_float v : row)
            dm->page_.data.push_back(v);
    dm->labels_ = y;
    return dm;
}

std::shared_ptr<DMatrix> DMatrix::from_matrix_view(
    const core::MatrixView& X,
    const std::vector<bst_float>& y)
{
    if (!X.data || X.rows == 0) throw std::runtime_error("DMatrix: empty X");
    if (X.cols == 0) throw std::runtime_error("DMatrix: zero columns");
    auto dm = std::make_shared<DMatrix>();
    dm->page_.n_rows = static_cast<bst_uint>(X.rows);
    dm->page_.n_cols = static_cast<bst_uint>(X.cols);
    dm->page_.data.resize(X.rows * X.cols);
    for (std::size_t r = 0; r < X.rows; ++r) {
        for (std::size_t c = 0; c < X.cols; ++c) {
            dm->page_.data[r * X.cols + c] = static_cast<bst_float>(X(r, c));
        }
    }
    dm->labels_ = y;
    return dm;
}

std::shared_ptr<DMatrix> DMatrix::from_flat(
    const std::vector<bst_float>& X_flat,
    bst_uint n_rows, bst_uint n_cols,
    const std::vector<bst_float>& y)
{
    auto dm = std::make_shared<DMatrix>();
    dm->page_.data   = X_flat;
    dm->page_.n_rows = n_rows;
    dm->page_.n_cols = n_cols;
    dm->labels_ = y;
    return dm;
}

const std::vector<std::vector<bst_uint>>& DMatrix::sorted_col_indices() const {
    if (!col_idx_built_) build_col_sorted_index();
    return sorted_col_idx_;
}

void DMatrix::build_col_sorted_index() const {
    bst_uint n_rows = page_.n_rows;
    bst_uint n_cols = page_.n_cols;
    sorted_col_idx_.resize(n_cols);
    std::vector<bst_uint> row_ids(n_rows);
    std::iota(row_ids.begin(), row_ids.end(), 0);

    for (bst_uint c = 0; c < n_cols; ++c) {
        sorted_col_idx_[c] = row_ids;
        std::sort(sorted_col_idx_[c].begin(), sorted_col_idx_[c].end(),
            [&](bst_uint a, bst_uint b) {
                return page_.at(a, c) < page_.at(b, c);
            });
    }
    col_idx_built_ = true;
}

void DMatrix::compute_statistics() {
    bst_uint n_rows = page_.n_rows;
    bst_uint n_cols = page_.n_cols;
    feat_mean_.assign(n_cols, 0.f);
    feat_std_.assign(n_cols, 0.f);

    for (bst_uint c = 0; c < n_cols; ++c) {
        double sum = 0.0;
        for (bst_uint r = 0; r < n_rows; ++r)
            sum += page_.at(r, c);
        feat_mean_[c] = static_cast<bst_float>(sum / n_rows);

        double var = 0.0;
        for (bst_uint r = 0; r < n_rows; ++r) {
            double diff = page_.at(r, c) - feat_mean_[c];
            var += diff * diff;
        }
        feat_std_[c] = static_cast<bst_float>(std::sqrt(var / n_rows));
    }
}

bst_float DMatrix::feature_mean(bst_uint col) const {
    if (feat_mean_.empty())
        throw std::runtime_error("compute_statistics() not called");
    return feat_mean_[col];
}

bst_float DMatrix::feature_std(bst_uint col) const {
    if (feat_std_.empty())
        throw std::runtime_error("compute_statistics() not called");
    return feat_std_[col];
}

void DMatrix::print_info() const {
    std::cout << "DMatrix: " << page_.n_rows << " rows × "
              << page_.n_cols << " features";
    if (has_labels())  std::cout << "  labels: " << labels_.size();
    if (has_weights()) std::cout << "  weights: " << weights_.size();
    std::cout << "\n";
}

} // namespace xgb
