#pragma once
#include "models/xgboost/core/types.hpp"
#include "core/matrix_view.hpp"
#include <vector>
#include <string>
#include <optional>
#include <memory>

namespace xgb {

//                        
//  SparsePage  (future: CSC storage for column blocks)
//  For now holds a dense row-major matrix.
//  Architecture mirrors XGBoost's SparsePage / CSCMatrix.
//                        
struct SparsePage {
    std::vector<bst_float> data;   // row-major: data[row * n_cols + col]
    bst_uint n_rows{0};
    bst_uint n_cols{0};

    bst_float at(bst_uint row, bst_uint col) const {
        return data[row * n_cols + col];
    }
    bst_float& at(bst_uint row, bst_uint col) {
        return data[row * n_cols + col];
    }
};

//                        
//  DMatrix
//  Primary dataset abstraction.
//  Holds features (X), labels (y), optional weights,
//  optional group info for ranking.
//
//  Future extensions:
//    - external memory (block-based) support
//    - column-sorted index for exact greedy (Sec. 4.1)
//    - missing-value mask
//                        
class DMatrix {
public:
    DMatrix() = default;

    // Construction               
    static std::shared_ptr<DMatrix> from_dense(
        const std::vector<std::vector<bst_float>>& X,
        const std::vector<bst_float>& y);
    static std::shared_ptr<DMatrix> from_matrix_view(
        const core::MatrixView& X,
        const std::vector<bst_float>& y);

    static std::shared_ptr<DMatrix> from_flat(
        const std::vector<bst_float>& X_flat,
        bst_uint n_rows,
        bst_uint n_cols,
        const std::vector<bst_float>& y);

    // Dimensions                
    bst_uint num_rows()     const { return page_.n_rows; }
    bst_uint num_cols()     const { return page_.n_cols; }
    bst_uint num_features() const { return page_.n_cols; }

    // Data access                
    bst_float feature(bst_uint row, bst_uint col) const {
        return page_.at(row, col);
    }
    const std::vector<bst_float>& labels()  const { return labels_; }
    const std::vector<bst_float>& weights() const { return weights_; }

    bool has_labels()  const { return !labels_.empty(); }
    bool has_weights() const { return !weights_.empty(); }

    // Column-sorted index (pre-sorted for exact greedy, Sec. 3.1 / 4.1)
    // sorted_indices_[j] = list of row indices sorted ascending by feature j
    const std::vector<std::vector<bst_uint>>& sorted_col_indices() const;

    void set_weights(const std::vector<bst_float>& w) { weights_ = w; }
    void set_feature_names(const std::vector<std::string>& names) {
        feature_names_ = names;
    }
    const std::vector<std::string>& feature_names() const {
        return feature_names_;
    }

    // Statistics                
    void compute_statistics();
    bst_float feature_mean(bst_uint col) const;
    bst_float feature_std(bst_uint col)  const;

    void print_info() const;

private:
    SparsePage page_;
    std::vector<bst_float> labels_;
    std::vector<bst_float> weights_;
    std::vector<std::string> feature_names_;

    // Cached sorted indices (lazy, built on first access)
    mutable std::vector<std::vector<bst_uint>> sorted_col_idx_;
    mutable bool col_idx_built_{false};

    // Per-feature stats (mean, std)
    std::vector<bst_float> feat_mean_;
    std::vector<bst_float> feat_std_;

    void build_col_sorted_index() const;
};

} // namespace xgb
