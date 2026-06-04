#pragma once
//                                        
//  csc_matrix.h  — Compressed Sparse Column (CSC) DMatrix  (§3.4, paper)
//
//  Motivation:
//    Dense row-major storage wastes memory and cache bandwidth for sparse
//    datasets (one-hot encoding, bag-of-words, recommender system matrices).
//    For a 99% sparse matrix, dense storage burns 100× more bandwidth per
//    non-zero than CSC.
//
//  CSC format:
//    values[]    — non-zero feature values,   length = nnz
//    row_index[] — row index of each value,   length = nnz
//    col_ptr[]   — start of each column,      length = n_cols + 1
//
//    Column j occupies values[col_ptr[j] .. col_ptr[j+1]-1].
//    Entries within each column are sorted by row_index ascending.
//
//  Integration:
//    • CSCMatrix is the internal storage inside CSCDMatrix.
//    • HistogramBuilder::build_histograms_csc() works column-by-column,
//      reading only non-zeros — O(nnz) instead of O(n_rows × n_cols).
//    • SplitEvaluator::find_best_split_csc() iterates sorted entries,
//      accumulating (G, H) without random row lookups.
//    • Missing values are handled implicitly: rows absent from a column
//      are routed via the default_left direction (Algorithm 3).
//                                        
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <stdexcept>
#include <cassert>

namespace xgb {

//                                        
//  CSCMatrix  — raw sparse storage (not a full DMatrix, but a sub-component)
//                                        
struct CSCMatrix {
    std::vector<bst_float> values;      // non-zero feature values
    std::vector<bst_uint>  row_index;   // corresponding row indices
    std::vector<bst_ulong> col_ptr;     // col_ptr[j] = start of column j in values[]

    bst_uint n_rows{0};
    bst_uint n_cols{0};

    bst_ulong nnz() const {
        return static_cast<bst_ulong>(values.size());
    }

    // Number of non-zeros in column j
    bst_ulong col_nnz(bst_uint j) const {
        assert(j < n_cols);
        return col_ptr[j + 1] - col_ptr[j];
    }

    // Pointer to first entry of column j
    const bst_float* col_values(bst_uint j) const {
        return values.data() + col_ptr[j];
    }
    const bst_uint* col_rows(bst_uint j) const {
        return row_index.data() + col_ptr[j];
    }

    double sparsity() const {
        if (n_rows == 0 || n_cols == 0) return 0.0;
        return 1.0 - static_cast<double>(nnz()) /
               (static_cast<double>(n_rows) * n_cols);
    }
};

//                                        
//  CSCDMatrix  — DMatrix-compatible wrapper around CSCMatrix
//
//  Exposes the same interface as DMatrix for feature access, but stores data
//  in CSC format.  Designed to be accepted by HistogramBuilder and
//  SplitEvaluator via overloaded build_histograms / find_best_split methods.
//
//  Construction:
//    from_dense()   — converts dense row-major matrix (for testing)
//    from_csc()     — constructs directly from pre-built CSC arrays
//    from_libsvm()  — parses libsvm/svmlight format (common for sparse data)
//                                        
class CSCDMatrix {
public:
    CSCDMatrix() = default;

    // Build from a dense row-major matrix (mostly for unit tests)
    static CSCDMatrix from_dense(
        const std::vector<std::vector<bst_float>>& X,
        const std::vector<bst_float>& y,
        bst_float missing = 0.f);

    // Build from pre-supplied CSC arrays
    static CSCDMatrix from_csc(
        std::vector<bst_float> values,
        std::vector<bst_uint>  row_index,
        std::vector<bst_ulong> col_ptr,
        bst_uint n_rows,
        bst_uint n_cols,
        const std::vector<bst_float>& y);

    // Parse libsvm format: "label feat:val feat:val ..."
    static CSCDMatrix from_libsvm(
        const std::string& filepath,
        bst_uint n_features = 0);

    // Dimension accessors (mirrors DMatrix interface)
    bst_uint num_rows()     const { return csc_.n_rows; }
    bst_uint num_cols()     const { return csc_.n_cols; }
    bst_uint num_features() const { return csc_.n_cols; }

    const std::vector<bst_float>& labels() const { return labels_; }

    // Sparse column access (used by HistogramBuilder/SplitEvaluator)
    const CSCMatrix& csc() const { return csc_; }

    // Dense feature access for prediction (falls back to 0.0 for missing)
    bst_float feature(bst_uint row, bst_uint col) const;

    double sparsity() const { return csc_.sparsity(); }

    void print_info() const;

private:
    CSCMatrix csc_;
    std::vector<bst_float> labels_;

    // For O(1) feature() lookup: per-row sorted list of (col, value) pairs
    // Built lazily on first feature() call.  Only constructed when predictions
    // are needed; not used by histogram/split routines.
    mutable bool row_index_built_{false};
    mutable std::vector<std::vector<std::pair<bst_uint, bst_float>>> row_lookup_;

    void build_row_lookup() const;
};

//                                        
//  HistogramBuilder CSC extension — free function overload
//
//  Works directly on CSCMatrix, iterating only non-zero entries per column.
//  Zero-valued (missing) entries contribute to the "right" bin implicitly
//  (they are routed by default_left at prediction time, as per Algorithm 3).
//
//  Complexity: O(nnz) instead of O(n_rows × n_cols).
//                                        
class HistogramBuilder;   // forward decl

// missing_value: the implicit value for absent CSC entries (0.0 by default).
// The aggregate (G,H) of rows absent from a column is deposited into the bin
// that missing_value maps to, ensuring total G/H == dense-path total.
std::vector<struct FeatureHistogram> build_histograms_csc(
    const HistogramBuilder& builder,
    const CSCDMatrix& dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>& row_mask,   // 1=included, 0=excluded (length = n_rows)
    const std::vector<bst_uint>& col_indices,
    bst_float missing_value = 0.0f);

} // namespace xgb
