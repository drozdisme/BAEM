//                                        
//  csc_matrix.cpp  — CSCDMatrix implementation
//                                        
#include "models/xgboost/data/csc_matrix.hpp"
#include "models/xgboost/tree/histogram_builder.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <stdexcept>

namespace xgb {

//                                        
//  from_dense  — convert dense row-major matrix → CSC
//
//  Algorithm:
//    Phase 1: count non-zeros per column → col_ptr (as counts first).
//    Phase 2: prefix-sum col_ptr → CSC offsets.
//    Phase 3: fill values / row_index by scanning rows.
//    Phase 4: sort each column's entries by row_index (for binary search).
//                                        
CSCDMatrix CSCDMatrix::from_dense(
    const std::vector<std::vector<bst_float>>& X,
    const std::vector<bst_float>& y,
    bst_float missing)
{
    if (X.empty()) return {};
    const bst_uint n_rows = static_cast<bst_uint>(X.size());
    const bst_uint n_cols = static_cast<bst_uint>(X[0].size());

    // Phase 1: count nnz per column
    std::vector<bst_ulong> counts(n_cols, 0);
    for (bst_uint r = 0; r < n_rows; ++r)
        for (bst_uint c = 0; c < n_cols; ++c)
            if (X[r][c] != missing) ++counts[c];

    // Phase 2: prefix sum → col_ptr
    std::vector<bst_ulong> col_ptr(n_cols + 1, 0);
    for (bst_uint c = 0; c < n_cols; ++c)
        col_ptr[c + 1] = col_ptr[c] + counts[c];

    const bst_ulong total_nnz = col_ptr[n_cols];
    std::vector<bst_float> values(total_nnz);
    std::vector<bst_uint>  row_index(total_nnz);

    // Phase 3: fill entries using write pointers
    std::vector<bst_ulong> write_pos = col_ptr;  // copy of col_ptr for writing
    for (bst_uint r = 0; r < n_rows; ++r) {
        for (bst_uint c = 0; c < n_cols; ++c) {
            if (X[r][c] != missing) {
                bst_ulong pos = write_pos[c]++;
                values[pos]    = X[r][c];
                row_index[pos] = r;
            }
        }
    }
    // Each column's entries are already in row-ascending order because we
    // iterated rows outer-most.

    CSCDMatrix dm;
    dm.csc_.values    = std::move(values);
    dm.csc_.row_index = std::move(row_index);
    dm.csc_.col_ptr   = std::move(col_ptr);
    dm.csc_.n_rows    = n_rows;
    dm.csc_.n_cols    = n_cols;
    dm.labels_        = y;
    return dm;
}

//                                        
//  from_csc  — direct construction from pre-built CSC arrays
//                                        
CSCDMatrix CSCDMatrix::from_csc(
    std::vector<bst_float> values,
    std::vector<bst_uint>  row_index,
    std::vector<bst_ulong> col_ptr,
    bst_uint n_rows,
    bst_uint n_cols,
    const std::vector<bst_float>& y)
{
    assert(col_ptr.size() == static_cast<size_t>(n_cols + 1));
    CSCDMatrix dm;
    dm.csc_.values    = std::move(values);
    dm.csc_.row_index = std::move(row_index);
    dm.csc_.col_ptr   = std::move(col_ptr);
    dm.csc_.n_rows    = n_rows;
    dm.csc_.n_cols    = n_cols;
    dm.labels_        = y;
    return dm;
}

//                                        
//  from_libsvm  — parse "label feat:val feat:val ..." format
//
//  LibSVM is the canonical format for sparse datasets (MNIST, RCV1, etc.).
//  We use a two-pass approach: first collect all (row, col, val) triples,
//  then sort by column and build CSC arrays.
//                                        
CSCDMatrix CSCDMatrix::from_libsvm(
    const std::string& filepath,
    bst_uint n_features)
{
    struct Triple { bst_uint row, col; bst_float val; };
    std::vector<Triple> triples;
    std::vector<bst_float> labels;
    bst_uint actual_max_col = 0;

    std::ifstream ifs(filepath);
    if (!ifs) throw std::runtime_error("Cannot open: " + filepath);

    std::string line;
    bst_uint row = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        bst_float label;
        ss >> label;
        labels.push_back(label);

        std::string token;
        while (ss >> token) {
            auto colon = token.find(':');
            if (colon == std::string::npos) continue;
            bst_uint col = static_cast<bst_uint>(std::stoul(token.substr(0, colon)));
            bst_float val = std::stof(token.substr(colon + 1));
            // LibSVM uses 1-based feature indices; convert to 0-based
            if (col > 0) --col;
            triples.push_back({ row, col, val });
            actual_max_col = std::max(actual_max_col, col + 1);
        }
        ++row;
    }

    const bst_uint n_rows = row;
    const bst_uint n_cols = (n_features > 0) ? n_features : actual_max_col;

    // Sort triples by (col, row) for CSC construction
    std::sort(triples.begin(), triples.end(),
        [](const Triple& a, const Triple& b) {
            return a.col < b.col || (a.col == b.col && a.row < b.row);
        });

    // Build CSC
    std::vector<bst_float> values(triples.size());
    std::vector<bst_uint>  row_index(triples.size());
    std::vector<bst_ulong> col_ptr(n_cols + 1, 0);

    for (size_t i = 0; i < triples.size(); ++i) {
        values[i]    = triples[i].val;
        row_index[i] = triples[i].row;
        ++col_ptr[triples[i].col + 1];
    }
    // Convert counts to prefix sums
    for (bst_uint c = 0; c < n_cols; ++c)
        col_ptr[c + 1] += col_ptr[c];

    return from_csc(std::move(values), std::move(row_index),
                    std::move(col_ptr), n_rows, n_cols, labels);
}

//                                        
//  feature()  — O(log nnz_per_row) dense lookup via row_lookup_ cache
//                                        
void CSCDMatrix::build_row_lookup() const {
    row_lookup_.assign(csc_.n_rows, {});
    for (bst_uint c = 0; c < csc_.n_cols; ++c) {
        bst_ulong start = csc_.col_ptr[c];
        bst_ulong end   = csc_.col_ptr[c + 1];
        for (bst_ulong k = start; k < end; ++k)
            row_lookup_[csc_.row_index[k]].emplace_back(c, csc_.values[k]);
    }
    // Sort each row's entry list by column index for binary search
    for (auto& row : row_lookup_)
        std::sort(row.begin(), row.end());
    row_index_built_ = true;
}

bst_float CSCDMatrix::feature(bst_uint row, bst_uint col) const {
    if (!row_index_built_) build_row_lookup();
    const auto& entries = row_lookup_[row];
    // Binary search for column
    auto it = std::lower_bound(entries.begin(), entries.end(),
        std::make_pair(col, 0.f),
        [](const std::pair<bst_uint,bst_float>& a,
           const std::pair<bst_uint,bst_float>& b) {
            return a.first < b.first;
        });
    if (it != entries.end() && it->first == col)
        return it->second;
    return 0.f;  // missing → treated as 0.0 (or handled by default_left)
}

void CSCDMatrix::print_info() const {
    std::cout << "CSCDMatrix: " << csc_.n_rows << " rows × "
              << csc_.n_cols << " cols, nnz=" << csc_.nnz()
              << " (sparsity=" << csc_.sparsity() * 100.0 << "%)\n";
}

//                                        
//  build_histograms_csc  — O(nnz + n_active) histogram construction
//
//  Standard build_histograms iterates rows × cols = O(n × d).
//  This version iterates only non-zeros per column — O(nnz) for the hot path.
//
//  Missing-value handling (critical correctness property):
//    Rows absent from column j carry the implicit missing_value (0.0 for
//    zero-stripped CSC). XGBoost does NOT discard these from histograms.
//    Their aggregate (G, H) is deposited into the bin that missing_value
//    maps to, so the total G/H per node equals the dense-path total.
//
//    Algorithm (O(n_active) pre-compute + O(nnz) scatter + O(log B) deposit):
//      1. Compute G_active / H_active once by summing over the row_mask.
//      2. Per column: scatter non-zero (value,row) pairs into histogram bins,
//         accumulating G_nonzero / H_nonzero.
//      3. G_missing = G_active − G_nonzero (residual from absent rows).
//      4. Bin-search missing_value (0.0) → deposit residual into that bin.
//
//    Guarantees:
//      ∀ column f:  Σ_b hist[f].bins[b].sum_grad == G_active   ✓
//                   Σ_b hist[f].bins[b].sum_hess == H_active   ✓
//                                        
std::vector<FeatureHistogram> build_histograms_csc(
    const HistogramBuilder& builder,
    const CSCDMatrix& dm,
    const std::vector<GradientPair>& grads,
    const std::vector<bst_uint>& row_mask,
    const std::vector<bst_uint>& col_indices,
    bst_float missing_value)
{
    const CSCMatrix& csc        = dm.csc();
    const auto&      cut_points = builder.cut_points();
    const bst_uint   n_mask     = static_cast<bst_uint>(row_mask.size());

    std::vector<FeatureHistogram> hists(csc.n_cols);

    // Initialise bin arrays for requested columns only
    for (bst_uint f : col_indices) {
        if (f >= static_cast<bst_uint>(cut_points.size())) continue;
        bst_uint n_bins = static_cast<bst_uint>(cut_points[f].size()) + 1;
        hists[f].bins.assign(n_bins, HistogramBin{});
        hists[f].cut_points = cut_points[f];
    }

    //   Step 1: total active gradient statistics (computed once)      
    double   G_active = 0.0, H_active = 0.0;
    bst_uint n_active = 0;
    for (bst_uint r = 0; r < n_mask; ++r) {
        if (!row_mask[r]) continue;
        G_active += grads[r].grad;
        H_active += grads[r].hess;
        ++n_active;
    }

    //   Steps 2–4: per-column scatter + missing-value deposit        
    for (bst_uint f : col_indices) {
        if (f >= csc.n_cols) continue;

        const bst_ulong col_start = csc.col_ptr[f];
        const bst_ulong col_end   = csc.col_ptr[f + 1];
        auto& hist = hists[f];

        // Step 2: scatter non-zero entries
        double   G_nonzero = 0.0, H_nonzero = 0.0;
        bst_uint n_nonzero = 0;

        for (bst_ulong k = col_start; k < col_end; ++k) {
            bst_uint row = csc.row_index[k];
            if (row >= n_mask || !row_mask[row]) continue;

            bst_float val = csc.values[k];
            const auto& cuts = hist.cut_points;
            bst_uint bin = static_cast<bst_uint>(
                std::upper_bound(cuts.begin(), cuts.end(), val) - cuts.begin());
            bin = std::min(bin, static_cast<bst_uint>(hist.bins.size() - 1));

            hist.bins[bin].add(grads[row]);
            G_nonzero += grads[row].grad;
            H_nonzero += grads[row].hess;
            ++n_nonzero;
        }

        // Step 3: residual for missing/zero rows
        const bst_uint n_missing = n_active - n_nonzero;
        if (n_missing == 0) continue;

        // Step 4: binary-search bin for missing_value, deposit residual
        const auto& cuts       = hist.cut_points;
        bst_uint missing_bin   = static_cast<bst_uint>(
            std::upper_bound(cuts.begin(), cuts.end(), missing_value) - cuts.begin());
        missing_bin = std::min(missing_bin,
                               static_cast<bst_uint>(hist.bins.size() - 1));

        hist.bins[missing_bin].sum_grad  += G_active - G_nonzero;
        hist.bins[missing_bin].sum_hess  += H_active - H_nonzero;
        hist.bins[missing_bin].count     += n_missing;
    }

    return hists;
}

} // namespace xgb