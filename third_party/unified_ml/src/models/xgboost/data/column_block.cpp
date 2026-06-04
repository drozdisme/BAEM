//                                        
//  column_block.cpp  — ColumnBlockStore implementation
//
//  Build procedure for each feature f:
//    1. Iterate over rows in chunks of kColumnBlockSize.
//    2. Fill entries[] with (value, row_idx) pairs.
//    3. Sort the chunk ascending by value (std::sort, O(B log B)).
//    4. Store in blocks_[f].
//
//  This means each block is independently sorted.  When SplitEvaluator walks
//  blocks for feature f, it reads them in order, accumulating (G, H) left-to-
//  right, as if processing one continuous sorted column — but with cache-
//  friendly sequential memory access within each block.
//
//  Memory: O(n_features × n_rows × 8 bytes) — same as a column-sorted index.
//                                        
#include "models/xgboost/data/column_block.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include <algorithm>
#include <numeric>

namespace xgb {

void ColumnBlockStore::build(const DMatrix& dm, bst_uint block_size) {
    const bst_uint n_feat = dm.num_features();
    const bst_uint n_rows = dm.num_rows();

    blocks_.resize(n_feat);

    for (bst_uint f = 0; f < n_feat; ++f) {
        blocks_[f].clear();

        // Iterate over row-chunks of size block_size
        for (bst_uint row_begin = 0; row_begin < n_rows; row_begin += block_size) {
            bst_uint row_end = std::min(row_begin + block_size, n_rows);

            ColumnBlock blk;
            blk.feature_idx = f;
            blk.row_begin   = row_begin;
            blk.row_end     = row_end;
            blk.entries.reserve(row_end - row_begin);

            // Fill (value, row) pairs for this chunk
            for (bst_uint r = row_begin; r < row_end; ++r) {
                blk.entries.push_back({ dm.feature(r, f), r });
            }

            // Sort ascending by feature value — enables sequential split scan
            std::sort(blk.entries.begin(), blk.entries.end());

            blocks_[f].push_back(std::move(blk));
        }
    }
}

} // namespace xgb
