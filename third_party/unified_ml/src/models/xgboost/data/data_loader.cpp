#include "models/xgboost/data/data_loader.hpp"
#include "models/xgboost/utils/csv_reader.hpp"
#include "models/xgboost/utils/logger.hpp"
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <random>

namespace xgb {

std::shared_ptr<DMatrix> DataLoader::load_csv(
    const std::string& path,
    const CSVLoadOptions& opts)
{
    utils::CSVReader reader(path, opts.delimiter);
    if (!reader.has_next())
        throw std::runtime_error("Empty CSV: " + path);

    std::vector<std::string> header;
    if (opts.has_header)
        header = reader.header();

    auto all_rows = reader.read_all();
    if (all_rows.empty())
        throw std::runtime_error("No data rows in CSV: " + path);

    bst_uint n_cols_total = static_cast<bst_uint>(all_rows[0].size());
    if (n_cols_total == 0)
        throw std::runtime_error("CSV has no columns");

    // Determine label column index
    int label_col = opts.label_column;
    if (label_col < 0)
        label_col = static_cast<int>(n_cols_total) - 1;

    bst_uint n_feature_cols = n_cols_total - 1;  // all except label
    bst_uint max_rows = (opts.max_rows > 0)
        ? std::min(opts.max_rows, static_cast<bst_uint>(all_rows.size()))
        : static_cast<bst_uint>(all_rows.size());

    std::vector<bst_float> X_flat;
    std::vector<bst_float> y;
    X_flat.reserve(max_rows * n_feature_cols);
    y.reserve(max_rows);

    for (bst_uint r = 0; r < max_rows; ++r) {
        const auto& row = all_rows[r];
        bst_uint feat_idx = 0;
        for (bst_uint c = 0; c < n_cols_total; ++c) {
            bst_float val = (c < row.size() && !row[c].empty())
                ? utils::CSVReader::parse_float(row[c])
                : opts.missing_value;
            if (static_cast<int>(c) == label_col)
                y.push_back(val);
            else {
                X_flat.push_back(std::isnan(val) ? opts.missing_value : val);
                ++feat_idx;
            }
        }
    }

    auto dm = DMatrix::from_flat(X_flat, max_rows, n_feature_cols, y);

    // Set feature names (header minus label column)
    if (!header.empty()) {
        std::vector<std::string> feat_names;
        for (bst_uint c = 0; c < n_cols_total; ++c)
            if (static_cast<int>(c) != label_col && c < header.size())
                feat_names.push_back(header[c]);
        dm->set_feature_names(feat_names);
    }

    XGB_LOG_INFO("Loaded " + path + " → " +
                 std::to_string(dm->num_rows()) + " rows, " +
                 std::to_string(dm->num_features()) + " features");
    return dm;
}

void DataLoader::save_csv(
    const std::string& path,
    const std::vector<bst_float>& predictions,
    const std::vector<std::string>& col_names)
{
    utils::CSVWriter writer(path);
    writer.write_header(col_names);
    for (bst_float v : predictions) {
        writer.write_row(std::vector<bst_float>{v});
    }
    XGB_LOG_INFO("Saved predictions → " + path);
}

void DataLoader::save_feature_importance(
    const std::string& path,
    const std::vector<std::string>& feature_names,
    const std::vector<bst_float>& importance)
{
    utils::CSVWriter writer(path);
    writer.write_header({"feature", "importance"});
    for (size_t i = 0; i < importance.size(); ++i) {
        std::string name = (i < feature_names.size())
            ? feature_names[i]
            : "f" + std::to_string(i);
        writer.write_row({name, std::to_string(importance[i])});
    }
    XGB_LOG_INFO("Saved feature importance → " + path);
}

std::pair<std::shared_ptr<DMatrix>, std::shared_ptr<DMatrix>>
DataLoader::train_test_split(
    const std::shared_ptr<DMatrix>& dm,
    float test_ratio,
    bst_uint seed)
{
    bst_uint n = dm->num_rows();
    bst_uint n_test = static_cast<bst_uint>(n * test_ratio);
    bst_uint n_train = n - n_test;

    // Shuffle indices
    std::vector<bst_uint> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    auto make_subset = [&](bst_uint from, bst_uint to) {
        bst_uint n_feat = dm->num_features();
        std::vector<bst_float> X_flat;
        std::vector<bst_float> y;
        X_flat.reserve((to - from) * n_feat);
        y.reserve(to - from);
        for (bst_uint i = from; i < to; ++i) {
            bst_uint r = idx[i];
            for (bst_uint c = 0; c < n_feat; ++c)
                X_flat.push_back(dm->feature(r, c));
            if (dm->has_labels())
                y.push_back(dm->labels()[r]);
        }
        return DMatrix::from_flat(X_flat, to - from, n_feat, y);
    };

    return {make_subset(0, n_train), make_subset(n_train, n)};
}

} // namespace xgb
