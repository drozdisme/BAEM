#pragma once
#include "models/xgboost/data/dmatrix.hpp"
#include <string>
#include <memory>

namespace xgb {

//                        
//  CSVLoadOptions
//                        
struct CSVLoadOptions {
    char   delimiter        {','};
    bool   has_header       {true};
    int    label_column     {-1};     // -1 = last column
    float  missing_value    {0.f};    // value to substitute for empty cells
    bst_uint max_rows       {0};      // 0 = load all
};

//                        
//  DataLoader
//  Responsible for reading data from disk and
//  converting it into DMatrix objects.
//
//  Future extensions:
//    - LibSVM format (sparse)
//    - Binary format (XGBoost .buffer)
//    - Arrow / Parquet
//                        
class DataLoader {
public:
    // Load CSV → DMatrix
    static std::shared_ptr<DMatrix> load_csv(
        const std::string& path,
        const CSVLoadOptions& opts = {});

    // Save predictions to CSV
    static void save_csv(
        const std::string& path,
        const std::vector<bst_float>& predictions,
        const std::vector<std::string>& col_names = {"prediction"});

    // Save feature importance to CSV
    static void save_feature_importance(
        const std::string& path,
        const std::vector<std::string>& feature_names,
        const std::vector<bst_float>& importance);

    // Utility: split DMatrix into train/test
    static std::pair<std::shared_ptr<DMatrix>, std::shared_ptr<DMatrix>>
    train_test_split(const std::shared_ptr<DMatrix>& dm,
                     float test_ratio = 0.2f,
                     bst_uint seed    = 42);

private:
    static std::vector<std::string> split_line(
        const std::string& line, char delim);
};

} // namespace xgb
