#pragma once
#include "models/xgboost/core/types.hpp"
#include <vector>
#include <string>
#include <fstream>

namespace xgb {
namespace utils {

//                        
//  CSVReader
//  Lightweight CSV parser.  No external deps.
//                        
class CSVReader {
public:
    struct Row {
        std::vector<std::string> cells;
        const std::string& operator[](size_t i) const { return cells[i]; }
        size_t size() const { return cells.size(); }
    };

    explicit CSVReader(const std::string& path, char delim = ',');
    ~CSVReader();

    bool has_next() const;
    Row  next_row();
    void reset();

    // Read all at once             
    std::vector<Row> read_all();

    std::vector<std::string> header() const { return header_; }
    bool has_header() const { return !header_.empty(); }

    // Static helper: parse float, return NaN on failure
    static bst_float parse_float(const std::string& s);

private:
    std::ifstream file_;
    char delim_;
    std::string current_line_;
    std::vector<std::string> header_;
    bool read_header_{false};

    static Row parse_line(const std::string& line, char delim);
};

//                        
//  CSVWriter
//                        
class CSVWriter {
public:
    explicit CSVWriter(const std::string& path, char delim = ',');
    ~CSVWriter();

    void write_header(const std::vector<std::string>& cols);
    void write_row(const std::vector<std::string>& cells);
    void write_row(const std::vector<bst_float>& values);

private:
    std::ofstream file_;
    char delim_;
};

} // namespace utils
} // namespace xgb
