#include "models/xgboost/utils/csv_reader.hpp"
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace xgb {
namespace utils {

// CSVReader            

CSVReader::CSVReader(const std::string& path, char delim)
  : delim_(delim)
{
  file_.open(path);
  if (!file_.is_open())
    throw std::runtime_error("CSVReader: cannot open " + path);

  // Always read first line to check for header
  if (std::getline(file_, current_line_)) {
    // Heuristic: if first cell cannot be parsed as float, treat as header
    auto row = parse_line(current_line_, delim_);
    if (!row.cells.empty()) {
    bst_float v = parse_float(row.cells[0]);
    if (std::isnan(v)) {
      // looks like a header
      header_ = row.cells;
      current_line_.clear();
    }
    // else: first line is data — keep current_line_ as next row
    }
  }
}

CSVReader::~CSVReader() {
  if (file_.is_open()) file_.close();
}

bool CSVReader::has_next() const {
  return !current_line_.empty() || file_.good();
}

CSVReader::Row CSVReader::next_row() {
  if (!current_line_.empty()) {
    auto row = parse_line(current_line_, delim_);
    current_line_.clear();
    return row;
  }
  std::string line;
  while (std::getline(file_, line)) {
    if (line.empty() || line == "\r") continue;
    return parse_line(line, delim_);
  }
  return {};
}

void CSVReader::reset() {
  file_.clear();
  file_.seekg(0, std::ios::beg);
  // skip header line if present
  if (!header_.empty()) {
    std::string dummy;
    std::getline(file_, dummy);
  }
  current_line_.clear();
}

std::vector<CSVReader::Row> CSVReader::read_all() {
  std::vector<Row> result;
  while (has_next()) {
    auto row = next_row();
    if (!row.cells.empty())
    result.push_back(std::move(row));
  }
  return result;
}

bst_float CSVReader::parse_float(const std::string& s) {
  if (s.empty() || s == "NA" || s == "nan" || s == "NaN" || s == "null")
    return std::numeric_limits<bst_float>::quiet_NaN();
  try {
    return std::stof(s);
  } catch (...) {
    return std::numeric_limits<bst_float>::quiet_NaN();
  }
}

CSVReader::Row CSVReader::parse_line(const std::string& line, char delim) {
  Row row;
  std::istringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, delim)) {
    // Trim CR
    if (!cell.empty() && cell.back() == '\r')
    cell.pop_back();
    row.cells.push_back(cell);
  }
  return row;
}

// CSVWriter            

CSVWriter::CSVWriter(const std::string& path, char delim)
  : delim_(delim)
{
  file_.open(path);
  if (!file_.is_open())
    throw std::runtime_error("CSVWriter: cannot open " + path);
}

CSVWriter::~CSVWriter() {
  if (file_.is_open()) file_.close();
}

void CSVWriter::write_header(const std::vector<std::string>& cols) {
  for (size_t i = 0; i < cols.size(); ++i) {
    if (i) file_ << delim_;
    file_ << cols[i];
  }
  file_ << "\n";
}

void CSVWriter::write_row(const std::vector<std::string>& cells) {
  for (size_t i = 0; i < cells.size(); ++i) {
    if (i) file_ << delim_;
    file_ << cells[i];
  }
  file_ << "\n";
}

void CSVWriter::write_row(const std::vector<bst_float>& values) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) file_ << delim_;
    file_ << values[i];
  }
  file_ << "\n";
}

} // namespace utils
} // namespace xgb
