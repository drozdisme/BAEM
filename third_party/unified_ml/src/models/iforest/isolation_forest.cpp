#include "models/iforest/isolation_forest.hpp"
#include "models/iforest/math_utils.hpp"
#include "models/iforest/random_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace iforest {

namespace {

constexpr char kMagic[8] = {'U','M','L','I','F','R','1','\0'};

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
  const std::size_t pos = out.size();
  out.resize(pos + sizeof(T));
  std::memcpy(out.data() + pos, &v, sizeof(T));
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
  if (off + sizeof(T) > in.size()) throw std::runtime_error("IsolationForest deserialize: truncated payload");
  T v{};
  std::memcpy(&v, in.data() + off, sizeof(T));
  off += sizeof(T);
  return v;
}

void serialize_node(std::vector<char>& out, const Node* node) {
  const bool present = node != nullptr;
  write_raw(out, present);
  if (!present) return;
  write_raw(out, node->feature_index);
  write_raw(out, node->split_value);
  write_raw(out, node->leaf_size);
  serialize_node(out, node->left.get());
  serialize_node(out, node->right.get());
}

std::unique_ptr<Node> deserialize_node(const std::vector<char>& in, std::size_t& off) {
  const bool present = read_raw<bool>(in, off);
  if (!present) return nullptr;
  auto node = std::make_unique<Node>();
  node->feature_index = read_raw<int>(in, off);
  node->split_value = read_raw<double>(in, off);
  node->leaf_size = read_raw<std::size_t>(in, off);
  node->left = deserialize_node(in, off);
  node->right = deserialize_node(in, off);
  return node;
}

std::vector<double> flatten_dataset_row_major(
  const std::vector<std::vector<double>>& data,
  std::size_t& out_rows,
  std::size_t& out_cols)
{
  if (data.empty()) {
    throw std::invalid_argument("Training data must not be empty.");
  }
  if (data[0].empty()) {
    throw std::invalid_argument("Each sample must have at least one feature.");
  }

  out_rows = data.size();
  out_cols = data[0].size();
  std::vector<double> flat(out_rows * out_cols);

  for (std::size_t r = 0; r < out_rows; ++r) {
    if (data[r].size() != out_cols) {
      throw std::invalid_argument(
        "Inconsistent feature dimension at row " + std::to_string(r) + ".");
    }
    std::copy(data[r].begin(), data[r].end(), flat.begin() + r * out_cols);
  }
  return flat;
}

} // namespace

IsolationForest::IsolationForest(
  int  n_trees,
  int  subsample_size,
  int  max_height,
  uint64_t seed)
  : n_trees_(n_trees)
  , subsample_size_(subsample_size)
  , max_height_(max_height)
  , seed_(seed)
  , n_samples_(0)
{
  if (n_trees <= 0)
    throw std::invalid_argument("n_trees must be a positive integer.");
  if (subsample_size <= 0)
    throw std::invalid_argument("subsample_size must be a positive integer.");
}

void IsolationForest::fit(const std::vector<std::vector<double>>& data) {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> flat_data = flatten_dataset_row_major(data, rows, cols);
  const core::MatrixView data_view(flat_data.data(), rows, cols, cols);

  n_samples_ = rows;
  n_features_ = cols;

  const int actual_sub = std::min(
    subsample_size_,
    static_cast<int>(n_samples_)
  );

  const int actual_height = (max_height_ > 0)
    ? max_height_
    : static_cast<int>(
    std::ceil(std::log2(static_cast<double>(actual_sub))));

  std::vector<std::size_t> population(n_samples_);
  std::iota(population.begin(), population.end(), std::size_t{0});

  trees_.resize(static_cast<std::size_t>(n_trees_));

  // FIX: Parallel tree building — each tree uses its own RNG seeded from seed_+t
  // so results are deterministic regardless of thread scheduling.
  #pragma omp parallel for schedule(dynamic, 1)
  for (int t = 0; t < n_trees_; ++t) {
    std::mt19937 local_rng(static_cast<std::mt19937::result_type>(seed_ + static_cast<uint64_t>(t)));
    auto subsample = sample_without_replacement(
    population,
    static_cast<std::size_t>(actual_sub),
    local_rng
    );
    trees_[t].build(data_view, subsample, local_rng, actual_height);
  }
}

double IsolationForest::score(const std::vector<double>& sample) const {
  if (trees_.empty())
    throw std::runtime_error(
    "Model has not been fitted. Call fit() before score().");
  if (sample.size() != n_features_)
    throw std::invalid_argument("score(): sample dimension mismatch.");

  double avg_path = 0.0;
  for (const ITree& tree : trees_) {
    avg_path += tree.path_length(sample.data());
  }
  avg_path /= static_cast<double>(trees_.size());

  const int actual_sub = std::min(
    subsample_size_,
    static_cast<int>(n_samples_)
  );
  const double cn = c(static_cast<std::size_t>(actual_sub));

  if (cn <= 0.0) return 0.5;

  return std::pow(2.0, -avg_path / cn);
}

bool IsolationForest::predict(
  const std::vector<double>& sample,
  double       threshold) const
{
  return score(sample) >= threshold;
}

// FIX: Parallel score_batch — main hot path, embarrassingly parallel
std::vector<double> IsolationForest::score_batch(
  const std::vector<std::vector<double>>& data) const
{
  if (trees_.empty())
    throw std::runtime_error(
      "Model has not been fitted. Call fit() before score_batch().");

  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> flat_data = flatten_dataset_row_major(data, rows, cols);
  if (cols != n_features_)
    throw std::invalid_argument("score_batch(): feature dimension mismatch.");

  const core::MatrixView view(flat_data.data(), rows, cols, cols);
  const std::size_t n = rows;
  std::vector<double> scores(n);

  const int actual_sub = std::min(
    subsample_size_,
    static_cast<int>(n_samples_)
  );
  const double cn = c(static_cast<std::size_t>(actual_sub));
  const int  nt = static_cast<int>(trees_.size());

  #pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    double avg_path = 0.0;
    const double* sample_ptr = view.data + static_cast<std::size_t>(i) * view.row_stride;
    for (int t = 0; t < nt; ++t)
    avg_path += trees_[t].path_length(sample_ptr);
    avg_path /= static_cast<double>(nt);
    scores[i] = (cn > 0.0) ? std::pow(2.0, -avg_path / cn) : 0.5;
  }
  return scores;
}

void IsolationForest::save(const std::string& filepath) const {
  if (trees_.empty())
    throw std::runtime_error("IsolationForest::save: model is not fitted");

  std::vector<char> payload;
  write_raw<int>(payload, n_trees_);
  write_raw<int>(payload, subsample_size_);
  write_raw<int>(payload, max_height_);
  write_raw<uint64_t>(payload, seed_);
  write_raw<std::size_t>(payload, n_samples_);
  write_raw<std::size_t>(payload, n_features_);
  const std::size_t tree_count = trees_.size();
  write_raw(payload, tree_count);
  for (const auto& tree : trees_) serialize_node(payload, tree.root());

  std::ofstream out(filepath, std::ios::binary);
  if (!out) throw std::runtime_error("IsolationForest::save: unable to open file");
  out.write(kMagic, sizeof(kMagic));
  const std::size_t payload_size = payload.size();
  out.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

IsolationForest IsolationForest::load(const std::string& filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in) throw std::runtime_error("IsolationForest::load: unable to open file");

  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("IsolationForest::load: invalid file header");

  std::size_t payload_size = 0;
  in.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
  std::vector<char> payload(payload_size);
  in.read(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (!in) throw std::runtime_error("IsolationForest::load: truncated payload");

  std::size_t off = 0;
  const int n_trees = read_raw<int>(payload, off);
  const int subsample_size = read_raw<int>(payload, off);
  const int max_height = read_raw<int>(payload, off);
  const uint64_t seed = read_raw<uint64_t>(payload, off);

  IsolationForest model(n_trees, subsample_size, max_height, seed);
  model.n_samples_ = read_raw<std::size_t>(payload, off);
  model.n_features_ = read_raw<std::size_t>(payload, off);

  const std::size_t tree_count = read_raw<std::size_t>(payload, off);
  model.trees_.resize(tree_count);
  for (std::size_t i = 0; i < tree_count; ++i) {
    model.trees_[i].set_root(deserialize_node(payload, off));
  }
  return model;
}

} // namespace iforest
