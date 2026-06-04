#include "models/rf/random_forest.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <fstream>
#include <cstring>
#include <cstdint>

namespace rf {

namespace {
constexpr char kMagic[8] = {'U','M','L','R','F','v','1','\0'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
  const char* p = reinterpret_cast<const char*>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

void write_node(std::vector<char>& out, const TreeNode* node) {
  const bool exists = (node != nullptr);
  write_raw(out, exists);
  if (!exists) return;
  write_raw(out, node->feature_index);
  write_raw(out, node->threshold);
  write_raw(out, node->is_leaf);
  write_raw(out, node->prediction);
  write_raw(out, node->impurity_decrease);
  write_raw(out, node->n_samples);
  write_node(out, node->left.get());
  write_node(out, node->right.get());
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
  if (off + sizeof(T) > in.size()) throw std::runtime_error("RF deserialize: truncated payload");
  T v{};
  std::memcpy(&v, in.data() + off, sizeof(T));
  off += sizeof(T);
  return v;
}

std::unique_ptr<TreeNode> read_node(const std::vector<char>& in, std::size_t& off) {
  const bool exists = read_raw<bool>(in, off);
  if (!exists) return nullptr;
  auto node = std::make_unique<TreeNode>();
  node->feature_index = read_raw<int>(in, off);
  node->threshold = read_raw<double>(in, off);
  node->is_leaf = read_raw<bool>(in, off);
  node->prediction = read_raw<double>(in, off);
  node->impurity_decrease = read_raw<double>(in, off);
  node->n_samples = read_raw<size_t>(in, off);
  node->left = read_node(in, off);
  node->right = read_node(in, off);
  return node;
}

std::uint64_t fnv1a64(const std::vector<char>& data) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : data) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}
} // namespace

void RandomForestParams::validate() const {
  if (n_estimators <= 0) throw std::invalid_argument("RandomForestParams: n_estimators must be > 0");
  if (max_depth == 0 || max_depth < -1) throw std::invalid_argument("RandomForestParams: max_depth must be -1 or > 0");
  if (min_samples_split < 2) throw std::invalid_argument("RandomForestParams: min_samples_split must be >= 2");
  if (min_samples_leaf < 1) throw std::invalid_argument("RandomForestParams: min_samples_leaf must be >= 1");
  if (min_impurity_decrease < 0.0) throw std::invalid_argument("RandomForestParams: min_impurity_decrease must be >= 0");
}

RandomForest::RandomForest(RandomForestParams params)
  : params_(std::move(params)) {
  params_.validate();
}

// ---------------------------------------------------------------------------
// CartParams factory for each tree
// ---------------------------------------------------------------------------
CartParams RandomForest::make_tree_params(uint64_t seed) const {
  CartParams cp;
  cp.max_depth      = params_.max_depth;
  cp.min_samples_split  = params_.min_samples_split;
  cp.min_samples_leaf   = params_.min_samples_leaf;
  cp.min_impurity_decrease  = params_.min_impurity_decrease;
  cp.max_features_strategy  = params_.max_features_strategy;
  cp.max_features     = params_.max_features;
  cp.random_seed    = seed;
  return cp;
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------
void RandomForest::fit(const Dataset& dataset) {
  task_   = dataset.task();
  n_classes_  = dataset.n_classes();
  n_features_ = dataset.n_features();

  size_t n = dataset.n_samples();
  trees_.clear();
  trees_.reserve(static_cast<size_t>(params_.n_estimators));

  // OOB storage
  bool do_oob = params_.compute_oob && params_.bootstrap;
  if (do_oob) {
    oob_sums_.assign(n, 0.0);
    oob_counts_.assign(n, 0);
    if (task_ == TaskType::Classification) {
    oob_votes_.assign(n, std::vector<int>(static_cast<size_t>(n_classes_), 0));
    }
  }

  RNG master_rng(params_.random_seed);

  for (int t = 0; t < params_.n_estimators; ++t) {
    uint64_t tree_seed = static_cast<uint64_t>(master_rng.uniform_int(0, 1 << 30));
    RNG tree_rng(tree_seed);

    std::vector<size_t> oob_idx;
    std::vector<size_t> train_idx;

    if (params_.bootstrap) {
    train_idx = dataset.bootstrap_sample(tree_rng, do_oob ? &oob_idx : nullptr);
    } else {
    train_idx = iota_indices(n);
    }

    auto tree = std::make_unique<CartTree>(make_tree_params(tree_seed + 1));
    tree->fit(dataset, train_idx);

    // Accumulate OOB predictions
    if (do_oob && !oob_idx.empty()) {
    for (size_t i : oob_idx) {
      double pred = tree->predict_one(dataset.row(i));
      if (task_ == TaskType::Classification) {
        int cls = static_cast<int>(pred);
        if (cls >= 0 && cls < n_classes_)
        ++oob_votes_[i][cls];
      } else {
        oob_sums_[i]  += pred;
        oob_counts_[i] += 1;
      }
    }
    }

    trees_.push_back(std::move(tree));

    if (params_.progress_callback)
    params_.progress_callback(t, params_.n_estimators);
  }

  if (do_oob) compute_oob_error(dataset);
}

// ---------------------------------------------------------------------------
// OOB error computation
// ---------------------------------------------------------------------------
void RandomForest::compute_oob_error(const Dataset& dataset) {
  size_t n = dataset.n_samples();
  size_t covered = 0;
  double err = 0.0;

  if (task_ == TaskType::Classification) {
    for (size_t i = 0; i < n; ++i) {
    const auto& votes = oob_votes_[i];
    int total = 0;
    for (int v : votes) total += v;
    if (total == 0) continue;
    ++covered;
    int pred_cls = static_cast<int>(
      std::max_element(votes.begin(), votes.end()) - votes.begin());
    if (pred_cls != static_cast<int>(dataset.y()[i])) ++err;
    }
    oob_error_ = (covered > 0) ? err / static_cast<double>(covered) : 0.0;
  } else {
    for (size_t i = 0; i < n; ++i) {
    if (oob_counts_[i] == 0) continue;
    ++covered;
    double pred = oob_sums_[i] / static_cast<double>(oob_counts_[i]);
    double d = pred - dataset.y()[i];
    err += d * d;
    }
    oob_error_ = (covered > 0) ? err / static_cast<double>(covered) : 0.0;
  }
  oob_computed_ = true;
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------
double RandomForest::predict_one(const std::vector<double>& x) const {
  if (trees_.empty()) throw std::runtime_error("RandomForest not fitted");

  if (task_ == TaskType::Classification) {
    std::vector<int> votes(static_cast<size_t>(n_classes_), 0);
    for (const auto& tree : trees_) {
    int cls = static_cast<int>(tree->predict_one(x));
    if (cls >= 0 && cls < n_classes_) ++votes[cls];
    }
    return static_cast<double>(
    std::max_element(votes.begin(), votes.end()) - votes.begin());
  } else {
    double sum = 0.0;
    for (const auto& tree : trees_) sum += tree->predict_one(x);
    return sum / static_cast<double>(trees_.size());
  }
}

std::vector<double> RandomForest::predict(const Dataset& dataset) const {
  std::vector<double> preds(dataset.n_samples());
  for (size_t i = 0; i < dataset.n_samples(); ++i)
    preds[i] = predict_one(dataset.row(i));
  return preds;
}

std::vector<double> RandomForest::predict(const std::vector<std::vector<double>>& X) const {
  std::vector<double> preds(X.size());
  for (size_t i = 0; i < X.size(); ++i)
    preds[i] = predict_one(X[i]);
  return preds;
}

std::vector<double> RandomForest::predict(const core::MatrixView& X) const {
  std::vector<double> preds(X.rows);
  std::vector<double> row(X.cols, 0.0);
  for (size_t i = 0; i < X.rows; ++i) {
    for (size_t j = 0; j < X.cols; ++j) row[j] = X(i, j);
    preds[i] = predict_one(row);
  }
  return preds;
}

std::vector<std::vector<double>> RandomForest::predict_proba(const Dataset& dataset) const {
  const auto flat = predict_proba_flat(dataset);
  const size_t n = dataset.n_samples();
  const size_t nc = static_cast<size_t>(n_classes_);
  std::vector<std::vector<double>> proba(n, std::vector<double>(nc, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t c = 0; c < nc; ++c) {
      proba[i][c] = flat[i * nc + c];
    }
  }
  return proba;
}

std::vector<double> RandomForest::predict_proba_flat(const Dataset& dataset) const {
  if (task_ != TaskType::Classification)
    throw std::runtime_error("predict_proba only available for classification");

  size_t n = dataset.n_samples();
  size_t nc = static_cast<size_t>(n_classes_);
  std::vector<double> proba(n * nc, 0.0);

  for (const auto& tree : trees_) {
    for (size_t i = 0; i < n; ++i) {
    int cls = static_cast<int>(tree->predict_one(dataset.row(i)));
    if (cls >= 0 && static_cast<size_t>(cls) < nc)
      proba[i * nc + static_cast<size_t>(cls)] += 1.0;
    }
  }
  double nt = static_cast<double>(trees_.size());
  for (double& p : proba) p /= nt;

  return proba;
}

// ---------------------------------------------------------------------------
// OOB error accessor
// ---------------------------------------------------------------------------
double RandomForest::oob_error() const {
  if (!oob_computed_)
    throw std::runtime_error("OOB error not computed. Set compute_oob=true and bootstrap=true.");
  return oob_error_;
}

// ---------------------------------------------------------------------------
// Feature importance
// ---------------------------------------------------------------------------
std::vector<double> RandomForest::feature_importances() const {
  if (trees_.empty() || n_features_ == 0) return {};

  std::vector<double> agg(n_features_, 0.0);
  for (const auto& tree : trees_) {
    auto fi = tree->feature_importances(n_features_);
    for (size_t j = 0; j < n_features_; ++j) agg[j] += fi[j];
  }
  double total = 0.0;
  for (double v : agg) total += v;
  if (total > 0.0) for (double& v : agg) v /= total;
  return agg;
}

void RandomForest::save(const std::string& filepath) const {
  if (trees_.empty()) throw std::runtime_error("RandomForest::save: model is not fitted");
  std::vector<char> payload;
  write_raw(payload, params_.n_estimators);
  write_raw(payload, params_.max_depth);
  write_raw(payload, params_.min_samples_split);
  write_raw(payload, params_.min_samples_leaf);
  write_raw(payload, params_.min_impurity_decrease);
  int mfs = static_cast<int>(params_.max_features_strategy);
  write_raw(payload, mfs);
  write_raw(payload, params_.max_features);
  write_raw(payload, params_.bootstrap);
  write_raw(payload, params_.compute_oob);
  write_raw(payload, params_.random_seed);
  int task = static_cast<int>(task_);
  write_raw(payload, task);
  write_raw(payload, n_classes_);
  write_raw(payload, n_features_);

  const std::size_t n_trees = trees_.size();
  write_raw(payload, n_trees);
  for (const auto& t : trees_) {
    const CartParams& cp = t->params();
    write_raw(payload, cp.max_depth);
    write_raw(payload, cp.min_samples_split);
    write_raw(payload, cp.min_samples_leaf);
    write_raw(payload, cp.min_impurity_decrease);
    write_raw(payload, cp.max_features);
    write_raw(payload, cp.random_seed);
    int cp_mfs = static_cast<int>(cp.max_features_strategy);
    write_raw(payload, cp_mfs);
    write_node(payload, t->root());
  }

  const std::uint64_t checksum = fnv1a64(payload);
  const std::uint64_t payload_size = static_cast<std::uint64_t>(payload.size());

  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs) throw std::runtime_error("RandomForest::save: cannot open file");
  ofs.write(kMagic, sizeof(kMagic));
  ofs.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
  ofs.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
  ofs.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
  ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

RandomForest RandomForest::load(const std::string& filepath) {
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs) throw std::runtime_error("RandomForest::load: cannot open file");

  char magic[8]{};
  std::uint32_t version = 0;
  std::uint64_t payload_size = 0, checksum = 0;
  ifs.read(magic, sizeof(magic));
  ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
  ifs.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
  ifs.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
    throw std::runtime_error("RandomForest::load: invalid magic");
  if (version != kVersion)
    throw std::runtime_error("RandomForest::load: unsupported version");

  std::vector<char> payload(static_cast<std::size_t>(payload_size));
  ifs.read(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (fnv1a64(payload) != checksum)
    throw std::runtime_error("RandomForest::load: checksum mismatch");

  std::size_t off = 0;
  RandomForestParams p;
  p.n_estimators = read_raw<int>(payload, off);
  p.max_depth = read_raw<int>(payload, off);
  p.min_samples_split = read_raw<int>(payload, off);
  p.min_samples_leaf = read_raw<int>(payload, off);
  p.min_impurity_decrease = read_raw<double>(payload, off);
  p.max_features_strategy = static_cast<CartParams::MaxFeaturesStrategy>(read_raw<int>(payload, off));
  p.max_features = read_raw<size_t>(payload, off);
  p.bootstrap = read_raw<bool>(payload, off);
  p.compute_oob = read_raw<bool>(payload, off);
  p.random_seed = read_raw<uint64_t>(payload, off);
  p.validate();

  RandomForest rf(p);
  rf.task_ = static_cast<TaskType>(read_raw<int>(payload, off));
  rf.n_classes_ = read_raw<int>(payload, off);
  rf.n_features_ = read_raw<size_t>(payload, off);
  const std::size_t n_trees = read_raw<size_t>(payload, off);
  rf.trees_.clear();
  rf.trees_.reserve(n_trees);

  for (std::size_t i = 0; i < n_trees; ++i) {
    CartParams cp;
    cp.max_depth = read_raw<int>(payload, off);
    cp.min_samples_split = read_raw<int>(payload, off);
    cp.min_samples_leaf = read_raw<int>(payload, off);
    cp.min_impurity_decrease = read_raw<double>(payload, off);
    cp.max_features = read_raw<size_t>(payload, off);
    cp.random_seed = read_raw<uint64_t>(payload, off);
    cp.max_features_strategy = static_cast<CartParams::MaxFeaturesStrategy>(read_raw<int>(payload, off));
    auto tree = std::make_unique<CartTree>(cp);
    tree->set_serialized_state(read_node(payload, off), rf.task_, rf.n_classes_, rf.n_features_);
    rf.trees_.push_back(std::move(tree));
  }
  return rf;
}

} // namespace rf
