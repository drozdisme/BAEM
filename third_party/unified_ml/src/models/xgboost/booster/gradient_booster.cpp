#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/data/dmatrix.hpp"
#include "models/xgboost/utils/logger.hpp"
#include "models/xgboost/utils/simple_json.hpp"
#include "models/xgboost/utils/binary_serializer.hpp"
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <cmath>

namespace xgb {

GradientBooster::GradientBooster(const BoosterConfig& cfg)
  : cfg_(cfg)
  , base_score_(cfg.base_score)
  , loss_fn_(LossFunction::create(cfg.objective, cfg.num_class))
  , rng_(static_cast<uint64_t>(cfg.seed) ^ 0xDEADBEEF12345678ULL)
{}

// Row/column sampling — now uses xoshiro256**     

std::vector<bst_uint> GradientBooster::sample_rows(bst_uint n_total) const {
  std::vector<bst_uint> idx(n_total);
  std::iota(idx.begin(), idx.end(), 0u);
  if (cfg_.tree.subsample >= 1.f) return idx;

  bst_uint n_sample = std::max(1u, static_cast<bst_uint>(n_total * cfg_.tree.subsample));
  for (bst_uint i = 0; i < n_sample; ++i) {
    bst_uint j = i + static_cast<bst_uint>(rng_.next_range(n_total - i));
    std::swap(idx[i], idx[j]);
  }
  idx.resize(n_sample);
  return idx;
}

std::vector<bst_uint> GradientBooster::sample_columns(bst_uint n_total) const {
  std::vector<bst_uint> idx(n_total);
  std::iota(idx.begin(), idx.end(), 0u);
  if (cfg_.tree.colsample_bytree >= 1.f) return idx;

  bst_uint n_sample = std::max(1u, static_cast<bst_uint>(n_total * cfg_.tree.colsample_bytree));
  for (bst_uint i = 0; i < n_sample; ++i) {
    bst_uint j = i + static_cast<bst_uint>(rng_.next_range(n_total - i));
    std::swap(idx[i], idx[j]);
  }
  idx.resize(n_sample);
  return idx;
}

// One boosting round         

bst_float GradientBooster::boost_one_round(
  const DMatrix& dm,
  std::vector<bst_float>& scores)
{
  const bst_uint n  = dm.num_rows();
  const bool multiclass = (cfg_.objective == "multi:softmax" ||
           cfg_.objective == "multi:softprob" ||
           cfg_.objective == "softmax");
  const bst_int nc  = multiclass ? std::max(cfg_.num_class, bst_int(2)) : 1;

  std::vector<GradientPair> grads;
  loss_fn_->compute_gradients(scores, dm.labels(), grads);

  auto col_idx = sample_columns(dm.num_features());

  TreeConfig tree_cfg = cfg_.tree;
  tree_cfg.nthread  = cfg_.nthread;

  if (multiclass) {
    // Build one tree per class; update the corresponding score slice.
    for (bst_int c = 0; c < nc; ++c) {
    // Extract per-class gradients into a contiguous vector.
    std::vector<GradientPair> class_grads(n);
    for (bst_uint i = 0; i < n; ++i)
      class_grads[i] = grads[i * nc + c];

    auto row_idx = sample_rows(n);
    tree_cfg.seed = cfg_.seed + static_cast<bst_uint>(trees_.size());

    auto tree = std::make_unique<DecisionTree>(tree_cfg);
    tree->build(dm, class_grads, row_idx, col_idx);

    // Bake current eta into leaves (same rationale as scalar case)
    tree->scale_leaves_by(cfg_.eta);

    // Update the class-c slice of the flat score buffer.
    for (bst_uint r = 0; r < n; ++r)
      scores[r * nc + c] += tree->predict(dm, r);

    trees_.push_back(std::move(tree));
    }
  } else {
    auto row_idx = sample_rows(n);
    tree_cfg.seed = cfg_.seed + static_cast<bst_uint>(trees_.size());

    auto tree = std::make_unique<DecisionTree>(tree_cfg);
    tree->build(dm, grads, row_idx, col_idx);

    // Bake the current learning-rate into the leaf values so that
    // predict() returns already-scaled values (no global cfg_.eta needed).
    tree->scale_leaves_by(cfg_.eta);

    for (bst_uint r = 0; r < n; ++r)
    scores[r] += tree->predict(dm, r);

    trees_.push_back(std::move(tree));
  }

  bst_float round_loss = loss_fn_->compute_loss(scores, dm.labels());
  return round_loss;
}

// Prediction           

bst_float GradientBooster::predict_raw(const DMatrix& dm, bst_uint row) const {
  bst_float score = base_score_;
  for (const auto& tree : trees_)
    score += tree->predict(dm, row); // eta already baked into leaves
  return score;
}

std::vector<bst_float> GradientBooster::predict_batch_raw(
  const DMatrix& dm, const std::vector<bst_uint>& row_indices) const
{
  const bool multiclass = (cfg_.objective == "multi:softmax" ||
           cfg_.objective == "multi:softprob" ||
           cfg_.objective == "softmax");
  const bst_int nc = multiclass ? std::max(cfg_.num_class, bst_int(2)) : 1;
  const bst_uint N  = static_cast<bst_uint>(row_indices.size());

  std::vector<bst_float> out(static_cast<size_t>(N) * nc, base_score_);

  if (multiclass) {
    // Trees are stored as: [r0_c0, r0_c1, ..., r0_cK-1, r1_c0, ...]
    // Tree index t belongs to class (t % nc).
    for (size_t t = 0; t < trees_.size(); ++t) {
    const bst_int c = static_cast<bst_int>(t) % nc;
    for (bst_uint i = 0; i < N; ++i)
      out[i * nc + c] += trees_[t]->predict(dm, row_indices[i]);
    }
  } else {
    for (const auto& tree : trees_)
    for (bst_uint i = 0; i < N; ++i)
      out[i] += tree->predict(dm, row_indices[i]); // eta already baked
  }
  return out;
}

// Feature importance         

std::vector<bst_float> GradientBooster::feature_importance(bst_uint n_features) const {
  std::vector<bst_float> imp(n_features, 0.f);
  for (const auto& tree : trees_)
    tree->compute_feature_importance(imp, n_features);
  return imp;
}

// Tree injection (for deserialisation)      

void GradientBooster::inject_tree(std::vector<TreeNode> nodes) {
  auto tree = std::make_unique<DecisionTree>(cfg_.tree);
  tree->load_nodes(std::move(nodes));
  trees_.push_back(std::move(tree));
}

// JSON Serialisation         

std::string GradientBooster::dump_model_json() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"base_score\": " << base_score_ << ",\n";
  oss << "  \"eta\": " << cfg_.eta << ",\n";
  oss << "  \"eta_baked\": true,\n"; // leaves already multiplied by eta
  oss << "  \"config\": {\n";
  oss << "  \"lambda\": "     << cfg_.tree.lambda     << ",\n";
  oss << "  \"alpha\": "    << cfg_.tree.alpha    << ",\n";
  oss << "  \"gamma\": "    << cfg_.tree.gamma    << ",\n";
  oss << "  \"max_depth\": "    << cfg_.tree.max_depth    << ",\n";
  oss << "  \"min_child_weight\": " << cfg_.tree.min_child_weight << ",\n";
  oss << "  \"objective\": \""  << cfg_.objective     << "\"\n";
  oss << "  },\n";
  oss << "  \"trees\": [\n";
  for (size_t i = 0; i < trees_.size(); ++i) {
    if (i) oss << ",\n";
    oss << trees_[i]->dump_json();
  }
  oss << "\n  ]\n}\n";
  return oss.str();
}

std::string GradientBooster::dump_model_text() const {
  std::ostringstream oss;
  oss << "XGBoost Model — " << trees_.size() << " trees\n";
  oss << "base_score=" << base_score_ << "  eta=" << cfg_.eta << "\n\n";
  for (size_t i = 0; i < trees_.size(); ++i) {
    oss << "  Tree " << i << "      \n";
    oss << trees_[i]->dump_text() << "\n";
  }
  return oss.str();
}

// save_model / load_model         

void GradientBooster::save_model(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("Cannot write model: " + path);
  f << dump_model_json();
}

namespace {
void parse_json_node(
  const JsonValue& jnode,
  std::vector<TreeNode>& nodes,
  NodeId parent_id,
  bst_int depth)
{
  NodeId nid = static_cast<NodeId>(jnode["nodeid"].as_int());
  if (nid >= static_cast<NodeId>(nodes.size()))
    nodes.resize(nid + 1);

  TreeNode& n = nodes[nid];
  n.id   = nid;
  n.parent = parent_id;
  n.depth  = depth;
  n.is_valid = true;

  if (jnode.has("leaf")) {
    n.is_leaf  = true;
    n.leaf_value = jnode["leaf"].as_float();
    n.child_left  = kInvalidNodeId;
    n.child_right = kInvalidNodeId;
  } else {
    n.is_leaf = false;
    const std::string& ss = jnode["split"].as_string();
    n.feature_idx = ss.size() > 1 ? static_cast<bst_uint>(std::stoul(ss.substr(1))) : 0u;
    n.split_value  = jnode["split_condition"].as_float();
    n.split_gain = jnode["gain"].as_float();
    n.default_left = jnode.has("default_left") ? jnode["default_left"].as_bool() : true;

    const auto& ch = jnode["children"].as_array();
    n.child_left  = static_cast<NodeId>(ch[0]["nodeid"].as_int());
    n.child_right = static_cast<NodeId>(ch[1]["nodeid"].as_int());
    parse_json_node(ch[0], nodes, nid, depth + 1);
    parse_json_node(ch[1], nodes, nid, depth + 1);
  }
}
} // namespace

void GradientBooster::load_model(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open model: " + path);

  std::string json_text((std::istreambuf_iterator<char>(f)),
         std::istreambuf_iterator<char>());

  JsonValue root = SimpleJson::parse(json_text);
  base_score_ = root["base_score"].as_float();
  cfg_.eta  = root["eta"].as_float();

  // If "eta_baked" is absent (old model format), the leaf values were stored
  // without eta applied, so we must bake eta in after loading.
  const bool eta_baked = root.has("eta_baked") && root["eta_baked"].as_bool();

  if (root.has("config")) {
    const auto& c = root["config"];
    if (c.has("lambda"))     cfg_.tree.lambda     = c["lambda"].as_float();
    if (c.has("alpha"))    cfg_.tree.alpha    = c["alpha"].as_float();
    if (c.has("gamma"))    cfg_.tree.gamma    = c["gamma"].as_float();
    if (c.has("max_depth"))    cfg_.tree.max_depth    = c["max_depth"].as_int();
    if (c.has("min_child_weight")) cfg_.tree.min_child_weight = c["min_child_weight"].as_float();
    if (c.has("objective"))    cfg_.objective     = c["objective"].as_string();
  }
  loss_fn_ = LossFunction::create(cfg_.objective, cfg_.num_class);

  trees_.clear();
  for (const auto& tj : root["trees"].as_array()) {
    auto tree = std::make_unique<DecisionTree>(cfg_.tree);
    std::vector<TreeNode> nodes;
    parse_json_node(tj, nodes, kInvalidNodeId, 0);
    tree->load_nodes(std::move(nodes));
    // Old format: bake eta into leaves now so predict() is consistent
    if (!eta_baked) tree->scale_leaves_by(cfg_.eta);
    trees_.push_back(std::move(tree));
  }
}

// Binary save/load (P4-7) — delegate to binary_serializer    

void GradientBooster::save_model_binary(const std::string& path) const {
  xgb::save_model_binary(*this, path);
}

void GradientBooster::load_model_binary(const std::string& path) {
  xgb::load_model_binary(*this, path);
}

} // namespace xgb
