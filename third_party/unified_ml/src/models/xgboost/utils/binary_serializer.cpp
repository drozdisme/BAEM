#include "models/xgboost/utils/binary_serializer.hpp"
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/tree/tree_node.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace xgb {

void save_model_binary(const GradientBooster& booster, const std::string& path) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("save_model_binary: cannot open " + path);

  // Write header
  BinaryHeader hdr;
  hdr.magic  = kBinaryMagic;
  hdr.version  = kBinaryVersion;
  hdr.num_trees  = booster.num_trees();
  hdr.base_score = booster.base_score();
  hdr.eta    = booster.config().eta;
  hdr.num_class  = static_cast<uint32_t>(booster.config().num_class);
  const std::string& obj = booster.config().objective;
#ifdef _MSC_VER
  strncpy_s(hdr.objective, sizeof(hdr.objective), obj.c_str(), _TRUNCATE);
#else
  std::strncpy(hdr.objective, obj.c_str(), sizeof(hdr.objective) - 1);
  hdr.objective[sizeof(hdr.objective) - 1] = '\0';
#endif

  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

  // Write each tree
  for (const auto& tree : booster.trees()) {
    const auto& nodes = tree->nodes();
    uint32_t n = static_cast<uint32_t>(nodes.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& nd : nodes) {
    NodeRecord rec;
    rec.node_id   = static_cast<int32_t>(nd.id);
    rec.left_child  = static_cast<int32_t>(nd.child_left);
    rec.right_child = static_cast<int32_t>(nd.child_right);
    rec.feature_idx = nd.feature_idx;
    rec.split_value = nd.split_value;
    rec.leaf_value  = nd.leaf_value;
    f.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    }
  }

  if (!f) throw std::runtime_error("save_model_binary: write error on " + path);
}

void load_model_binary(GradientBooster& booster, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("load_model_binary: cannot open " + path);

  BinaryHeader hdr;
  f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!f || hdr.magic != kBinaryMagic)
    throw std::runtime_error("load_model_binary: invalid magic / corrupt file");
  if (hdr.version != kBinaryVersion)
    throw std::runtime_error("load_model_binary: unsupported version");

  // Update booster metadata via mutable config
  booster.mutable_config().eta    = hdr.eta;
  booster.mutable_config().num_class  = static_cast<bst_int>(hdr.num_class);
  booster.mutable_config().objective  = std::string(hdr.objective);
  // base_score is stored in the booster directly; we rebuild via JSON path
  // For binary: re-use the tree injection path

  // Read trees
  for (uint32_t t = 0; t < hdr.num_trees; ++t) {
    uint32_t n_nodes = 0;
    f.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
    if (!f) throw std::runtime_error("load_model_binary: unexpected EOF at tree " + std::to_string(t));

    std::vector<TreeNode> nodes(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i) {
    NodeRecord rec;
    f.read(reinterpret_cast<char*>(&rec), sizeof(rec));
    if (!f) throw std::runtime_error("load_model_binary: unexpected EOF at node " + std::to_string(i));
    nodes[i].id    = static_cast<NodeId>(rec.node_id);
    nodes[i].child_left  = static_cast<NodeId>(rec.left_child);
    nodes[i].child_right = static_cast<NodeId>(rec.right_child);
    nodes[i].feature_idx = rec.feature_idx;
    nodes[i].split_value = rec.split_value;
    nodes[i].leaf_value  = rec.leaf_value;
    nodes[i].is_leaf   = (rec.left_child == -1 && rec.right_child == -1);
    }
    // Inject tree into booster
    booster.inject_tree(std::move(nodes));
  }
}

} // namespace xgb
