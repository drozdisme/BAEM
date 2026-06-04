#pragma once

#include <cstddef>
#include <memory>

namespace iforest {

/**
 * A single node inside an isolation tree.
 *
 * Internal nodes:  feature_index >= 0, left/right are non-null.
 * Leaf nodes:      feature_index == -1, leaf_size holds partition count.
 */
struct Node {
    int         feature_index = -1;   ///< Feature used for split; -1 signals a leaf.
    double      split_value   = 0.0;  ///< Threshold: left child if value < split_value.
    std::size_t leaf_size     = 0;    ///< Number of training samples at this leaf.

    std::unique_ptr<Node> left;       ///< sample[feature] <  split_value
    std::unique_ptr<Node> right;      ///< sample[feature] >= split_value

    /// True when this node is a leaf (no further split performed).
    [[nodiscard]] bool is_leaf() const noexcept { return feature_index == -1; }
};

} // namespace iforest
