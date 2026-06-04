#pragma once

#include "core/matrix_view.hpp"
#include "models/iforest/node.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

namespace iforest {

/**
 * A single isolation tree (iTree).
 *
 * Constructed by recursively partitioning a subsample of the training data
 * using randomly chosen features and random split thresholds, until each
 * partition contains at most one point or the maximum depth is reached.
 */
class ITree {
public:
    ITree() = default;

    ITree(const ITree&)            = delete;
    ITree& operator=(const ITree&) = delete;
    ITree(ITree&&)                 = default;
    ITree& operator=(ITree&&)      = default;

    /**
     * Build the tree from a subsample of the training data.
     *
     * @param data       Full dataset (all rows are stored here).
     * @param indices    Indices into `data` that form this tree's subsample.
     * @param rng        Caller-supplied RNG (state is advanced in place).
     * @param max_height Maximum recursion depth (tree height limit).
     */
    void build(
        const std::vector<std::vector<double>>& data,
        const std::vector<std::size_t>&         indices,
        std::mt19937&                           rng,
        int                                     max_height
    );

    void build(
        const core::MatrixView&         data,
        const std::vector<std::size_t>& indices,
        std::mt19937&                   rng,
        int                             max_height
    );

    /**
     * Compute the path length h(x) for a sample through this tree.
     *
     * At a leaf node of size n the adjustment c(n) is added to approximate
     * the expected remaining path in an imagined, deeper tree.
     *
     * @throws std::runtime_error if build() was not called first.
     */
    [[nodiscard]] double path_length(const std::vector<double>& sample) const;
    [[nodiscard]] double path_length(const double* sample_ptr) const;
    [[nodiscard]] const Node* root() const noexcept { return root_.get(); }
    void set_root(std::unique_ptr<Node> root) { root_ = std::move(root); }

private:
    std::unique_ptr<Node> root_;

    std::unique_ptr<Node> build_recursive(
        const std::vector<std::vector<double>>& data,
        const std::vector<std::size_t>&         indices,
        int                                     current_depth,
        int                                     max_height,
        std::mt19937&                           rng
    );

    std::unique_ptr<Node> build_recursive(
        const core::MatrixView&         data,
        const std::vector<std::size_t>& indices,
        int                             current_depth,
        int                             max_height,
        std::mt19937&                   rng
    );

    [[nodiscard]] double path_length_recursive(
        const Node*                node,
        const std::vector<double>& sample,
        double                     current_depth
    ) const;

    [[nodiscard]] double path_length_recursive(
        const Node*      node,
        const double*    sample_ptr,
        double           current_depth
    ) const;
};

} // namespace iforest
