#pragma once

#include "models/iforest/itree.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace unified_ml { class AdvancedModelArtifact; }

namespace iforest {

/**
 * Isolation Forest — unsupervised anomaly detector.
 *
 * Reference: Liu, Ting & Zhou, "Isolation Forest", ICDM 2008.
 *
 * Anomaly scores lie in [0, 1]:
 *   ~1.0  →  highly anomalous (isolated in very few splits)
 *   ~0.5  →  indistinguishable from normal observations
 *   ~0.0  →  very deep / dense region (strongly normal)
 *
 * The model is deterministic given a fixed `seed`.
 */
class IsolationForest {
public:
    /**
     * @param n_trees        Number of isolation trees to build.
     * @param subsample_size Rows sampled (without replacement) per tree.
     *                       Values of 256 work well for most datasets.
     * @param max_height     Tree depth limit.  Pass -1 (default) for
     *                       automatic: ceil(log2(subsample_size)).
     * @param seed           RNG seed — same seed → identical results.
     *
     * @throws std::invalid_argument if n_trees <= 0 or subsample_size <= 0.
     */
    explicit IsolationForest(
        int      n_trees        = 100,
        int      subsample_size = 256,
        int      max_height     = -1,
        uint64_t seed           = 42
    );

    /**
     * Train the model on the given dataset.
     *
     * Each inner vector is one sample; all samples must have the same
     * number of features.
     *
     * @throws std::invalid_argument on empty data or inconsistent dimensions.
     */
    void fit(const std::vector<std::vector<double>>& data);

    /**
     * Return the anomaly score for a single sample in [0, 1].
     *
     * @throws std::runtime_error if fit() has not been called.
     */
    [[nodiscard]] double score(const std::vector<double>& sample) const;

    /**
     * Return true when score(sample) >= threshold.
     *
     * The default threshold of 0.5 flags the upper half of the score range,
     * which corresponds to points that are easier to isolate than average.
     */
    [[nodiscard]] bool predict(
        const std::vector<double>& sample,
        double                     threshold = 0.5
    ) const;

    /**
     * Compute anomaly scores for every sample in `data`.
     * Returned vector has the same length as `data`.
     */
    [[nodiscard]] std::vector<double> score_batch(
        const std::vector<std::vector<double>>& data
    ) const;

    void save(const std::string& filepath) const;
    static IsolationForest load(const std::string& filepath);

private:
    int      n_trees_;
    int      subsample_size_;
    int      max_height_;
    uint64_t seed_;
    std::size_t n_samples_ = 0;
    std::size_t n_features_ = 0;

    std::vector<ITree> trees_;

    friend class ::unified_ml::AdvancedModelArtifact;
};

} // namespace iforest
