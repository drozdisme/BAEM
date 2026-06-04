#pragma once

#include "models/dbscan/point.hpp"
#include <vector>

namespace dbscan {

/**
 * @brief DBSCAN — Density-Based Spatial Clustering of Applications with Noise.
 *
 * Reference: Ester et al., "A Density-Based Algorithm for Discovering Clusters
 *            in Large Spatial Databases with Noise", KDD 1996.
 *
 * Complexity: O(n²) — naive pairwise distance approach, suitable for small-to-
 * medium datasets. The design is intentionally kept simple and extension-friendly.
 *
 * Usage:
 * @code
 *   DBSCAN dbscan(0.5, 3);           // epsilon=0.5, minPts=3
 *   auto labels = dbscan.fit(points); // returns label per point
 * @endcode
 */
class DBSCAN {
public:
    /**
     * @param epsilon  Neighbourhood radius.  Two points are "neighbours" if
     *                 their Euclidean distance is <= epsilon.
     * @param minPts   Minimum number of neighbours (including the point itself)
     *                 required to classify a point as a *core point*.
     */
    DBSCAN(double epsilon, int minPts);

    /**
     * @brief Runs the DBSCAN algorithm on the given dataset.
     *
     * Each point receives a label:
     *   >= 0  → cluster id (0-based)
     *   -1    → noise
     *
     * The returned vector has the same size as `points` and uses the same
     * index ordering.  The internal state of the Point objects is NOT mutated;
     * labels are only returned through this vector.
     *
     * @throws std::invalid_argument if any two points have different dimensionality.
     * @return Vector of integer labels, one per input point.
     */
    std::vector<int> fit(const std::vector<Point>& points);

    // --- Accessors ---

    double epsilon()  const noexcept { return epsilon_; }
    int    minPts()   const noexcept { return minPts_;  }

    /// Number of clusters found in the last fit() call (excluding noise).
    int numClusters() const noexcept { return numClusters_; }

    /// Number of noise points in the last fit() call.
    int numNoise()    const noexcept { return numNoise_;    }

private:
    // ---- Algorithm internals ----

    /**
     * @brief Expands cluster `clusterId` starting from core point `pointIdx`.
     *
     * Iteratively processes the seed set, assigning all density-reachable
     * points to the cluster.
     */
    void expandCluster(const std::vector<Point>& points,
                       std::vector<int>&         labels,
                       std::size_t               pointIdx,
                       int                       clusterId,
                       const std::vector<std::size_t>& initialNeighbours);

    // ---- Parameters ----
    double epsilon_;
    int    minPts_;

    // ---- State (valid after fit()) ----
    int numClusters_ = 0;
    int numNoise_    = 0;
};

} // namespace dbscan
