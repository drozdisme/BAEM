#include "models/dbscan/dbscan.hpp"
#include "models/dbscan/utils.hpp"

#include <stdexcept>
#include <queue>
#include <algorithm>

namespace dbscan {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DBSCAN::DBSCAN(double epsilon, int minPts)
  : epsilon_(epsilon), minPts_(minPts)
{
  if (epsilon <= 0.0) {
    throw std::invalid_argument("DBSCAN: epsilon must be > 0.");
  }
  if (minPts <= 0) {
    throw std::invalid_argument("DBSCAN: minPts must be > 0.");
  }
}

// ---------------------------------------------------------------------------
// fit()
// ---------------------------------------------------------------------------

std::vector<int> DBSCAN::fit(const std::vector<Point>& points) {
  if (points.empty()) {
    numClusters_ = 0;
    numNoise_  = 0;
    return {};
  }

  // Validate that all points share the same dimensionality.
  const std::size_t dims = points[0].dims();
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (points[i].dims() != dims) {
    throw std::invalid_argument(
      "DBSCAN::fit: all points must have the same dimensionality.");
    }
  }

  // Working label array; UNVISITED until processed.
  std::vector<int> labels(points.size(), Point::UNVISITED);

  int currentCluster = 0;

  for (std::size_t i = 0; i < points.size(); ++i) {
    // Skip already-assigned points.
    if (labels[i] != Point::UNVISITED) continue;

    // Query the epsilon-neighbourhood of point i.
    auto neighbours = utils::regionQuery(points, i, epsilon_);

    // Not enough neighbours → mark as noise (may be revised later).
    if (static_cast<int>(neighbours.size()) < minPts_) {
    labels[i] = Point::NOISE;
    continue;
    }

    // Core point: start a new cluster and expand it.
    expandCluster(points, labels, i, currentCluster, neighbours);
    ++currentCluster;
  }

  // Collect statistics.
  numClusters_ = currentCluster;
  numNoise_ = 0;
  for (int lbl : labels) {
    if (lbl == Point::NOISE) ++numNoise_;
  }

  return labels;
}

// ---------------------------------------------------------------------------
// expandCluster()
// ---------------------------------------------------------------------------

void DBSCAN::expandCluster(const std::vector<Point>& points,
          std::vector<int>&   labels,
          std::size_t     pointIdx,
          int         clusterId,
          const std::vector<std::size_t>& initialNeighbours)
{
  // Assign the core point itself.
  labels[pointIdx] = clusterId;

  // Use a queue to process the seed set iteratively (BFS-style expansion).
  // This avoids recursion depth issues on large datasets.
  std::queue<std::size_t> seeds;
  for (std::size_t idx : initialNeighbours) {
    if (idx != pointIdx) seeds.push(idx);
  }

  while (!seeds.empty()) {
    const std::size_t current = seeds.front();
    seeds.pop();

    // If this point was previously labelled noise it is density-reachable
    // from a core point, so it becomes a border point of this cluster.
    if (labels[current] == Point::NOISE) {
    labels[current] = clusterId;
    continue; // Border points do not expand the cluster further.
    }

    // Skip already-assigned points (assigned to this or another cluster).
    if (labels[current] != Point::UNVISITED) continue;

    // Assign to current cluster.
    labels[current] = clusterId;

    // Query neighbours of this point.
    auto neighbours = utils::regionQuery(points, current, epsilon_);

    // If it is also a core point, add its unvisited neighbours to the queue.
    if (static_cast<int>(neighbours.size()) >= minPts_) {
    for (std::size_t nb : neighbours) {
      if (labels[nb] == Point::UNVISITED || labels[nb] == Point::NOISE) {
        seeds.push(nb);
      }
    }
    }
  }
}

} // namespace dbscan
