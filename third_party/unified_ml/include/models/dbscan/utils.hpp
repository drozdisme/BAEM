#pragma once

#include "models/dbscan/point.hpp"
#include <vector>
#include <string>
#include <ostream>
#include <unordered_map>

namespace dbscan {
namespace utils {

/**
 * @brief Computes the squared Euclidean distance between two points.
 *
 * Using squared distance avoids a sqrt call, which is useful when only
 * comparing distances to a threshold (epsilon²).
 *
 * @throws std::invalid_argument if points have different dimensionality.
 */
double squaredEuclideanDistance(const Point& a, const Point& b);

/**
 * @brief Computes the Euclidean distance between two points.
 *
 * @throws std::invalid_argument if points have different dimensionality.
 */
double euclideanDistance(const Point& a, const Point& b);

/**
 * @brief Returns all indices in `points` whose distance from `center` is <= epsilon.
 *
 * Uses squared distance comparison to avoid sqrt where possible.
 */
std::vector<std::size_t> regionQuery(const std::vector<Point>& points,
                                     std::size_t               centerIdx,
                                     double                    epsilon);

/**
 * @brief Groups point indices by their cluster label.
 *
 * Returns a map where:
 *   key = cluster label (>= 0)
 *   value = sorted vector of point indices belonging to that cluster
 * Noise points (label == -1) are excluded.
 */
std::unordered_map<int, std::vector<std::size_t>>
groupByCluster(const std::vector<Point>& points);

/**
 * @brief Prints a human-readable cluster summary to the given output stream.
 */
void printClusterSummary(const std::vector<Point>& points, std::ostream& out);

/**
 * @brief Prints every point with its assigned label.
 */
void printPointLabels(const std::vector<Point>& points, std::ostream& out);

} // namespace utils
} // namespace dbscan
