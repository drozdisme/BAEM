#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <ostream>

namespace dbscan {

/**
 * @brief Represents an N-dimensional point in Euclidean space.
 *
 * Coordinates are stored as doubles. Dimensionality is determined at
 * construction time and is fixed for the lifetime of the object.
 */
class Point {
public:
    // Label assigned after DBSCAN fit. UNVISITED = not yet processed.
    static constexpr int UNVISITED = -2;
    static constexpr int NOISE     = -1;

    /// Construct from a vector of coordinates.
    explicit Point(std::vector<double> coords)
        : coords_(std::move(coords))
        , label_(UNVISITED)
    {
        if (coords_.empty()) {
            throw std::invalid_argument("Point must have at least one dimension.");
        }
    }

    /// Convenience constructor from initializer list.
    Point(std::initializer_list<double> init)
        : Point(std::vector<double>(init))
    {}

    /// Number of dimensions.
    std::size_t dims() const noexcept { return coords_.size(); }

    /// Read-only access to coordinates.
    const std::vector<double>& coords() const noexcept { return coords_; }

    /// Coordinate by index (checked).
    double operator[](std::size_t i) const { return coords_.at(i); }

    /// Cluster label assigned by DBSCAN (-1 = noise, >=0 = cluster id).
    int  label() const noexcept { return label_; }
    void setLabel(int l) noexcept { label_ = l; }

    bool isVisited() const noexcept { return label_ != UNVISITED; }

    /// Pretty-print.
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(";
        for (std::size_t i = 0; i < p.coords_.size(); ++i) {
            os << p.coords_[i];
            if (i + 1 < p.coords_.size()) os << ", ";
        }
        os << ")";
        return os;
    }

private:
    std::vector<double> coords_;
    int                 label_;
};

} // namespace dbscan
