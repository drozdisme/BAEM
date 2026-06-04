#include "models/pca/pca.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pca {

namespace {
constexpr char kMagic[8] = {'U','M','L','P','C','A','1','\0'};

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
  const char* p = reinterpret_cast<const char*>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
  if (off + sizeof(T) > in.size()) throw std::runtime_error("PCA deserialize: truncated payload");
  T v{};
  std::memcpy(&v, in.data() + off, sizeof(T));
  off += sizeof(T);
  return v;
}
}

PCA::PCA(std::size_t n_components) : n_components_(n_components) {}

namespace {
core::Matrix to_matrix(const core::MatrixView& view) {
  core::Matrix out(view.rows, view.cols, 0.0);
  for (std::size_t i = 0; i < view.rows; ++i)
    for (std::size_t j = 0; j < view.cols; ++j)
      out(i, j) = view(i, j);
  return out;
}
}  // namespace

void PCA::fit(const core::Matrix& X) {
  const std::size_t n = X.rows(), d = X.cols();
  if (n < 2) throw std::invalid_argument("PCA::fit: need at least 2 samples");
  if (d == 0) throw std::invalid_argument("PCA::fit: data has no features");
  n_features_ = d;

  // Mean            
  mean_ = core::Vector(d, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < d; ++j) mean_[j] += X(i, j);
  for (std::size_t j = 0; j < d; ++j) mean_[j] /= static_cast<double>(n);

  core::Matrix Xc = center(X);
  const double denom = static_cast<double>(n - 1);

  // Covariance — deterministic sequential accumulation     
  // Intentionally NOT parallelised: floating-point reduction order must be
  // identical across calls for the same input data.  OMP thread scheduling
  // makes the critical-section merge order non-deterministic, which produces
  // subtly different covariance values across fits.  For near-degenerate
  // eigenvalues this breaks the INV-PCA-PLAT reproducibility invariant.
  // For typical d (≤ a few hundred features) this loop is cheap.
  core::Matrix cov(d, d, 0.0);
  {
    std::vector<double> local(d * d, 0.0);
    for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t i = 0; i < d; ++i) {
      const double xki = Xc(k, i);
      for (std::size_t j = i; j < d; ++j)
        local[i * d + j] += xki * Xc(k, j);
    }
    }
    for (std::size_t i = 0; i < d; ++i)
    for (std::size_t j = i; j < d; ++j) {
      const double v = local[i * d + j] / denom;
      cov(i, j) = v;
      if (j != i) cov(j, i) = v;
    }
  }

  auto [eigenvalues, eigenvectors] = core::jacobi_eigen(cov);

  total_variance_ = 0.0;
  for (std::size_t i = 0; i < d; ++i)
    total_variance_ += std::max(0.0, eigenvalues[i]);

  std::size_t k = (n_components_ == 0 || n_components_ > d) ? d : n_components_;
  n_components_ = k;
  eigenvalues_ = core::Vector(k);
  components_  = core::Matrix(k, d);

  for (std::size_t i = 0; i < k; ++i) {
    eigenvalues_[i] = std::max(0.0, eigenvalues[i]);
    for (std::size_t j = 0; j < d; ++j) components_(i, j) = eigenvectors(j, i);
    double max_abs = 0.0, sign = 1.0;
    for (std::size_t j = 0; j < d; ++j) {
    double a = std::abs(components_(i, j));
    if (a > max_abs) { max_abs = a; sign = (components_(i, j) < 0.0) ? -1.0 : 1.0; }
    }
    if (sign < 0.0)
    for (std::size_t j = 0; j < d; ++j) components_(i, j) = -components_(i, j);
  }
  fitted_ = true;
}

void PCA::fit(const core::MatrixView& X) {
  fit(to_matrix(X));
}

core::Matrix PCA::transform(const core::Matrix& X, std::size_t n_comp) const {
  check_fitted();
  if (X.cols() != n_features_)
    throw std::invalid_argument("PCA::transform: feature dimension mismatch");
  const std::size_t k  = (n_comp == 0) ? n_components_ : std::min(n_comp, n_components_);
  const std::size_t ns = X.rows();
  core::Matrix Xc = center(X);
  core::Matrix Z(ns, k, 0.0);

  // Parallel projection         
  // Each sample s is independent: Z[s,i] = dot(Xc[s,:], components[i,:])
  #pragma omp parallel for schedule(static)
  for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(ns); ++s) {
    for (std::size_t i = 0; i < k; ++i) {
    double acc = 0.0;
    #pragma omp simd reduction(+:acc)
    for (std::size_t j = 0; j < n_features_; ++j)
      acc += Xc(s, j) * components_(i, j);
    Z(s, i) = acc;
    }
  }
  return Z;
}

core::Matrix PCA::transform(const core::MatrixView& X, std::size_t n_comp) const {
  return transform(to_matrix(X), n_comp);
}

core::Matrix PCA::fit_transform(const core::Matrix& X, std::size_t n_comp) {
  fit(X); return transform(X, n_comp);
}

core::Matrix PCA::inverse_transform(const core::Matrix& Z) const {
  check_fitted();
  const std::size_t k = Z.cols();
  if (k > n_components_)
    throw std::invalid_argument("PCA::inverse_transform: Z has more columns than fitted components");
  const std::size_t ns = Z.rows();
  core::Matrix X_approx(ns, n_features_);

  #pragma omp parallel for schedule(static)
  for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(ns); ++s)
    for (std::size_t j = 0; j < n_features_; ++j) {
    double val = mean_[j];
    #pragma omp simd reduction(+:val)
    for (std::size_t i = 0; i < k; ++i) val += Z(s, i) * components_(i, j);
    X_approx(s, j) = val;
    }
  return X_approx;
}

const core::Matrix& PCA::components()   const { check_fitted(); return components_; }
const core::Vector& PCA::explained_variance() const { check_fitted(); return eigenvalues_; }
const core::Vector& PCA::mean()     const { check_fitted(); return mean_; }

core::Vector PCA::explained_variance_ratio() const {
  check_fitted();
  core::Vector ratio(eigenvalues_.size(), 0.0);
  if (total_variance_ > 0.0)
    for (std::size_t i = 0; i < eigenvalues_.size(); ++i)
    ratio[i] = eigenvalues_[i] / total_variance_;
  return ratio;
}

void PCA::save(const std::string& filepath) const {
  check_fitted();

  std::vector<char> payload;
  write_raw(payload, n_components_);
  write_raw(payload, n_features_);
  write_raw(payload, total_variance_);
  write_raw(payload, fitted_);

  const std::size_t mean_size = mean_.size();
  write_raw(payload, mean_size);
  for (std::size_t i = 0; i < mean_size; ++i) write_raw(payload, mean_[i]);

  write_raw(payload, components_.rows());
  write_raw(payload, components_.cols());
  for (std::size_t r = 0; r < components_.rows(); ++r)
    for (std::size_t c = 0; c < components_.cols(); ++c)
      write_raw(payload, components_(r, c));

  const std::size_t eig_size = eigenvalues_.size();
  write_raw(payload, eig_size);
  for (std::size_t i = 0; i < eig_size; ++i) write_raw(payload, eigenvalues_[i]);

  std::ofstream out(filepath, std::ios::binary);
  if (!out) throw std::runtime_error("PCA::save: unable to open file");
  out.write(kMagic, sizeof(kMagic));
  const std::size_t payload_size = payload.size();
  out.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

PCA PCA::load(const std::string& filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in) throw std::runtime_error("PCA::load: unable to open file");

  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("PCA::load: invalid file header");

  std::size_t payload_size = 0;
  in.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
  std::vector<char> payload(payload_size);
  in.read(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (!in) throw std::runtime_error("PCA::load: truncated payload");

  std::size_t off = 0;
  PCA model;
  model.n_components_ = read_raw<std::size_t>(payload, off);
  model.n_features_ = read_raw<std::size_t>(payload, off);
  model.total_variance_ = read_raw<double>(payload, off);
  model.fitted_ = read_raw<bool>(payload, off);

  const std::size_t mean_size = read_raw<std::size_t>(payload, off);
  model.mean_ = core::Vector(mean_size, 0.0);
  for (std::size_t i = 0; i < mean_size; ++i) model.mean_[i] = read_raw<double>(payload, off);

  const std::size_t comp_rows = read_raw<std::size_t>(payload, off);
  const std::size_t comp_cols = read_raw<std::size_t>(payload, off);
  model.components_ = core::Matrix(comp_rows, comp_cols, 0.0);
  for (std::size_t r = 0; r < comp_rows; ++r)
    for (std::size_t c = 0; c < comp_cols; ++c)
      model.components_(r, c) = read_raw<double>(payload, off);

  const std::size_t eig_size = read_raw<std::size_t>(payload, off);
  model.eigenvalues_ = core::Vector(eig_size, 0.0);
  for (std::size_t i = 0; i < eig_size; ++i) model.eigenvalues_[i] = read_raw<double>(payload, off);

  return model;
}

void PCA::check_fitted() const {
  if (!fitted_) throw std::runtime_error("PCA: not fitted — call fit() first");
}

core::Matrix PCA::center(const core::Matrix& X) const {
  const std::size_t nr = X.rows(), nc = X.cols();
  core::Matrix Xc(nr, nc);
  // Vectorised centering — each row is independent
  #pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nr); ++i) {
    #pragma omp simd
    for (std::size_t j = 0; j < nc; ++j)
    Xc(i, j) = X(i, j) - mean_[j];
  }
  return Xc;
}

} // namespace pca
