#include "models/gp/gaussian_process.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gp {

namespace {

constexpr char kMagic[8] = {'U','M','L','G','P','v','7','\0'};
constexpr std::uint32_t kVersion = 7;

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
    if (off + sizeof(T) > in.size()) throw std::runtime_error("GP deserialize: truncated payload");
    T v{};
    std::memcpy(&v, in.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
}

std::uint64_t fnv1a64(const std::vector<char>& data) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : data) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

std::vector<double> cholesky_factorize(const std::vector<double>& a,
                                       std::size_t n,
                                       double base_jitter) {
    double jitter = std::max(0.0, base_jitter);

    for (int attempt = 0; attempt < 10; ++attempt) {
        std::vector<double> L(n * n, 0.0);
        bool ok = true;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double sum = a[i * n + j];
                if (i == j) sum += jitter;
                for (std::size_t k = 0; k < j; ++k) sum -= L[i * n + k] * L[j * n + k];

                if (i == j) {
                    if (!(sum > 0.0) || !std::isfinite(sum)) {
                        ok = false;
                        break;
                    }
                    L[i * n + j] = std::sqrt(sum);
                } else {
                    const double denom = L[j * n + j];
                    if (!(denom > 0.0) || !std::isfinite(denom)) {
                        ok = false;
                        break;
                    }
                    L[i * n + j] = sum / denom;
                }
            }
            if (!ok) break;
        }

        if (ok) return L;
        jitter = (jitter == 0.0) ? 1e-12 : jitter * 10.0;
    }

    throw std::runtime_error("GP::fit: covariance matrix is not numerically positive definite even after jitter escalation");
}

}  // namespace

GaussianProcessRegressor::GaussianProcessRegressor(const GPParams& params) : params_(params) {
    if (!(params_.length_scale > 0.0)) throw std::invalid_argument("GP: length_scale must be > 0");
    if (!(params_.signal_variance > 0.0)) throw std::invalid_argument("GP: signal_variance must be > 0");
    if (params_.noise_variance < 0.0) throw std::invalid_argument("GP: noise_variance must be >= 0");
    if (params_.jitter < 0.0) throw std::invalid_argument("GP: jitter must be >= 0");
    if (params_.max_training_points == 0) throw std::invalid_argument("GP: max_training_points must be > 0");
    if (params_.output_correlation < -0.999 || params_.output_correlation > 0.999) throw std::invalid_argument("GP: output_correlation must be in [-0.999, 0.999]");
    if (params_.inducing_points == 0) throw std::invalid_argument("GP: inducing_points must be > 0");
    if (!(params_.alpha > 0.0)) throw std::invalid_argument("GP: alpha must be > 0");
    if (!(params_.period > 0.0)) throw std::invalid_argument("GP: period must be > 0");
}

double GaussianProcessRegressor::kernel_eval(const std::vector<double>& a,
                                             const std::vector<double>& b) const {
    double dist2 = 0.0;
    double dist = 0.0;
    double dot = 0.0;
    double periodic_sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        dist2 += d * d;
        dot += a[i] * b[i];
        periodic_sum += std::sin(M_PI * d / params_.period) * std::sin(M_PI * d / params_.period);
    }
    dist = std::sqrt(dist2);

    switch (params_.kernel) {
        case KernelType::Linear:
            return params_.signal_variance * dot;
        case KernelType::Matern32: {
            const double r = std::sqrt(3.0) * dist / params_.length_scale;
            return params_.signal_variance * (1.0 + r) * std::exp(-r);
        }
        case KernelType::Matern52: {
            const double r = std::sqrt(5.0) * dist / params_.length_scale;
            return params_.signal_variance * (1.0 + r + (5.0 * dist2) / (3.0 * params_.length_scale * params_.length_scale)) * std::exp(-r);
        }
        case KernelType::RationalQuadratic: {
            const double base = 1.0 + dist2 / (2.0 * params_.alpha * params_.length_scale * params_.length_scale);
            return params_.signal_variance * std::pow(base, -params_.alpha);
        }
        case KernelType::Periodic:
            return params_.signal_variance * std::exp(-2.0 * periodic_sum / (params_.length_scale * params_.length_scale));
        case KernelType::RBFPlusLinear:
            return params_.signal_variance * std::exp(-0.5 * dist2 / (params_.length_scale * params_.length_scale)) + 0.5 * params_.signal_variance * dot;
        case KernelType::RBFPlusPeriodic:
            return params_.signal_variance * std::exp(-0.5 * dist2 / (params_.length_scale * params_.length_scale)) + 0.5 * params_.signal_variance * std::exp(-2.0 * periodic_sum / (params_.length_scale * params_.length_scale));
        case KernelType::RBF:
        default:
            return params_.signal_variance * std::exp(-0.5 * dist2 / (params_.length_scale * params_.length_scale));
    }
}

std::vector<double> GaussianProcessRegressor::solve_lower_triangular(const std::vector<double>& L,
                                                                     const std::vector<double>& b) const {
    const std::size_t n = b.size();
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = b[i];
        for (std::size_t j = 0; j < i; ++j) sum -= L[i * n + j] * x[j];
        const double diag = L[i * n + i];
        if (!(diag > 0.0) || !std::isfinite(diag)) throw std::runtime_error("GP: lower-triangular solve encountered invalid diagonal");
        x[i] = sum / diag;
    }
    return x;
}

std::vector<double> GaussianProcessRegressor::solve_upper_triangular(const std::vector<double>& L,
                                                                     const std::vector<double>& b) const {
    const std::size_t n = b.size();
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) sum -= L[j * n + i] * x[j];
        const double diag = L[i * n + i];
        if (!(diag > 0.0) || !std::isfinite(diag)) throw std::runtime_error("GP: upper-triangular solve encountered invalid diagonal");
        x[i] = sum / diag;
    }
    return x;
}

std::vector<std::size_t> GaussianProcessRegressor::select_active_set(std::size_t n) const {
    const std::size_t limit = std::min(n, params_.inducing_points);
    std::vector<std::size_t> idx(limit, 0);
    if (limit == n) {
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    }
    const double step = static_cast<double>(n - 1) / static_cast<double>(limit - 1);
    for (std::size_t i = 0; i < limit; ++i) idx[i] = static_cast<std::size_t>(std::round(i * step));
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    while (idx.size() < limit) idx.push_back(idx.back() + 1 < n ? idx.back() + 1 : idx.back());
    return idx;
}

void GaussianProcessRegressor::fit_internal(const std::vector<std::vector<double>>& X,
                                            const std::vector<std::vector<double>>& Y) {
    if (X.empty() || X.size() != Y.size()) throw std::invalid_argument("GP::fit: invalid dataset");
    feature_dim_ = X.front().size();
    if (feature_dim_ == 0) throw std::invalid_argument("GP::fit: feature dimension must be > 0");
    output_dim_ = Y.front().size();
    if (output_dim_ == 0) throw std::invalid_argument("GP::fit: output dimension must be > 0");
    for (const auto& row : X) if (row.size() != feature_dim_) throw std::invalid_argument("GP::fit: inconsistent feature dims");
    for (const auto& row : Y) if (row.size() != output_dim_) throw std::invalid_argument("GP::fit: inconsistent target dims");

    X_train_ = X;
    Y_train_ = Y;
    target_mean_.assign(output_dim_, 0.0);
    target_scale_.assign(output_dim_, 1.0);

    std::vector<std::vector<double>> normalized_Y = Y;
    if (params_.normalize_targets) {
        for (std::size_t out = 0; out < output_dim_; ++out) {
            double mean = 0.0;
            for (const auto& row : Y) mean += row[out];
            mean /= static_cast<double>(Y.size());
            double var = 0.0;
            for (const auto& row : Y) {
                const double d = row[out] - mean;
                var += d * d;
            }
            var /= static_cast<double>(Y.size());
            target_mean_[out] = mean;
            target_scale_[out] = std::sqrt(std::max(var, 1e-12));
            for (auto& row : normalized_Y) row[out] = (row[out] - mean) / target_scale_[out];
        }
    }

    std::vector<std::size_t> active_idx;
    if (params_.approximation == ApproximationType::Exact) {
        if (X.size() > params_.max_training_points) throw std::invalid_argument("GP::fit: dataset exceeds max_training_points contract for exact mode");
        active_idx.resize(X.size());
        std::iota(active_idx.begin(), active_idx.end(), 0);
    } else {
        active_idx = select_active_set(X.size());
    }

    active_X_.assign(active_idx.size(), {});
    active_Y_.assign(active_idx.size(), std::vector<double>(output_dim_, 0.0));
    for (std::size_t i = 0; i < active_idx.size(); ++i) {
        active_X_[i] = X[active_idx[i]];
        active_Y_[i] = normalized_Y[active_idx[i]];
    }

    const std::size_t n = active_X_.size();
    std::vector<double> K(n * n, 0.0);
    inducing_diag_correction_.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double kv = kernel_eval(active_X_[i], active_X_[j]);
            K[i * n + j] = kv;
            K[j * n + i] = kv;
        }
        K[i * n + i] += params_.noise_variance;
        if (params_.approximation != ApproximationType::Exact && params_.use_inducing_correction) {
            const double self_k = kernel_eval(active_X_[i], active_X_[i]);
            inducing_diag_correction_[i] = std::max(0.0, self_k - K[i * n + i] + params_.noise_variance);
            K[i * n + i] += inducing_diag_correction_[i];
        }
    }

    cholesky_L_ = cholesky_factorize(K, n, params_.jitter);
    alpha_vectors_.assign(output_dim_, std::vector<double>(n, 0.0));
    for (std::size_t out = 0; out < output_dim_; ++out) {
        std::vector<double> y_vec(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) y_vec[i] = active_Y_[i][out];
        alpha_vectors_[out] = solve_upper_triangular(cholesky_L_, solve_lower_triangular(cholesky_L_, y_vec));
    }

    if (output_dim_ > 1 && coregionalization_matrix_.empty()) {
        coregionalization_matrix_.assign(output_dim_, std::vector<double>(output_dim_, 0.0));
        for (std::size_t i = 0; i < output_dim_; ++i) {
            coregionalization_matrix_[i][i] = 1.0;
            for (std::size_t j = i + 1; j < output_dim_; ++j) {
                coregionalization_matrix_[i][j] = params_.output_correlation;
                coregionalization_matrix_[j][i] = params_.output_correlation;
            }
        }
    }

    last_log_marginal_likelihood_ = 0.0;
    for (std::size_t out = 0; out < output_dim_; ++out) {
        std::vector<double> y_vec(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) y_vec[i] = active_Y_[i][out];
        double quad = 0.0;
        for (std::size_t i = 0; i < n; ++i) quad += y_vec[i] * alpha_vectors_[out][i];
        double log_det = 0.0;
        for (std::size_t i = 0; i < n; ++i) log_det += std::log(cholesky_L_[i * n + i]);
        last_log_marginal_likelihood_ += -0.5 * quad - log_det - 0.5 * static_cast<double>(n) * std::log(2.0 * M_PI);
    }

    fitted_ = true;
}

void GaussianProcessRegressor::fit(const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y) {
    std::vector<std::vector<double>> Y(y.size(), std::vector<double>(1, 0.0));
    for (std::size_t i = 0; i < y.size(); ++i) Y[i][0] = (params_.task == TaskType::BinaryClassification) ? (y[i] > 0.0 ? 1.0 : -1.0) : y[i];
    fit_internal(X, Y);
}

void GaussianProcessRegressor::fit_multi_output(const std::vector<std::vector<double>>& X,
                                                const std::vector<std::vector<double>>& Y) {
    fit_internal(X, Y);
}



void GaussianProcessRegressor::append_observation(const std::vector<double>& x,
                                                 double y,
                                                 bool refit) {
    append_multi_output_observation(x, {y}, refit);
}

void GaussianProcessRegressor::append_multi_output_observation(const std::vector<double>& x,
                                                              const std::vector<double>& y,
                                                              bool refit) {
    if (fitted_) {
        if (x.size() != feature_dim_) throw std::invalid_argument("GP::append_multi_output_observation: feature dimension mismatch");
        if (y.size() != output_dim_) throw std::invalid_argument("GP::append_multi_output_observation: output dimension mismatch");
    }
    X_train_.push_back(x);
    Y_train_.push_back(y);
    if (refit) fit_internal(X_train_, Y_train_);
}

void GaussianProcessRegressor::set_coregionalization_matrix(const std::vector<std::vector<double>>& B) {
    if (B.empty() || B.size() != B.front().size()) throw std::invalid_argument("GP::set_coregionalization_matrix: matrix must be square");
    for (const auto& row : B) if (row.size() != B.size()) throw std::invalid_argument("GP::set_coregionalization_matrix: inconsistent row size");
    coregionalization_matrix_ = B;
}

void GaussianProcessRegressor::optimize_hyperparameters(const std::vector<std::vector<double>>& X,
                                                        const std::vector<double>& y,
                                                        const HyperparameterSearchSpace& search_space) {
    if (search_space.length_scales.empty() || search_space.signal_variances.empty() || search_space.noise_variances.empty()) {
        throw std::invalid_argument("GP::optimize_hyperparameters: search space must be non-empty");
    }
    GPParams best = params_;
    double best_ll = -std::numeric_limits<double>::infinity();
    for (double ls : search_space.length_scales) {
        for (double sv : search_space.signal_variances) {
            for (double nv : search_space.noise_variances) {
                GPParams candidate = params_;
                candidate.length_scale = ls;
                candidate.signal_variance = sv;
                candidate.noise_variance = nv;
                GaussianProcessRegressor probe(candidate);
                probe.fit(X, y);
                const double ll = probe.log_marginal_likelihood();
                if (ll > best_ll) {
                    best_ll = ll;
                    best = candidate;
                }
            }
        }
    }
    params_ = best;
    fit(X, y);
}

std::vector<double> GaussianProcessRegressor::predict_mean_variance_internal(const std::vector<double>& x,
                                                                             std::size_t output_index,
                                                                             double* variance) const {
    const std::size_t n = active_X_.size();
    std::vector<double> k(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) k[i] = kernel_eval(active_X_[i], x);

    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += k[i] * alpha_vectors_[output_index][i];
    mean = mean * target_scale_[output_index] + target_mean_[output_index];

    if (variance != nullptr) {
        const auto v = solve_lower_triangular(cholesky_L_, k);
        double quad = 0.0;
        for (double value : v) quad += value * value;
        double var = kernel_eval(x, x) - quad;
        if (!inducing_diag_correction_.empty()) {
            const double avg_corr = std::accumulate(inducing_diag_correction_.begin(), inducing_diag_correction_.end(), 0.0) / static_cast<double>(inducing_diag_correction_.size());
            var += 0.1 * avg_corr;
        }
        if (var < 0.0 && var > -1e-10) var = 0.0;
        if (var < 0.0 || !std::isfinite(var)) throw std::runtime_error("GP::predict_one: predictive variance became invalid");
        *variance = var * target_scale_[output_index] * target_scale_[output_index];
    }

    return {mean};
}


std::vector<std::vector<double>> GaussianProcessRegressor::build_output_correlation_matrix(const std::vector<double>& variances) const {
    std::vector<std::vector<double>> cov(output_dim_, std::vector<double>(output_dim_, 0.0));
    for (std::size_t i = 0; i < output_dim_; ++i) {
        cov[i][i] = variances[i];
        for (std::size_t j = i + 1; j < output_dim_; ++j) {
            const double corr = (!coregionalization_matrix_.empty() && i < coregionalization_matrix_.size() && j < coregionalization_matrix_[i].size()) ? coregionalization_matrix_[i][j] : params_.output_correlation;
            const double shared = corr * std::sqrt(std::max(0.0, variances[i] * variances[j]));
            cov[i][j] = shared;
            cov[j][i] = shared;
        }
    }
    return cov;
}

Prediction GaussianProcessRegressor::predict_one(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("GP::predict_one: model is not fitted");
    if (x.size() != feature_dim_) throw std::invalid_argument("GP::predict_one: feature dimension mismatch");
    if (output_dim_ != 1) throw std::runtime_error("GP::predict_one: use multi-output API for vector targets");

    double variance = 0.0;
    const double mean = predict_mean_variance_internal(x, 0, &variance).front();
    Prediction pred;
    pred.mean = mean;
    pred.variance = variance;
    if (params_.task == TaskType::BinaryClassification) {
        pred.probability = 1.0 / (1.0 + std::exp(-mean / std::sqrt(1.0 + variance)));
        pred.label = pred.probability >= 0.5 ? 1 : 0;
    }
    return pred;
}

std::vector<double> GaussianProcessRegressor::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> out(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = predict_one(X[i]).mean;
    return out;
}

std::vector<Prediction> GaussianProcessRegressor::predict_with_uncertainty(const std::vector<std::vector<double>>& X) const {
    std::vector<Prediction> out(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = predict_one(X[i]);
    return out;
}

std::vector<double> GaussianProcessRegressor::predict_multi_output(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("GP::predict_multi_output: model is not fitted");
    if (x.size() != feature_dim_) throw std::invalid_argument("GP::predict_multi_output: feature dimension mismatch");
    std::vector<double> out(output_dim_, 0.0);
    for (std::size_t i = 0; i < output_dim_; ++i) out[i] = predict_mean_variance_internal(x, i, nullptr).front();
    return out;
}

std::vector<std::vector<double>> GaussianProcessRegressor::predict_multi_output_covariance(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("GP::predict_multi_output_covariance: model is not fitted");
    if (x.size() != feature_dim_) throw std::invalid_argument("GP::predict_multi_output_covariance: feature dimension mismatch");
    std::vector<double> vars(output_dim_, 0.0);
    for (std::size_t i = 0; i < output_dim_; ++i) {
        (void)predict_mean_variance_internal(x, i, &vars[i]);
    }
    return build_output_correlation_matrix(vars);
}

std::vector<std::vector<double>> GaussianProcessRegressor::predict_multi_output(const std::vector<std::vector<double>>& X) const {
    std::vector<std::vector<double>> out(X.size(), std::vector<double>(output_dim_, 0.0));
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = predict_multi_output(X[i]);
    return out;
}


std::vector<double> GaussianProcessRegressor::predict_class_probabilities(const std::vector<std::vector<double>>& X) const {
    if (params_.task != TaskType::BinaryClassification) throw std::runtime_error("GP::predict_class_probabilities: task is not binary classification");
    std::vector<double> out(X.size(), 0.0);
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = predict_one(X[i]).probability;
    return out;
}

double GaussianProcessRegressor::log_marginal_likelihood() const {
    if (!fitted_) throw std::runtime_error("GP::log_marginal_likelihood: model is not fitted");
    return last_log_marginal_likelihood_;
}

AcquisitionResult GaussianProcessRegressor::select_next_point_ucb(const std::vector<std::vector<double>>& candidates,
                                                                  double beta) const {
    if (!fitted_) throw std::runtime_error("GP::select_next_point_ucb: model is not fitted");
    if (candidates.empty()) throw std::invalid_argument("GP::select_next_point_ucb: candidates must be non-empty");
    AcquisitionResult best;
    best.score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto pred = predict_one(candidates[i]);
        const double score = pred.mean + beta * std::sqrt(std::max(0.0, pred.variance));
        if (score > best.score) best = {i, score, pred};
    }
    return best;
}

AcquisitionResult GaussianProcessRegressor::select_next_point_expected_improvement(const std::vector<std::vector<double>>& candidates,
                                                                                   double best_observed,
                                                                                   double xi) const {
    if (!fitted_) throw std::runtime_error("GP::select_next_point_expected_improvement: model is not fitted");
    if (candidates.empty()) throw std::invalid_argument("GP::select_next_point_expected_improvement: candidates must be non-empty");
    AcquisitionResult best;
    best.score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto pred = predict_one(candidates[i]);
        const double sigma = std::sqrt(std::max(0.0, pred.variance));
        double score = 0.0;
        if (sigma > 1e-12) {
            const double z = (pred.mean - best_observed - xi) / sigma;
            const double pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
            const double cdf = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
            score = (pred.mean - best_observed - xi) * cdf + sigma * pdf;
        }
        if (score > best.score) best = {i, score, pred};
    }
    return best;
}

void GaussianProcessRegressor::save(const std::string& filepath) const {
    if (!fitted_) throw std::runtime_error("GP::save: model is not fitted");

    std::vector<char> payload;
    write_raw(payload, static_cast<int>(params_.kernel));
    write_raw(payload, static_cast<int>(params_.approximation));
    write_raw(payload, static_cast<int>(params_.task));
    write_raw(payload, params_.length_scale);
    write_raw(payload, params_.signal_variance);
    write_raw(payload, params_.noise_variance);
    write_raw(payload, params_.jitter);
    write_raw(payload, params_.alpha);
    write_raw(payload, params_.period);
    write_raw(payload, params_.max_training_points);
    write_raw(payload, params_.inducing_points);
    write_raw(payload, params_.normalize_targets);
    write_raw(payload, params_.output_correlation);
    write_raw(payload, params_.use_inducing_correction);
    write_raw(payload, fitted_);
    write_raw(payload, feature_dim_);
    write_raw(payload, output_dim_);
    write_raw(payload, last_log_marginal_likelihood_);

    write_raw(payload, X_train_.size());
    for (const auto& row : X_train_) for (double v : row) write_raw(payload, v);
    write_raw(payload, Y_train_.size());
    for (const auto& row : Y_train_) for (double v : row) write_raw(payload, v);

    write_raw(payload, active_X_.size());
    for (const auto& row : active_X_) for (double v : row) write_raw(payload, v);
    for (const auto& row : active_Y_) for (double v : row) write_raw(payload, v);

    write_raw(payload, target_mean_.size());
    for (double v : target_mean_) write_raw(payload, v);
    write_raw(payload, target_scale_.size());
    for (double v : target_scale_) write_raw(payload, v);

    write_raw(payload, cholesky_L_.size());
    for (double v : cholesky_L_) write_raw(payload, v);
    write_raw(payload, inducing_diag_correction_.size());
    for (double v : inducing_diag_correction_) write_raw(payload, v);
    write_raw(payload, coregionalization_matrix_.size());
    for (const auto& row : coregionalization_matrix_) for (double v : row) write_raw(payload, v);
    write_raw(payload, alpha_vectors_.size());
    for (const auto& row : alpha_vectors_) {
        write_raw(payload, row.size());
        for (double v : row) write_raw(payload, v);
    }

    const std::uint64_t checksum = fnv1a64(payload);
    const std::uint64_t payload_size = static_cast<std::uint64_t>(payload.size());

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("GP::save: cannot open file");
    ofs.write(kMagic, sizeof(kMagic));
    ofs.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    ofs.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
    ofs.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

GaussianProcessRegressor GaussianProcessRegressor::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) throw std::runtime_error("GP::load: cannot open file");

    char magic[8]{};
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0, checksum = 0;
    ifs.read(magic, sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
    ifs.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("GP::load: invalid magic");
    if (version != kVersion) throw std::runtime_error("GP::load: unsupported version");

    std::vector<char> payload(static_cast<std::size_t>(payload_size));
    ifs.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (fnv1a64(payload) != checksum) throw std::runtime_error("GP::load: checksum mismatch");

    std::size_t off = 0;
    GPParams params;
    params.kernel = static_cast<KernelType>(read_raw<int>(payload, off));
    params.approximation = static_cast<ApproximationType>(read_raw<int>(payload, off));
    params.task = static_cast<TaskType>(read_raw<int>(payload, off));
    params.length_scale = read_raw<double>(payload, off);
    params.signal_variance = read_raw<double>(payload, off);
    params.noise_variance = read_raw<double>(payload, off);
    params.jitter = read_raw<double>(payload, off);
    params.alpha = read_raw<double>(payload, off);
    params.period = read_raw<double>(payload, off);
    params.max_training_points = read_raw<std::size_t>(payload, off);
    params.inducing_points = read_raw<std::size_t>(payload, off);
    params.normalize_targets = read_raw<bool>(payload, off);
    params.output_correlation = read_raw<double>(payload, off);
    params.use_inducing_correction = read_raw<bool>(payload, off);
    const bool fitted = read_raw<bool>(payload, off);

    GaussianProcessRegressor model(params);
    model.fitted_ = fitted;
    model.feature_dim_ = read_raw<std::size_t>(payload, off);
    model.output_dim_ = read_raw<std::size_t>(payload, off);
    model.last_log_marginal_likelihood_ = read_raw<double>(payload, off);

    const std::size_t x_train_rows = read_raw<std::size_t>(payload, off);
    model.X_train_.assign(x_train_rows, std::vector<double>(model.feature_dim_, 0.0));
    for (auto& row : model.X_train_) for (double& v : row) v = read_raw<double>(payload, off);

    const std::size_t y_train_rows = read_raw<std::size_t>(payload, off);
    model.Y_train_.assign(y_train_rows, std::vector<double>(model.output_dim_, 0.0));
    for (auto& row : model.Y_train_) for (double& v : row) v = read_raw<double>(payload, off);

    const std::size_t active_rows = read_raw<std::size_t>(payload, off);
    model.active_X_.assign(active_rows, std::vector<double>(model.feature_dim_, 0.0));
    for (auto& row : model.active_X_) for (double& v : row) v = read_raw<double>(payload, off);
    model.active_Y_.assign(active_rows, std::vector<double>(model.output_dim_, 0.0));
    for (auto& row : model.active_Y_) for (double& v : row) v = read_raw<double>(payload, off);

    const std::size_t mean_size = read_raw<std::size_t>(payload, off);
    model.target_mean_.assign(mean_size, 0.0);
    for (double& v : model.target_mean_) v = read_raw<double>(payload, off);
    const std::size_t scale_size = read_raw<std::size_t>(payload, off);
    model.target_scale_.assign(scale_size, 1.0);
    for (double& v : model.target_scale_) v = read_raw<double>(payload, off);

    const std::size_t l_size = read_raw<std::size_t>(payload, off);
    model.cholesky_L_.assign(l_size, 0.0);
    for (double& v : model.cholesky_L_) v = read_raw<double>(payload, off);

    const std::size_t corr_size = read_raw<std::size_t>(payload, off);
    model.inducing_diag_correction_.assign(corr_size, 0.0);
    for (double& v : model.inducing_diag_correction_) v = read_raw<double>(payload, off);

    const std::size_t coreg_rows = read_raw<std::size_t>(payload, off);
    model.coregionalization_matrix_.assign(coreg_rows, std::vector<double>(coreg_rows, 0.0));
    for (auto& row : model.coregionalization_matrix_) for (double& v : row) v = read_raw<double>(payload, off);

    const std::size_t alpha_rows = read_raw<std::size_t>(payload, off);
    model.alpha_vectors_.assign(alpha_rows, {});
    for (auto& row : model.alpha_vectors_) {
        const std::size_t row_size = read_raw<std::size_t>(payload, off);
        row.assign(row_size, 0.0);
        for (double& v : row) v = read_raw<double>(payload, off);
    }

    return model;
}

}  // namespace gp
