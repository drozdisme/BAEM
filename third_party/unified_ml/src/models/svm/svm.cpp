#include "models/svm/svm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace svm {

namespace {

constexpr char kMagic[8] = {'U','M','L','S','V','M','4','\0'};
constexpr std::uint32_t kVersion = 4;

constexpr double kTiny = 1e-12;
constexpr double kKTKTol = 1e-3;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
    if (off + sizeof(T) > in.size()) throw std::runtime_error("SVM deserialize: truncated payload");
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

std::vector<int> unique_sorted(const std::vector<int>& y) {
    std::vector<int> c = y;
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

}  // namespace

SVM::SVM(const SVMParams& params) : params_(params) {
    if (!(params_.C > 0.0)) throw std::invalid_argument("SVM: C must be > 0");
    if (!(params_.learning_rate > 0.0)) throw std::invalid_argument("SVM: learning_rate must be > 0");
    if (params_.epochs == 0) throw std::invalid_argument("SVM: epochs must be > 0");
    if ((params_.kernel == KernelType::RBF || params_.kernel == KernelType::Polynomial || params_.kernel == KernelType::Sigmoid) && !(params_.gamma > 0.0)) {
        throw std::invalid_argument("SVM: gamma must be > 0 for selected kernel");
    }
    if (params_.degree <= 0) throw std::invalid_argument("SVM: degree must be > 0");
    if (params_.epsilon < 0.0) throw std::invalid_argument("SVM: epsilon must be >= 0");
    if (params_.nu <= 0.0 || params_.nu > 1.0) throw std::invalid_argument("SVM: nu must be in (0,1]");
    if (params_.quantile_tau <= 0.0 || params_.quantile_tau >= 1.0) throw std::invalid_argument("SVM: quantile_tau must be in (0,1)");
}

double SVM::kernel_eval(const std::vector<double>& a, const std::vector<double>& b) const {
    const double linear = dot(a, b);
    if (params_.kernel == KernelType::Linear) return linear;

    double dist2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        dist2 += d * d;
    }

    switch (params_.kernel) {
        case KernelType::Polynomial:
            return std::pow(params_.gamma * linear + params_.coef0, params_.degree);
        case KernelType::Sigmoid:
            return std::tanh(params_.gamma * linear + params_.coef0);
        case KernelType::RBF:
            return std::exp(-params_.gamma * dist2);
        case KernelType::Linear:
        default:
            return linear;
    }
}

void SVM::fit_binary_model(BinaryModel& model,
                           const std::vector<std::vector<double>>& X,
                           const std::vector<int>& y_binary,
                           const std::vector<double>* sample_weights) const {
    const std::size_t n = X.size();
    const std::size_t d = X.front().size();
    const auto weights = sanitize_sample_weights(sample_weights, n);
    model.w.clear();
    model.support_X.clear();
    model.support_y.clear();
    model.alpha.clear();
    model.error_cache.clear();
    model.b = 0.0;
    model.rho = 0.0;

    if (params_.optimization != SVMOptimization::PrimalSGD && params_.kernel != KernelType::Linear) {
        fit_binary_model_dual(model, X, y_binary, sample_weights, params_.optimization == SVMOptimization::SMOStyle);
        return;
    }

    if (params_.kernel == KernelType::Linear) {
        model.w.assign(d, 0.0);
        const double base_C = params_.variant == SVMVariant::Nu ? std::max(1e-6, params_.nu * params_.C * 0.5) : params_.C;
        for (std::size_t epoch = 0; epoch < params_.epochs; ++epoch) {
            for (std::size_t i = 0; i < n; ++i) {
                double margin = y_binary[i] * (dot(model.w, X[i]) + model.b);
                for (std::size_t j = 0; j < d; ++j) model.w[j] *= (1.0 - params_.learning_rate);
                if (margin < 1.0) {
                    const double sample_weight = (y_binary[i] > 0 ? model.positive_weight : model.negative_weight) * weights[i];
                    for (std::size_t j = 0; j < d; ++j) {
                        model.w[j] += params_.learning_rate * base_C * sample_weight * y_binary[i] * X[i][j];
                    }
                    if (params_.fit_intercept) {
                        model.b += params_.learning_rate * base_C * sample_weight * y_binary[i];
                    }
                }
            }
        }
    } else {
        model.support_X = X;
        model.support_y = y_binary;
        model.alpha.assign(n, 0.0);
        model.error_cache.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) model.error_cache[i] = -static_cast<double>(y_binary[i]);
        const std::vector<double> upper_bounds = sanitize_sample_weights(sample_weights, n);
        fit_binary_model_dual(model, X, y_binary, sample_weights, false);
    }
}

bool SVM::violates_kkt(const BinaryModel& model,
                      const std::vector<int>& y_binary,
                      std::size_t i,
                      const std::vector<double>& upper_bounds,
                      double tolerance) const {
    const double yi = static_cast<double>(y_binary[i]);
    const double ai = model.alpha[i];
    const double Ei = model.error_cache[i];
    const double yfi = yi * (Ei + yi);
    if (ai < upper_bounds[i] - tolerance && yfi < 1.0 - tolerance) return true;
    if (ai > tolerance && yfi > 1.0 + tolerance) return true;
    return false;
}

std::size_t SVM::select_second_index(const BinaryModel& model,
                                    const std::vector<int>& y_binary,
                                    std::size_t i,
                                    const std::vector<double>& upper_bounds) const {
    const double Ei = model.error_cache[i];
    std::size_t best = i;
    double best_gap = -1.0;
    for (std::size_t j = 0; j < model.alpha.size(); ++j) {
        if (j == i) continue;
        if (!violates_kkt(model, y_binary, j, upper_bounds, 1e-2) && std::abs(model.alpha[j]) < kTiny) continue;
        const double gap = std::abs(Ei - model.error_cache[j]);
        if (gap > best_gap) {
            best_gap = gap;
            best = j;
        }
    }
    if (best != i) return best;
    return (i + 1) % model.alpha.size();
}

bool SVM::smo_pair_update(BinaryModel& model,
                         const std::vector<std::vector<double>>& X,
                         std::size_t i,
                         std::size_t j,
                         const std::vector<double>& upper_bounds) const {
    if (i == j) return false;
    const double yi = static_cast<double>(model.support_y[i]);
    const double yj = static_cast<double>(model.support_y[j]);
    const double ai_old = model.alpha[i];
    const double aj_old = model.alpha[j];
    const double Ei = model.error_cache[i];
    const double Ej = model.error_cache[j];

    double L = 0.0;
    double H = 0.0;
    if (yi != yj) {
        L = std::max(0.0, aj_old - ai_old);
        H = std::min(upper_bounds[j], upper_bounds[j] - ai_old + aj_old + upper_bounds[i]);
    } else {
        L = std::max(0.0, ai_old + aj_old - upper_bounds[i]);
        H = std::min(upper_bounds[j], ai_old + aj_old);
    }
    if (H - L <= kTiny) return false;

    const double kii = kernel_eval(X[i], X[i]);
    const double kjj = kernel_eval(X[j], X[j]);
    const double kij = kernel_eval(X[i], X[j]);
    double eta = 2.0 * kij - kii - kjj;
    if (eta >= 0.0) eta = -kTiny;

    double aj_new = aj_old - yj * (Ei - Ej) / eta;
    aj_new = std::clamp(aj_new, L, H);
    if (std::abs(aj_new - aj_old) < 1e-8) return false;
    const double ai_new = ai_old + yi * yj * (aj_old - aj_new);

    const double b1 = model.b - Ei - yi * (ai_new - ai_old) * kii - yj * (aj_new - aj_old) * kij;
    const double b2 = model.b - Ej - yi * (ai_new - ai_old) * kij - yj * (aj_new - aj_old) * kjj;

    model.alpha[i] = std::clamp(ai_new, 0.0, upper_bounds[i]);
    model.alpha[j] = std::clamp(aj_new, 0.0, upper_bounds[j]);
    if (model.alpha[i] > kTiny && model.alpha[i] < upper_bounds[i] - kTiny) model.b = b1;
    else if (model.alpha[j] > kTiny && model.alpha[j] < upper_bounds[j] - kTiny) model.b = b2;
    else model.b = 0.5 * (b1 + b2);

    for (std::size_t k = 0; k < X.size(); ++k) {
        const double kik = kernel_eval(X[i], X[k]);
        const double kjk = kernel_eval(X[j], X[k]);
        model.error_cache[k] += yi * (model.alpha[i] - ai_old) * kik + yj * (model.alpha[j] - aj_old) * kjk + (model.b - 0.5 * (b1 + b2));
    }
    return true;
}

void SVM::fit_binary_model_dual(BinaryModel& model,
                               const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y_binary,
                               const std::vector<double>* sample_weights,
                               bool smo_style) const {
    const std::size_t n = X.size();
    const auto weights = sanitize_sample_weights(sample_weights, n);
    model.support_X = X;
    model.support_y = y_binary;
    model.alpha.assign(n, 0.0);
    model.error_cache.assign(n, 0.0);
    model.w.clear();
    model.b = 0.0;
    const double nu_mass = std::clamp(params_.nu * static_cast<double>(n) * 0.5, 1.0, static_cast<double>(n));
    std::vector<double> upper_bounds(n, params_.C);
    double total_bound = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double cls_weight = y_binary[i] > 0 ? model.positive_weight : model.negative_weight;
        upper_bounds[i] = params_.C * weights[i] * cls_weight;
        total_bound += upper_bounds[i];
        if (params_.variant == SVMVariant::Nu) {
            model.alpha[i] = std::min(upper_bounds[i], nu_mass / static_cast<double>(n));
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        double fx = model.b;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::abs(model.alpha[j]) <= kTiny) continue;
            fx += model.alpha[j] * static_cast<double>(y_binary[j]) * kernel_eval(X[j], X[i]);
        }
        model.error_cache[i] = fx - static_cast<double>(y_binary[i]);
    }

    bool examine_all = true;
    std::size_t passes_without_change = 0;
    for (std::size_t epoch = 0; epoch < params_.epochs && passes_without_change < 6; ++epoch) {
        std::size_t num_changed = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!examine_all && !violates_kkt(model, y_binary, i, upper_bounds, kKTKTol)) continue;
            const std::size_t j = smo_style ? select_second_index(model, y_binary, i, upper_bounds)
                                            : ((i + epoch + 1) % n);
            if (smo_pair_update(model, X, i, j, upper_bounds)) ++num_changed;
        }
        if (params_.variant == SVMVariant::Nu) {
            double alpha_sum = std::accumulate(model.alpha.begin(), model.alpha.end(), 0.0);
            if (alpha_sum > kTiny) {
                const double target_sum = std::min(total_bound, nu_mass);
                const double scale = target_sum / alpha_sum;
                for (std::size_t i = 0; i < n; ++i) model.alpha[i] = std::clamp(model.alpha[i] * scale, 0.0, upper_bounds[i]);
            }
        }
        if (examine_all) examine_all = false;
        else if (num_changed == 0) examine_all = true;
        passes_without_change = (num_changed == 0) ? (passes_without_change + 1) : 0;
    }

    double rho_sum = 0.0;
    double rho_count = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (model.alpha[i] <= kTiny || model.alpha[i] >= upper_bounds[i] - kTiny) continue;
        double margin = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            if (std::abs(model.alpha[k]) <= kTiny) continue;
            margin += model.alpha[k] * static_cast<double>(y_binary[k]) * kernel_eval(X[k], X[i]);
        }
        rho_sum += static_cast<double>(y_binary[i]) - margin;
        rho_count += 1.0;
    }
    model.rho = rho_count > 0.0 ? rho_sum / rho_count : model.b;
    model.b = model.rho;
}

double SVM::binary_decision_function(const BinaryModel& model,
                                     const std::vector<double>& x) const {
    if (!model.w.empty()) {
        return dot(model.w, x) + model.b;
    }

    double s = model.b;
    for (std::size_t i = 0; i < model.support_X.size(); ++i) {
        if (model.alpha[i] != 0.0) {
            s += model.alpha[i] * model.support_y[i] * kernel_eval(model.support_X[i], x);
        }
    }
    return s;
}

void SVM::calibrate_probabilities(BinaryModel& model,
                                  const std::vector<std::vector<double>>& X,
                                  const std::vector<int>& y_binary) const {
    std::vector<double> scores(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) scores[i] = binary_decision_function(model, X[i]);

    double a = 0.0;
    double b = 0.0;
    const double lr = 0.05;
    const double reg = 1e-4;

    for (int iter = 0; iter < 200; ++iter) {
        double ga = 0.0;
        double gb = 0.0;
        for (std::size_t i = 0; i < scores.size(); ++i) {
            const double yi = y_binary[i] > 0 ? 1.0 : 0.0;
            const double z = std::clamp(a * scores[i] + b, -50.0, 50.0);
            const double p = 1.0 / (1.0 + std::exp(-z));
            const double diff = p - yi;
            ga += diff * scores[i];
            gb += diff;
        }
        ga = ga / static_cast<double>(scores.size()) + reg * a;
        gb /= static_cast<double>(scores.size());
        a -= lr * ga;
        b -= lr * gb;
    }

    model.prob_a = a;
    model.prob_b = b;
}

double SVM::kernel_svr_predict(const std::vector<double>& x) const {
    double s = regression_b_;
    for (std::size_t i = 0; i < regression_support_X_.size(); ++i) {
        const double coeff = (i < regression_alpha_star_.size()) ? (regression_alpha_[i] - regression_alpha_star_[i])
                                                                 : regression_alpha_[i];
        if (coeff != 0.0) s += coeff * kernel_eval(regression_support_X_[i], x);
    }
    return s;
}

std::vector<double> SVM::sanitize_sample_weights(const std::vector<double>* sample_weights,
                                                std::size_t n) {
    if (sample_weights == nullptr || sample_weights->empty()) return std::vector<double>(n, 1.0);
    if (sample_weights->size() != n) throw std::invalid_argument("SVM: sample_weights size mismatch");
    std::vector<double> out = *sample_weights;
    for (double& w : out) {
        if (!(w > 0.0)) w = 1.0;
    }
    return out;
}

double SVM::class_weight_for_index(std::size_t class_index) const {
    if (class_index < params_.class_weights.size() && params_.class_weights[class_index] > 0.0) return params_.class_weights[class_index];
    return 1.0;
}

double SVM::binary_probability(const BinaryModel& model,
                               double score) const {
    const double z = std::clamp(model.prob_a * score + model.prob_b, -50.0, 50.0);
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<double> SVM::softmax_stable(const std::vector<double>& scores) {
    if (scores.empty()) return {};
    const double max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> exps(scores.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        exps[i] = std::exp(scores[i] - max_score);
        sum += exps[i];
    }
    if (!(sum > 0.0)) {
        return std::vector<double>(scores.size(), 1.0 / static_cast<double>(scores.size()));
    }
    for (double& v : exps) v /= sum;
    return exps;
}

void SVM::fit(const std::vector<std::vector<double>>& X,
              const std::vector<int>& y) {
    fit(X, y, params_.sample_weights);
}

void SVM::fit(const std::vector<std::vector<double>>& X,
              const std::vector<int>& y,
              const std::vector<double>& sample_weights) {
    if (X.empty() || X.size() != y.size()) throw std::invalid_argument("SVM::fit: invalid dataset");
    if (params_.mode != SVMMode::Classification) throw std::runtime_error("SVM::fit: model is configured for regression");
    feature_dim_ = X.front().size();
    if (feature_dim_ == 0) throw std::invalid_argument("SVM::fit: feature dimension must be > 0");
    for (const auto& row : X) {
        if (row.size() != feature_dim_) throw std::invalid_argument("SVM::fit: inconsistent feature dims");
    }

    classes_ = unique_sorted(y);
    if (classes_.size() < 2) throw std::invalid_argument("SVM::fit: need at least two classes");

    binary_models_.clear();

    if (classes_.size() == 2) {
        BinaryModel model;
        model.negative_label = classes_[0];
        model.positive_label = classes_[1];
        model.negative_weight = class_weight_for_index(0);
        model.positive_weight = class_weight_for_index(1);
        std::vector<int> y_binary(y.size(), -1);
        for (std::size_t i = 0; i < y.size(); ++i) y_binary[i] = (y[i] == model.positive_label) ? 1 : -1;
        fit_binary_model(model, X, y_binary, &sample_weights);
        calibrate_probabilities(model, X, y_binary);
        binary_models_.push_back(std::move(model));
    } else if (params_.multiclass_strategy == MultiClassStrategy::OneVsRest) {
        for (int cls : classes_) {
            BinaryModel model;
            model.negative_label = -1;
            model.positive_label = cls;
            const auto class_it = std::find(classes_.begin(), classes_.end(), cls);
            const std::size_t class_idx = static_cast<std::size_t>(std::distance(classes_.begin(), class_it));
            model.positive_weight = class_weight_for_index(class_idx);
            model.negative_weight = 1.0;
            std::vector<int> y_binary(y.size(), -1);
            for (std::size_t i = 0; i < y.size(); ++i) y_binary[i] = (y[i] == cls) ? 1 : -1;
            fit_binary_model(model, X, y_binary, &sample_weights);
            calibrate_probabilities(model, X, y_binary);
            binary_models_.push_back(std::move(model));
        }
    } else {
        for (std::size_t i = 0; i < classes_.size(); ++i) {
            for (std::size_t j = i + 1; j < classes_.size(); ++j) {
                BinaryModel model;
                model.negative_label = classes_[i];
                model.positive_label = classes_[j];
                model.negative_weight = class_weight_for_index(i);
                model.positive_weight = class_weight_for_index(j);
                std::vector<std::vector<double>> X_pair;
                std::vector<int> y_pair;
                std::vector<double> w_pair;
                for (std::size_t k = 0; k < y.size(); ++k) {
                    if (y[k] == classes_[i] || y[k] == classes_[j]) {
                        X_pair.push_back(X[k]);
                        y_pair.push_back(y[k] == classes_[j] ? 1 : -1);
                        if (!sample_weights.empty()) w_pair.push_back(sample_weights[k]);
                    }
                }
                fit_binary_model(model, X_pair, y_pair, w_pair.empty() ? nullptr : &w_pair);
                calibrate_probabilities(model, X_pair, y_pair);
                binary_models_.push_back(std::move(model));
            }
        }
    }

    fitted_ = true;
}

void SVM::fit_regression(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y) {
    fit_regression(X, y, params_.sample_weights);
}

void SVM::calibrate_regression_confidence(const std::vector<std::vector<double>>& X,
                                         const std::vector<double>& y) const {
    if (X.empty()) {
        regression_confidence_scale_ = 1.0;
        regression_residual_variance_ = 0.0;
        return;
    }
    double sq = 0.0;
    double mean_std = 0.0;
    double covered = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        const double pred = predict_regression(X[i]);
        const double resid = pred - y[i];
        const double stddev = std::max(1e-6, regression_predictive_stddev(X[i]));
        sq += resid * resid;
        mean_std += stddev;
        if (std::abs(resid) <= 1.96 * stddev) covered += 1.0;
    }
    regression_residual_variance_ = sq / static_cast<double>(X.size());
    mean_std /= static_cast<double>(X.size());
    const double empirical_std = std::sqrt(std::max(0.0, regression_residual_variance_));
    regression_confidence_scale_ = empirical_std / std::max(1e-6, mean_std);
    if (!(regression_confidence_scale_ > 0.0) || !std::isfinite(regression_confidence_scale_)) regression_confidence_scale_ = 1.0;
}

double SVM::regression_predictive_stddev(const std::vector<double>& x) const {
    if (regression_support_X_.empty()) {
        return std::sqrt(std::max(1e-12, regression_residual_variance_));
    }
    double self_k = kernel_eval(x, x);
    double mass = 0.0;
    double leverage = 0.0;
    for (std::size_t i = 0; i < regression_support_X_.size(); ++i) {
        const double coeff = (i < regression_alpha_star_.size()) ? std::abs(regression_alpha_[i] - regression_alpha_star_[i])
                                                                 : std::abs(regression_alpha_[i]);
        if (coeff <= kTiny) continue;
        mass += coeff;
        leverage += coeff * std::abs(kernel_eval(regression_support_X_[i], x));
    }
    const double normalized = leverage / std::max(kTiny, mass);
    const double geometric = std::max(0.0, self_k - normalized);
    const double base = std::max(1e-8, regression_residual_variance_ + geometric);
    return std::sqrt(base) * std::max(1e-6, regression_confidence_scale_);
}

void SVM::fit_regression(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y,
                        const std::vector<double>& sample_weights) {
    if (params_.mode != SVMMode::Regression) throw std::runtime_error("SVM::fit_regression: model is not configured for regression");
    if (X.empty() || X.size() != y.size()) throw std::invalid_argument("SVM::fit_regression: invalid dataset");
    feature_dim_ = X.front().size();
    if (feature_dim_ == 0) throw std::invalid_argument("SVM::fit_regression: feature dimension must be > 0");
    for (const auto& row : X) if (row.size() != feature_dim_) throw std::invalid_argument("SVM::fit_regression: inconsistent feature dims");
    regression_w_.assign(feature_dim_, 0.0);
    regression_b_ = 0.0;
    regression_support_X_.clear();
    regression_alpha_.clear();
    regression_alpha_star_.clear();
    regression_targets_ = y;
    regression_sample_weights_ = sanitize_sample_weights(&sample_weights, X.size());
    regression_confidence_scale_ = 1.0;
    regression_residual_variance_ = 0.0;
    classes_.clear();
    binary_models_.clear();
    if (params_.optimization != SVMOptimization::PrimalSGD && params_.kernel != KernelType::Linear) {
        fit_regression_dual(X, y, regression_sample_weights_);
    } else if (params_.kernel == KernelType::Linear) {
        const double base_C = params_.variant == SVMVariant::Nu ? std::max(1e-6, params_.nu * params_.C) : params_.C;
        for (std::size_t epoch = 0; epoch < params_.epochs; ++epoch) {
            for (std::size_t i = 0; i < X.size(); ++i) {
                const double pred = dot(regression_w_, X[i]) + regression_b_;
                const double err = pred - y[i];
                for (std::size_t j = 0; j < feature_dim_; ++j) regression_w_[j] *= (1.0 - params_.learning_rate);
                if (std::abs(err) > params_.epsilon) {
                    const double sw = regression_sample_weights_[i];
                    const double q = std::clamp(params_.quantile_tau, 1e-4, 1.0 - 1e-4);
                    const double asym = err > 0.0 ? q : (1.0 - q);
                    const double grad_sign = err > 0.0 ? 1.0 : -1.0;
                    for (std::size_t j = 0; j < feature_dim_; ++j) regression_w_[j] -= params_.learning_rate * base_C * sw * asym * grad_sign * X[i][j];
                    if (params_.fit_intercept) regression_b_ -= params_.learning_rate * base_C * sw * asym * grad_sign;
                }
            }
        }
    } else {
        fit_regression_dual(X, y, regression_sample_weights_);
    }
    fitted_ = true;
    calibrate_regression_confidence(X, y);
}

void SVM::fit_regression_dual(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& y,
                             const std::vector<double>& sample_weights) {
    if (params_.variant == SVMVariant::Nu) fit_regression_dual_nu(X, y, sample_weights);
    else fit_regression_dual_epsilon(X, y, sample_weights);
}

void SVM::fit_regression_dual_epsilon(const std::vector<std::vector<double>>& X,
                                     const std::vector<double>& y,
                                     const std::vector<double>& sample_weights) {
    regression_support_X_ = X;
    regression_alpha_.assign(X.size(), 0.0);
    regression_alpha_star_.assign(X.size(), 0.0);
    const double tau_pos = std::clamp(params_.quantile_tau, 1e-4, 1.0 - 1e-4);
    const double tau_neg = 1.0 - tau_pos;
    for (std::size_t epoch = 0; epoch < params_.epochs; ++epoch) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            const double pred = kernel_svr_predict(X[i]);
            const double err = pred - y[i];
            const double Ci = params_.C * sample_weights[i];
            if (err > params_.epsilon) {
                regression_alpha_star_[i] = std::clamp(regression_alpha_star_[i] + params_.learning_rate * Ci * tau_pos * (err - params_.epsilon), 0.0, Ci);
                regression_alpha_[i] = std::max(0.0, regression_alpha_[i] - params_.learning_rate * Ci * 0.25);
            } else if (err < -params_.epsilon) {
                regression_alpha_[i] = std::clamp(regression_alpha_[i] + params_.learning_rate * Ci * tau_neg * (-err - params_.epsilon), 0.0, Ci);
                regression_alpha_star_[i] = std::max(0.0, regression_alpha_star_[i] - params_.learning_rate * Ci * 0.25);
            } else {
                regression_alpha_[i] *= 0.999;
                regression_alpha_star_[i] *= 0.999;
            }
        }
        double bias_sum = 0.0;
        double bias_count = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            const double coeff = regression_alpha_[i] - regression_alpha_star_[i];
            if (std::abs(coeff) <= 1e-8) continue;
            bias_sum += y[i] - (kernel_svr_predict(X[i]) - regression_b_);
            bias_count += 1.0;
        }
        regression_b_ = bias_count > 0.0 ? bias_sum / bias_count : 0.0;
    }
}

void SVM::fit_regression_dual_nu(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y,
                                const std::vector<double>& sample_weights) {
    regression_support_X_ = X;
    regression_alpha_.assign(X.size(), 0.0);
    regression_alpha_star_.assign(X.size(), 0.0);
    const double total_budget = std::max(1e-6, params_.nu * params_.C * static_cast<double>(X.size()));
    const double tau_pos = std::clamp(params_.quantile_tau, 1e-4, 1.0 - 1e-4);
    const double tau_neg = 1.0 - tau_pos;
    for (std::size_t epoch = 0; epoch < params_.epochs; ++epoch) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            const double pred = kernel_svr_predict(X[i]);
            const double err = pred - y[i];
            const double Ci = params_.C * sample_weights[i];
            if (err > 0.0) {
                regression_alpha_star_[i] = std::clamp(regression_alpha_star_[i] + params_.learning_rate * tau_pos * std::abs(err), 0.0, Ci);
                regression_alpha_[i] *= 0.999;
            } else {
                regression_alpha_[i] = std::clamp(regression_alpha_[i] + params_.learning_rate * tau_neg * std::abs(err), 0.0, Ci);
                regression_alpha_star_[i] *= 0.999;
            }
        }
        double total = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) total += regression_alpha_[i] + regression_alpha_star_[i];
        if (total > kTiny) {
            const double scale = total_budget / total;
            for (std::size_t i = 0; i < X.size(); ++i) {
                const double Ci = params_.C * sample_weights[i];
                regression_alpha_[i] = std::clamp(regression_alpha_[i] * scale, 0.0, Ci);
                regression_alpha_star_[i] = std::clamp(regression_alpha_star_[i] * scale, 0.0, Ci);
            }
        }
        double signed_sum = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) signed_sum += regression_alpha_[i] - regression_alpha_star_[i];
        if (std::abs(signed_sum) > kTiny) {
            const double corr = signed_sum / static_cast<double>(X.size());
            for (std::size_t i = 0; i < X.size(); ++i) {
                const double Ci = params_.C * sample_weights[i];
                regression_alpha_[i] = std::clamp(regression_alpha_[i] - 0.5 * corr, 0.0, Ci);
                regression_alpha_star_[i] = std::clamp(regression_alpha_star_[i] + 0.5 * corr, 0.0, Ci);
            }
        }
        double bias_sum = 0.0;
        double bias_count = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            const double coeff = regression_alpha_[i] - regression_alpha_star_[i];
            if (std::abs(coeff) <= 1e-8) continue;
            bias_sum += y[i] - (kernel_svr_predict(X[i]) - regression_b_);
            bias_count += 1.0;
        }
        regression_b_ = bias_count > 0.0 ? bias_sum / bias_count : 0.0;
    }
}

double SVM::decision_function(std::span<const double> x) const {
    return decision_function(std::vector<double>(x.begin(), x.end()));
}

double SVM::decision_function(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("SVM::decision_function: model is not fitted");
    if (x.size() != feature_dim_) throw std::invalid_argument("SVM::decision_function: feature dimension mismatch");
    if (is_regressor()) return params_.kernel == KernelType::Linear ? dot(regression_w_, x) + regression_b_ : kernel_svr_predict(x);
    if (binary_models_.empty()) throw std::runtime_error("SVM::decision_function: no trained models");
    return binary_decision_function(binary_models_.front(), x);
}

BinaryDecision SVM::predict_binary(std::span<const double> x) const {
    return predict_binary(std::vector<double>(x.begin(), x.end()));
}

BinaryDecision SVM::predict_binary(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("SVM::predict_binary: model is not fitted");
    if (classes_.size() != 2) throw std::runtime_error("SVM::predict_binary: model is multiclass");
    const auto score = binary_decision_function(binary_models_.front(), x);
    const auto prob = binary_probability(binary_models_.front(), score);
    return {score >= 0.0 ? binary_models_.front().positive_label : binary_models_.front().negative_label, score, prob, std::abs(score)};
}

std::vector<double> SVM::predict_scores(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("SVM::predict_scores: model is not fitted");
    if (x.size() != feature_dim_) throw std::invalid_argument("SVM::predict_scores: feature dimension mismatch");

    std::vector<double> scores(classes_.size(), 0.0);
    if (classes_.size() == 2) {
        const double score = binary_decision_function(binary_models_.front(), x);
        scores[0] = -score;
        scores[1] = score;
        return scores;
    }

    if (params_.multiclass_strategy == MultiClassStrategy::OneVsRest) {
        for (std::size_t i = 0; i < binary_models_.size(); ++i) scores[i] = binary_decision_function(binary_models_[i], x);
        return scores;
    }

    std::vector<double> vote_scores(classes_.size(), 0.0);
    std::size_t model_idx = 0;
    for (std::size_t i = 0; i < classes_.size(); ++i) {
        for (std::size_t j = i + 1; j < classes_.size(); ++j, ++model_idx) {
            const auto& model = binary_models_[model_idx];
            const double score = binary_decision_function(model, x);
            vote_scores[i] += (score < 0.0) ? 1.0 : 0.0;
            vote_scores[j] += (score >= 0.0) ? 1.0 : 0.0;
            scores[i] -= score;
            scores[j] += score;
        }
    }
    for (std::size_t i = 0; i < scores.size(); ++i) scores[i] += vote_scores[i];
    return scores;
}

std::vector<double> SVM::multiclass_probability_coupling(const std::vector<double>& pairwise_scores) const {
    return softmax_stable(pairwise_scores);
}

std::vector<double> SVM::predict_proba(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("SVM::predict_proba: model is not fitted");
    if (classes_.size() == 2) {
        const auto pred = predict_binary(x);
        return {1.0 - pred.probability, pred.probability};
    }

    if (params_.multiclass_strategy == MultiClassStrategy::OneVsRest) {
        std::vector<double> probs(binary_models_.size(), 0.0);
        for (std::size_t i = 0; i < binary_models_.size(); ++i) {
            const double score = binary_decision_function(binary_models_[i], x);
            probs[i] = binary_probability(binary_models_[i], score);
        }
        double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (!(sum > 0.0)) return std::vector<double>(probs.size(), 1.0 / static_cast<double>(probs.size()));
        for (double& v : probs) v /= sum;
        return probs;
    }

    return multiclass_probability_coupling(predict_scores(x));
}

std::vector<std::vector<double>> SVM::predict_proba(const std::vector<std::vector<double>>& X) const {
    std::vector<std::vector<double>> out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(predict_proba(row));
    return out;
}

MultiClassPrediction SVM::predict_multiclass(const std::vector<double>& x) const {
    if (!fitted_) throw std::runtime_error("SVM::predict_multiclass: model is not fitted");
    auto scores = predict_scores(x);
    auto probs = predict_proba(x);
    const auto it = std::max_element(probs.begin(), probs.end());
    const std::size_t idx = static_cast<std::size_t>(std::distance(probs.begin(), it));
    return {classes_[idx], classes_, std::move(scores), std::move(probs)};
}

std::vector<MultiClassPrediction> SVM::predict_multiclass(const std::vector<std::vector<double>>& X) const {
    std::vector<MultiClassPrediction> out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(predict_multiclass(row));
    return out;
}

int SVM::predict(std::span<const double> x) const {
    return predict(std::vector<double>(x.begin(), x.end()));
}

int SVM::predict(const std::vector<double>& x) const {
    if (is_regressor()) throw std::runtime_error("SVM::predict: classification API called on regressor");
    return predict_multiclass(x).label;
}

double SVM::predict_regression(std::span<const double> x) const {
    return predict_regression(std::vector<double>(x.begin(), x.end()));
}

double SVM::predict_regression(const std::vector<double>& x) const {
    if (!is_regressor()) throw std::runtime_error("SVM::predict_regression: model is not a regressor");
    return decision_function(x);
}

std::vector<double> SVM::predict_regression(const std::vector<std::vector<double>>& X) const {
    std::vector<double> out(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = predict_regression(X[i]);
    return out;
}

std::size_t SVM::support_vector_count() const noexcept {
    std::size_t count = 0;
    for (const auto& model : binary_models_) {
        if (!model.w.empty()) continue;
        for (double a : model.alpha) if (std::abs(a) > 1e-10) ++count;
    }
    return count;
}

std::vector<int> SVM::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> out(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = predict(X[i]);
    return out;
}

TrainingDiagnostics SVM::diagnostics(const std::vector<std::vector<double>>& X,
                                    const std::vector<int>& y) const {
    if (!fitted_) throw std::runtime_error("SVM::diagnostics: model is not fitted");
    if (X.size() != y.size()) throw std::invalid_argument("SVM::diagnostics: invalid dataset");
    TrainingDiagnostics d;
    if (X.empty()) return d;
    auto pred = predict(X);
    for (std::size_t i = 0; i < X.size(); ++i) {
        const double score = decision_function(X[i]);
        const int yi = (classes_.size() == 2) ? (y[i] == classes_.back() ? 1 : -1) : (pred[i] == y[i] ? 1 : -1);
        const double margin = yi * score;
        d.average_margin += margin;
        d.average_hinge_loss += std::max(0.0, 1.0 - margin);
        d.training_accuracy += pred[i] == y[i] ? 1.0 : 0.0;
    }
    const double n = static_cast<double>(X.size());
    d.average_margin /= n;
    d.average_hinge_loss /= n;
    d.training_accuracy /= n;
    return d;
}

RegressionDiagnostics SVM::regression_diagnostics(const std::vector<std::vector<double>>& X,
                                                 const std::vector<double>& y) const {
    if (!fitted_ || !is_regressor()) throw std::runtime_error("SVM::regression_diagnostics: model is not a fitted regressor");
    if (X.size() != y.size()) throw std::invalid_argument("SVM::regression_diagnostics: invalid dataset");
    RegressionDiagnostics d;
    if (X.empty()) return d;
    double sq = 0.0, ab = 0.0, mean = 0.0, support = 0.0, coverage = 0.0, std_sum = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        const double pred = predict_regression(X[i]);
        const double r = pred - y[i];
        const double pred_std = regression_predictive_stddev(X[i]);
        sq += r * r;
        ab += std::abs(r);
        mean += r;
        std_sum += pred_std;
        if (std::abs(r) <= 1.96 * pred_std) coverage += 1.0;
    }
    if (!regression_alpha_.empty()) {
        for (std::size_t i = 0; i < regression_alpha_.size(); ++i) {
            const double coeff = (i < regression_alpha_star_.size()) ? std::abs(regression_alpha_[i] - regression_alpha_star_[i])
                                                                     : std::abs(regression_alpha_[i]);
            if (coeff > 1e-10) support += 1.0;
        }
        d.support_fraction = support / static_cast<double>(regression_alpha_.size());
    } else if (!regression_w_.empty()) {
        d.support_fraction = 1.0;
    }
    d.rmse = std::sqrt(sq / static_cast<double>(X.size()));
    d.mae = ab / static_cast<double>(X.size());
    d.mean_residual = mean / static_cast<double>(X.size());
    d.residual_stddev = std::sqrt(std::max(0.0, sq / static_cast<double>(X.size()) - d.mean_residual * d.mean_residual));
    d.calibrated_interval_scale = regression_confidence_scale_;
    d.nominal_coverage_95 = coverage / static_cast<double>(X.size());
    d.mean_predictive_stddev = std_sum / static_cast<double>(X.size());
    return d;
}

void SVM::save(const std::string& filepath) const {
    if (!fitted_) throw std::runtime_error("SVM::save: model is not fitted");

    std::vector<char> payload;
    write_raw(payload, params_.C);
    write_raw(payload, params_.learning_rate);
    write_raw(payload, params_.epochs);
    write_raw(payload, static_cast<int>(params_.kernel));
    write_raw(payload, params_.gamma);
    write_raw(payload, params_.coef0);
    write_raw(payload, params_.degree);
    write_raw(payload, params_.fit_intercept);
    write_raw(payload, params_.deterministic);
    write_raw(payload, static_cast<int>(params_.multiclass_strategy));
    write_raw(payload, static_cast<int>(params_.mode));
    write_raw(payload, params_.epsilon);
    write_raw(payload, static_cast<int>(params_.optimization));
    write_raw(payload, static_cast<int>(params_.variant));
    write_raw(payload, params_.nu);
    write_raw(payload, params_.quantile_tau);
    write_raw(payload, params_.class_weights.size());
    for (double v : params_.class_weights) write_raw(payload, v);
    write_raw(payload, fitted_);
    write_raw(payload, feature_dim_);
    write_raw(payload, regression_w_.size());
    for (double v : regression_w_) write_raw(payload, v);
    write_raw(payload, regression_b_);
    write_raw(payload, regression_confidence_scale_);
    write_raw(payload, regression_residual_variance_);
    write_raw(payload, regression_support_X_.size());
    for (const auto& row : regression_support_X_) for (double v : row) write_raw(payload, v);
    write_raw(payload, regression_alpha_.size());
    for (double v : regression_alpha_) write_raw(payload, v);
    write_raw(payload, regression_alpha_star_.size());
    for (double v : regression_alpha_star_) write_raw(payload, v);
    write_raw(payload, regression_targets_.size());
    for (double v : regression_targets_) write_raw(payload, v);

    write_raw(payload, classes_.size());
    for (int cls : classes_) write_raw(payload, cls);

    write_raw(payload, binary_models_.size());
    for (const auto& model : binary_models_) {
        write_raw(payload, model.positive_label);
        write_raw(payload, model.negative_label);
        write_raw(payload, model.b);
        write_raw(payload, model.prob_a);
        write_raw(payload, model.prob_b);
        write_raw(payload, model.positive_weight);
        write_raw(payload, model.negative_weight);
        write_raw(payload, model.rho);

        write_raw(payload, model.w.size());
        for (double v : model.w) write_raw(payload, v);

        const std::size_t n_sv = model.support_X.size();
        const std::size_t d = n_sv ? model.support_X.front().size() : 0;
        write_raw(payload, n_sv);
        write_raw(payload, d);
        for (const auto& row : model.support_X) for (double v : row) write_raw(payload, v);
        write_raw(payload, model.support_y.size());
        for (int v : model.support_y) write_raw(payload, v);
        write_raw(payload, model.alpha.size());
        for (double v : model.alpha) write_raw(payload, v);
        write_raw(payload, model.error_cache.size());
        for (double v : model.error_cache) write_raw(payload, v);
    }

    const std::uint64_t checksum = fnv1a64(payload);
    const std::uint64_t payload_size = static_cast<std::uint64_t>(payload.size());

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("SVM::save: cannot open file");
    ofs.write(kMagic, sizeof(kMagic));
    ofs.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    ofs.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
    ofs.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

SVM SVM::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) throw std::runtime_error("SVM::load: cannot open file");

    char magic[8]{};
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0, checksum = 0;
    ifs.read(magic, sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
    ifs.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("SVM::load: invalid magic");
    if (version != kVersion) throw std::runtime_error("SVM::load: unsupported version");

    std::vector<char> payload(static_cast<std::size_t>(payload_size));
    ifs.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (fnv1a64(payload) != checksum) throw std::runtime_error("SVM::load: checksum mismatch");

    std::size_t off = 0;
    SVMParams params;
    params.C = read_raw<double>(payload, off);
    params.learning_rate = read_raw<double>(payload, off);
    params.epochs = read_raw<std::size_t>(payload, off);
    params.kernel = static_cast<KernelType>(read_raw<int>(payload, off));
    params.gamma = read_raw<double>(payload, off);
    params.coef0 = read_raw<double>(payload, off);
    params.degree = read_raw<int>(payload, off);
    params.fit_intercept = read_raw<bool>(payload, off);
    params.deterministic = read_raw<bool>(payload, off);
    params.multiclass_strategy = static_cast<MultiClassStrategy>(read_raw<int>(payload, off));
    params.mode = static_cast<SVMMode>(read_raw<int>(payload, off));
    params.epsilon = read_raw<double>(payload, off);
    params.optimization = static_cast<SVMOptimization>(read_raw<int>(payload, off));
    params.variant = static_cast<SVMVariant>(read_raw<int>(payload, off));
    params.nu = read_raw<double>(payload, off);
    params.quantile_tau = read_raw<double>(payload, off);
    const std::size_t class_weight_size = read_raw<std::size_t>(payload, off);
    params.class_weights.assign(class_weight_size, 0.0);
    for (double& v : params.class_weights) v = read_raw<double>(payload, off);
    const bool fitted = read_raw<bool>(payload, off);

    SVM model(params);
    model.fitted_ = fitted;
    model.feature_dim_ = read_raw<std::size_t>(payload, off);
    const std::size_t reg_w_size = read_raw<std::size_t>(payload, off);
    model.regression_w_.assign(reg_w_size, 0.0);
    for (double& v : model.regression_w_) v = read_raw<double>(payload, off);
    model.regression_b_ = read_raw<double>(payload, off);
    model.regression_confidence_scale_ = read_raw<double>(payload, off);
    model.regression_residual_variance_ = read_raw<double>(payload, off);
    const std::size_t reg_sv_size = read_raw<std::size_t>(payload, off);
    model.regression_support_X_.assign(reg_sv_size, std::vector<double>(model.feature_dim_, 0.0));
    for (auto& row : model.regression_support_X_) for (double& v : row) v = read_raw<double>(payload, off);
    const std::size_t reg_alpha_size = read_raw<std::size_t>(payload, off);
    model.regression_alpha_.assign(reg_alpha_size, 0.0);
    for (double& v : model.regression_alpha_) v = read_raw<double>(payload, off);
    const std::size_t reg_alpha_star_size = read_raw<std::size_t>(payload, off);
    model.regression_alpha_star_.assign(reg_alpha_star_size, 0.0);
    for (double& v : model.regression_alpha_star_) v = read_raw<double>(payload, off);
    const std::size_t reg_target_size = read_raw<std::size_t>(payload, off);
    model.regression_targets_.assign(reg_target_size, 0.0);
    for (double& v : model.regression_targets_) v = read_raw<double>(payload, off);

    const std::size_t class_count = read_raw<std::size_t>(payload, off);
    model.classes_.assign(class_count, 0);
    for (int& cls : model.classes_) cls = read_raw<int>(payload, off);

    const std::size_t binary_count = read_raw<std::size_t>(payload, off);
    model.binary_models_.assign(binary_count, {});
    for (auto& binary : model.binary_models_) {
        binary.positive_label = read_raw<int>(payload, off);
        binary.negative_label = read_raw<int>(payload, off);
        binary.b = read_raw<double>(payload, off);
        binary.prob_a = read_raw<double>(payload, off);
        binary.prob_b = read_raw<double>(payload, off);
        binary.positive_weight = read_raw<double>(payload, off);
        binary.negative_weight = read_raw<double>(payload, off);
        binary.rho = read_raw<double>(payload, off);

        const std::size_t w_size = read_raw<std::size_t>(payload, off);
        binary.w.assign(w_size, 0.0);
        for (double& v : binary.w) v = read_raw<double>(payload, off);

        const std::size_t n_sv = read_raw<std::size_t>(payload, off);
        const std::size_t d = read_raw<std::size_t>(payload, off);
        binary.support_X.assign(n_sv, std::vector<double>(d, 0.0));
        for (auto& row : binary.support_X) for (double& v : row) v = read_raw<double>(payload, off);

        const std::size_t y_size = read_raw<std::size_t>(payload, off);
        binary.support_y.assign(y_size, 0);
        for (int& v : binary.support_y) v = read_raw<int>(payload, off);

        const std::size_t alpha_size = read_raw<std::size_t>(payload, off);
        binary.alpha.assign(alpha_size, 0.0);
        for (double& v : binary.alpha) v = read_raw<double>(payload, off);

        const std::size_t err_size = read_raw<std::size_t>(payload, off);
        binary.error_cache.assign(err_size, 0.0);
        for (double& v : binary.error_cache) v = read_raw<double>(payload, off);
    }
    return model;
}

}  // namespace svm
