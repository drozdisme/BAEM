#include "models/sindy/sindy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace sindy {

namespace {

constexpr char kMagic[8] = {'U','M','L','S','I','N','5','\0'};
constexpr std::uint32_t kVersion = 5;

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
    if (off + sizeof(T) > in.size()) throw std::runtime_error("SINDy deserialize: truncated payload");
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

std::vector<double> linspace(double lo, double hi, std::size_t n) {
    std::vector<double> out(n, lo);
    if (n <= 1) return out;
    const double step = (hi - lo) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) out[i] = lo + step * static_cast<double>(i);
    return out;
}

}  // namespace

SINDy::SINDy(const SINDyParams& params) : params_(params) {
    if (params_.polynomial_order < 1 || params_.polynomial_order > 4) throw std::invalid_argument("SINDy: supported polynomial_order range is [1, 4]");
    if (params_.threshold < 0.0) throw std::invalid_argument("SINDy: threshold must be >= 0");
    if (params_.max_iterations <= 0) throw std::invalid_argument("SINDy: max_iterations must be > 0");
    if (params_.ridge < 0.0) throw std::invalid_argument("SINDy: ridge must be >= 0");
    if (params_.weak_window == 0) throw std::invalid_argument("SINDy: weak_window must be > 0");
    if (params_.quadrature_substeps == 0) throw std::invalid_argument("SINDy: quadrature_substeps must be > 0");
    if (params_.support_path_steps == 0) throw std::invalid_argument("SINDy: support_path_steps must be > 0");
    if (!(params_.ensemble_subsample_ratio > 0.0 && params_.ensemble_subsample_ratio <= 1.0)) throw std::invalid_argument("SINDy: ensemble_subsample_ratio must be in (0, 1]");
    if (!(params_.stability_threshold >= 0.0 && params_.stability_threshold <= 1.0)) throw std::invalid_argument("SINDy: stability_threshold must be in [0, 1]");
}

void SINDy::append_polynomial_features(const std::vector<double>& state,
                                       const std::vector<double>* control,
                                       std::vector<double>& phi,
                                       std::vector<std::string>* names) const {
    for (std::size_t i = 0; i < state.size(); ++i) {
        phi.push_back(state[i]);
        if (names) names->push_back("x" + std::to_string(i));
    }
    if (control != nullptr) {
        for (std::size_t i = 0; i < control->size(); ++i) {
            phi.push_back((*control)[i]);
            if (names) names->push_back("u" + std::to_string(i));
        }
    }

    const std::size_t total_dim = state.size() + (control ? control->size() : 0);
    std::vector<double> all = state;
    if (control) all.insert(all.end(), control->begin(), control->end());
    std::vector<std::string> labels;
    for (std::size_t i = 0; i < state.size(); ++i) labels.push_back("x" + std::to_string(i));
    if (control) for (std::size_t i = 0; i < control->size(); ++i) labels.push_back("u" + std::to_string(i));

    if (params_.polynomial_order >= 2) {
        for (std::size_t i = 0; i < total_dim; ++i)
            for (std::size_t j = i; j < total_dim; ++j) {
                phi.push_back(all[i] * all[j]);
                if (names) names->push_back(labels[i] + "*" + labels[j]);
            }
    }
    if (params_.polynomial_order >= 3) {
        for (std::size_t i = 0; i < total_dim; ++i)
            for (std::size_t j = i; j < total_dim; ++j)
                for (std::size_t k = j; k < total_dim; ++k) {
                    phi.push_back(all[i] * all[j] * all[k]);
                    if (names) names->push_back(labels[i] + "*" + labels[j] + "*" + labels[k]);
                }
    }
    if (params_.polynomial_order >= 4) {
        for (std::size_t i = 0; i < total_dim; ++i)
            for (std::size_t j = i; j < total_dim; ++j)
                for (std::size_t k = j; k < total_dim; ++k)
                    for (std::size_t l = k; l < total_dim; ++l) {
                        phi.push_back(all[i] * all[j] * all[k] * all[l]);
                        if (names) names->push_back(labels[i] + "*" + labels[j] + "*" + labels[k] + "*" + labels[l]);
                    }
    }
}

std::vector<std::string> SINDy::build_feature_names(std::size_t state_dim,
                                                    std::size_t control_dim) const {
    std::vector<std::string> names;
    if (params_.include_bias) names.push_back("1");
    std::vector<double> state_dummy(state_dim, 1.0);
    std::vector<double> control_dummy(control_dim, 1.0);
    std::vector<double> tmp;
    append_polynomial_features(state_dummy, control_dim ? &control_dummy : nullptr, tmp, &names);

    if (params_.include_trig || params_.library == FeatureLibraryType::PolynomialTrig || params_.library == FeatureLibraryType::Generalized) {
        for (std::size_t i = 0; i < state_dim; ++i) {
            names.push_back("sin(x" + std::to_string(i) + ")");
            names.push_back("cos(x" + std::to_string(i) + ")");
        }
        for (std::size_t i = 0; i < control_dim; ++i) {
            names.push_back("sin(u" + std::to_string(i) + ")");
            names.push_back("cos(u" + std::to_string(i) + ")");
        }
    }
    if (params_.include_pairwise_sin_cos) {
        for (std::size_t i = 0; i < state_dim; ++i)
            for (std::size_t j = i + 1; j < state_dim; ++j) {
                names.push_back("sin(x" + std::to_string(i) + "-x" + std::to_string(j) + ")");
                names.push_back("cos(x" + std::to_string(i) + "-x" + std::to_string(j) + ")");
            }
    }
    if (params_.include_inverse || params_.library == FeatureLibraryType::Generalized) {
        for (std::size_t i = 0; i < state_dim; ++i) names.push_back("inv(1+|x" + std::to_string(i) + "|)");
        for (std::size_t i = 0; i < control_dim; ++i) names.push_back("inv(1+|u" + std::to_string(i) + "|)");
    }
    return names;
}

std::vector<double> SINDy::build_features(const std::vector<double>& x,
                                          const std::vector<double>* u) const {
    std::vector<double> phi;
    if (params_.include_bias) phi.push_back(1.0);
    append_polynomial_features(x, u, phi, nullptr);

    if (params_.include_trig || params_.library == FeatureLibraryType::PolynomialTrig || params_.library == FeatureLibraryType::Generalized) {
        for (double v : x) {
            phi.push_back(std::sin(v));
            phi.push_back(std::cos(v));
        }
        if (u) {
            for (double v : *u) {
                phi.push_back(std::sin(v));
                phi.push_back(std::cos(v));
            }
        }
    }
    if (params_.include_pairwise_sin_cos) {
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = i + 1; j < x.size(); ++j) {
                phi.push_back(std::sin(x[i] - x[j]));
                phi.push_back(std::cos(x[i] - x[j]));
            }
    }
    if (params_.include_inverse || params_.library == FeatureLibraryType::Generalized) {
        for (double v : x) phi.push_back(1.0 / (1.0 + std::abs(v)));
        if (u) for (double v : *u) phi.push_back(1.0 / (1.0 + std::abs(v)));
    }
    return phi;
}

std::vector<double> SINDy::solve_normal_eq(const std::vector<std::vector<double>>& A,
                                           const std::vector<double>& b,
                                           double ridge_override) const {
    const std::size_t m = A.size();
    const std::size_t n = A.front().size();
    const double ridge = ridge_override >= 0.0 ? ridge_override : params_.ridge;
    const std::size_t rows = m + n;
    std::vector<std::vector<double>> Aaug(rows, std::vector<double>(n, 0.0));
    std::vector<double> baug(rows, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        Aaug[i] = A[i];
        baug[i] = b[i];
    }
    const double rs = std::sqrt(std::max(0.0, ridge));
    for (std::size_t i = 0; i < n; ++i) Aaug[m + i][i] = rs;

    std::vector<std::vector<double>> Q(rows, std::vector<double>(n, 0.0));
    std::vector<double> R(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> v(rows, 0.0);
        for (std::size_t i = 0; i < rows; ++i) v[i] = Aaug[i][j];
        for (std::size_t k = 0; k < j; ++k) {
            double r = 0.0;
            for (std::size_t i = 0; i < rows; ++i) r += Q[i][k] * v[i];
            R[k * n + j] = r;
            for (std::size_t i = 0; i < rows; ++i) v[i] -= r * Q[i][k];
        }
        double norm = 0.0;
        for (double vv : v) norm += vv * vv;
        norm = std::sqrt(norm);
        R[j * n + j] = norm;
        if (norm < 1e-12) continue;
        for (std::size_t i = 0; i < rows; ++i) Q[i][j] = v[i] / norm;
    }

    std::vector<double> Qtb(n, 0.0), x(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < rows; ++i) Qtb[j] += Q[i][j] * baug[i];
    }
    for (std::size_t ii = n; ii-- > 0;) {
        double rhs = Qtb[ii];
        for (std::size_t j = ii + 1; j < n; ++j) rhs -= R[ii * n + j] * x[j];
        const double d = R[ii * n + ii];
        x[ii] = (std::abs(d) < 1e-12) ? 0.0 : rhs / d;
    }
    return x;
}

std::vector<double> SINDy::weak_test_function(std::size_t family,
                                              std::size_t window_size) const {
    std::vector<double> phi(window_size, 0.0);
    if (window_size == 0) return phi;
    const double denom = std::max(1.0, static_cast<double>(window_size - 1));
    for (std::size_t i = 0; i < window_size; ++i) {
        const double s = static_cast<double>(i) / denom;
        switch (family % 4) {
            case 0:
                phi[i] = s * (1.0 - s);
                break;
            case 1:
                phi[i] = std::sin(M_PI * s);
                break;
            case 2:
                phi[i] = std::sin(2.0 * M_PI * s);
                break;
            default:
                phi[i] = (1.0 - 2.0 * std::abs(s - 0.5));
                break;
        }
    }
    return phi;
}

std::vector<std::vector<double>> SINDy::estimate_derivatives_weak_variational(const std::vector<std::vector<double>>& X,
                                                                               double dt) const {
    if (X.size() < 3) throw std::invalid_argument("SINDy::estimate_derivatives_weak_variational: need at least 3 samples");
    const std::size_t n = X.size();
    const std::size_t d = X.front().size();
    const std::size_t window = std::max<std::size_t>(2, params_.weak_window);
    const std::size_t substeps = std::max<std::size_t>(1, params_.quadrature_substeps);
    std::vector<std::vector<double>> Xdot(n, std::vector<double>(d, 0.0));
    std::vector<double> weight_sum(n, 0.0);

    for (std::size_t center = 0; center < n; ++center) {
        const std::size_t lo = (center > window ? center - window : 0);
        const std::size_t hi = std::min(n - 1, center + window);
        const std::size_t span = hi - lo + 1;
        for (std::size_t family = 0; family < std::max<std::size_t>(1, params_.weak_test_functions); ++family) {
            const auto phi = weak_test_function(family, span);
            double mean_phi = std::accumulate(phi.begin(), phi.end(), 0.0) / static_cast<double>(span);
            std::vector<double> phi_norm = phi;
            for (double& v : phi_norm) v -= mean_phi;
            double l2 = 0.0;
            for (double v : phi_norm) l2 += v * v;
            l2 = std::sqrt(std::max(1e-12, l2));
            for (double& v : phi_norm) v /= l2;
            std::vector<double> dphi(span, 0.0);
            for (std::size_t k = 1; k + 1 < span; ++k) dphi[k] = (phi_norm[k + 1] - phi_norm[k - 1]) / (2.0 * dt);
            if (span > 1) {
                dphi[0] = (phi_norm[1] - phi_norm[0]) / dt;
                dphi[span - 1] = (phi_norm[span - 1] - phi_norm[span - 2]) / dt;
            }
            for (std::size_t dim = 0; dim < d; ++dim) {
                double numerator = 0.0;
                double denom = 0.0;
                for (std::size_t local = 0; local < span - 1; ++local) {
                    for (std::size_t q = 0; q < substeps; ++q) {
                        const double t = (static_cast<double>(q) + 0.5) / static_cast<double>(substeps);
                        const double xq = (1.0 - t) * X[lo + local][dim] + t * X[lo + local + 1][dim];
                        const double wq = (1.0 - t) * dphi[local] + t * dphi[local + 1];
                        numerator += -xq * wq * (dt / static_cast<double>(substeps));
                        const double phiq = (1.0 - t) * phi_norm[local] + t * phi_norm[local + 1];
                        denom += phiq * phiq * (dt / static_cast<double>(substeps));
                    }
                }
                const double estimate = numerator / std::max(1e-8, denom);
                Xdot[center][dim] += estimate;
            }
            weight_sum[center] += 1.0;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        const double norm = std::max(1.0, weight_sum[i]);
        for (std::size_t j = 0; j < d; ++j) Xdot[i][j] /= norm;
    }
    return Xdot;
}

std::vector<std::vector<double>> SINDy::fit_sparse_system(const std::vector<std::vector<double>>& Theta,
                                                         const std::vector<std::vector<double>>& Xdot) const {
    std::vector<std::vector<double>> coeffs(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    const std::size_t m = Theta.size();
    const std::size_t val_count = (m >= 20 ? std::max<std::size_t>(1, m / 5) : 0);
    const std::size_t train_count = m - val_count;
    std::vector<std::vector<double>> Theta_train;
    std::vector<std::vector<double>> Theta_val;
    Theta_train.reserve(train_count);
    Theta_val.reserve(val_count);
    for (std::size_t i = 0; i < m; ++i) {
        if (i < train_count) Theta_train.push_back(Theta[i]);
        else Theta_val.push_back(Theta[i]);
    }

    std::vector<double> threshold_grid;
    if (params_.threshold <= 1e-9) {
        threshold_grid = {0.0, 1e-6, 5e-6, 1e-5};
    } else {
        threshold_grid = {
            params_.threshold * 0.25,
            params_.threshold * 0.5,
            params_.threshold,
            params_.threshold * 1.5,
            params_.threshold * 2.0
        };
    }

    for (std::size_t target = 0; target < input_dim_; ++target) {
        std::vector<double> rhs(Xdot.size());
        for (std::size_t i = 0; i < Xdot.size(); ++i) rhs[i] = Xdot[i][target];
        std::vector<double> rhs_train(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(train_count));
        std::vector<double> rhs_val(rhs.begin() + static_cast<std::ptrdiff_t>(train_count), rhs.end());

        auto fit_with_threshold = [&](double thr) {
            auto xi_local = solve_normal_eq(Theta_train.empty() ? Theta : Theta_train,
                                            rhs_train.empty() ? rhs : rhs_train);
            for (int iter = 0; iter < params_.max_iterations; ++iter) {
                std::vector<std::size_t> active;
                for (std::size_t j = 0; j < xi_local.size(); ++j) if (std::abs(xi_local[j]) >= thr) active.push_back(j);
                if (active.empty()) {
                    std::fill(xi_local.begin(), xi_local.end(), 0.0);
                    break;
                }
                const auto& theta_ref = Theta_train.empty() ? Theta : Theta_train;
                std::vector<std::vector<double>> Theta_reduced(theta_ref.size(), std::vector<double>(active.size()));
                for (std::size_t i = 0; i < theta_ref.size(); ++i)
                    for (std::size_t j = 0; j < active.size(); ++j)
                        Theta_reduced[i][j] = theta_ref[i][active[j]];
                const auto& rhs_ref = rhs_train.empty() ? rhs : rhs_train;
                const double ridge = params_.unbias_after_support ? 0.0 : params_.ridge;
                auto xi_reduced = solve_normal_eq(Theta_reduced, rhs_ref, ridge);
                std::vector<double> next(xi_local.size(), 0.0);
                for (std::size_t j = 0; j < active.size(); ++j) next[active[j]] = xi_reduced[j];
                xi_local = std::move(next);
            }
            return xi_local;
        };

        double best_score = std::numeric_limits<double>::infinity();
        std::vector<double> best_xi(feature_names_.size(), 0.0);
        for (double thr : threshold_grid) {
            auto xi = fit_with_threshold(thr);
            double mse = 0.0;
            const auto& theta_eval = Theta_val.empty() ? Theta : Theta_val;
            const auto& rhs_eval = rhs_val.empty() ? rhs : rhs_val;
            for (std::size_t i = 0; i < theta_eval.size(); ++i) {
                double pred = 0.0;
                for (std::size_t j = 0; j < xi.size(); ++j) pred += theta_eval[i][j] * xi[j];
                const double e = pred - rhs_eval[i];
                mse += e * e;
            }
            mse /= std::max<std::size_t>(1, theta_eval.size());
            std::size_t active = 0;
            for (double c : xi) if (std::abs(c) >= thr) ++active;
            const double score = mse + 1e-5 * static_cast<double>(active);
            if (score < best_score) {
                best_score = score;
                best_xi = std::move(xi);
            }
        }
        coeffs[target] = std::move(best_xi);
    }
    return coeffs;
}

namespace {

double derivative_fit_r2(const std::vector<std::vector<double>>& y_true,
                         const std::vector<std::vector<double>>& y_pred) {
    if (y_true.empty()) return 0.0;
    const std::size_t n = y_true.size();
    const std::size_t d = y_true.front().size();
    double mean = 0.0;
    double cnt = 0.0;
    for (const auto& row : y_true) {
        for (double v : row) {
            mean += v;
            cnt += 1.0;
        }
    }
    mean /= std::max(1.0, cnt);

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            const double e = y_true[i][j] - y_pred[i][j];
            const double c = y_true[i][j] - mean;
            ss_res += e * e;
            ss_tot += c * c;
        }
    }
    if (ss_tot < 1e-12) return 1.0;
    return 1.0 - ss_res / ss_tot;
}

std::size_t count_active_terms(const std::vector<std::vector<double>>& coeffs,
                               double threshold) {
    std::size_t active = 0;
    for (const auto& row : coeffs) {
        for (double c : row) {
            if (std::abs(c) >= threshold) ++active;
        }
    }
    return active;
}

}  // namespace

void SINDy::populate_support_path() {
    stability_report_.support_path_thresholds.assign(input_dim_, {});
    stability_report_.support_path_coefficients.assign(input_dim_, {});
    const auto thresholds = linspace(0.0, std::max(params_.threshold, 1e-6) * 2.5, params_.support_path_steps);
    for (std::size_t t = 0; t < input_dim_; ++t) {
        stability_report_.support_path_thresholds[t] = thresholds;
        stability_report_.support_path_coefficients[t].assign(thresholds.size(), std::vector<double>(feature_names_.size(), 0.0));
        for (std::size_t s = 0; s < thresholds.size(); ++s) {
            const double thr = thresholds[s];
            for (std::size_t j = 0; j < feature_names_.size(); ++j) {
                if (std::abs(coeffs_[t][j]) >= thr) stability_report_.support_path_coefficients[t][s][j] = coeffs_[t][j];
            }
        }
    }
}

void SINDy::populate_model_selection_summary() {
    stability_report_.model_selection_summary.clear();
    for (std::size_t t = 0; t < input_dim_; ++t) {
        std::ostringstream oss;
        std::size_t active = 0;
        double total_importance = 0.0;
        for (std::size_t j = 0; j < feature_names_.size(); ++j) {
            if (std::abs(coeffs_[t][j]) >= params_.threshold) ++active;
            if (t < stability_report_.inclusion_importance.size() && j < stability_report_.inclusion_importance[t].size()) {
                total_importance += stability_report_.inclusion_importance[t][j];
            }
        }
        oss << "state " << t << ": active_terms=" << active
            << ", avg_importance=" << (feature_names_.empty() ? 0.0 : total_importance / static_cast<double>(feature_names_.size()));
        stability_report_.model_selection_summary.push_back(oss.str());
    }
}

std::vector<std::vector<double>> SINDy::evaluate_bootstrap_rmse(const std::vector<std::vector<std::vector<double>>>& ensemble_coeffs,
                                                                const std::vector<std::vector<double>>& X,
                                                                const std::vector<std::vector<double>>* U,
                                                                const std::vector<std::vector<double>>& Xdot) const {
    std::vector<std::vector<double>> rmse(input_dim_, std::vector<double>(ensemble_coeffs.size(), 0.0));
    for (std::size_t e = 0; e < ensemble_coeffs.size(); ++e) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            const auto phi = build_features(X[i], U ? &(*U)[i] : nullptr);
            for (std::size_t t = 0; t < input_dim_; ++t) {
                double pred = 0.0;
                for (std::size_t j = 0; j < phi.size(); ++j) pred += ensemble_coeffs[e][t][j] * phi[j];
                const double err = pred - Xdot[i][t];
                rmse[t][e] += err * err;
            }
        }
        for (std::size_t t = 0; t < input_dim_; ++t) rmse[t][e] = std::sqrt(rmse[t][e] / std::max<std::size_t>(1, X.size()));
    }
    return rmse;
}

std::vector<std::vector<double>> SINDy::compute_control_alignment(const std::vector<std::vector<double>>& U,
                                                                  const std::vector<std::vector<double>>& Xdot) const {
    if (U.empty()) return {};
    std::vector<std::vector<double>> out(input_dim_, std::vector<double>(control_dim_, 0.0));
    for (std::size_t t = 0; t < input_dim_; ++t) {
        for (std::size_t c = 0; c < control_dim_; ++c) {
            double num = 0.0;
            double du = 0.0;
            double dx = 0.0;
            double mean_u = 0.0;
            double mean_x = 0.0;
            for (std::size_t i = 0; i < U.size(); ++i) {
                mean_u += U[i][c];
                mean_x += Xdot[i][t];
            }
            mean_u /= static_cast<double>(U.size());
            mean_x /= static_cast<double>(U.size());
            for (std::size_t i = 0; i < U.size(); ++i) {
                const double uu = U[i][c] - mean_u;
                const double xx = Xdot[i][t] - mean_x;
                num += uu * xx;
                du += uu * uu;
                dx += xx * xx;
            }
            out[t][c] = num / std::sqrt(std::max(1e-12, du * dx));
        }
    }
    return out;
}

void SINDy::fit_internal(const std::vector<std::vector<double>>& X,
                         const std::vector<std::vector<double>>* U,
                         const std::vector<std::vector<double>>& Xdot) {
    if (X.empty() || X.size() != Xdot.size()) throw std::invalid_argument("SINDy::fit: invalid dataset");
    input_dim_ = X.front().size();
    if (input_dim_ == 0) throw std::invalid_argument("SINDy::fit: input dimension must be > 0");
    control_dim_ = U ? U->front().size() : 0;
    for (const auto& row : X) if (row.size() != input_dim_) throw std::invalid_argument("SINDy::fit: inconsistent X dims");
    for (const auto& row : Xdot) if (row.size() != input_dim_) throw std::invalid_argument("SINDy::fit: Xdot dim must match X dim");
    if (U) {
        if (U->size() != X.size()) throw std::invalid_argument("SINDy::fit_with_control: U size must match X size");
        for (const auto& row : *U) if (row.size() != control_dim_) throw std::invalid_argument("SINDy::fit_with_control: inconsistent U dims");
    }

    feature_names_ = build_feature_names(input_dim_, control_dim_);
    std::vector<std::vector<double>> Theta(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) Theta[i] = build_features(X[i], U ? &(*U)[i] : nullptr);
    std::vector<double> theta_scale(feature_names_.size(), 1.0);
    for (std::size_t j = 0; j < feature_names_.size(); ++j) {
        double ss = 0.0;
        for (std::size_t i = 0; i < Theta.size(); ++i) ss += Theta[i][j] * Theta[i][j];
        theta_scale[j] = std::sqrt(ss / std::max<std::size_t>(1, Theta.size()));
        if (theta_scale[j] < 1e-12) theta_scale[j] = 1.0;
        for (std::size_t i = 0; i < Theta.size(); ++i) Theta[i][j] /= theta_scale[j];
    }
    coeffs_ = fit_sparse_system(Theta, Xdot);
    for (std::size_t t = 0; t < input_dim_; ++t)
        for (std::size_t j = 0; j < feature_names_.size(); ++j)
            coeffs_[t][j] /= theta_scale[j];

    // Adaptive fallback: pure polynomial libraries often underfit oscillatory/decaying systems.
    // If training derivative R^2 is poor and trig features were not explicitly enabled, retry once with trig terms.
    const bool may_retry_with_trig =
        (params_.library == FeatureLibraryType::Polynomial) &&
        !params_.include_trig &&
        !params_.include_inverse &&
        !params_.include_pairwise_sin_cos;
    if (may_retry_with_trig) {
        std::vector<std::vector<double>> pred_base(X.size(), std::vector<double>(input_dim_, 0.0));
        for (std::size_t i = 0; i < X.size(); ++i) {
            const auto phi = build_features(X[i], U ? &(*U)[i] : nullptr);
            for (std::size_t t = 0; t < input_dim_; ++t) {
                for (std::size_t j = 0; j < phi.size(); ++j) pred_base[i][t] += coeffs_[t][j] * phi[j];
            }
        }
        const double r2_base = derivative_fit_r2(Xdot, pred_base);
        if (r2_base < 0.70) {
            SINDy best_model = *this;
            double best_score = r2_base - 5e-4 * static_cast<double>(count_active_terms(coeffs_, params_.threshold));
            std::vector<SINDyParams> candidates;
            {
                auto p = params_;
                p.include_trig = true;
                p.library = FeatureLibraryType::PolynomialTrig;
                candidates.push_back(p);
            }
            {
                auto p = params_;
                p.include_trig = true;
                p.include_pairwise_sin_cos = true;
                p.library = FeatureLibraryType::Generalized;
                p.threshold = std::max(1e-6, params_.threshold * 0.7);
                candidates.push_back(p);
            }
            {
                auto p = params_;
                p.include_trig = true;
                p.include_inverse = true;
                p.library = FeatureLibraryType::Generalized;
                p.threshold = std::max(1e-6, params_.threshold * 0.6);
                candidates.push_back(p);
            }
            for (const auto& candidate : candidates) {
                SINDy retry(candidate);
                retry.fit_internal(X, U, Xdot);
                std::vector<std::vector<double>> pred_retry(X.size(), std::vector<double>(input_dim_, 0.0));
                for (std::size_t i = 0; i < X.size(); ++i) {
                    const auto phi = retry.build_features(X[i], U ? &(*U)[i] : nullptr);
                    for (std::size_t t = 0; t < input_dim_; ++t) {
                        for (std::size_t j = 0; j < phi.size(); ++j) pred_retry[i][t] += retry.coeffs_[t][j] * phi[j];
                    }
                }
                const double r2_retry = derivative_fit_r2(Xdot, pred_retry);
                const double complexity = static_cast<double>(count_active_terms(retry.coeffs_, retry.params_.threshold));
                const double score = r2_retry - 5e-4 * complexity;
                if (score > best_score + 1e-3) {
                    best_score = score;
                    best_model = std::move(retry);
                }
            }
            if (best_score > r2_base + 1e-3) {
                *this = std::move(best_model);
                return;
            }
        }
    }
    ensemble_coefficients_cache_.clear();
    stability_report_ = {};
    stability_report_.feature_names = feature_names_;
    stability_report_.selection_frequency.assign(input_dim_, std::vector<double>(feature_names_.size(), 1.0));
    stability_report_.mean_coefficients = coeffs_;
    stability_report_.coefficient_stddev.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_.ci_lower = coeffs_;
    stability_report_.ci_upper = coeffs_;
    stability_report_.inclusion_importance.assign(input_dim_, std::vector<double>(feature_names_.size(), 1.0));
    if (U) stability_report_.aligned_control_variation = compute_control_alignment(*U, Xdot);
    populate_support_path();
    populate_model_selection_summary();
    fitted_ = true;
}

void SINDy::fit(const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>>& Xdot) {
    fit_internal(X, nullptr, Xdot);
}

void SINDy::fit_ensemble_internal(const std::vector<std::vector<double>>& X,
                                  const std::vector<std::vector<double>>* U,
                                  const std::vector<std::vector<double>>& Xdot) {
    if (params_.ensemble_models <= 1) {
        fit_internal(X, U, Xdot);
        return;
    }
    if (X.empty() || X.size() != Xdot.size()) throw std::invalid_argument("SINDy::fit_ensemble: invalid dataset");
    input_dim_ = X.front().size();
    if (input_dim_ == 0) throw std::invalid_argument("SINDy::fit_ensemble: input dimension must be > 0");
    control_dim_ = U ? U->front().size() : 0;
    for (const auto& row : X) if (row.size() != input_dim_) throw std::invalid_argument("SINDy::fit_ensemble: inconsistent X dims");
    for (const auto& row : Xdot) if (row.size() != input_dim_) throw std::invalid_argument("SINDy::fit_ensemble: Xdot dim must match X dim");
    if (U) {
        if (U->size() != X.size()) throw std::invalid_argument("SINDy::fit_ensemble_with_control: U size must match X size");
        for (const auto& row : *U) if (row.size() != control_dim_) throw std::invalid_argument("SINDy::fit_ensemble_with_control: inconsistent U dims");
    }
    feature_names_ = build_feature_names(input_dim_, control_dim_);
    const std::size_t m = std::max<std::size_t>(1, static_cast<std::size_t>(std::round(params_.ensemble_subsample_ratio * X.size())));
    std::random_device rd;
    std::mt19937 rng(rd());
    std::vector<std::vector<double>> coeff_sum(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    std::vector<std::vector<double>> coeff_sq_sum(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    std::vector<std::vector<double>> support_count(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    ensemble_coefficients_cache_.clear();
    for (std::size_t e = 0; e < params_.ensemble_models; ++e) {
        std::vector<std::vector<double>> Xsub;
        std::vector<std::vector<double>> Xdotsub;
        std::vector<std::vector<double>> Usub;
        std::uniform_int_distribution<std::size_t> dist(0, X.size() - 1);
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t idx = dist(rng);
            Xsub.push_back(X[idx]);
            Xdotsub.push_back(Xdot[idx]);
            if (U) Usub.push_back((*U)[idx]);
        }
        std::vector<std::vector<double>> Theta(Xsub.size());
        for (std::size_t i = 0; i < Xsub.size(); ++i) Theta[i] = build_features(Xsub[i], U ? &Usub[i] : nullptr);
        auto coeffs = fit_sparse_system(Theta, Xdotsub);
        ensemble_coefficients_cache_.push_back(coeffs);
        for (std::size_t t = 0; t < input_dim_; ++t) {
            for (std::size_t j = 0; j < feature_names_.size(); ++j) {
                coeff_sum[t][j] += coeffs[t][j];
                coeff_sq_sum[t][j] += coeffs[t][j] * coeffs[t][j];
                if (std::abs(coeffs[t][j]) >= params_.threshold) support_count[t][j] += 1.0;
            }
        }
    }
    coeffs_.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_ = {};
    stability_report_.feature_names = feature_names_;
    stability_report_.selection_frequency.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_.mean_coefficients.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_.coefficient_stddev.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_.ci_lower.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_.ci_upper.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    stability_report_.inclusion_importance.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
    for (std::size_t t = 0; t < input_dim_; ++t) {
        for (std::size_t j = 0; j < feature_names_.size(); ++j) {
            const double freq = support_count[t][j] / static_cast<double>(params_.ensemble_models);
            const double mean = coeff_sum[t][j] / static_cast<double>(params_.ensemble_models);
            const double var = std::max(0.0, coeff_sq_sum[t][j] / static_cast<double>(params_.ensemble_models) - mean * mean);
            const double stddev = std::sqrt(var);
            stability_report_.selection_frequency[t][j] = freq;
            stability_report_.mean_coefficients[t][j] = mean;
            stability_report_.coefficient_stddev[t][j] = stddev;
            stability_report_.ci_lower[t][j] = mean - 1.96 * stddev;
            stability_report_.ci_upper[t][j] = mean + 1.96 * stddev;
            stability_report_.inclusion_importance[t][j] = freq * std::abs(mean) / std::max(1e-8, stddev + 1e-6);
            if (freq >= params_.stability_threshold) coeffs_[t][j] = mean;
        }
    }
    if (params_.bootstrap_samples > 0) {
        auto rmse = evaluate_bootstrap_rmse(ensemble_coefficients_cache_, X, U, Xdot);
        stability_report_.bootstrap_derivative_rmse = rmse;
        stability_report_.bootstrap_state_rmse = rmse;
    }
    if (U) stability_report_.aligned_control_variation = compute_control_alignment(*U, Xdot);
    populate_support_path();
    populate_model_selection_summary();
    fitted_ = true;
}

void SINDy::fit_with_control(const std::vector<std::vector<double>>& X,
                             const std::vector<std::vector<double>>& U,
                             const std::vector<std::vector<double>>& Xdot) {
    fit_internal(X, &U, Xdot);
}

std::vector<std::vector<double>> SINDy::estimate_derivatives(const std::vector<std::vector<double>>& X,
                                                             double dt) const {
    if (X.size() < 2) throw std::invalid_argument("SINDy::estimate_derivatives: need at least 2 samples");
    if (!(dt > 0.0)) throw std::invalid_argument("SINDy::estimate_derivatives: dt must be > 0");
    const std::size_t n = X.size();
    const std::size_t d = X.front().size();
    std::vector<std::vector<double>> Xsmooth = X;
    if (params_.derivative_mode == DerivativeMode::SmoothedFiniteDifference) {
        const std::size_t w = std::max<std::size_t>(1, params_.smoothing_window);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                double acc = 0.0;
                double cnt = 0.0;
                const std::size_t lo = (i > w ? i - w : 0);
                const std::size_t hi = std::min(n - 1, i + w);
                for (std::size_t k = lo; k <= hi; ++k) { acc += X[k][j]; cnt += 1.0; }
                Xsmooth[i][j] = acc / std::max(1.0, cnt);
            }
        }
    }
    std::vector<std::vector<double>> Xdot(n, std::vector<double>(d, 0.0));
    if (params_.derivative_mode == DerivativeMode::WeakIntegral) {
        return estimate_derivatives_weak_variational(Xsmooth, dt);
    }
    for (std::size_t j = 0; j < d; ++j) {
        if (n >= 5) {
            Xdot[0][j] = (-25.0 * Xsmooth[0][j] + 48.0 * Xsmooth[1][j] - 36.0 * Xsmooth[2][j] + 16.0 * Xsmooth[3][j] - 3.0 * Xsmooth[4][j]) / (12.0 * dt);
            Xdot[1][j] = (-3.0 * Xsmooth[0][j] - 10.0 * Xsmooth[1][j] + 18.0 * Xsmooth[2][j] - 6.0 * Xsmooth[3][j] + Xsmooth[4][j]) / (12.0 * dt);
            for (std::size_t i = 2; i + 2 < n; ++i) {
                Xdot[i][j] = (Xsmooth[i - 2][j] - 8.0 * Xsmooth[i - 1][j] + 8.0 * Xsmooth[i + 1][j] - Xsmooth[i + 2][j]) / (12.0 * dt);
            }
            Xdot[n - 2][j] = (3.0 * Xsmooth[n - 1][j] + 10.0 * Xsmooth[n - 2][j] - 18.0 * Xsmooth[n - 3][j] + 6.0 * Xsmooth[n - 4][j] - Xsmooth[n - 5][j]) / (12.0 * dt);
            Xdot[n - 1][j] = (25.0 * Xsmooth[n - 1][j] - 48.0 * Xsmooth[n - 2][j] + 36.0 * Xsmooth[n - 3][j] - 16.0 * Xsmooth[n - 4][j] + 3.0 * Xsmooth[n - 5][j]) / (12.0 * dt);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t im1 = (i == 0 ? 0 : i - 1);
                const std::size_t ip1 = (i + 1 >= n ? n - 1 : i + 1);
                const double denom = (ip1 == im1 ? dt : (ip1 - im1) * dt);
                Xdot[i][j] = (Xsmooth[ip1][j] - Xsmooth[im1][j]) / denom;
            }
        }
    }
    return Xdot;
}

std::vector<std::vector<double>> SINDy::estimate_derivatives_multi_test(const std::vector<std::vector<double>>& X,
                                                                        double dt) const {
    auto base = estimate_derivatives(X, dt);
    if (params_.derivative_mode != DerivativeMode::WeakIntegral || params_.weak_test_functions <= 1) return base;
    auto out = base;
    for (std::size_t m = 1; m < params_.weak_test_functions; ++m) {
        auto alt = estimate_derivatives_weak_variational(X, dt);
        for (std::size_t i = 0; i < X.size(); ++i)
            for (std::size_t j = 0; j < X[i].size(); ++j)
                out[i][j] += alt[i][j];
    }
    const double denom = static_cast<double>(params_.weak_test_functions);
    for (auto& row : out) for (double& v : row) v /= denom;
    return out;
}

void SINDy::fit_from_trajectory(const std::vector<std::vector<double>>& X,
                                double dt) {
    fit_internal(X, nullptr, estimate_derivatives_multi_test(X, dt));
}

void SINDy::fit_from_trajectory_with_control(const std::vector<std::vector<double>>& X,
                                             const std::vector<std::vector<double>>& U,
                                             double dt) {
    fit_internal(X, &U, estimate_derivatives_multi_test(X, dt));
}

void SINDy::fit_multi_trajectory(const std::vector<std::vector<std::vector<double>>>& trajectories,
                                 double dt) {
    if (trajectories.empty()) throw std::invalid_argument("SINDy::fit_multi_trajectory: empty trajectories");
    if (params_.ensemble_models > 1 && params_.bag_trajectories) {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<std::size_t> dist(0, trajectories.size() - 1);
        std::vector<std::vector<std::vector<double>>> ensemble_coeffs;
        for (std::size_t e = 0; e < params_.ensemble_models; ++e) {
            std::vector<std::vector<double>> Xsub, Xdotsub;
            const std::size_t pick = std::max<std::size_t>(1, static_cast<std::size_t>(std::round(params_.ensemble_subsample_ratio * trajectories.size())));
            for (std::size_t p = 0; p < pick; ++p) {
                const auto& tr = trajectories[dist(rng)];
                auto d = estimate_derivatives_multi_test(tr, dt);
                Xsub.insert(Xsub.end(), tr.begin(), tr.end());
                Xdotsub.insert(Xdotsub.end(), d.begin(), d.end());
            }
            fit_internal(Xsub, nullptr, Xdotsub);
            ensemble_coeffs.push_back(coeffs_);
        }
        coeffs_.assign(input_dim_, std::vector<double>(feature_names_.size(), 0.0));
        for (std::size_t t = 0; t < input_dim_; ++t)
            for (std::size_t j = 0; j < feature_names_.size(); ++j) {
                double s = 0.0;
                for (const auto& c : ensemble_coeffs) s += c[t][j];
                coeffs_[t][j] = s / static_cast<double>(ensemble_coeffs.size());
            }
        fitted_ = true;
        populate_support_path();
        populate_model_selection_summary();
        return;
    }
    std::vector<std::vector<double>> Xall, Xdotall;
    for (const auto& tr : trajectories) {
        auto d = estimate_derivatives_multi_test(tr, dt);
        Xall.insert(Xall.end(), tr.begin(), tr.end());
        Xdotall.insert(Xdotall.end(), d.begin(), d.end());
    }
    if (params_.ensemble_models > 1 && params_.bag_trajectories) fit_ensemble_internal(Xall, nullptr, Xdotall);
    else fit_internal(Xall, nullptr, Xdotall);
}

void SINDy::fit_multi_trajectory_with_control(const std::vector<std::vector<std::vector<double>>>& trajectories,
                                              const std::vector<std::vector<std::vector<double>>>& controls,
                                              double dt) {
    std::vector<std::vector<double>> Xall, Xdotall, Uall;
    if (trajectories.size() != controls.size()) throw std::invalid_argument("SINDy::fit_multi_trajectory_with_control: trajectory/control count mismatch");
    for (std::size_t t = 0; t < trajectories.size(); ++t) {
        if (trajectories[t].size() != controls[t].size()) throw std::invalid_argument("SINDy::fit_multi_trajectory_with_control: trajectory/control length mismatch");
        auto d = estimate_derivatives_multi_test(trajectories[t], dt);
        if (params_.align_multi_trajectory_controls && trajectories[t].size() > 1) {
            const std::size_t common = std::min(trajectories[t].size(), controls[t].size());
            for (std::size_t i = 0; i < common; ++i) {
                Xall.push_back(trajectories[t][i]);
                Xdotall.push_back(d[i]);
                Uall.push_back(controls[t][i]);
            }
        } else {
            Xall.insert(Xall.end(), trajectories[t].begin(), trajectories[t].end());
            Xdotall.insert(Xdotall.end(), d.begin(), d.end());
            Uall.insert(Uall.end(), controls[t].begin(), controls[t].end());
        }
    }
    if (params_.ensemble_models > 1 && params_.bag_trajectories) fit_ensemble_internal(Xall, &Uall, Xdotall);
    else fit_internal(Xall, &Uall, Xdotall);
}

void SINDy::fit_ensemble(const std::vector<std::vector<double>>& X,
                         const std::vector<std::vector<double>>& Xdot) {
    fit_ensemble_internal(X, nullptr, Xdot);
}

void SINDy::fit_ensemble_with_control(const std::vector<std::vector<double>>& X,
                                      const std::vector<std::vector<double>>& U,
                                      const std::vector<std::vector<double>>& Xdot) {
    fit_ensemble_internal(X, &U, Xdot);
}

std::vector<std::vector<double>> SINDy::predict_derivative(const std::vector<std::vector<double>>& X) const {
    if (has_control()) throw std::runtime_error("SINDy::predict_derivative: model expects control inputs");
    return predict_derivative_with_control(X, {});
}

std::vector<std::vector<double>> SINDy::predict_derivative_with_control(const std::vector<std::vector<double>>& X,
                                                                        const std::vector<std::vector<double>>& U) const {
    if (!fitted_) throw std::runtime_error("SINDy::predict_derivative: model is not fitted");
    if (has_control() && U.size() != X.size()) throw std::invalid_argument("SINDy::predict_derivative_with_control: U size must match X size");
    std::vector<std::vector<double>> out(X.size(), std::vector<double>(input_dim_, 0.0));
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != input_dim_) throw std::invalid_argument("SINDy::predict_derivative: feature dimension mismatch");
        const std::vector<double>* u_ptr = nullptr;
        if (has_control()) {
            if (U[i].size() != control_dim_) throw std::invalid_argument("SINDy::predict_derivative_with_control: control dimension mismatch");
            u_ptr = &U[i];
        }
        auto phi = build_features(X[i], u_ptr);
        for (std::size_t t = 0; t < input_dim_; ++t)
            for (std::size_t j = 0; j < phi.size(); ++j)
                out[i][t] += coeffs_[t][j] * phi[j];
    }
    return out;
}

std::vector<double> SINDy::derivative_single(const std::vector<double>& x,
                                             const std::vector<double>* u) const {
    const auto phi = build_features(x, u);
    std::vector<double> out(input_dim_, 0.0);
    for (std::size_t t = 0; t < input_dim_; ++t)
        for (std::size_t j = 0; j < phi.size(); ++j)
            out[t] += coeffs_[t][j] * phi[j];
    return out;
}

std::vector<std::vector<double>> SINDy::simulate(const std::vector<double>& x0,
                                                 double dt,
                                                 std::size_t steps) const {
    if (has_control()) throw std::runtime_error("SINDy::simulate: model expects control sequence");
    if (!fitted_) throw std::runtime_error("SINDy::simulate: model is not fitted");
    if (x0.size() != input_dim_) throw std::invalid_argument("SINDy::simulate: initial state dimension mismatch");
    if (!(dt > 0.0)) throw std::invalid_argument("SINDy::simulate: dt must be > 0");
    std::vector<std::vector<double>> states(steps + 1, std::vector<double>(input_dim_, 0.0));
    states[0] = x0;
    for (std::size_t step = 0; step < steps; ++step) {
        if (params_.integrator == IntegratorType::Euler) {
            auto k1 = derivative_single(states[step], nullptr);
            states[step + 1] = states[step];
            for (std::size_t j = 0; j < input_dim_; ++j) states[step + 1][j] += dt * k1[j];
        } else {
            auto k1 = derivative_single(states[step], nullptr);
            std::vector<double> x2 = states[step], x3 = states[step], x4 = states[step];
            for (std::size_t j = 0; j < input_dim_; ++j) x2[j] += 0.5 * dt * k1[j];
            auto k2 = derivative_single(x2, nullptr);
            for (std::size_t j = 0; j < input_dim_; ++j) x3[j] += 0.5 * dt * k2[j];
            auto k3 = derivative_single(x3, nullptr);
            for (std::size_t j = 0; j < input_dim_; ++j) x4[j] += dt * k3[j];
            auto k4 = derivative_single(x4, nullptr);
            states[step + 1] = states[step];
            for (std::size_t j = 0; j < input_dim_; ++j) states[step + 1][j] += dt * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) / 6.0;
        }
    }
    return states;
}

std::vector<std::vector<double>> SINDy::simulate_with_control(const std::vector<double>& x0,
                                                              const std::vector<std::vector<double>>& U,
                                                              double dt) const {
    if (!fitted_) throw std::runtime_error("SINDy::simulate_with_control: model is not fitted");
    if (x0.size() != input_dim_) throw std::invalid_argument("SINDy::simulate_with_control: initial state dimension mismatch");
    if (!(dt > 0.0)) throw std::invalid_argument("SINDy::simulate_with_control: dt must be > 0");
    if (has_control() && U.empty()) throw std::invalid_argument("SINDy::simulate_with_control: control sequence required");

    const std::size_t steps = has_control() ? U.size() : 0;
    std::vector<std::vector<double>> states(steps + 1, std::vector<double>(input_dim_, 0.0));
    states[0] = x0;
    for (std::size_t step = 0; step < steps; ++step) {
        const auto* u = &U[step];
        if (has_control() && u->size() != control_dim_) throw std::invalid_argument("SINDy::simulate_with_control: control dimension mismatch");
        if (params_.integrator == IntegratorType::Euler) {
            auto k1 = derivative_single(states[step], u);
            states[step + 1] = states[step];
            for (std::size_t j = 0; j < input_dim_; ++j) states[step + 1][j] += dt * k1[j];
        } else {
            auto k1 = derivative_single(states[step], u);
            std::vector<double> x2 = states[step], x3 = states[step], x4 = states[step];
            for (std::size_t j = 0; j < input_dim_; ++j) x2[j] += 0.5 * dt * k1[j];
            auto k2 = derivative_single(x2, u);
            for (std::size_t j = 0; j < input_dim_; ++j) x3[j] += 0.5 * dt * k2[j];
            auto k3 = derivative_single(x3, u);
            for (std::size_t j = 0; j < input_dim_; ++j) x4[j] += dt * k3[j];
            auto k4 = derivative_single(x4, u);
            states[step + 1] = states[step];
            for (std::size_t j = 0; j < input_dim_; ++j) states[step + 1][j] += dt * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) / 6.0;
        }
    }
    return states;
}

std::vector<std::string> SINDy::equations() const {
    if (!fitted_) throw std::runtime_error("SINDy::equations: model is not fitted");
    std::vector<std::string> eqs(input_dim_);
    for (std::size_t t = 0; t < input_dim_; ++t) {
        std::ostringstream oss;
        oss << "dx" << t << "/dt = ";
        bool first = true;
        for (std::size_t j = 0; j < coeffs_[t].size(); ++j) {
            const double c = coeffs_[t][j];
            if (std::abs(c) < params_.threshold) continue;
            if (!first) oss << " + ";
            oss << c << "*" << feature_names_[j];
            first = false;
        }
        if (first) oss << "0";
        eqs[t] = oss.str();
    }
    return eqs;
}

void SINDy::save(const std::string& filepath) const {
    if (!fitted_) throw std::runtime_error("SINDy::save: model is not fitted");

    std::vector<char> payload;
    write_raw(payload, params_.polynomial_order);
    write_raw(payload, params_.threshold);
    write_raw(payload, params_.max_iterations);
    write_raw(payload, params_.include_bias);
    write_raw(payload, params_.include_trig);
    write_raw(payload, params_.include_inverse);
    write_raw(payload, params_.include_pairwise_sin_cos);
    write_raw(payload, params_.ridge);
    write_raw(payload, params_.unbias_after_support);
    write_raw(payload, static_cast<int>(params_.library));
    write_raw(payload, static_cast<int>(params_.integrator));
    write_raw(payload, static_cast<int>(params_.derivative_mode));
    write_raw(payload, params_.smoothing_window);
    write_raw(payload, params_.weak_window);
    write_raw(payload, params_.ensemble_models);
    write_raw(payload, params_.ensemble_subsample_ratio);
    write_raw(payload, params_.stability_threshold);
    write_raw(payload, params_.bag_trajectories);
    write_raw(payload, params_.weak_test_functions);
    write_raw(payload, params_.bootstrap_samples);
    write_raw(payload, params_.quadrature_substeps);
    write_raw(payload, params_.support_path_steps);
    write_raw(payload, params_.align_multi_trajectory_controls);
    write_raw(payload, fitted_);
    write_raw(payload, input_dim_);
    write_raw(payload, control_dim_);

    const std::size_t targets = coeffs_.size();
    const std::size_t features = targets ? coeffs_.front().size() : 0;
    write_raw(payload, targets);
    write_raw(payload, features);
    for (const auto& row : coeffs_) for (double v : row) write_raw(payload, v);

    const std::uint64_t checksum = fnv1a64(payload);
    const std::uint64_t payload_size = static_cast<std::uint64_t>(payload.size());

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("SINDy::save: cannot open file");
    ofs.write(kMagic, sizeof(kMagic));
    ofs.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    ofs.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
    ofs.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

SINDy SINDy::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) throw std::runtime_error("SINDy::load: cannot open file");

    char magic[8]{};
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0, checksum = 0;
    ifs.read(magic, sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
    ifs.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("SINDy::load: invalid magic");
    if (version != kVersion) throw std::runtime_error("SINDy::load: unsupported version");

    std::vector<char> payload(static_cast<std::size_t>(payload_size));
    ifs.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (fnv1a64(payload) != checksum) throw std::runtime_error("SINDy::load: checksum mismatch");

    std::size_t off = 0;
    SINDyParams params;
    params.polynomial_order = read_raw<int>(payload, off);
    params.threshold = read_raw<double>(payload, off);
    params.max_iterations = read_raw<int>(payload, off);
    params.include_bias = read_raw<bool>(payload, off);
    params.include_trig = read_raw<bool>(payload, off);
    params.include_inverse = read_raw<bool>(payload, off);
    params.include_pairwise_sin_cos = read_raw<bool>(payload, off);
    params.ridge = read_raw<double>(payload, off);
    params.unbias_after_support = read_raw<bool>(payload, off);
    params.library = static_cast<FeatureLibraryType>(read_raw<int>(payload, off));
    params.integrator = static_cast<IntegratorType>(read_raw<int>(payload, off));
    params.derivative_mode = static_cast<DerivativeMode>(read_raw<int>(payload, off));
    params.smoothing_window = read_raw<std::size_t>(payload, off);
    params.weak_window = read_raw<std::size_t>(payload, off);
    params.ensemble_models = read_raw<std::size_t>(payload, off);
    params.ensemble_subsample_ratio = read_raw<double>(payload, off);
    params.stability_threshold = read_raw<double>(payload, off);
    params.bag_trajectories = read_raw<bool>(payload, off);
    params.weak_test_functions = read_raw<std::size_t>(payload, off);
    params.bootstrap_samples = read_raw<std::size_t>(payload, off);
    params.quadrature_substeps = read_raw<std::size_t>(payload, off);
    params.support_path_steps = read_raw<std::size_t>(payload, off);
    params.align_multi_trajectory_controls = read_raw<bool>(payload, off);
    const bool fitted = read_raw<bool>(payload, off);

    SINDy model(params);
    model.fitted_ = fitted;
    model.input_dim_ = read_raw<std::size_t>(payload, off);
    model.control_dim_ = read_raw<std::size_t>(payload, off);
    const std::size_t targets = read_raw<std::size_t>(payload, off);
    const std::size_t features = read_raw<std::size_t>(payload, off);
    model.feature_names_ = model.build_feature_names(model.input_dim_, model.control_dim_);
    model.coeffs_.assign(targets, std::vector<double>(features, 0.0));
    for (auto& row : model.coeffs_) for (double& v : row) v = read_raw<double>(payload, off);
    model.stability_report_.feature_names = model.feature_names_;
    model.stability_report_.selection_frequency.assign(targets, std::vector<double>(features, 1.0));
    model.stability_report_.mean_coefficients = model.coeffs_;
    model.stability_report_.coefficient_stddev.assign(targets, std::vector<double>(features, 0.0));
    model.stability_report_.ci_lower = model.coeffs_;
    model.stability_report_.ci_upper = model.coeffs_;
    model.stability_report_.inclusion_importance.assign(targets, std::vector<double>(features, 1.0));
    model.populate_support_path();
    model.populate_model_selection_summary();
    return model;
}

}  // namespace sindy
