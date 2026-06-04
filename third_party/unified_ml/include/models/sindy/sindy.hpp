#pragma once

#include "ucao/engine_policy.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace sindy {

enum class FeatureLibraryType {
    Polynomial,
    PolynomialTrig,
    Generalized
};

enum class IntegratorType {
    Euler,
    RK4
};

enum class DerivativeMode {
    Direct,
    SmoothedFiniteDifference,
    WeakIntegral
};

struct SINDyParams {
    int polynomial_order = 2;
    double threshold = 0.05;
    int max_iterations = 8;
    bool include_bias = true;
    bool include_trig = false;
    bool include_inverse = false;
    bool include_pairwise_sin_cos = false;
    double ridge = 1e-8;
    bool unbias_after_support = true;
    FeatureLibraryType library = FeatureLibraryType::Polynomial;
    IntegratorType integrator = IntegratorType::RK4;
    DerivativeMode derivative_mode = DerivativeMode::Direct;
    std::size_t smoothing_window = 5;
    std::size_t weak_window = 3;
    std::size_t ensemble_models = 1;
    double ensemble_subsample_ratio = 1.0;
    double stability_threshold = 0.5;
    bool bag_trajectories = false;
    std::size_t weak_test_functions = 3;
    std::size_t bootstrap_samples = 0;
    std::size_t quadrature_substeps = 4;
    std::size_t support_path_steps = 6;
    bool align_multi_trajectory_controls = true;
};

struct StabilityReport {
    std::vector<std::string> feature_names;
    std::vector<std::vector<double>> selection_frequency;
    std::vector<std::vector<double>> mean_coefficients;
    std::vector<std::vector<double>> coefficient_stddev;
    std::vector<std::vector<double>> ci_lower;
    std::vector<std::vector<double>> ci_upper;
    std::vector<std::vector<double>> inclusion_importance;
    std::vector<std::vector<double>> support_path_thresholds;
    std::vector<std::vector<std::vector<double>>> support_path_coefficients;
    std::vector<std::vector<double>> bootstrap_state_rmse;
    std::vector<std::vector<double>> bootstrap_derivative_rmse;
    std::vector<std::vector<double>> aligned_control_variation;
    std::vector<std::string> model_selection_summary;
};

class SINDy : public ucao::engine::PolicyBound<ucao::engine::ModelFamily::SINDy> {
public:
    explicit SINDy(const SINDyParams& params = {});

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<std::vector<double>>& Xdot);
    void fit_with_control(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& U,
                          const std::vector<std::vector<double>>& Xdot);
    void fit_from_trajectory(const std::vector<std::vector<double>>& X,
                             double dt);
    void fit_from_trajectory_with_control(const std::vector<std::vector<double>>& X,
                                          const std::vector<std::vector<double>>& U,
                                          double dt);
    void fit_multi_trajectory(const std::vector<std::vector<std::vector<double>>>& trajectories,
                              double dt);
    void fit_multi_trajectory_with_control(const std::vector<std::vector<std::vector<double>>>& trajectories,
                                           const std::vector<std::vector<std::vector<double>>>& controls,
                                           double dt);
    void fit_ensemble(const std::vector<std::vector<double>>& X,
                      const std::vector<std::vector<double>>& Xdot);
    void fit_ensemble_with_control(const std::vector<std::vector<double>>& X,
                                   const std::vector<std::vector<double>>& U,
                                   const std::vector<std::vector<double>>& Xdot);

    std::vector<std::vector<double>> predict_derivative(const std::vector<std::vector<double>>& X) const;
    std::vector<std::vector<double>> predict_derivative_with_control(const std::vector<std::vector<double>>& X,
                                                                     const std::vector<std::vector<double>>& U) const;
    std::vector<std::vector<double>> simulate(const std::vector<double>& x0,
                                              double dt,
                                              std::size_t steps) const;
    std::vector<std::vector<double>> simulate_with_control(const std::vector<double>& x0,
                                                           const std::vector<std::vector<double>>& U,
                                                           double dt) const;
    std::vector<std::string> equations() const;
    StabilityReport stability_report() const { return stability_report_; }

    bool is_fitted() const noexcept { return fitted_; }
    bool has_control() const noexcept { return control_dim_ > 0; }
    std::size_t feature_count() const noexcept { return feature_names_.size(); }

    void save(const std::string& filepath) const;
    static SINDy load(const std::string& filepath);

private:
    std::vector<double> build_features(const std::vector<double>& x,
                                       const std::vector<double>* u = nullptr) const;
    std::vector<std::string> build_feature_names(std::size_t state_dim,
                                                 std::size_t control_dim) const;
    std::vector<double> solve_normal_eq(const std::vector<std::vector<double>>& A,
                                        const std::vector<double>& b,
                                        double ridge_override = -1.0) const;
    std::vector<std::vector<double>> fit_sparse_system(const std::vector<std::vector<double>>& Theta,
                                                       const std::vector<std::vector<double>>& Xdot) const;
    void append_polynomial_features(const std::vector<double>& state,
                                    const std::vector<double>* control,
                                    std::vector<double>& phi,
                                    std::vector<std::string>* names) const;
    void fit_internal(const std::vector<std::vector<double>>& X,
                      const std::vector<std::vector<double>>* U,
                      const std::vector<std::vector<double>>& Xdot);
    void fit_ensemble_internal(const std::vector<std::vector<double>>& X,
                               const std::vector<std::vector<double>>* U,
                               const std::vector<std::vector<double>>& Xdot);
    std::vector<double> derivative_single(const std::vector<double>& x,
                                          const std::vector<double>* u = nullptr) const;
    std::vector<std::vector<double>> estimate_derivatives(const std::vector<std::vector<double>>& X,
                                                          double dt) const;
    std::vector<std::vector<double>> estimate_derivatives_multi_test(const std::vector<std::vector<double>>& X,
                                                                     double dt) const;
    std::vector<std::vector<double>> estimate_derivatives_weak_variational(const std::vector<std::vector<double>>& X,
                                                                           double dt) const;
    std::vector<double> weak_test_function(std::size_t family,
                                           std::size_t window_size) const;
    void populate_support_path();
    void populate_model_selection_summary();
    std::vector<std::vector<double>> evaluate_bootstrap_rmse(const std::vector<std::vector<std::vector<double>>>& ensemble_coeffs,
                                                             const std::vector<std::vector<double>>& X,
                                                             const std::vector<std::vector<double>>* U,
                                                             const std::vector<std::vector<double>>& Xdot) const;
    std::vector<std::vector<double>> compute_control_alignment(const std::vector<std::vector<double>>& U,
                                                               const std::vector<std::vector<double>>& Xdot) const;

    SINDyParams params_;
    bool fitted_{false};
    std::size_t input_dim_{0};
    std::size_t control_dim_{0};
    std::vector<std::string> feature_names_;
    std::vector<std::vector<double>> coeffs_;
    StabilityReport stability_report_;
    std::vector<std::vector<std::vector<double>>> ensemble_coefficients_cache_;
};

}  // namespace sindy
