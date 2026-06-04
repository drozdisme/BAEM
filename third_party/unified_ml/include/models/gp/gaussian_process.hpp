#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gp {

enum class KernelType {
    RBF,
    Matern32,
    Matern52,
    RationalQuadratic,
    Linear,
    Periodic,
    RBFPlusLinear,
    RBFPlusPeriodic
};

enum class ApproximationType {
    Exact,
    SubsetOfRegressors,
    SparseNyström
};

enum class TaskType {
    Regression,
    BinaryClassification
};

struct GPParams {
    KernelType kernel = KernelType::RBF;
    double length_scale = 1.0;
    double signal_variance = 1.0;
    double noise_variance = 1e-6;
    double jitter = 1e-10;
    double alpha = 1.0;
    double period = 1.0;
    std::size_t max_training_points = 4096;
    std::size_t inducing_points = 256;
    ApproximationType approximation = ApproximationType::Exact;
    TaskType task = TaskType::Regression;
    bool normalize_targets = true;
    double output_correlation = 0.0;
    bool use_inducing_correction = true;
};

struct Prediction {
    double mean = 0.0;
    double variance = 0.0;
    double probability = 0.5;
    int label = 0;
};

struct HyperparameterSearchSpace {
    std::vector<double> length_scales;
    std::vector<double> signal_variances;
    std::vector<double> noise_variances;
};

struct AcquisitionResult {
    std::size_t index = 0;
    double score = 0.0;
    Prediction prediction;
};

class GaussianProcessRegressor {
public:
    explicit GaussianProcessRegressor(const GPParams& params = {});

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);
    void fit_multi_output(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& Y);
    void append_observation(const std::vector<double>& x,
                            double y,
                            bool refit = true);
    void append_multi_output_observation(const std::vector<double>& x,
                                         const std::vector<double>& y,
                                         bool refit = true);
    void set_coregionalization_matrix(const std::vector<std::vector<double>>& B);
    void optimize_hyperparameters(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& y,
                                  const HyperparameterSearchSpace& search_space);

    Prediction predict_one(const std::vector<double>& x) const;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    std::vector<Prediction> predict_with_uncertainty(const std::vector<std::vector<double>>& X) const;
    std::vector<double> predict_multi_output(const std::vector<double>& x) const;
    std::vector<std::vector<double>> predict_multi_output(const std::vector<std::vector<double>>& X) const;
    std::vector<std::vector<double>> predict_multi_output_covariance(const std::vector<double>& x) const;
    std::vector<double> predict_class_probabilities(const std::vector<std::vector<double>>& X) const;
    double log_marginal_likelihood() const;
    AcquisitionResult select_next_point_ucb(const std::vector<std::vector<double>>& candidates,
                                            double beta = 2.0) const;
    AcquisitionResult select_next_point_expected_improvement(const std::vector<std::vector<double>>& candidates,
                                                             double best_observed,
                                                             double xi = 0.01) const;

    bool is_fitted() const noexcept { return fitted_; }
    bool is_multi_output() const noexcept { return output_dim_ > 1; }
    std::size_t training_point_count() const noexcept { return active_X_.size(); }
    std::size_t output_dim() const noexcept { return output_dim_; }
    std::size_t active_point_count() const noexcept { return active_X_.size(); }

    void save(const std::string& filepath) const;
    static GaussianProcessRegressor load(const std::string& filepath);

private:
    double kernel_eval(const std::vector<double>& a,
                       const std::vector<double>& b) const;
    std::vector<double> solve_lower_triangular(const std::vector<double>& L,
                                               const std::vector<double>& b) const;
    std::vector<double> solve_upper_triangular(const std::vector<double>& L,
                                               const std::vector<double>& b) const;
    void fit_internal(const std::vector<std::vector<double>>& X,
                      const std::vector<std::vector<double>>& Y);
    std::vector<std::size_t> select_active_set(std::size_t n) const;
    std::vector<double> predict_mean_variance_internal(const std::vector<double>& x,
                                                       std::size_t output_index,
                                                       double* variance) const;
    std::vector<std::vector<double>> build_output_correlation_matrix(const std::vector<double>& variances) const;

    GPParams params_;
    bool fitted_{false};
    std::size_t feature_dim_{0};
    std::size_t output_dim_{0};
    double last_log_marginal_likelihood_{0.0};
    std::vector<std::vector<double>> X_train_;
    std::vector<std::vector<double>> Y_train_;
    std::vector<std::vector<double>> active_X_;
    std::vector<std::vector<double>> active_Y_;
    std::vector<double> target_mean_;
    std::vector<double> target_scale_;
    std::vector<double> cholesky_L_;
    std::vector<std::vector<double>> alpha_vectors_;
    std::vector<double> inducing_diag_correction_;
    std::vector<std::vector<double>> coregionalization_matrix_;
};

}  // namespace gp
