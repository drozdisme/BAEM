#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svm {

/**
 * @brief Kernel family used by the support vector machine.
 */

enum class KernelType {
    Linear,
    Polynomial,
    RBF,
    Sigmoid
};

enum class MultiClassStrategy {
    OneVsRest,
    OneVsOne
};

enum class SVMMode {
    Classification,
    Regression
};

enum class SVMOptimization {
    PrimalSGD,
    DualCoordinate,
    SMOStyle
};

enum class SVMVariant {
    C,
    Nu
};

struct SVMParams {
    double C = 1.0;
    double learning_rate = 1e-2;
    std::size_t epochs = 200;
    KernelType kernel = KernelType::Linear;
    double gamma = 1.0;
    double coef0 = 0.0;
    int degree = 3;
    bool fit_intercept = true;
    bool deterministic = true;
    MultiClassStrategy multiclass_strategy = MultiClassStrategy::OneVsRest;
    std::vector<double> class_weights;
    std::vector<double> sample_weights;
    SVMMode mode = SVMMode::Classification;
    double epsilon = 0.1;
    SVMOptimization optimization = SVMOptimization::PrimalSGD;
    SVMVariant variant = SVMVariant::C;
    double nu = 0.5;
    double quantile_tau = 0.5;
};

struct BinaryDecision {
    int label = -1;
    double decision_value = 0.0;
    double probability = 0.5;
    double margin = 0.0;
};

struct MultiClassPrediction {
    int label = -1;
    std::vector<int> classes;
    std::vector<double> scores;
    std::vector<double> probabilities;
};

struct TrainingDiagnostics {
    double average_hinge_loss = 0.0;
    double average_margin = 0.0;
    double training_accuracy = 0.0;
};

struct RegressionDiagnostics {
    double rmse = 0.0;
    double mae = 0.0;
    double mean_residual = 0.0;
    double support_fraction = 0.0;
    double residual_stddev = 0.0;
    double calibrated_interval_scale = 1.0;
    double nominal_coverage_95 = 0.0;
    double mean_predictive_stddev = 0.0;
};

class SVM {
public:
    /**
     * @brief Construct an SVM instance with explicit solver parameters.
     * @param params Solver, kernel, and calibration settings.
     */
    explicit SVM(const SVMParams& params = {});

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             const std::vector<double>& sample_weights);
    void fit_regression(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y);
    void fit_regression(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y,
                        const std::vector<double>& sample_weights);

    /**
     * @brief Evaluate the signed decision function for one sample.
     * @param x Feature vector.
     * @return Raw decision value before thresholding.
     */
    double decision_function(const std::vector<double>& x) const;
    /**
     * @brief Evaluate the signed decision function for one sample span.
     * @param x Feature view.
     * @return Raw decision value before thresholding.
     */
    double decision_function(std::span<const double> x) const;
    /**
     * @brief Predict the class label for one sample.
     * @param x Feature vector.
     * @return Predicted class label.
     */
    int predict(const std::vector<double>& x) const;
    /**
     * @brief Predict the class label for one sample span.
     * @param x Feature view.
     * @return Predicted class label.
     */
    int predict(std::span<const double> x) const;
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    /**
     * @brief Predict a scalar regression response for one sample.
     * @param x Feature vector.
     * @return Predicted regression value.
     */
    double predict_regression(const std::vector<double>& x) const;
    /**
     * @brief Predict a scalar regression response for one sample span.
     * @param x Feature view.
     * @return Predicted regression value.
     */
    double predict_regression(std::span<const double> x) const;
    std::vector<double> predict_regression(const std::vector<std::vector<double>>& X) const;

    /**
     * @brief Predict binary decision details for one sample.
     * @param x Feature vector.
     * @return Decision metadata including margin and calibrated probability.
     */
    BinaryDecision predict_binary(const std::vector<double>& x) const;
    /**
     * @brief Predict binary decision details for one sample span.
     * @param x Feature view.
     * @return Decision metadata including margin and calibrated probability.
     */
    BinaryDecision predict_binary(std::span<const double> x) const;
    std::vector<double> predict_scores(const std::vector<double>& x) const;
    std::vector<double> predict_proba(const std::vector<double>& x) const;
    std::vector<std::vector<double>> predict_proba(const std::vector<std::vector<double>>& X) const;
    MultiClassPrediction predict_multiclass(const std::vector<double>& x) const;
    std::vector<MultiClassPrediction> predict_multiclass(const std::vector<std::vector<double>>& X) const;

    bool is_fitted() const noexcept { return fitted_; }
    bool is_multiclass() const noexcept { return classes_.size() > 2; }
    bool is_regressor() const noexcept { return params_.mode == SVMMode::Regression; }
    std::size_t n_classes() const noexcept { return classes_.size(); }
    const std::vector<int>& classes() const noexcept { return classes_; }
    std::size_t support_vector_count() const noexcept;
    TrainingDiagnostics diagnostics(const std::vector<std::vector<double>>& X,
                                    const std::vector<int>& y) const;
    RegressionDiagnostics regression_diagnostics(const std::vector<std::vector<double>>& X,
                                                 const std::vector<double>& y) const;

    void save(const std::string& filepath) const;
    static SVM load(const std::string& filepath);

private:
    struct BinaryModel {
        std::vector<double> w;
        double b = 0.0;
        std::vector<std::vector<double>> support_X;
        std::vector<int> support_y;
        std::vector<double> alpha;
        std::vector<double> error_cache;
        int positive_label = 1;
        int negative_label = -1;
        double positive_weight = 1.0;
        double negative_weight = 1.0;
        double prob_a = 0.0;
        double prob_b = 0.0;
        double rho = 0.0;
    };

    double kernel_eval(const std::vector<double>& a,
                       const std::vector<double>& b) const;
    void fit_binary_model(BinaryModel& model,
                          const std::vector<std::vector<double>>& X,
                          const std::vector<int>& y_binary,
                          const std::vector<double>* sample_weights) const;
    void fit_binary_model_dual(BinaryModel& model,
                               const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y_binary,
                               const std::vector<double>* sample_weights,
                               bool smo_style) const;
    bool smo_pair_update(BinaryModel& model,
                         const std::vector<std::vector<double>>& X,
                         std::size_t i,
                         std::size_t j,
                         const std::vector<double>& upper_bounds) const;
    std::size_t select_second_index(const BinaryModel& model,
                                    const std::vector<int>& y_binary,
                                    std::size_t i,
                                    const std::vector<double>& upper_bounds) const;
    bool violates_kkt(const BinaryModel& model,
                      const std::vector<int>& y_binary,
                      std::size_t i,
                      const std::vector<double>& upper_bounds,
                      double tolerance = 1e-3) const;
    double binary_decision_function(const BinaryModel& model,
                                    const std::vector<double>& x) const;
    double binary_probability(const BinaryModel& model,
                              double score) const;
    void calibrate_probabilities(BinaryModel& model,
                                 const std::vector<std::vector<double>>& X,
                                 const std::vector<int>& y_binary) const;
    std::vector<double> multiclass_probability_coupling(const std::vector<double>& pairwise_scores) const;
    static std::vector<double> softmax_stable(const std::vector<double>& scores);
    double class_weight_for_index(std::size_t class_index) const;
    double kernel_svr_predict(const std::vector<double>& x) const;
    void fit_regression_dual(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& y,
                             const std::vector<double>& sample_weights);
    void fit_regression_dual_epsilon(const std::vector<std::vector<double>>& X,
                                     const std::vector<double>& y,
                                     const std::vector<double>& sample_weights);
    void fit_regression_dual_nu(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y,
                                const std::vector<double>& sample_weights);
    double regression_predictive_stddev(const std::vector<double>& x) const;
    void calibrate_regression_confidence(const std::vector<std::vector<double>>& X,
                                         const std::vector<double>& y) const;
    static std::vector<double> sanitize_sample_weights(const std::vector<double>* sample_weights,
                                                       std::size_t n);

    SVMParams params_;
    bool fitted_{false};
    std::size_t feature_dim_{0};
    std::vector<int> classes_;
    std::vector<BinaryModel> binary_models_;
    std::vector<double> regression_w_;
    double regression_b_{0.0};
    std::vector<std::vector<double>> regression_support_X_;
    std::vector<double> regression_alpha_;
    std::vector<double> regression_alpha_star_;
    std::vector<double> regression_targets_;
    std::vector<double> regression_sample_weights_;
    mutable double regression_confidence_scale_{1.0};
    mutable double regression_residual_variance_{0.0};
};

}  // namespace svm
