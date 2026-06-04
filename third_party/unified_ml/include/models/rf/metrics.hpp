#pragma once

#include <string>
#include <vector>

namespace rf {

/**
 * @brief Aggregated classification metrics suitable for external reporting.
 */
struct ClassificationReport {
    double accuracy = 0.0;
    double macro_f1 = 0.0;
    std::vector<double> precision;
    std::vector<double> recall;
};

/**
 * @brief Aggregated regression metrics suitable for external reporting.
 */
struct RegressionReport {
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
};


/**
 * @brief Compute the fraction of correctly predicted labels.
 * @param y_true Ground-truth labels.
 * @param y_pred Predicted labels.
 * @return Classification accuracy in the inclusive range [0, 1].
 */
double accuracy(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/**
 * @brief Build the confusion matrix for a fixed number of classes.
 * @param y_true Ground-truth labels.
 * @param y_pred Predicted labels.
 * @param n_classes Number of classes represented in the matrix.
 * @return Dense confusion matrix indexed as [true_class][pred_class].
 */
std::vector<std::vector<int>> confusion_matrix(const std::vector<double>& y_true,
                                                const std::vector<double>& y_pred,
                                                int n_classes);

/**
 * @brief Compute per-class precision values.
 * @param y_true Ground-truth labels.
 * @param y_pred Predicted labels.
 * @param n_classes Number of classes.
 * @return Precision for each class.
 */
std::vector<double> precision_per_class(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred,
                                         int n_classes);

/**
 * @brief Compute per-class recall values.
 * @param y_true Ground-truth labels.
 * @param y_pred Predicted labels.
 * @param n_classes Number of classes.
 * @return Recall for each class.
 */
std::vector<double> recall_per_class(const std::vector<double>& y_true,
                                      const std::vector<double>& y_pred,
                                      int n_classes);

/**
 * @brief Compute the macro-averaged F1 score.
 * @param y_true Ground-truth labels.
 * @param y_pred Predicted labels.
 * @param n_classes Number of classes.
 * @return Macro-averaged F1 score.
 */
double f1_macro(const std::vector<double>& y_true,
                const std::vector<double>& y_pred,
                int n_classes);

/**
 * @brief Compute mean squared error.
 * @param y_true Ground-truth regression targets.
 * @param y_pred Predicted targets.
 * @return Mean squared error.
 */
double mse(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/**
 * @brief Compute root mean squared error.
 * @param y_true Ground-truth regression targets.
 * @param y_pred Predicted targets.
 * @return Root mean squared error.
 */
double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/**
 * @brief Compute mean absolute error.
 * @param y_true Ground-truth regression targets.
 * @param y_pred Predicted targets.
 * @return Mean absolute error.
 */
double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/**
 * @brief Compute the coefficient of determination.
 * @param y_true Ground-truth regression targets.
 * @param y_pred Predicted targets.
 * @return R-squared score.
 */
double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/**
 * @brief Aggregate common classification metrics into a reusable report object.
 * @param y_true Ground-truth labels.
 * @param y_pred Predicted labels.
 * @param n_classes Number of classes.
 * @return Structured classification report suitable for application-owned formatting.
 */
ClassificationReport classification_report(const std::vector<double>& y_true,
                                           const std::vector<double>& y_pred,
                                           int n_classes);

/**
 * @brief Aggregate common regression metrics into a reusable report object.
 * @param y_true Ground-truth regression targets.
 * @param y_pred Predicted targets.
 * @return Structured regression report suitable for application-owned formatting.
 */
RegressionReport regression_report(const std::vector<double>& y_true,
                                   const std::vector<double>& y_pred);

} // namespace rf
