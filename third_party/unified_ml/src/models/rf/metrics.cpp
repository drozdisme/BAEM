#include "models/rf/metrics.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace rf {

static void check_sizes(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("metrics: y_true and y_pred size mismatch");
    if (a.empty())
        throw std::invalid_argument("metrics: empty vectors");
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------
double accuracy(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    check_sizes(y_true, y_pred);
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (static_cast<int>(y_true[i]) == static_cast<int>(y_pred[i])) ++correct;
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

std::vector<std::vector<int>> confusion_matrix(const std::vector<double>& y_true,
                                                const std::vector<double>& y_pred,
                                                int n_classes) {
    check_sizes(y_true, y_pred);
    std::vector<std::vector<int>> cm(static_cast<size_t>(n_classes),
                                      std::vector<int>(static_cast<size_t>(n_classes), 0));
    for (size_t i = 0; i < y_true.size(); ++i) {
        int t = static_cast<int>(y_true[i]);
        int p = static_cast<int>(y_pred[i]);
        if (t >= 0 && t < n_classes && p >= 0 && p < n_classes)
            ++cm[static_cast<size_t>(t)][static_cast<size_t>(p)];
    }
    return cm;
}

std::vector<double> precision_per_class(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred,
                                         int n_classes) {
    auto cm = confusion_matrix(y_true, y_pred, n_classes);
    std::vector<double> prec(static_cast<size_t>(n_classes), 0.0);
    for (int c = 0; c < n_classes; ++c) {
        int col_sum = 0;
        for (int r = 0; r < n_classes; ++r) col_sum += cm[static_cast<size_t>(r)][static_cast<size_t>(c)];
        if (col_sum > 0)
            prec[static_cast<size_t>(c)] = static_cast<double>(cm[static_cast<size_t>(c)][static_cast<size_t>(c)]) / static_cast<double>(col_sum);
    }
    return prec;
}

std::vector<double> recall_per_class(const std::vector<double>& y_true,
                                      const std::vector<double>& y_pred,
                                      int n_classes) {
    auto cm = confusion_matrix(y_true, y_pred, n_classes);
    std::vector<double> rec(static_cast<size_t>(n_classes), 0.0);
    for (int c = 0; c < n_classes; ++c) {
        int row_sum = 0;
        for (int col = 0; col < n_classes; ++col) row_sum += cm[static_cast<size_t>(c)][static_cast<size_t>(col)];
        if (row_sum > 0)
            rec[static_cast<size_t>(c)] = static_cast<double>(cm[static_cast<size_t>(c)][static_cast<size_t>(c)]) / static_cast<double>(row_sum);
    }
    return rec;
}

double f1_macro(const std::vector<double>& y_true,
                const std::vector<double>& y_pred,
                int n_classes) {
    auto prec = precision_per_class(y_true, y_pred, n_classes);
    auto rec  = recall_per_class   (y_true, y_pred, n_classes);
    double sum = 0.0;
    int count = 0;
    for (int c = 0; c < n_classes; ++c) {
        double denom = prec[static_cast<size_t>(c)] + rec[static_cast<size_t>(c)];
        if (denom > 0.0) {
            sum += 2.0 * prec[static_cast<size_t>(c)] * rec[static_cast<size_t>(c)] / denom;
            ++count;
        }
    }
    return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

// ---------------------------------------------------------------------------
// Regression
// ---------------------------------------------------------------------------
double mse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    check_sizes(y_true, y_pred);
    double s = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double d = y_true[i] - y_pred[i];
        s += d * d;
    }
    return s / static_cast<double>(y_true.size());
}

double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return std::sqrt(mse(y_true, y_pred));
}

double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    check_sizes(y_true, y_pred);
    double s = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        s += std::abs(y_true[i] - y_pred[i]);
    return s / static_cast<double>(y_true.size());
}

double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    check_sizes(y_true, y_pred);
    double mean_y = 0.0;
    for (double v : y_true) mean_y += v;
    mean_y /= static_cast<double>(y_true.size());

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double dt = y_true[i] - mean_y;
        double dr = y_true[i] - y_pred[i];
        ss_tot += dt * dt;
        ss_res += dr * dr;
    }
    return (ss_tot < 1e-15) ? 0.0 : 1.0 - ss_res / ss_tot;
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------
ClassificationReport classification_report(const std::vector<double>& y_true,
                                           const std::vector<double>& y_pred,
                                           int n_classes) {
    ClassificationReport report;
    report.precision = precision_per_class(y_true, y_pred, n_classes);
    report.recall = recall_per_class(y_true, y_pred, n_classes);
    report.accuracy = accuracy(y_true, y_pred);
    report.macro_f1 = f1_macro(y_true, y_pred, n_classes);
    return report;
}

RegressionReport regression_report(const std::vector<double>& y_true,
                                   const std::vector<double>& y_pred) {
    RegressionReport report;
    report.mse = mse(y_true, y_pred);
    report.rmse = rmse(y_true, y_pred);
    report.mae = mae(y_true, y_pred);
    report.r2 = r2_score(y_true, y_pred);
    return report;
}

} // namespace rf
