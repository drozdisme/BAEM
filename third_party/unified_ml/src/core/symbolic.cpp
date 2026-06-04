#include "core/symbolic.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace core::symbolic {
namespace {

[[nodiscard]] bool is_binary_matrix(core::MatrixView view, double tolerance) noexcept {
    for (std::size_t r = 0; r < view.rows; ++r) {
        for (std::size_t c = 0; c < view.cols; ++c) {
            const double value = view(r, c);
            if (std::abs(value) > tolerance && std::abs(value - 1.0) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] std::vector<std::vector<double>> to_rows(core::MatrixView view) {
    std::vector<std::vector<double>> rows(view.rows, std::vector<double>(view.cols, 0.0));
    for (std::size_t r = 0; r < view.rows; ++r) {
        for (std::size_t c = 0; c < view.cols; ++c) {
            rows[r][c] = view(r, c);
        }
    }
    return rows;
}

[[nodiscard]] std::vector<::sle::TrainingExample> to_training_examples(core::MatrixView X, core::MatrixView Y) {
    if (X.rows != Y.rows) {
        throw std::invalid_argument("core::symbolic::fit: X/Y row mismatch for discrete dispatch");
    }
    std::vector<::sle::TrainingExample> dataset;
    dataset.reserve(X.rows);
    for (std::size_t r = 0; r < X.rows; ++r) {
        std::vector<::sle::BitVector> inputs;
        inputs.reserve(X.cols);
        for (std::size_t c = 0; c < X.cols; ++c) {
            ::sle::BitVector bit(1);
            bit.set(0, X(r, c) > 0.5);
            inputs.push_back(std::move(bit));
        }
        ::sle::BitVector target(Y.cols == 0 ? 1 : Y.cols);
        if (Y.cols == 0) {
            target.set(0, false);
        } else {
            for (std::size_t c = 0; c < Y.cols; ++c) {
                target.set(c, Y(r, c) > 0.5);
            }
        }
        dataset.push_back(::sle::TrainingExample{std::move(inputs), std::move(target)});
    }
    return dataset;
}

[[nodiscard]] std::string export_discrete_latex(const DiscreteFitResult& result) {
    std::ostringstream oss;
    oss << "\\text{BooleanCascade}(inputs=" << result.circuit.input_count()
        << ", gates=" << result.circuit.gate_count() << ")";
    return oss.str();
}

[[nodiscard]] std::string export_continuous_latex(const ContinuousFitResult& result) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < result.equations.size(); ++i) {
        if (i) oss << "\\n";
        oss << result.equations[i];
    }
    return oss.str();
}

[[nodiscard]] std::string export_discrete_cpp(const DiscreteFitResult& result, std::string_view function_name) {
    std::ostringstream oss;
    oss << "bool " << function_name << "(const std::vector<bool>& inputs) {\n";
    oss << "  // BooleanCascade: inputs=" << result.circuit.input_count() << ", gates=" << result.circuit.gate_count() << "\n";
    oss << "  return false;\n}";
    return oss.str();
}

[[nodiscard]] std::string export_continuous_cpp(const ContinuousFitResult& result, std::string_view function_name) {
    std::ostringstream oss;
    oss << "std::vector<double> " << function_name << "(const std::vector<double>& x) {\n";
    oss << "  // SINDy equations\n";
    for (const auto& eq : result.equations) {
        oss << "  // " << eq << "\n";
    }
    oss << "  return {};\n}";
    return oss.str();
}

} // namespace

DispatchInfo analyze_domain(const AutoFitRequest& request) noexcept {
    const bool x_binary = is_binary_matrix(request.X, request.binary_tolerance);
    const bool y_binary = is_binary_matrix(request.Y, request.binary_tolerance);
    return DispatchInfo{
        (x_binary && y_binary) ? DispatchInfo::Domain::Discrete : DispatchInfo::Domain::Continuous,
        x_binary,
        y_binary,
    };
}

ContinuousFitResult fit_continuous(const ContinuousFitRequest& request) {
    auto X = to_rows(request.X);
    auto dXdt = to_rows(request.dXdt);
    ContinuousModel model(request.params);
    model.fit(X, dXdt);
    return ContinuousFitResult{std::move(model), model.equations()};
}

DiscreteFitResult fit_discrete(const DiscreteFitRequest& request) {
    auto trained = ::sle::train_full_engine(request.dataset, request.config, {});
    return DiscreteFitResult{trained.model, trained.model.base};
}

FitResult fit(const AutoFitRequest& request) {
    const auto dispatch = analyze_domain(request);
    if (dispatch.domain == DispatchInfo::Domain::Discrete) {
        return fit_discrete(DiscreteFitRequest{to_training_examples(request.X, request.Y), request.discrete_config});
    }
    return fit_continuous(ContinuousFitRequest{request.X, request.Y, request.continuous_params});
}

std::string export_model(const FitResult& result, const ExportOptions& options) {
    switch (options.format) {
        case ExportOptions::Format::LaTeX:
            return export_latex(result);
        case ExportOptions::Format::Cpp:
            return export_cpp(result, options.function_name);
    }
    return {};
}

std::string export_latex(const FitResult& result) {
    return std::visit([](const auto& model) -> std::string {
        using T = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<T, ContinuousFitResult>) {
            return export_continuous_latex(model);
        } else {
            return export_discrete_latex(model);
        }
    }, result);
}

std::string export_cpp(const FitResult& result, std::string_view function_name) {
    return std::visit([function_name](const auto& model) -> std::string {
        using T = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<T, ContinuousFitResult>) {
            return export_continuous_cpp(model, function_name);
        } else {
            return export_discrete_cpp(model, function_name);
        }
    }, result);
}

} // namespace core::symbolic
