#pragma once

#include "core/matrix_view.hpp"
#include "models/sindy/sindy.hpp"
#include "models/sle/circuit.hpp"
#include "models/sle/full_engine.hpp"

#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace core::symbolic {

using DiscreteEngine = core::models::sle::FullEngine;
using DiscreteEngineConfig = core::models::sle::EngineConfig;
using ContinuousModel = ::sindy::SINDy;
using DiscreteCircuit = core::models::sle::BooleanCascade;

struct ContinuousFitRequest {
    core::MatrixView X;
    core::MatrixView dXdt;
    ::sindy::SINDyParams params{};
};

struct DiscreteFitRequest {
    std::vector<::sle::TrainingExample> dataset;
    DiscreteEngineConfig config{};
};

struct AutoFitRequest {
    core::MatrixView X;
    core::MatrixView Y;
    ::sindy::SINDyParams continuous_params{};
    DiscreteEngineConfig discrete_config{};
    double binary_tolerance = 1e-9;
};

struct DispatchInfo {
    enum class Domain {
        Continuous,
        Discrete,
    };

    Domain domain = Domain::Continuous;
    bool input_is_binary = false;
    bool target_is_binary = false;
};

struct ContinuousFitResult {
    ContinuousModel model;
    std::vector<std::string> equations;
};

struct DiscreteFitResult {
    DiscreteEngine model;
    DiscreteCircuit circuit;
};

using FitResult = std::variant<ContinuousFitResult, DiscreteFitResult>;

[[nodiscard]] DispatchInfo analyze_domain(const AutoFitRequest& request) noexcept;
[[nodiscard]] ContinuousFitResult fit_continuous(const ContinuousFitRequest& request);
[[nodiscard]] DiscreteFitResult fit_discrete(const DiscreteFitRequest& request);
[[nodiscard]] FitResult fit(const AutoFitRequest& request);

struct ExportOptions {
    enum class Format {
        LaTeX,
        Cpp,
    };

    Format format = Format::LaTeX;
    std::string function_name = "symbolic_model";
};

[[nodiscard]] std::string export_model(const FitResult& result, const ExportOptions& options = {});
[[nodiscard]] std::string export_latex(const FitResult& result);
[[nodiscard]] std::string export_cpp(const FitResult& result, std::string_view function_name = "symbolic_model");

} // namespace core::symbolic
