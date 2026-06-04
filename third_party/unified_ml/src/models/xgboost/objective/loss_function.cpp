#include "models/xgboost/objective/loss_function.hpp"
#include "models/xgboost/objective/softmax_loss_adapter.hpp"
#include "models/xgboost/objective/squared_error.hpp"
#include "models/xgboost/objective/logistic_loss.hpp"
#include "models/xgboost/objective/tweedie_regression.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace xgb {

//   Default implementations                         

std::vector<bst_float> LossFunction::transform_batch(
    const std::vector<bst_float>& margins) const
{
    std::vector<bst_float> out(margins.size());
    for (size_t i = 0; i < margins.size(); ++i)
        out[i] = transform(margins[i]);
    return out;
}

bst_float LossFunction::base_score(
    const std::vector<bst_float>& labels) const
{
    if (labels.empty()) return 0.5f;
    double s = 0.0;
    for (bst_float v : labels) s += v;
    return static_cast<bst_float>(s / labels.size());
}

//   Factory                                 
// Supported objective names:
//   reg:squarederror | reg:linear  — squared error regression
//   binary:logistic                — binary logistic regression
//   reg:tweedie | tweedie          — Tweedie regression (p ∈ (1,2))
//   multi:softmax                  — see MultiSoftmaxObjective (separate class)
//   rank:ndcg                      — see LambdaMARTObjective (separate class)

std::unique_ptr<LossFunction> LossFunction::create(const std::string& name,
                                                    bst_int num_class) {
    if (name == "reg:squarederror" || name == "reg:linear" || name == "squarederror")
        return std::make_unique<SquaredError>();
    if (name == "binary:logistic" || name == "logistic")
        return std::make_unique<LogisticLoss>();
    if (name == "reg:tweedie" || name == "tweedie")
        return std::make_unique<TweedieRegressionObjective>(1.5f);
    if (name == "multi:softmax" || name == "multi:softprob" || name == "softmax")
        return std::make_unique<SoftmaxLossAdapter>(std::max(num_class, bst_int(2)));
    throw std::runtime_error(
        "Unknown objective: '" + name + "'. "
        "Supported: reg:squarederror, binary:logistic, reg:tweedie, "
        "multi:softmax, rank:ndcg.");
}

} // namespace xgb
