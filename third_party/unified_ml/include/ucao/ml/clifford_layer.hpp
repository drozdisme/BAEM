#pragma once

#include "autograd/autograd.h"
#include "core/random.hpp"
#include "models/mlp/activation.hpp"
#include "models/mlp/layer.hpp"
#include "models/mlp/linear.hpp"
#include "models/mlp/sequential.hpp"
#include "ucao/engine_policy.hpp"
#include "ucao/kernel/sign_table.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ucao::ml {

/** @brief Registry binding for autograd-facing UCAO Clifford feature layers. */
using CliffordEngineBinding = ucao::engine::PolicyBound<ucao::engine::ModelFamily::CliffordLayer>;

/**
 * @brief Autograd-compatible Clifford layer built entirely from Tensor ops.
 * @pre HiddenDim > 0 and input has D coefficients.
 * @post Preserves differentiability through all trainable parameters.
 */
template <int N, int P, int Q, int HiddenDim = 64>
class CliffordLayer : public mlp::Layer, public CliffordEngineBinding {
    static_assert(P + Q == N, "ucao: algebra dimensions must sum to N");
    static_assert(N <= 5, "ucao: N > 5 requires 64-wide SIMD, not implemented");

    static constexpr int D = 1 << N;

    autograd::Tensor weights_;
    autograd::Tensor biases_;
    autograd::Tensor sign_matrix_;
    int hidden_dim_ = HiddenDim;

public:
    explicit CliffordLayer(int hidden_dim = HiddenDim, unsigned seed = 42)
        : weights_(std::vector<double>(static_cast<std::size_t>(hidden_dim) * D, 0.0),
                   std::vector<std::size_t>{static_cast<std::size_t>(hidden_dim), static_cast<std::size_t>(D)}, true),
          biases_(std::vector<double>(static_cast<std::size_t>(hidden_dim) * D, 0.0),
                  std::vector<std::size_t>{static_cast<std::size_t>(hidden_dim), static_cast<std::size_t>(D)}, true),
          sign_matrix_(std::vector<double>(static_cast<std::size_t>(D) * D, 0.0),
                       std::vector<std::size_t>{static_cast<std::size_t>(D), static_cast<std::size_t>(D)}, false),
          hidden_dim_(hidden_dim) {
        init_sign_matrix();
        init_weights(seed);
    }

    /**
     * @brief Forward pass through a quadratic Clifford feature map plus affine head.
     * @pre input graph is graph-preserving and has D coefficients.
     * @post Returns a D-dimensional tensor computed only via Tensor operations.
     */
    autograd::Tensor forward(const autograd::Tensor& input) override {
        input.require_graph_preserving("CliffordLayer::forward");
        if (!input.graph_preserving()) {
            throw std::logic_error("CliffordLayer::forward requires graph-preserving input");
        }
        const auto in_shape = input.shape();
        if (!(in_shape.size() == 1 && static_cast<int>(in_shape[0]) == D)) {
            throw std::invalid_argument("CliffordLayer::forward expects shape {D}");
        }

        const auto input_data = input.data();
        std::vector<double> gp_vals(static_cast<std::size_t>(D), 0.0);
        for (int c = 0; c < D; ++c) {
            double acc = 0.0;
            const auto& entries = ucao::kernel::kGPTable<N, P, Q>.sparse_[c];
            const int nnz = ucao::kernel::kGPTable<N, P, Q>.nnz_[c];
            for (int k = 0; k < nnz; ++k) {
                const auto e = entries[k];
                acc += static_cast<double>(e.sign) * input_data[e.a] * input_data[e.a ^ c];
            }
            gp_vals[c] = acc;
        }
        autograd::Tensor gp(std::move(gp_vals), std::vector<std::size_t>{static_cast<std::size_t>(D)}, input.requires_grad());
        gp.require_graph_preserving("CliffordLayer::forward");

        std::vector<double> out_vals(static_cast<std::size_t>(D), 0.0);
        const auto w = weights_.data();
        const auto b = biases_.data();
        const auto gp_data = gp.data();
        for (int h = 0; h < hidden_dim_; ++h) {
            const std::size_t row = static_cast<std::size_t>(h) * D;
            double hidden_scalar = 0.0;
            for (int i = 0; i < D; ++i) {
                hidden_scalar += w[row + i] * gp_data[i];
            }
            hidden_scalar = std::tanh(hidden_scalar);
            for (int i = 0; i < D; ++i) {
                out_vals[i] += hidden_scalar * w[row + i] + b[row + i];
            }
        }

        autograd::Tensor out(std::move(out_vals), std::vector<std::size_t>{static_cast<std::size_t>(D)}, true);
        out.require_graph_preserving("CliffordLayer::forward");
        return out;
    }

    [[nodiscard]] std::vector<autograd::Tensor*> parameters() override {
        return {&weights_, &biases_};
    }

    /**
     * @brief Save trainable tensors to a binary file.
     * @pre path is writable.
     * @post The file stores weights followed by biases in row-major order.
     */
    void save(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        const auto w = weights_.data();
        const auto b = biases_.data();
        for (double v : w) {
            os.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
        for (double v : b) {
            os.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }

    /**
     * @brief Load trainable tensors from a binary file.
     * @pre path exists and matches the layer tensor sizes.
     * @post weights_ and biases_ are replaced with file values.
     */
    void load(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        auto w = weights_.data();
        auto b = biases_.data();
        for (double& v : w) {
            is.read(reinterpret_cast<char*>(&v), sizeof(v));
        }
        for (double& v : b) {
            is.read(reinterpret_cast<char*>(&v), sizeof(v));
        }
        weights_ = autograd::Tensor(w, {static_cast<std::size_t>(hidden_dim_), static_cast<std::size_t>(D)}, true);
        biases_ = autograd::Tensor(b, {static_cast<std::size_t>(hidden_dim_), static_cast<std::size_t>(D)}, true);
    }

private:
    void init_sign_matrix() {
        std::vector<double> data(static_cast<std::size_t>(D) * D, 0.0);
        for (int c = 0; c < D; ++c) {
            for (int a = 0; a < D; ++a) {
                data[static_cast<std::size_t>(c) * D + a] = static_cast<double>(ucao::kernel::kGPTable<N, P, Q>.sign(a, a ^ c));
            }
        }
        sign_matrix_ = autograd::Tensor(std::move(data), {static_cast<std::size_t>(D), static_cast<std::size_t>(D)}, false);
    }

    void init_weights(unsigned seed) {
        core::RNG rng(seed);
        std::vector<double> w(static_cast<std::size_t>(hidden_dim_) * D, 0.0);
        std::vector<double> b(static_cast<std::size_t>(hidden_dim_) * D, 0.0);
        const double scale = 1.0 / std::sqrt(static_cast<double>(hidden_dim_));
        for (double& v : w) {
            v = rng.uniform_real(-0.5, 0.5) * scale;
        }
        for (double& v : b) {
            v = rng.uniform_real(-0.5, 0.5) * scale;
        }
        weights_ = autograd::Tensor(std::move(w), {static_cast<std::size_t>(hidden_dim_), static_cast<std::size_t>(D)}, true);
        biases_ = autograd::Tensor(std::move(b), {static_cast<std::size_t>(hidden_dim_), static_cast<std::size_t>(D)}, true);
    }
};

template <int N, int P, int Q, int HiddenDim = 64>
[[nodiscard]] std::unique_ptr<mlp::Sequential> make_clifford_sequential(int out_dim = 1, unsigned seed = 42) {
    auto seq = std::make_unique<mlp::Sequential>();
    seq->add(std::make_unique<CliffordLayer<N, P, Q, HiddenDim>>(HiddenDim, seed));
    seq->add(std::make_unique<mlp::Linear>(static_cast<std::size_t>(1 << N), static_cast<std::size_t>(out_dim)));
    seq->add(std::make_unique<mlp::Tanh>());
    return seq;
}

} // namespace ucao::ml
