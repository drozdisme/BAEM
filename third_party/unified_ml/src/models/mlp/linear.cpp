#include "models/mlp/linear.hpp"
#include "core/hpc_kernels.hpp"
#include "autograd/tensor.h"
#include "autograd/node.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <memory>

namespace mlp {

Linear::Linear(std::size_t in_f, std::size_t out_f, bool use_bias)
    : in_features_(in_f), out_features_(out_f), use_bias_(use_bias)
    , weight_({0.0},{1},false), bias_({0.0},{1},false)
{
    if (in_f  == 0) throw std::invalid_argument("Linear: in_features must be > 0");
    if (out_f == 0) throw std::invalid_argument("Linear: out_features must be > 0");

    // Xavier / Glorot uniform  U(-limit, +limit)
    const double limit = std::sqrt(6.0 / (static_cast<double>(in_f) + static_cast<double>(out_f)));
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-limit, +limit);

    std::vector<double> w_data(out_f * in_f);
    for (auto& v : w_data) v = dist(rng);
    weight_ = autograd::Tensor(std::move(w_data), {out_f, in_f}, true);

    std::vector<double> b_data(out_f, 0.0);
    bias_ = autograd::Tensor(std::move(b_data), {out_f}, true);
}

//    forward                                 
//
// OLD path: transpose(W) → allocates {in,out} tensor + Node (full weight copy)
//           matmul(x, Wt)  → allocates output tensor + Node
//           y + bias        → allocates output tensor + Node
//           ↳  3 heap allocs, 1 full weight copy per layer per step
//
// NEW path: fused_gemv_bias_act(W, x, b, y)
//           computes y[i] = dot(W[i,:], x) + b[i] in one kernel, zero copies.
//           One Node with analytically-fused backward (outer + gemvT).
//           ↳  1 node alloc, 0 weight copies
//
// For batch input [B, in_f]: fused_batch_forward dispatches parallel GEMV.
//                                       
autograd::Tensor Linear::forward(const autograd::Tensor& input)
{
    if (input.ndim() < 1)
        throw std::invalid_argument("Linear::forward: input must have at least 1 dimension");
    if (input.shape().back() != in_features_)
        throw std::invalid_argument("Linear::forward: input last dim != in_features");

    const std::size_t in_f  = in_features_;
    const std::size_t out_f = out_features_;
    const bool is_batch     = (input.ndim() >= 2);
    const std::size_t batch = is_batch ? input.shape()[0] : 1;

    //   Forward: y = W @ x + b  (W is [out_f × in_f], no transpose needed)  
    std::vector<double> out_data(batch * out_f);
    const double* W = weight_.data().data();
    const double* x = input.data().data();

    if (use_bias_) {
        const double* b = bias_.data().data();
        if (batch == 1)
            hpc::fused_gemv_bias_act(W, x, b, out_data.data(),
                                     out_f, in_f, hpc::FusedAct::IDENTITY);
        else
            hpc::fused_batch_forward(W, b, x, out_data.data(),
                                     batch, out_f, in_f, hpc::FusedAct::IDENTITY);
    } else {
        // No bias: plain GEMV for each sample
        for (std::size_t s = 0; s < batch; ++s) {
            double* ys = out_data.data() + s * out_f;
            const double* xs = x + s * in_f;
            for (std::size_t i = 0; i < out_f; ++i)
                ys[i] = hpc::dot_product(W + i * in_f, xs, in_f);
        }
    }

    // Output shape: {batch, out_f} for batch inputs, {out_f} for vectors
    std::vector<std::size_t> out_shape =
        is_batch ? std::vector<std::size_t>{batch, out_f}
                 : std::vector<std::size_t>{out_f};

    const bool req = weight_.requires_grad()
                  || (use_bias_ && bias_.requires_grad())
                  || input.requires_grad();

    autograd::Tensor out(std::move(out_data), out_shape, false);
    if (!req) return out;

    //   Build fused gradient node                        
    auto out_node = std::make_shared<autograd::Node>(batch * out_f);
    out_node->is_leaf = false;

    auto w_node = weight_.node();
    auto b_node = use_bias_ ? bias_.node() : nullptr;
    auto x_node = input.node();

    if (w_node) out_node->inputs.push_back(w_node);
    if (b_node) out_node->inputs.push_back(b_node);
    if (x_node && x_node != w_node) out_node->inputs.push_back(x_node);

    // Capture tensors by value (cheap: shared storage) to avoid deep copies
    // of input/weight buffers on every forward pass.
    const autograd::Tensor input_capture  = input;
    const autograd::Tensor weight_capture = weight_;

    out_node->backward_fn = [out_node,
                               w_node, b_node, x_node,
                               input_capture,
                               weight_capture,
                               batch, in_f, out_f]()
    {
        const auto& go = out_node->grad;    // shape [batch * out_f]
        const double* x_ptr = input_capture.data().data();

        // dW[i, j] += sum_b  go[b*out_f + i] * x[b*in_f + j]
        if (w_node) {
            double* dW = w_node->grad.data();
            for (std::size_t s = 0; s < batch; ++s)
                hpc::outer_hpc(go.data()     + s * out_f,
                               x_ptr + s * in_f,
                               dW, out_f, in_f);
        }

        // db[i] += sum_b  go[b*out_f + i]
        if (b_node) {
            double* db = b_node->grad.data();
            for (std::size_t s = 0; s < batch; ++s) {
                const double* go_s = go.data() + s * out_f;
#pragma omp simd
                for (std::size_t i = 0; i < out_f; ++i) db[i] += go_s[i];
            }
        }

        // dx[b, j] += sum_i  W[i, j] * go[b*out_f + i]  = W^T @ go
        if (x_node) {
            double* dx = x_node->grad.data();
            const double* W_ptr = weight_capture.data().data();
            for (std::size_t s = 0; s < batch; ++s)
                hpc::gemvT_hpc(W_ptr,
                               go.data() + s * out_f,
                               dx + s * in_f,
                               out_f, in_f);
        }
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

std::vector<autograd::Tensor*> Linear::parameters()
{
    if (use_bias_) return {&weight_, &bias_};
    return {&weight_};
}

} // namespace mlp
