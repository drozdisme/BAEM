#include "models/deep_onet/model.hpp"
// model.cpp — DeepONet forward pass and parameter management.
//
// Architecture recap
//          
//
//   Input:   u_batch [batch, m]    — function samples at sensor points
//            y_batch [batch, d]    — query coordinates
//
//   Branch:  b = BranchNet(u_batch)     → [batch, p]
//   Trunk:   t = TrunkNet (y_batch)     → [batch, p]
//
//   Merge:   element-wise product b ⊙ t → [batch, p]
//            sum over p-axis            → [batch, 1]
//
//   Output:  pred = sum(b ⊙ t, dim=-1) + output_bias
//
// Gradient flow
//        
//
//   loss.backward() differentiates through:
//     sum   → engine's sum() op, grad = ones broadcast
//     mul   → engine's mul() op, grad_b = upstream * t, grad_t = upstream * b
//     both branch and trunk layers via matmul / add / activation backward_fns
//
// The output_bias is a scalar leaf that accumulates dL/d(bias) = sum(upstream).
// It is initialised to zero (a common convention in DeepONet).
//
// Batched element-wise merge
//              
// b and t are both [batch, p].  engine's mul() requires identical shapes,
// which is satisfied here.  The row-wise dot-product is then computed as
// sum(b * t) over the p dimension.  We implement this by:
//   1. element-wise mul -> [batch, p]
//   2. sum over the whole tensor (gives a scalar for batch=1, or we use
//      the per-row approach below for batch > 1)
//
// For batch > 1 we reshape and use matmul:
//   dot(b_i, t_i) for each row i can be computed as
//     diag(b @ t^T) but that is O(batch^2).
//   Better: element-wise b * t → [batch,p], then sum along p with a
//   matmul against a ones vector of shape [p, 1]:
//     result = (b * t) @ ones_p   →  [batch, 1]
//   This keeps everything in the autograd graph.

#include "models/deep_onet/model.hpp"

#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace deep_onet {

namespace {

autograd::Tensor row_sum_2d(const autograd::Tensor& x) {
    if (x.ndim() != 2)
        throw std::invalid_argument("row_sum_2d: expected a 2-D tensor");

    const std::size_t batch = x.shape()[0];
    const std::size_t cols  = x.shape()[1];
    std::vector<double> out_data(batch, 0.0);
    for (std::size_t i = 0; i < batch; ++i) {
        const double* row = x.data().data() + i * cols;
        double acc = 0.0;
        for (std::size_t j = 0; j < cols; ++j) acc += row[j];
        out_data[i] = acc;
    }

    autograd::Tensor out(std::move(out_data), {batch, 1}, false);
    if (!x.requires_grad()) return out;

    auto out_node = std::make_shared<autograd::Node>(batch);
    auto x_node = x.node();
    out_node->is_leaf = false;
    out_node->inputs.push_back(x_node);

    out_node->backward_fn = [out_node, x_node, batch, cols]() {
        const auto& go = out_node->grad;  // [batch,1] flattened to [batch]
        auto& gx = x_node->grad;          // [batch,cols]
        for (std::size_t i = 0; i < batch; ++i) {
            const double gi = go[i];
            double* row = gx.data() + i * cols;
            for (std::size_t j = 0; j < cols; ++j) row[j] += gi;
        }
    };

    out_node->vjp_fn = [x_node, batch, cols, x_shape = x.shape()](const std::any& g) -> std::any {
        const autograd::Tensor& gup = std::any_cast<const autograd::Tensor&>(g);
        std::vector<double> gx(batch * cols, 0.0);
        for (std::size_t i = 0; i < batch; ++i) {
            const double gi = gup.data()[i];
            for (std::size_t j = 0; j < cols; ++j) gx[i * cols + j] = gi;
        }
        std::unordered_map<autograd::Node*, autograd::Tensor> res;
        res.emplace(x_node.get(), autograd::Tensor(std::move(gx), x_shape, false));
        return std::make_any<std::unordered_map<autograd::Node*, autograd::Tensor>>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

}  // namespace

//                                        

DeepONet::DeepONet(std::size_t                    branch_input_dim,
                   const std::vector<std::size_t>& branch_hidden_dims,
                   std::size_t                    trunk_input_dim,
                   const std::vector<std::size_t>& trunk_hidden_dims,
                   std::size_t                    latent_dim,
                   Activation                     branch_act,
                   Activation                     trunk_act)
    : latent_dim_(latent_dim)
    , branch_(std::make_unique<BranchNet>(branch_input_dim, branch_hidden_dims,
                                          latent_dim, branch_act))
    , trunk_ (std::make_unique<TrunkNet> (trunk_input_dim,  trunk_hidden_dims,
                                          latent_dim, trunk_act))
{
    if (latent_dim == 0)
        throw std::invalid_argument("DeepONet: latent_dim must be > 0");

    // Scalar output bias, initialised to 0.
    branch_latent_gain_ = std::make_unique<autograd::Tensor>(
        std::vector<double>(latent_dim_, 1.0),
        std::vector<std::size_t>{1, latent_dim_},
        /*requires_grad=*/true);
    branch_latent_bias_ = std::make_unique<autograd::Tensor>(
        std::vector<double>(latent_dim_, 0.0),
        std::vector<std::size_t>{1, latent_dim_},
        /*requires_grad=*/true);
    branch_adapter_mix_ = std::make_unique<autograd::Tensor>(
        std::vector<double>{0.0},
        std::vector<std::size_t>{1},
        /*requires_grad=*/true);
    trunk_latent_gain_ = std::make_unique<autograd::Tensor>(
        std::vector<double>(latent_dim_, 1.0),
        std::vector<std::size_t>{1, latent_dim_},
        /*requires_grad=*/true);
    trunk_latent_bias_ = std::make_unique<autograd::Tensor>(
        std::vector<double>(latent_dim_, 0.0),
        std::vector<std::size_t>{1, latent_dim_},
        /*requires_grad=*/true);
    trunk_adapter_mix_ = std::make_unique<autograd::Tensor>(
        std::vector<double>{0.0},
        std::vector<std::size_t>{1},
        /*requires_grad=*/true);
    output_scale_ = std::make_unique<autograd::Tensor>(
        std::vector<double>{1.0},
        std::vector<std::size_t>{1},
        /*requires_grad=*/true);
    output_bias_ = std::make_unique<autograd::Tensor>(
        std::vector<double>{0.0},
        std::vector<std::size_t>{1},
        /*requires_grad=*/true);

}

//                                        

autograd::Tensor DeepONet::forward(const autograd::Tensor& u_batch,
                                   const autograd::Tensor& y_batch) const
{
    //   Validate input shapes                         
    if (u_batch.ndim() != 2)
        throw std::invalid_argument("DeepONet::forward: u_batch must be 2-D [batch, m]");
    if (y_batch.ndim() != 2)
        throw std::invalid_argument("DeepONet::forward: y_batch must be 2-D [batch, d]");

    const std::size_t batch = u_batch.shape()[0];
    if (y_batch.shape()[0] != batch)
        throw std::invalid_argument(
            "DeepONet::forward: u_batch and y_batch must have the same batch size");
    if (u_batch.shape()[1] != branch_->input_dim())
        throw std::invalid_argument(
            "DeepONet::forward: u_batch feature dimension mismatch");
    if (y_batch.shape()[1] != trunk_->input_dim())
        throw std::invalid_argument(
            "DeepONet::forward: y_batch feature dimension mismatch");

    //   Sub-network forward passes                       
    // b : [batch, p]   — branch coefficients
    autograd::Tensor b = branch_->forward(u_batch);
    // t : [batch, p]   — trunk  basis values
    autograd::Tensor t = trunk_->forward(y_batch);
    autograd::Tensor b_affine = b * *branch_latent_gain_ + *branch_latent_bias_;
    autograd::Tensor t_affine = t * *trunk_latent_gain_ + *trunk_latent_bias_;
    b = b + *branch_adapter_mix_ * (b_affine - b);
    t = t + *trunk_adapter_mix_ * (t_affine - t);

    //   Merge: dot product per row                       
    // Step 1: element-wise product → [batch, p]
    autograd::Tensor bt = b * t;

    // Step 2: reduce along latent dimension in O(batch * p).
    autograd::Tensor dot = row_sum_2d(bt);

    //   Add scalar output bias                         
    // output_bias_ has shape [1]; broadcast over [batch, 1] via add.
    autograd::Tensor pred = dot * *output_scale_ + *output_bias_;

    return pred;  // [batch, 1]
}

//                                        

std::vector<autograd::Tensor*> DeepONet::parameters()
{
    auto p = branch_->parameters();
    auto q = trunk_->parameters();
    p.insert(p.end(), q.begin(), q.end());
    p.push_back(branch_latent_gain_.get());
    p.push_back(branch_latent_bias_.get());
    p.push_back(branch_adapter_mix_.get());
    p.push_back(trunk_latent_gain_.get());
    p.push_back(trunk_latent_bias_.get());
    p.push_back(trunk_adapter_mix_.get());
    p.push_back(output_scale_.get());
    p.push_back(output_bias_.get());
    return p;
}

void DeepONet::zero_grad()
{
    branch_->zero_grad();
    trunk_->zero_grad();
    branch_latent_gain_->zero_grad();
    branch_latent_bias_->zero_grad();
    branch_adapter_mix_->zero_grad();
    trunk_latent_gain_->zero_grad();
    trunk_latent_bias_->zero_grad();
    trunk_adapter_mix_->zero_grad();
    output_scale_->zero_grad();
    output_bias_->zero_grad();
}

} // namespace deep_onet
