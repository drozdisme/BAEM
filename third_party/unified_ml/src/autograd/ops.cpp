// ops.cpp — Differentiable tensor operations.
//
// Each op follows the same pattern:
//   1. Compute forward value.
//   2. If any input requires grad, build a non-leaf Node with:
//        backward_fn – fast path (raw double arithmetic)
//        vjp_fn      – VJP path (Tensor arithmetic for create_graph mode)
//   3. Attach the Node to the output Tensor.
//
// VJP convention:
//   Each vjp_fn receives the upstream gradient as std::any(const Tensor&)
//   and returns std::any(unordered_map<Node*, Tensor>) mapping every input
//   node to its gradient contribution.  When an input node appears twice
//   (x*x), both contributions are accumulated in the map before return.
//
// Broadcasting (add, sub):
//   Full NumPy-style broadcasting is supported.  The backward reduces the
//   output gradient back to the shape of each operand by summing over the
//   broadcast dimensions.
//
// Gradient derivations
//           
//  add     C = A + B      dA += dC              dB += dC
//  sub     C = A - B      dA += dC              dB -= dC
//  mul     C = A ⊙ B      dA += dC * B          dB += dC * A
//  matmul  C = A @ B      dA = dC @ Bᵀ          dB = Aᵀ @ dC
//  sum     s = Σ A        dA[i] += ds
//  mean    m = Σ A / N    dA[i] += dm / N
//  relu    R = max(0,A)   dA[i] += dR[i] if A[i]>0 else 0
//  pow     P = A^e        dA[i] += dP[i] * e * A[i]^(e-1)

#include "autograd/tensor.h"
#include "autograd/functional.h"   // for GradMap typedef via full includes
#include "core/hpc_kernels.hpp"

#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace autograd {

// ════════════════════════════════════════════════════════════════════════════
//  Shared type alias
// ════════════════════════════════════════════════════════════════════════════

using GradMap = std::unordered_map<Node*, Tensor>;

// ════════════════════════════════════════════════════════════════════════════
//  File-local helpers
// ════════════════════════════════════════════════════════════════════════════

namespace {

//   Broadcasting helpers                            

/// Compute NumPy-style broadcast output shape.
/// Throws if shapes are incompatible.
std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t>& sa,
                                          const std::vector<std::size_t>& sb)
{
    std::size_t ndim = std::max(sa.size(), sb.size());
    std::vector<std::size_t> out(ndim);
    for (int i = 0; i < (int)ndim; ++i) {
        int ai = (int)sa.size() - (int)ndim + i;
        int bi = (int)sb.size() - (int)ndim + i;
        std::size_t da = (ai >= 0) ? sa[ai] : 1;
        std::size_t db = (bi >= 0) ? sb[bi] : 1;
        if (da != db && da != 1 && db != 1) {
            std::ostringstream oss;
            oss << "shapes are not broadcast-compatible at dim " << i;
            throw std::invalid_argument(oss.str());
        }
        out[i] = std::max(da, db);
    }
    return out;
}

/// Compute the strides for a shape (row-major).
std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape)
{
    std::size_t ndim = shape.size();
    std::vector<std::size_t> strides(ndim, 1);
    for (int d = (int)ndim - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * shape[d + 1];
    return strides;
}

/// Reduce a gradient tensor (shape out_shape) back to in_shape by summing
/// over all broadcast dimensions.
///
/// Uses an odometer (multi-index carry counter) so each output element costs
/// only O(ndim) additions instead of O(ndim) integer divisions/modulos.
std::vector<double> reduce_broadcast(const std::vector<double>& grad_out,
                                      const std::vector<std::size_t>& out_shape,
                                      const std::vector<std::size_t>& in_shape)
{
    if (in_shape.size() > out_shape.size()) {
        throw std::invalid_argument("reduce_broadcast: input rank cannot exceed output rank");
    }
    const std::size_t pad = out_shape.size() - in_shape.size();
    for (std::size_t i = 0; i < in_shape.size(); ++i) {
        const std::size_t out_dim = out_shape[pad + i];
        const std::size_t in_dim  = in_shape[i];
        if (in_dim != 1 && in_dim != out_dim) {
            throw std::invalid_argument("reduce_broadcast: non-broadcast-compatible shapes");
        }
    }

    std::size_t in_numel = 1;
    for (std::size_t d : in_shape) in_numel *= d;

    std::vector<double> result(in_numel, 0.0);
    const std::size_t out_numel = grad_out.size();
    const std::size_t ndim      = out_shape.size();
    const std::size_t rank_pad  = ndim - in_shape.size();

    // Padded in_shape with leading 1s for broadcast dims.
    std::vector<std::size_t> in_padded(ndim, 1);
    for (std::size_t i = 0; i < in_shape.size(); ++i)
        in_padded[rank_pad + i] = in_shape[i];

    // Precompute effective input strides (0 for broadcast dimensions).
    // Computing once here avoids O(out_numel) calls to compute_strides.
    auto raw_in_strides = compute_strides(in_padded);
    std::vector<std::size_t> eff_in_strides(ndim);
    for (std::size_t d = 0; d < ndim; ++d)
        eff_in_strides[d] = (in_padded[d] == 1) ? 0 : raw_in_strides[d];

    // Odometer walk: each step costs O(ndim) additions, no division/modulo.
    std::vector<std::size_t> multi(ndim, 0);
    for (std::size_t flat = 0; flat < out_numel; ++flat) {
        std::size_t in_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d)
            in_flat += multi[d] * eff_in_strides[d];

        result[in_flat] += grad_out[flat];

        // Carry-increment the odometer from the last dimension.
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++multi[d] < out_shape[d]) break;
            multi[d] = 0;
        }
    }
    return result;
}

/// Precompute, for each output-batch flat index, the corresponding
/// flat batch indices in operands A and B under NumPy broadcast rules.
/// This avoids repeated division/modulo work and temporary allocations
/// inside hot matmul loops.
void build_broadcast_batch_maps(const std::vector<std::size_t>& out_batch,
                                const std::vector<std::size_t>& a_batch,
                                const std::vector<std::size_t>& b_batch,
                                std::vector<std::size_t>& a_map,
                                std::vector<std::size_t>& b_map)
{
    std::size_t batch_size = 1;
    for (std::size_t d : out_batch) batch_size *= d;

    a_map.assign(batch_size, 0);
    b_map.assign(batch_size, 0);
    if (batch_size == 0) return;

    const std::size_t ndim = out_batch.size();
    if (ndim == 0) {
        a_map[0] = 0;
        b_map[0] = 0;
        return;
    }

    const std::size_t a_pad = ndim - a_batch.size();
    const std::size_t b_pad = ndim - b_batch.size();

    std::vector<std::size_t> a_padded(ndim, 1), b_padded(ndim, 1);
    for (std::size_t i = 0; i < a_batch.size(); ++i) a_padded[a_pad + i] = a_batch[i];
    for (std::size_t i = 0; i < b_batch.size(); ++i) b_padded[b_pad + i] = b_batch[i];

    const auto a_strides = compute_strides(a_padded);
    const auto b_strides = compute_strides(b_padded);

    std::vector<std::size_t> multi(ndim, 0);
    for (std::size_t idx = 0; idx < batch_size; ++idx) {
        std::size_t a_flat = 0, b_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            if (a_padded[d] != 1) a_flat += multi[d] * a_strides[d];
            if (b_padded[d] != 1) b_flat += multi[d] * b_strides[d];
        }
        a_map[idx] = a_flat;
        b_map[idx] = b_flat;

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++multi[d] < out_batch[d]) break;
            multi[d] = 0;
        }
    }
}

//   GradMap helpers                              

/// Add `contrib` to `map[key]`, creating the entry if it does not exist.
void gmap_add(GradMap& map, Node* key, const Tensor& contrib) {
    auto it = map.find(key);
    if (it == map.end())
        map.emplace(key, contrib);
    else
        it->second = it->second + contrib;
}

struct MatmulBackwardScratch {
    std::vector<double> bT;
    std::vector<double> aT;
    std::vector<double> dA_tmp;
    std::vector<double> dB_tmp;
};

}  // anonymous namespace

// ════════════════════════════════════════════════════════════════════════════
//  add  (with broadcast)
// ════════════════════════════════════════════════════════════════════════════

Tensor add(const Tensor& a, const Tensor& b)
{
    // Fast path: identical shapes.
    if (a.shape() == b.shape()) {
        const std::size_t N = a.numel();
        std::vector<double> out_data(N);
        for (std::size_t i = 0; i < N; ++i)
            out_data[i] = a.value_flat(i) + b.value_flat(i);

        const bool req = a.requires_grad() || b.requires_grad();
        Tensor out(std::move(out_data), a.shape(), false);
        if (!req) return out;

        auto out_node = std::make_shared<Node>(N);
        out_node->is_leaf = false;
        auto a_node = a.node(), b_node = b.node();
        if (a_node) out_node->inputs.push_back(a_node);
        if (b_node && b_node != a_node) out_node->inputs.push_back(b_node);

        std::weak_ptr<Node> out_weak = out_node;
        out_node->backward_fn = [out_weak, a_node, b_node]() {
            auto out_locked = out_weak.lock();
            if (!out_locked) return;
            const auto& go = out_locked->grad;
            if (a_node) for (std::size_t i = 0; i < go.size(); ++i) a_node->grad[i] += go[i];
            if (b_node) for (std::size_t i = 0; i < go.size(); ++i) b_node->grad[i] += go[i];
        };

        out_node->vjp_fn = [a_node, b_node](const std::any& g) -> std::any {
            const Tensor& gup = std::any_cast<const Tensor&>(g);
            GradMap res;
            if (a_node) gmap_add(res, a_node.get(), gup);
            if (b_node) gmap_add(res, b_node.get(), gup);
            return std::make_any<GradMap>(std::move(res));
        };

        out.set_node(out_node);
        out.set_requires_grad(true);
        return out;
    }

    // Broadcast path.
    auto out_shape = broadcast_shape(a.shape(), b.shape());
    std::size_t out_numel = 1;
    for (std::size_t d : out_shape) out_numel *= d;

    // Precompute effective strides for a and b once (broadcast dims get stride 0).
    // This avoids O(out_numel) calls to broadcast_flat_index, each of which
    // allocates vectors and performs ndim integer divisions.
    const std::size_t ndim = out_shape.size();
    const std::size_t a_pad = ndim - a.shape().size();
    const std::size_t b_pad = ndim - b.shape().size();

    std::vector<std::size_t> a_padded(ndim, 1), b_padded(ndim, 1);
    for (std::size_t i = 0; i < a.shape().size(); ++i) a_padded[a_pad + i] = a.shape()[i];
    for (std::size_t i = 0; i < b.shape().size(); ++i) b_padded[b_pad + i] = b.shape()[i];

    {
        auto a_raw = compute_strides(a_padded), b_raw = compute_strides(b_padded);
        std::vector<std::size_t> a_eff(ndim), b_eff(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            a_eff[d] = (a_padded[d] == 1) ? 0 : a_raw[d];
            b_eff[d] = (b_padded[d] == 1) ? 0 : b_raw[d];
        }

        std::vector<double> out_data(out_numel);
        std::vector<std::size_t> multi(ndim, 0);
        for (std::size_t i = 0; i < out_numel; ++i) {
            std::size_t ia = 0, ib = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                ia += multi[d] * a_eff[d];
                ib += multi[d] * b_eff[d];
            }
            out_data[i] = a.value_flat(ia) + b.value_flat(ib);
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                if (++multi[d] < out_shape[d]) break;
                multi[d] = 0;
            }
        }

        const bool req = a.requires_grad() || b.requires_grad();
        Tensor out(std::move(out_data), out_shape, false);
        if (!req) return out;

        auto out_node = std::make_shared<Node>(out_numel);
        out_node->is_leaf = false;
        auto a_node = a.node(), b_node = b.node();
        auto a_shape = a.shape(), b_shape = b.shape();

        if (a_node) out_node->inputs.push_back(a_node);
        if (b_node && b_node != a_node) out_node->inputs.push_back(b_node);

        std::weak_ptr<Node> out_weak = out_node;
        out_node->backward_fn = [out_weak, a_node, b_node, out_shape, a_shape, b_shape]() {
            auto out_locked = out_weak.lock();
            if (!out_locked) return;
            const auto& go = out_locked->grad;
            if (a_node) {
                auto ga = reduce_broadcast(go, out_shape, a_shape);
                for (std::size_t i = 0; i < ga.size(); ++i) a_node->grad[i] += ga[i];
            }
            if (b_node) {
                auto gb = reduce_broadcast(go, out_shape, b_shape);
                for (std::size_t i = 0; i < gb.size(); ++i) b_node->grad[i] += gb[i];
            }
        };

        out_node->vjp_fn = [a_node, b_node, out_shape, a_shape, b_shape]
                           (const std::any& g) -> std::any {
            const Tensor& gup = std::any_cast<const Tensor&>(g);
            GradMap res;
            if (a_node) {
                auto rd = reduce_broadcast(gup.data(), out_shape, a_shape);
                gmap_add(res, a_node.get(), Tensor(rd, a_shape, false));
            }
            if (b_node) {
                auto rd = reduce_broadcast(gup.data(), out_shape, b_shape);
                gmap_add(res, b_node.get(), Tensor(rd, b_shape, false));
            }
            return std::make_any<GradMap>(std::move(res));
        };

        out.set_node(out_node);
        out.set_requires_grad(true);
        return out;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  sub  (with broadcast, delegates to add)
// ════════════════════════════════════════════════════════════════════════════

Tensor sub(const Tensor& a, const Tensor& b)
{
    // sub(a,b) = add(a, -b).  Negate b and use add.
    Tensor neg_b = -b;   // unary negation via tensor.cpp operator-()
    return add(a, neg_b);
}

// ════════════════════════════════════════════════════════════════════════════
//  mul  (element-wise, with NumPy-style broadcast)
//
//  Mirrors add/sub: same-shape inputs use a fast path; mismatched shapes go
//  through the broadcast path that pre-expands both operands and then reduces
//  the output gradient back to each operand's shape.
//
//  VJP note: the broadcast path captures the fully-expanded operand values at
//  forward time and uses them in raw-double arithmetic during the backward.
//  Second-order differentiation through a broadcast reduction is therefore NOT
//  supported (same limitation as add's broadcast VJP).  For non-broadcast or
//  scalar-broadcast use cases, second-order works correctly via the same-shape
//  VJP path that calls the differentiable mul() recursively.
// ════════════════════════════════════════════════════════════════════════════

Tensor mul(const Tensor& a, const Tensor& b)
{
    const bool same_input_node = (a.node() && b.node() && a.node() == b.node());
    //   Broadcast path (shapes differ)                     
    if (a.shape() != b.shape()) {
        auto out_shape = broadcast_shape(a.shape(), b.shape());   // throws if incompatible
        std::size_t out_numel = 1;
        for (std::size_t d : out_shape) out_numel *= d;

        const std::size_t ndim  = out_shape.size();
        const std::size_t a_pad = ndim - a.shape().size();
        const std::size_t b_pad = ndim - b.shape().size();

        std::vector<std::size_t> a_padded(ndim, 1), b_padded(ndim, 1);
        for (std::size_t i = 0; i < a.shape().size(); ++i) a_padded[a_pad + i] = a.shape()[i];
        for (std::size_t i = 0; i < b.shape().size(); ++i) b_padded[b_pad + i] = b.shape()[i];

        auto a_raw = compute_strides(a_padded), b_raw = compute_strides(b_padded);
        std::vector<std::size_t> a_eff(ndim), b_eff(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            a_eff[d] = (a_padded[d] == 1) ? 0 : a_raw[d];
            b_eff[d] = (b_padded[d] == 1) ? 0 : b_raw[d];
        }

        // Forward: compute result and snapshot expanded values for backward.
        std::vector<double> out_data(out_numel);
        std::vector<double> a_expand(out_numel), b_expand(out_numel);
        {
            std::vector<std::size_t> multi(ndim, 0);
            for (std::size_t i = 0; i < out_numel; ++i) {
                std::size_t ia = 0, ib = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    ia += multi[d] * a_eff[d];
                    ib += multi[d] * b_eff[d];
                }
                a_expand[i] = a.value_flat(ia);
                b_expand[i] = b.value_flat(ib);
                out_data[i] = a_expand[i] * b_expand[i];
                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    if (++multi[d] < out_shape[d]) break;
                    multi[d] = 0;
                }
            }
        }

        const bool req = a.requires_grad() || b.requires_grad();
        Tensor out(std::move(out_data), out_shape, false);
        if (!req) return out;

        auto out_node = std::make_shared<Node>(out_numel);
        out_node->is_leaf = false;
        auto a_node = a.node(), b_node = b.node();
        auto a_shape = a.shape(), b_shape = b.shape();

        if (a_node) out_node->inputs.push_back(a_node);
        if (b_node && b_node != a_node) out_node->inputs.push_back(b_node);

        // Fast-path backward: grad_a = reduce(grad_out ⊙ b_expand, a_shape)
        std::weak_ptr<Node> out_weak = out_node;
        out_node->backward_fn = [out_weak, a_node, b_node,
                                  out_shape, a_shape, b_shape,
                                  a_expand, b_expand]() {
            auto out_locked = out_weak.lock();
            if (!out_locked) return;
            const auto& go = out_locked->grad;
            if (a_node) {
                std::vector<double> tmp(go.size());
                for (std::size_t i = 0; i < go.size(); ++i) tmp[i] = go[i] * b_expand[i];
                auto ga = reduce_broadcast(tmp, out_shape, a_shape);
                for (std::size_t i = 0; i < ga.size(); ++i) a_node->grad[i] += ga[i];
            }
            if (b_node && b_node != a_node) {
                std::vector<double> tmp(go.size());
                for (std::size_t i = 0; i < go.size(); ++i) tmp[i] = go[i] * a_expand[i];
                auto gb = reduce_broadcast(tmp, out_shape, b_shape);
                for (std::size_t i = 0; i < gb.size(); ++i) b_node->grad[i] += gb[i];
            } else if (b_node && b_node == a_node) {
                std::vector<double> tmp(go.size());
                for (std::size_t i = 0; i < go.size(); ++i) tmp[i] = go[i] * (a_expand[i] + b_expand[i]);
                auto ga = reduce_broadcast(tmp, out_shape, a_shape);
                for (std::size_t i = 0; i < ga.size(); ++i) a_node->grad[i] += ga[i];
            }
        };

        // VJP: same raw-data approach — 2nd-order through the broadcast
        // reduction itself is not supported (same limit as add's broadcast VJP).
        out_node->vjp_fn = [a_node, b_node, out_shape, a_shape, b_shape,
                             a_expand, b_expand](const std::any& g) -> std::any {
            const Tensor& gup = std::any_cast<const Tensor&>(g);
            GradMap res;
            if (a_node) {
                std::vector<double> tmp(gup.numel());
                for (std::size_t i = 0; i < gup.numel(); ++i) tmp[i] = gup.data()[i] * b_expand[i];
                auto ga = reduce_broadcast(tmp, out_shape, a_shape);
                gmap_add(res, a_node.get(), Tensor(ga, a_shape, false));
            }
            if (b_node && b_node != a_node) {
                std::vector<double> tmp(gup.numel());
                for (std::size_t i = 0; i < gup.numel(); ++i) tmp[i] = gup.data()[i] * a_expand[i];
                auto gb = reduce_broadcast(tmp, out_shape, b_shape);
                gmap_add(res, b_node.get(), Tensor(gb, b_shape, false));
            } else if (b_node && b_node == a_node) {
                std::vector<double> tmp(gup.numel());
                for (std::size_t i = 0; i < gup.numel(); ++i) tmp[i] = gup.data()[i] * (a_expand[i] + b_expand[i]);
                auto ga = reduce_broadcast(tmp, out_shape, a_shape);
                gmap_add(res, a_node.get(), Tensor(ga, a_shape, false));
            }
            return std::make_any<GradMap>(std::move(res));
        };

        out.set_node(out_node);
        out.set_requires_grad(true);
        return out;
    }

    //   Fast path: identical shapes                       
    const std::size_t N = a.numel();

    //   Forward                                
    std::vector<double> out_data(N);
    for (std::size_t i = 0; i < N; ++i)
        out_data[i] = a.value_flat(i) * b.value_flat(i);

    const bool req = a.requires_grad() || b.requires_grad();
    Tensor out(std::move(out_data), a.shape(), false);
    if (!req) return out;

    //   Build node                               
    auto out_node = std::make_shared<Node>(N);
    out_node->is_leaf = false;

    auto a_node = a.node(), b_node = b.node();

    // Deduplicate inputs (x*x: same node, insert once).
    if (a_node) out_node->inputs.push_back(a_node);
    if (b_node && b_node != a_node) out_node->inputs.push_back(b_node);

    // Capture full Tensors (shares nodes; data only copied once).
    Tensor a_copy = a, b_copy = b;

    //   Fast-path backward                           
    //  grad_a[i] += grad_out[i] * b_snap[i]
    //  grad_b[i] += grad_out[i] * a_snap[i]
    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, b_node,
                              a_copy, b_copy, same_input_node]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        const auto& ad = a_copy.data();
        const auto& bd = b_copy.data();
        if (a_node) {
            if (same_input_node) {
                for (std::size_t i = 0; i < go.size(); ++i)
                    a_node->grad[i] += go[i] * (ad[i] + bd[i]);
            } else {
                for (std::size_t i = 0; i < go.size(); ++i) a_node->grad[i] += go[i] * bd[i];
            }
        }
        if (b_node && b_node != a_node)
            for (std::size_t i = 0; i < go.size(); ++i) b_node->grad[i] += go[i] * ad[i];
    };

    //   VJP                                  
    //  grad_a = grad_c ⊙ b_copy   (b_copy keeps requires_grad for 2nd-order)
    //  grad_b = grad_c ⊙ a_copy
    out_node->vjp_fn = [a_node, b_node, a_copy, b_copy, same_input_node]
                       (const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        GradMap res;
        if (a_node) {
            if (same_input_node) {
                gmap_add(res, a_node.get(), mul(gup, a_copy + b_copy));
            } else {
                gmap_add(res, a_node.get(), mul(gup, b_copy));
            }
        }
        if (b_node && b_node != a_node) gmap_add(res, b_node.get(), mul(gup, a_copy));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  transpose  ([..., m, n] → [..., n, m])
//
//  Forward:  swap last two dims.
//  Backward: another transpose (transpose is its own inverse adjoint).
// ════════════════════════════════════════════════════════════════════════════

Tensor transpose(const Tensor& a)
{
    if (a.ndim() < 2)
        throw std::invalid_argument(
            "transpose: tensor must have at least 2 dimensions (got " +
            std::to_string(a.ndim()) + ")");

    //   New shape: swap last two dims                     
    auto out_shape   = a.shape();
    const std::size_t nd = out_shape.size();
    std::swap(out_shape[nd - 2], out_shape[nd - 1]);

    const std::size_t m = a.shape()[nd - 2];   // rows in each 2-D slice
    const std::size_t n = a.shape()[nd - 1];   // cols in each 2-D slice

    // batch_size = product of all dims except the last two
    std::size_t batch = 1;
    for (std::size_t d = 0; d < nd - 2; ++d) batch *= a.shape()[d];

    //   Forward                                
    // Cache-blocked transpose: process TILE×TILE sub-matrices so that both
    // the read (src row) and write (dst column) fit in L1/L2 cache.
    // Plain ij: src[i*n+j] is stride-1 (good) but dst[j*m+i] is stride-m
    // (scatter).  Blocking amortises the scatter cost over a TILE-wide strip.
    constexpr std::size_t TILE = 32;
    std::vector<double> out_data(a.numel());
    for (std::size_t b = 0; b < batch; ++b) {
        const double* src = a.data().data() + b * m * n;
        double*       dst = out_data.data() + b * n * m;
        for (std::size_t ii = 0; ii < m; ii += TILE)
            for (std::size_t jj = 0; jj < n; jj += TILE) {
                const std::size_t i_end = std::min(ii + TILE, m);
                const std::size_t j_end = std::min(jj + TILE, n);
                for (std::size_t i = ii; i < i_end; ++i)
                    for (std::size_t j = jj; j < j_end; ++j)
                        dst[j * m + i] = src[i * n + j];
            }
    }

    Tensor out(std::move(out_data), out_shape, false);
    if (!a.requires_grad()) return out;

    //   Build node                               
    auto out_node = std::make_shared<Node>(a.numel());
    auto a_node   = a.node();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    //   Fast backward: grad_a = transpose(grad_out) — same blocking       
    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, m, n, batch]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        auto& ga       = a_node->grad;
        constexpr std::size_t BLK = 32;
        for (std::size_t b = 0; b < batch; ++b) {
            const double* src = go.data() + b * n * m;
            double*       dst = ga.data() + b * m * n;
            for (std::size_t ii = 0; ii < n; ii += BLK)
                for (std::size_t jj = 0; jj < m; jj += BLK) {
                    const std::size_t i_end = std::min(ii + BLK, n);
                    const std::size_t j_end = std::min(jj + BLK, m);
                    for (std::size_t i = ii; i < i_end; ++i)
                        for (std::size_t j = jj; j < j_end; ++j)
                            dst[j * n + i] += src[i * m + j];
                }
        }
    };

    //   VJP: grad_a = transpose(grad_out) — re-uses the differentiable op   
    out_node->vjp_fn = [a_node](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        GradMap res;
        if (a_node) res.emplace(a_node.get(), transpose(gup));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  matmul  (batched: [..., m, k] @ [..., k, n]  →  [..., m, n])
//
//  Batch dimensions are broadcast-compatible (NumPy rules).
//  1-D inputs are treated as matrices by inserting/removing a length-1 dim:
//    [k]        →  treated as [1, k]  (left arg)   ; leading dim removed from output
//    [k]        →  treated as [k, 1]  (right arg)  ; trailing dim removed from output
//
//  Gradients:
//    dL/dA[b] = dL/dC[b] @ B[b]ᵀ   → matmul(dC, transpose(B))
//    dL/dB[b] = A[b]ᵀ @ dL/dC[b]   → matmul(transpose(A), dC)
//  VJP uses the differentiable transpose + matmul ops so the backward pass
//  itself builds a graph (second-order derivatives are supported).
// ════════════════════════════════════════════════════════════════════════════

Tensor matmul(const Tensor& a, const Tensor& b)
{
    //   Dimension validation                         
    if (a.ndim() == 0 || b.ndim() == 0)
        throw std::invalid_argument("matmul: scalar inputs not supported");

    // Promote 1-D inputs to 2-D (PyTorch convention).
    bool a_was_1d = (a.ndim() == 1);
    bool b_was_1d = (b.ndim() == 1);

    // Build effective (possibly promoted) shapes.
    std::vector<std::size_t> a_eff = a_was_1d
        ? std::vector<std::size_t>{1, a.shape()[0]}
        : a.shape();
    std::vector<std::size_t> b_eff = b_was_1d
        ? std::vector<std::size_t>{b.shape()[0], 1}
        : b.shape();

    if (a_eff.size() < 2 || b_eff.size() < 2)
        throw std::invalid_argument("matmul: inputs must be at least 1-D");

    const std::size_t nd_a = a_eff.size();
    const std::size_t nd_b = b_eff.size();

    const std::size_t m  = a_eff[nd_a - 2];
    const std::size_t k  = a_eff[nd_a - 1];
    const std::size_t k2 = b_eff[nd_b - 2];
    const std::size_t n  = b_eff[nd_b - 1];

    if (k != k2)
        throw std::invalid_argument(
            "matmul: inner dimensions must agree (k=" + std::to_string(k) +
            " vs k2=" + std::to_string(k2) + ")");

    //   Batch shape broadcast                         
    // Batch prefix = all dims except the last two.
    std::vector<std::size_t> a_batch(a_eff.begin(), a_eff.end() - 2);
    std::vector<std::size_t> b_batch(b_eff.begin(), b_eff.end() - 2);
    std::vector<std::size_t> out_batch = broadcast_shape(a_batch, b_batch);

    std::size_t batch_size = 1;
    for (std::size_t d : out_batch) batch_size *= d;

    // Precompute broadcast mapping once for forward + backward.
    std::vector<std::size_t> a_batch_map, b_batch_map;
    build_broadcast_batch_maps(out_batch, a_batch, b_batch, a_batch_map, b_batch_map);

    //   Forward                                
    // Output shape: out_batch + [m, n]
    std::vector<std::size_t> out_shape = out_batch;
    out_shape.push_back(m);
    out_shape.push_back(n);

    std::vector<double> out_data(batch_size * m * n, 0.0);
    const bool a_dense = a.is_contiguous() && a.offset() == 0;
    const bool b_dense = b.is_contiguous() && b.offset() == 0;
    std::vector<double> a_buf_storage;
    std::vector<double> b_buf_storage;
    const double* a_buf = nullptr;
    const double* b_buf = nullptr;
    if (a_dense) {
        a_buf = a.data().data();
    } else {
        a_buf_storage.resize(a.numel());
        for (std::size_t i = 0; i < a.numel(); ++i) a_buf_storage[i] = a.value_flat(i);
        a_buf = a_buf_storage.data();
    }
    if (b_dense) {
        b_buf = b.data().data();
    } else {
        b_buf_storage.resize(b.numel());
        for (std::size_t i = 0; i < b.numel(); ++i) b_buf_storage[i] = b.value_flat(i);
        b_buf = b_buf_storage.data();
    }

    // For each output batch index, map it back to A and B batch indices
    // (handling broadcast where a batch dim == 1 → always index 0).
    // We reuse broadcast_flat_index by treating batch dims separately.
    for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        // Flat index into A's batch and B's batch.
        const std::size_t a_b = a_batch_map[b_idx];
        const std::size_t b_b = b_batch_map[b_idx];

        const double* A_ptr = a_buf + a_b * m * k;
        const double* B_ptr = b_buf + b_b * k * n;
        double*       C_ptr = out_data.data() + b_idx * m * n;

        hpc::gemm_hpc(A_ptr, B_ptr, C_ptr, m, k, n, k, n, n);
    }

    // Remove dimensions that were added by 1-D promotion.
    if (a_was_1d) {
        // Remove second-to-last dim (m==1).
        out_shape.erase(out_shape.end() - 2);
    }
    if (b_was_1d) {
        // Remove last dim (n==1).
        out_shape.erase(out_shape.end() - 1);
    }
    if (out_shape.empty()) {
        // Both were 1-D: result is a scalar.
        out_shape = {};
    }

    const bool req = a.requires_grad() || b.requires_grad();
    Tensor out(std::move(out_data), out_shape, false);
    if (!req) return out;

    //   Build node                               
    auto out_node = std::make_shared<Node>(out.numel());
    out_node->is_leaf = false;

    auto a_node = a.node(), b_node = b.node();
    if (a_node) out_node->inputs.push_back(a_node);
    if (b_node && b_node != a_node) out_node->inputs.push_back(b_node);

    // Snap forward data for the fast path.
    std::vector<double> a_snap(a.numel());
    std::vector<double> b_snap(b.numel());
    for (std::size_t i = 0; i < a.numel(); ++i) a_snap[i] = a.value_flat(i);
    for (std::size_t i = 0; i < b.numel(); ++i) b_snap[i] = b.value_flat(i);
    // Keep Tensor copies for the VJP path (preserves requires_grad / graph).
    // We store them with the *effective* 2-D+ shapes for transpose to work.
    Tensor a_vjp = a_was_1d
        ? Tensor(a.data(), a_eff, a.requires_grad())
        : a;
    Tensor b_vjp = b_was_1d
        ? Tensor(b.data(), b_eff, b.requires_grad())
        : b;
    if (a_was_1d && a.node()) a_vjp.set_node(a.node());
    if (b_was_1d && b.node()) b_vjp.set_node(b.node());

    //   Fast-path backward                           
    //  dL/dA[b] = dL/dC[b] @ B[b]ᵀ
    //  dL/dB[b] = A[b]ᵀ @ dL/dC[b]
    //
    //  Batch broadcasting: accumulate when a batch dim == 1.
    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, b_node,
                              a, b, a_dense, b_dense,
                              a_snap, b_snap,
                              m, k, n, batch_size,
                              a_batch_map, b_batch_map]()
    {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        static thread_local MatmulBackwardScratch scratch;
        scratch.bT.resize(n * k);
        scratch.aT.resize(k * m);
        if (!a_dense) scratch.dA_tmp.assign(m * k, 0.0);
        if (!b_dense) scratch.dB_tmp.assign(k * n, 0.0);

        for (std::size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            const std::size_t a_b = a_batch_map[b_idx];
            const std::size_t b_b = b_batch_map[b_idx];

            const double* dC = go.data()     + b_idx * m * n;
            const double* A  = a_snap.data() + a_b   * m * k;
            const double* B  = b_snap.data() + b_b   * k * n;

            if (a_node) {
                // dA += dC @ B^T
                for (std::size_t row = 0; row < k; ++row)
                    for (std::size_t col = 0; col < n; ++col)
                        scratch.bT[col * k + row] = B[row * n + col];
                if (a_dense) {
                    double* dA = a_node->grad.data() + a_b * m * k;
                    hpc::gemm_hpc(dC, scratch.bT.data(), dA, m, n, k, n, k, k);
                } else {
                    std::fill(scratch.dA_tmp.begin(), scratch.dA_tmp.end(), 0.0);
                    hpc::gemm_hpc(dC, scratch.bT.data(), scratch.dA_tmp.data(), m, n, k, n, k, k);
                    for (std::size_t idx = 0; idx < scratch.dA_tmp.size(); ++idx)
                        a_node->grad[a.flat_to_storage(a_b * m * k + idx)] += scratch.dA_tmp[idx];
                }
            }
            if (b_node) {
                // dB += A^T @ dC
                for (std::size_t row = 0; row < m; ++row)
                    for (std::size_t col = 0; col < k; ++col)
                        scratch.aT[col * m + row] = A[row * k + col];
                if (b_dense) {
                    double* dB = b_node->grad.data() + b_b * k * n;
                    hpc::gemm_hpc(scratch.aT.data(), dC, dB, k, m, n, m, n, n);
                } else {
                    std::fill(scratch.dB_tmp.begin(), scratch.dB_tmp.end(), 0.0);
                    hpc::gemm_hpc(scratch.aT.data(), dC, scratch.dB_tmp.data(), k, m, n, m, n, n);
                    for (std::size_t idx = 0; idx < scratch.dB_tmp.size(); ++idx)
                        b_node->grad[b.flat_to_storage(b_b * k * n + idx)] += scratch.dB_tmp[idx];
                }
            }
        }
    };

    //   VJP (create_graph / second-order)                   
    //  dA = dC @ Bᵀ    ← matmul(dC_eff, transpose(B_eff))
    //  dB = Aᵀ @ dC    ← matmul(transpose(A_eff), dC_eff)
    //
    //  Both are expressed as differentiable Tensor ops so the backward pass
    //  itself builds a computation graph.
    //
    //  If either input was originally 1-D, we reshape dC to remove the
    //  extra dimension before multiplying so shapes stay consistent.
    out_node->vjp_fn = [a_node, b_node, a_vjp, b_vjp,
                         a_was_1d, b_was_1d, m, k, n,
                         out_batch, a_batch, b_batch]
                        (const std::any& g) -> std::any
    {
        const Tensor& gup_orig = std::any_cast<const Tensor&>(g);

        // Restore the effective output shape [..., m, n] for gup.
        std::vector<std::size_t> eff_out_shape = out_batch;
        eff_out_shape.push_back(m);
        eff_out_shape.push_back(n);
        Tensor gup = gup_orig.reshape(eff_out_shape);

        GradMap res;

        if (a_node) {
            // dA_eff = gup @ Bᵀ   shape: [..., m, k]
            Tensor bT     = transpose(b_vjp);          // [..., n, k]
            Tensor dA_eff = matmul(gup, bT);            // [..., m, k]
            // Reduce broadcast back to a's original batch shape.
            // If no broadcast occurred (shapes already match), return dA_eff
            // directly to keep the computation graph intact for create_graph.
            // If broadcast reduction is needed, we accept the limitation that
            // second-order through the reduction is not tracked (same as
            // add/mul broadcast VJPs): construct with requires_grad=false to
            // avoid creating a disconnected leaf node.
            Tensor dA;
            if (a_was_1d) {
                dA = dA_eff.reshape({k});   // collapse the m==1 leading dim
            } else if (dA_eff.shape() == a_vjp.shape()) {
                dA = dA_eff;                // no broadcast: preserve graph
            } else {
                auto rd = reduce_broadcast(dA_eff.data(), dA_eff.shape(), a_vjp.shape());
                dA = Tensor(rd, a_vjp.shape(), false);  // broadcast reduction; graph not tracked
            }
            gmap_add(res, a_node.get(), dA);
        }
        if (b_node) {
            // dB_eff = Aᵀ @ gup   shape: [..., k, n]
            Tensor aT     = transpose(a_vjp);           // [..., k, m]
            Tensor dB_eff = matmul(aT, gup);             // [..., k, n]
            Tensor dB;
            if (b_was_1d) {
                dB = dB_eff.reshape({n});   // collapse the n==1 trailing dim
            } else if (dB_eff.shape() == b_vjp.shape()) {
                dB = dB_eff;                // no broadcast: preserve graph
            } else {
                auto rd = reduce_broadcast(dB_eff.data(), dB_eff.shape(), b_vjp.shape());
                dB = Tensor(rd, b_vjp.shape(), false);  // broadcast reduction; graph not tracked
            }
            gmap_add(res, b_node.get(), dB);
        }

        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  sum  (reduce all elements → scalar)
// ════════════════════════════════════════════════════════════════════════════

Tensor sum(const Tensor& a)
{
    double s = 0.0;
    for (std::size_t i = 0; i < a.numel(); ++i) s += a.value_flat(i);
    Tensor out(s, false);
    if (!a.requires_grad()) return out;

    auto out_node  = std::make_shared<Node>(1);
    auto a_node    = a.node();
    const std::size_t N = a.numel();
    auto a_shape   = a.shape();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, N]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        double go = out_locked->grad[0];
        for (std::size_t i = 0; i < N; ++i) a_node->grad[i] += go;
    };

    // VJP: grad_a[i] = grad_out (broadcast scalar to all elements)
    // sum is linear, so second derivative through sum is 0; we don't
    // need to track grad_out's graph through this broadcast.
    out_node->vjp_fn = [a_node, N, a_shape](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        double scalar_g = gup.item();
        GradMap res;
        if (a_node) {
            std::vector<double> ga_data(N, scalar_g);
            res.emplace(a_node.get(), Tensor(ga_data, a_shape, false));
        }
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  mean  (average of all elements → scalar)
// ════════════════════════════════════════════════════════════════════════════

Tensor mean(const Tensor& a)
{
    const std::size_t N = a.numel();
    double s = 0.0;
    for (std::size_t i = 0; i < a.numel(); ++i) s += a.value_flat(i);
    Tensor out(s / static_cast<double>(N), false);
    if (!a.requires_grad()) return out;

    auto out_node = std::make_shared<Node>(1);
    auto a_node   = a.node();
    auto a_shape  = a.shape();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, N]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        double go = out_locked->grad[0] / static_cast<double>(N);
        for (std::size_t i = 0; i < N; ++i) a_node->grad[i] += go;
    };

    out_node->vjp_fn = [a_node, N, a_shape](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        double scalar_g = gup.item() / static_cast<double>(N);
        GradMap res;
        if (a_node) {
            std::vector<double> ga_data(N, scalar_g);
            res.emplace(a_node.get(), Tensor(ga_data, a_shape, false));
        }
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  relu  (element-wise max(0, x))
// ════════════════════════════════════════════════════════════════════════════

Tensor relu(const Tensor& a)
{
    const std::size_t N = a.numel();
    std::vector<double> out_data(N);
    for (std::size_t i = 0; i < N; ++i)
        out_data[i] = std::max(0.0, a.value_flat(i));
    Tensor out(out_data, a.shape(), false);
    if (!a.requires_grad()) return out;

    auto out_node = std::make_shared<Node>(N);
    auto a_node   = a.node();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::vector<double> a_snap = a.data();

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, a_snap]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        for (std::size_t i = 0; i < go.size(); ++i)
            if (a_snap[i] > 0.0) a_node->grad[i] += go[i];
    };

    out_node->vjp_fn = [a_node, a_snap](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        std::vector<double> mask(a_snap.size());
        for (std::size_t i = 0; i < a_snap.size(); ++i)
            mask[i] = (a_snap[i] > 0.0) ? 1.0 : 0.0;
        Tensor mask_t(mask, {mask.size()}, false);
        GradMap res;
        if (a_node) res.emplace(a_node.get(), mul(gup, mask_t));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  pow  (element-wise x^exp)
//
//  Forward:   out[i] = a[i]^e
//  Backward:  grad_a[i] += grad_out[i] * e * a[i]^(e-1)
// ════════════════════════════════════════════════════════════════════════════

Tensor pow(const Tensor& a, double e)
{
    const std::size_t N = a.numel();
    std::vector<double> out_data(N);
    for (std::size_t i = 0; i < N; ++i)
        out_data[i] = std::pow(a.value_flat(i), e);
    Tensor out(out_data, a.shape(), false);
    if (!a.requires_grad()) return out;

    auto out_node = std::make_shared<Node>(N);
    auto a_node   = a.node();
    auto a_shape  = a.shape();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::vector<double> a_snap = a.data();

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, a_snap, e]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        for (std::size_t i = 0; i < go.size(); ++i)
            a_node->grad[i] += go[i] * e * std::pow(a_snap[i], e - 1.0);
    };

    // VJP for pow: grad_a = grad_out * e * a^(e-1)
    // We compute this using Tensor ops so it is differentiable.
    Tensor a_copy = a;  // keeps requires_grad=true for second-order
    out_node->vjp_fn = [a_node, a_copy, a_shape, e](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        // d_coeff = e * a^(e-1) as a Tensor (using a_copy for graph tracking)
        Tensor d_coeff = pow(a_copy, e - 1.0) * e;
        GradMap res;
        if (a_node) res.emplace(a_node.get(), mul(gup, d_coeff));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

Tensor exp(const Tensor& a)
{
    const std::size_t N = a.numel();
    std::vector<double> out_data(N);
    for (std::size_t i = 0; i < N; ++i)
        out_data[i] = std::exp(a.value_flat(i));
    Tensor out(out_data, a.shape(), false);
    if (!a.requires_grad()) return out;

    auto out_node = std::make_shared<Node>(N);
    auto a_node   = a.node();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, out_data]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        for (std::size_t i = 0; i < go.size(); ++i)
            a_node->grad[i] += go[i] * out_data[i];
    };

    Tensor a_copy = a;
    out_node->vjp_fn = [a_node, a_copy](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        GradMap res;
        if (a_node) res.emplace(a_node.get(), mul(gup, exp(a_copy)));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

Tensor log(const Tensor& a)
{
    const std::size_t N = a.numel();
    std::vector<double> out_data(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double v = a.value_flat(i);
        if (v <= 0.0) throw std::domain_error("log: all inputs must be strictly positive");
        out_data[i] = std::log(v);
    }
    Tensor out(out_data, a.shape(), false);
    if (!a.requires_grad()) return out;

    auto out_node = std::make_shared<Node>(N);
    auto a_node   = a.node();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::vector<double> a_snap = a.data();
    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, a_snap]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        for (std::size_t i = 0; i < go.size(); ++i)
            a_node->grad[i] += go[i] / a_snap[i];
    };

    Tensor a_copy = a;
    out_node->vjp_fn = [a_node, a_copy](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        GradMap res;
        if (a_node) res.emplace(a_node.get(), mul(gup, pow(a_copy, -1.0)));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

Tensor sqrt(const Tensor& a)
{
    return pow(a, 0.5);
}

Tensor softmax(const Tensor& a)
{
    if (a.numel() == 0) throw std::invalid_argument("softmax: empty tensor");
    const auto& shape = a.shape();
    const std::size_t last_dim = shape.empty() ? 1 : shape.back();
    const std::size_t rows = a.numel() / last_dim;
    std::vector<double> out_data(a.numel());

    for (std::size_t r = 0; r < rows; ++r) {
        double row_max = a.value_flat(r * last_dim);
        for (std::size_t c = 1; c < last_dim; ++c)
            row_max = std::max(row_max, a.value_flat(r * last_dim + c));
        double denom = 0.0;
        for (std::size_t c = 0; c < last_dim; ++c) {
            const std::size_t idx = r * last_dim + c;
            out_data[idx] = std::exp(a.value_flat(idx) - row_max);
            denom += out_data[idx];
        }
        for (std::size_t c = 0; c < last_dim; ++c)
            out_data[r * last_dim + c] /= denom;
    }

    Tensor out(out_data, shape, false);
    if (!a.requires_grad()) return out;

    auto out_node = std::make_shared<Node>(a.numel());
    auto a_node   = a.node();
    out_node->is_leaf = false;
    out_node->inputs.push_back(a_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, a_node, out_data, last_dim, rows]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        for (std::size_t r = 0; r < rows; ++r) {
            double dot = 0.0;
            for (std::size_t c = 0; c < last_dim; ++c) {
                const std::size_t idx = r * last_dim + c;
                dot += go[idx] * out_data[idx];
            }
            for (std::size_t c = 0; c < last_dim; ++c) {
                const std::size_t idx = r * last_dim + c;
                a_node->grad[idx] += out_data[idx] * (go[idx] - dot);
            }
        }
    };

    Tensor a_copy = a;
    out_node->vjp_fn = [a_node, a_copy](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        Tensor s = softmax(a_copy);
        const auto& shp = s.shape();
        const std::size_t ld = shp.empty() ? 1 : shp.back();
        const std::size_t rs = s.numel() / ld;
        std::vector<double> grad_data(s.numel(), 0.0);
        for (std::size_t r = 0; r < rs; ++r) {
            double dot = 0.0;
            for (std::size_t c = 0; c < ld; ++c)
                dot += gup.value_flat(r * ld + c) * s.value_flat(r * ld + c);
            for (std::size_t c = 0; c < ld; ++c) {
                const std::size_t idx = r * ld + c;
                grad_data[idx] = s.value_flat(idx) * (gup.value_flat(idx) - dot);
            }
        }
        GradMap res;
        if (a_node) res.emplace(a_node.get(), Tensor(std::move(grad_data), shp, false));
        return std::make_any<GradMap>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps)
{
    if (x.ndim() == 0) throw std::invalid_argument("layer_norm: x must have rank >= 1");
    const std::size_t hidden = x.shape().back();
    if (gamma.numel() != hidden || beta.numel() != hidden)
        throw std::invalid_argument("layer_norm: gamma and beta must match last dim");

    const std::size_t rows = x.numel() / hidden;
    std::vector<double> out_data(x.numel());
    std::vector<double> xhat(x.numel());
    std::vector<double> inv_std(rows);

    for (std::size_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (std::size_t c = 0; c < hidden; ++c) mean += x.value_flat(r * hidden + c);
        mean /= static_cast<double>(hidden);
        double var = 0.0;
        for (std::size_t c = 0; c < hidden; ++c) {
            const double d = x.value_flat(r * hidden + c) - mean;
            var += d * d;
        }
        var /= static_cast<double>(hidden);
        inv_std[r] = 1.0 / std::sqrt(var + eps);
        for (std::size_t c = 0; c < hidden; ++c) {
            const std::size_t idx = r * hidden + c;
            xhat[idx] = (x.value_flat(idx) - mean) * inv_std[r];
            out_data[idx] = xhat[idx] * gamma.value_flat(c) + beta.value_flat(c);
        }
    }

    Tensor out(out_data, x.shape(), false);
    if (!(x.requires_grad() || gamma.requires_grad() || beta.requires_grad())) return out;

    auto out_node = std::make_shared<Node>(x.numel());
    auto x_node = x.node();
    auto g_node = gamma.node();
    auto b_node = beta.node();
    out_node->is_leaf = false;
    if (x_node) out_node->inputs.push_back(x_node);
    if (g_node) out_node->inputs.push_back(g_node);
    if (b_node) out_node->inputs.push_back(b_node);

    std::weak_ptr<Node> out_weak = out_node;
    Tensor gamma_copy = gamma;
    out_node->backward_fn = [out_weak, x_node, g_node, b_node, gamma_copy, xhat, inv_std, hidden, rows]() mutable {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        const auto& go = out_locked->grad;
        std::vector<double> dgamma(hidden, 0.0), dbeta(hidden, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            double sum_dy = 0.0, sum_dy_xhat = 0.0;
            for (std::size_t c = 0; c < hidden; ++c) {
                const std::size_t idx = r * hidden + c;
                const double dy = go[idx];
                dgamma[c] += dy * xhat[idx];
                dbeta[c] += dy;
                sum_dy += dy * gamma_copy.value_flat(c);
                sum_dy_xhat += dy * gamma_copy.value_flat(c) * xhat[idx];
            }
            if (x_node) {
                for (std::size_t c = 0; c < hidden; ++c) {
                    const std::size_t idx = r * hidden + c;
                    const double dy = go[idx] * gamma_copy.value_flat(c);
                    x_node->grad[idx] += (inv_std[r] / static_cast<double>(hidden)) *
                        (static_cast<double>(hidden) * dy - sum_dy - xhat[idx] * sum_dy_xhat);
                }
            }
        }
        if (g_node) for (std::size_t c = 0; c < hidden; ++c) g_node->grad[c] += dgamma[c];
        if (b_node) for (std::size_t c = 0; c < hidden; ++c) b_node->grad[c] += dbeta[c];
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

}  // namespace autograd
