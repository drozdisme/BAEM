// activations.cpp — SIMD-optimized activation functions.
//
// KEY OPTIMIZATIONS vs baseline:
//  1. Loop body works on raw aligned pointers — no per-element bounds checks
//  2. #pragma omp simd enables AVX2/AVX-512 vectorization (8x doubles at once)
//  3. Precomputed constants (e.g. 1.0) hoisted out of loops
//  4. backward_fn uses same SIMD-annotated loops
//  5. tanh: out_data captured by value in closure; backward is a single SIMD pass

#include "core/activations.hpp"
#include "autograd/node.h"
#include "autograd/tensor.h"

#include <any>
#include <cmath>
#include <memory>
#include <unordered_map>

namespace core {

using GradMap = std::unordered_map<autograd::Node*, autograd::Tensor>;

// ============================================================================
// exp_act
// ============================================================================
autograd::Tensor exp_act(const autograd::Tensor& x) {
    const std::size_t N = x.numel();
    std::vector<double> out_data(N);
    const double* __restrict__ xp = x.data().data();
    double*       __restrict__ op = out_data.data();
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i) op[i] = std::exp(xp[i]);

    const bool rg = x.requires_grad();
    autograd::Tensor out(out_data, x.shape(), rg);
    if (!rg) return out;

    auto node = std::make_shared<autograd::Node>(N);
    node->is_leaf = false;
    auto x_node = x.node();
    if (x_node) node->inputs.push_back(x_node);

    node->backward_fn = [x_node, out_data = std::move(out_data), node_ptr = node.get()]() {
        if (!x_node) return;
        const double* __restrict__ g  = node_ptr->grad.data();
        const double* __restrict__ od = out_data.data();
        double*       __restrict__ dx = x_node->grad.data();
        const std::size_t N2 = node_ptr->grad.size();
#pragma omp simd
        for (std::size_t i = 0; i < N2; ++i) dx[i] += g[i] * od[i];
    };

    autograd::Tensor x_cap = x;
    node->vjp_fn = [x_cap](const std::any& up_any) -> std::any {
        const auto& up = std::any_cast<const autograd::Tensor&>(up_any);
        GradMap result;
        if (!x_cap.node()) return std::any(result);
        autograd::Tensor ex = exp_act(x_cap);
        result[x_cap.node().get()] = autograd::mul(up, ex);
        return std::any(result);
    };

    out.set_node(std::move(node));
    return out;
}

// ============================================================================
// tanh_act  — SIMD-vectorized forward and backward
// ============================================================================
autograd::Tensor tanh_act(const autograd::Tensor& x) {
    const std::size_t N = x.numel();
    std::vector<double> out_data(N);
    const double* __restrict__ xp = x.data().data();
    double*       __restrict__ op = out_data.data();
    // std::tanh is vectorized by glibc with -ffast-math on x86
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i) op[i] = std::tanh(xp[i]);

    const bool rg = x.requires_grad();
    autograd::Tensor out(out_data, x.shape(), rg);
    if (!rg) return out;

    auto node = std::make_shared<autograd::Node>(N);
    node->is_leaf = false;
    auto x_node = x.node();
    if (x_node) node->inputs.push_back(x_node);

    // Backward: dx[i] += g[i] * (1 - tanh(x[i])²)
    // We store tanh(x) as out_data (already computed).
    node->backward_fn = [x_node, out_data, node_ptr = node.get()]() mutable {
        if (!x_node) return;
        const double* __restrict__ g  = node_ptr->grad.data();
        const double* __restrict__ t  = out_data.data();   // tanh values
        double*       __restrict__ dx = x_node->grad.data();
        const std::size_t N2 = node_ptr->grad.size();
        // d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
        // This loop: 2 FMA + 1 mul = 3 FLOP/elem → fully vectorizable
#pragma omp simd
        for (std::size_t i = 0; i < N2; ++i)
            dx[i] += g[i] * (1.0 - t[i] * t[i]);
    };

    autograd::Tensor x_cap = x;
    node->vjp_fn = [x_cap](const std::any& up_any) -> std::any {
        const auto& up = std::any_cast<const autograd::Tensor&>(up_any);
        GradMap result;
        if (!x_cap.node()) return std::any(result);
        autograd::Tensor t     = tanh_act(x_cap);
        autograd::Tensor t_sq  = autograd::mul(t, t);
        autograd::Tensor sech2 = 1.0 - t_sq;
        result[x_cap.node().get()] = autograd::mul(up, sech2);
        return std::any(result);
    };

    out.set_node(std::move(node));
    return out;
}

// ============================================================================
// sigmoid_act  — SIMD-vectorized
// ============================================================================
autograd::Tensor sigmoid_act(const autograd::Tensor& x) {
    const std::size_t N = x.numel();
    std::vector<double> out_data(N);
    const double* __restrict__ xp = x.data().data();
    double*       __restrict__ op = out_data.data();
    // σ(x) = 1/(1+exp(-x)); -ffast-math enables SIMD exp
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i)
        op[i] = 1.0 / (1.0 + std::exp(-xp[i]));

    const bool rg = x.requires_grad();
    autograd::Tensor out(out_data, x.shape(), rg);
    if (!rg) return out;

    auto node = std::make_shared<autograd::Node>(N);
    node->is_leaf = false;
    auto x_node = x.node();
    if (x_node) node->inputs.push_back(x_node);

    // Backward: dx[i] += g[i] * s[i] * (1 - s[i])
    node->backward_fn = [x_node, out_data, node_ptr = node.get()]() mutable {
        if (!x_node) return;
        const double* __restrict__ g  = node_ptr->grad.data();
        const double* __restrict__ s  = out_data.data();   // sigmoid values
        double*       __restrict__ dx = x_node->grad.data();
        const std::size_t N2 = node_ptr->grad.size();
#pragma omp simd
        for (std::size_t i = 0; i < N2; ++i)
            dx[i] += g[i] * s[i] * (1.0 - s[i]);
    };

    autograd::Tensor x_cap = x;
    node->vjp_fn = [x_cap](const std::any& up_any) -> std::any {
        const auto& up = std::any_cast<const autograd::Tensor&>(up_any);
        GradMap result;
        if (!x_cap.node()) return std::any(result);
        autograd::Tensor s      = sigmoid_act(x_cap);
        autograd::Tensor omos   = 1.0 - s;
        autograd::Tensor dsig   = autograd::mul(s, omos);
        result[x_cap.node().get()] = autograd::mul(up, dsig);
        return std::any(result);
    };

    out.set_node(std::move(node));
    return out;
}

autograd::Tensor gelu_act(const autograd::Tensor& x) {
    const double kAlpha = std::sqrt(2.0 / 3.14159265358979323846);
    const double kBeta  = 0.044715;
    const std::size_t N = x.numel();
    std::vector<double> out_data(N);
    const double* __restrict__ xp = x.data().data();
    double*       __restrict__ op = out_data.data();
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i) {
        const double x3 = xp[i] * xp[i] * xp[i];
        const double u  = kAlpha * (xp[i] + kBeta * x3);
        op[i] = 0.5 * xp[i] * (1.0 + std::tanh(u));
    }

    const bool rg = x.requires_grad();
    autograd::Tensor out(out_data, x.shape(), rg);
    if (!rg) return out;

    auto node = std::make_shared<autograd::Node>(N);
    node->is_leaf = false;
    auto x_node = x.node();
    if (x_node) node->inputs.push_back(x_node);

    node->backward_fn = [x_node, x_cap = x, node_ptr = node.get()]() mutable {
        if (!x_node) return;
        const double* __restrict__ g  = node_ptr->grad.data();
        const double* __restrict__ xp = x_cap.data().data();
        double*       __restrict__ dx = x_node->grad.data();
        const std::size_t N2 = node_ptr->grad.size();
        const double a = std::sqrt(2.0 / 3.14159265358979323846);
        const double b = 0.044715;
#pragma omp simd
        for (std::size_t i = 0; i < N2; ++i) {
            const double x = xp[i];
            const double x2 = x * x;
            const double u = a * (x + b * x * x2);
            const double t = std::tanh(u);
            const double sech2 = 1.0 - t * t;
            const double du = a * (1.0 + 3.0 * b * x2);
            const double dgelu = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du;
            dx[i] += g[i] * dgelu;
        }
    };

    node->vjp_fn = [x_cap = x](const std::any& up_any) -> std::any {
        const auto& up = std::any_cast<const autograd::Tensor&>(up_any);
        GradMap result;
        if (!x_cap.node()) return std::any(result);

        const double a = std::sqrt(2.0 / 3.14159265358979323846);
        const double b = 0.044715;
        auto x2 = autograd::mul(x_cap, x_cap);
        auto x3 = autograd::mul(x2, x_cap);
        auto u  = (x_cap + x3 * b) * a;
        auto t  = tanh_act(u);
        auto sech2 = 1.0 - autograd::mul(t, t);
        auto du = (1.0 + x2 * (3.0 * b)) * a;
        auto dgelu = 0.5 * (1.0 + t) + 0.5 * autograd::mul(x_cap, autograd::mul(sech2, du));
        result[x_cap.node().get()] = autograd::mul(up, dgelu);
        return std::any(result);
    };

    out.set_node(std::move(node));
    return out;
}

} // namespace core
