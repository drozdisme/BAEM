// functional.h — Higher-level autograd utilities.
//
//  grad()           Compute the gradient of a scalar output w.r.t. a leaf
//                   tensor, returning the result as a Tensor.  With
//                   create_graph=true the returned Tensor itself has
//                   requires_grad=true, enabling second-order differentiation.
//
//  numerical_grad() Estimate gradients via central finite differences.
//                   Primary use: gradient correctness checking (gradcheck).

#pragma once

#include "autograd/tensor.h"

#include <functional>

namespace autograd {
namespace functional {

/// Compute dL/d(input) where L = output (scalar).
///
/// @param output        A scalar (numel==1) Tensor computed from input.
/// @param input         A leaf Tensor with requires_grad=true.
/// @param retain_graph  Keep the backward graph alive after this call.
/// @param create_graph  Build a new computation graph for the gradient,
///                      enabling second-order differentiation.
///
/// @returns  The gradient as a Tensor with the same shape as input.
///           If create_graph=true, the returned Tensor has requires_grad=true.
Tensor grad(const Tensor& output,
            const Tensor& input,
            bool retain_graph  = false,
            bool create_graph  = false);

/// Estimate gradient via central finite differences:
///   approx_grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
///
/// Useful for verifying analytical gradients.
/// @param f        A function Tensor -> scalar Tensor.
/// @param x        Point at which to evaluate the gradient.
/// @param epsilon  Finite-difference step size.
Tensor numerical_grad(std::function<Tensor(const Tensor&)> f,
                      const Tensor& x,
                      double epsilon = 1e-5);

/// Convenience: relative maximum error between two tensors of the same shape.
/// Returns max(|a[i] - b[i]| / max(1, |a[i]|, |b[i]|)).
double max_rel_error(const Tensor& a, const Tensor& b);

}  // namespace functional
}  // namespace autograd
