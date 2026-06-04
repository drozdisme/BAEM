// tensor.h — Multidimensional array with reverse-mode gradient tracking.
//
// Design decisions
//         
//  • Each Tensor that requires_grad owns a shared_ptr<Node>.
//    Non-differentiable tensors carry node_ == nullptr.
//
//  • Leaf tensors (user-created) have is_leaf=true, no backward_fn.
//    Their grad accumulates in node_->grad.
//
//  • Intermediate tensors (results of ops) have is_leaf=false and a
//    backward_fn that propagates their accumulated grad to each input node.
//    Ops also install a vjp_fn for create_graph backward support.
//
//  • Data is owned by the Tensor (plain std::vector<double>).
//    backward_fn lambdas capture the input Tensors by value so the backward
//    pass remains valid even if originals go out of scope.
//
//  • Broadcasting: full NumPy-style broadcast is supported for add/sub.
//    mul and matmul batch-broadcast their batch dimensions; the inner two
//    dims follow standard matrix-product rules.
//
//  • matmul supports any number of batch dimensions:
//      [..., m, k] @ [..., k, n]  →  [..., m, n]
//    where the batch prefix is broadcast-compatible.

#pragma once

#include "autograd/aligned_buffer.hpp"
#include "autograd/node.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace autograd {

class Tensor {
public:
    //   Constructors                             
    Tensor() = default;

    /// General constructor.
    /// @param data          Flat row-major buffer; size must equal ∏(shape).
    /// @param shape         Dimension extents.  Empty ({}) => scalar.
    /// @param requires_grad Track this tensor in the computational graph?
    Tensor(std::vector<double>      data,
           std::vector<std::size_t> shape,
           bool                     requires_grad = false);

    /// Scalar tensor (shape = {}).
    explicit Tensor(double value, bool requires_grad = false);

    //   Data accessors                            
    VectorCompatProxy               data()  const noexcept { return VectorCompatProxy(data_); }
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    std::size_t                     offset() const noexcept { return offset_; }
    const double*                   data_ptr() const noexcept { return data_->data() + offset_; }
    double*                         data_ptr() noexcept { return data_->data() + offset_; }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t                     ndim()  const noexcept { return shape_.size(); }
    std::size_t                     numel() const noexcept;
    bool                            requires_grad() const noexcept { return requires_grad_; }
    bool                            is_scalar() const noexcept;
    bool                            is_contiguous() const noexcept;
    double                          value_flat(std::size_t flat) const;
    std::size_t                     flat_to_storage(std::size_t flat) const;
    double                          item() const;

    //   Gradient accessors                           
    /// Accumulated gradient after backward(). Throws if !requires_grad().
    const std::vector<double>& grad() const;

    /// The computational-graph node; nullptr iff !requires_grad().
    std::shared_ptr<Node> node() const noexcept { return node_; }

    /// True when this tensor participates in a graph-preserving differentiable path.
    bool graph_preserving() const noexcept {
        return !node_ || node_->graph_integrity == GraphIntegrity::Preserved;
    }

    /// Throws if this tensor was materialized through a graph-bypassing path.
    void require_graph_preserving(const char* op) const {
        if (node_) node_->require_graph_integrity(op);
    }

    /// Tensor-valued gradient set by backward(create_graph=true).
    /// Returns nullptr if not yet computed via create_graph.
    std::shared_ptr<Tensor> grad_tensor() const;

    //   Autograd                               
    /// Run reverse-mode autodiff from this scalar tensor.
    ///
    /// @param retain_graph  Keep backward closures alive after the pass so
    ///                      backward() can be called again (remember to
    ///                      zero_grad() leaf tensors first to avoid
    ///                      gradient accumulation across passes).
    ///
    /// @param create_graph  Build a new computation graph during the backward
    ///                      pass.  After this call, grad_tensor() on each leaf
    ///                      returns a Tensor with requires_grad=true, enabling
    ///                      second-order differentiation.
    void backward(bool retain_graph = false, bool create_graph = false);

    /// Reset accumulated gradient to zero on this tensor's node.
    void zero_grad();

    //   Shape utilities                            
    /// Reshape to new_shape.  Total elements must match.
    Tensor reshape(std::vector<std::size_t> new_shape) const;

    /// Flatten to a 1-D tensor.
    Tensor flatten() const;
    Tensor transpose_view(std::size_t dim0, std::size_t dim1) const;
    Tensor slice_view(std::size_t dim, std::size_t start, std::size_t end, std::size_t step = 1) const;

    //   Arithmetic operators                          
    // Tensor × Tensor  (broadcast-compatible shapes for +/-; same shape for *)
    Tensor operator+(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;

    // Tensor × scalar
    Tensor operator+(double s) const;
    Tensor operator-(double s) const;
    Tensor operator*(double s) const;
    Tensor operator/(double s) const;

    // Unary negation
    Tensor operator-() const;

    //   Internal helpers (used by ops.cpp / tensor.cpp)            
    void set_node(std::shared_ptr<Node> n) noexcept { node_ = std::move(n); }
    void set_requires_grad(bool v)         noexcept { requires_grad_ = v; }

private:
    std::size_t storage_index(std::size_t flat) const;
    static std::vector<std::size_t> make_contiguous_strides(const std::vector<std::size_t>& shape);

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t offset_{0};
    SharedAlignedBuffer       data_{std::make_shared<AlignedBuffer>()};
    bool                     requires_grad_{false};
    std::shared_ptr<Node>    node_;          // null iff !requires_grad_
};

//   Free-function differentiable operations                  
// Primary implementations live in ops.cpp.
//
// Differentiable module contract:
//   - Modules that expose trainable parameters must compose their forward pass
//     from graph-preserving Tensor operations only.
//   - Materializing intermediate buffers into raw doubles and then constructing
//     a fresh Tensor with requires_grad=true is NOT graph-preserving.
//   - A module that cannot preserve graph semantics must either stay outside the
//     stable differentiable surface or throw explicitly when gradients are requested.

/// Element-wise addition with NumPy-style broadcasting.
Tensor add(const Tensor& a, const Tensor& b);

/// Element-wise subtraction with NumPy-style broadcasting.
Tensor sub(const Tensor& a, const Tensor& b);

/// Element-wise multiplication.  Shapes must be identical.
Tensor mul(const Tensor& a, const Tensor& b);

/// Batched matrix multiplication.
///
/// Supports any number of leading batch dimensions:
///   [*, m, k] @ [*, k, n]  →  [*, m, n]
/// where the batch prefix (*) is broadcast-compatible (NumPy rules).
/// Special cases:
///   1-D [k] treated as a row vector [1, k];  extra dim removed from output.
///   0-D (scalar) is rejected.
///
/// Has both a fast-path backward_fn (raw doubles) and a full VJP closure
/// (Tensor arithmetic) so second-order derivatives are supported.
Tensor matmul(const Tensor& a, const Tensor& b);

/// Transpose: reverse the last two dimensions.
///
///   2-D: [m, n]       → [n, m]
///   N-D: [..., m, n]  → [..., n, m]   (batch dims unchanged)
///
/// This is a differentiable view-like op; gradient is another transpose.
Tensor transpose(const Tensor& a);

/// Reduce all elements to a scalar sum.
Tensor sum(const Tensor& a);

/// Reduce all elements to a scalar mean.
Tensor mean(const Tensor& a);

/// Element-wise ReLU: max(0, x).
Tensor relu(const Tensor& a);

/// Element-wise power: x^exp  (only differentiable at x != 0 for non-integer exp).
Tensor pow(const Tensor& a, double exp);

/// Element-wise exp.
Tensor exp(const Tensor& a);

/// Element-wise log.
Tensor log(const Tensor& a);

/// Element-wise sqrt.
Tensor sqrt(const Tensor& a);

/// Softmax along the last dimension. For 1-D tensors, normalizes the vector.
/// For 2-D+ tensors, applies independently to each row of the flattened prefix.
Tensor softmax(const Tensor& a);

/// Layer normalization over the last dimension.
/// Returns gamma * (x - mean) / sqrt(var + eps) + beta.
Tensor layer_norm(const Tensor& x,
                  const Tensor& gamma,
                  const Tensor& beta,
                  double eps = 1e-5);

//   Scalar-on-left convenience wrappers                    
inline Tensor operator+(double s, const Tensor& t) { return t + s; }
inline Tensor operator*(double s, const Tensor& t) { return t * s; }
inline Tensor operator-(double s, const Tensor& t) { return (-t) + s; }

}  // namespace autograd
