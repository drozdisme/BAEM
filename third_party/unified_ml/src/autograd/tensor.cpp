// tensor.cpp — Tensor construction, backward pass, scalar-op backward lambdas,
//              shape utilities (reshape / flatten), and new reductions.
//
// The fast-path backward() runs raw-double arithmetic for speed.
// For create_graph backward (higher-order derivatives), see functional.cpp.

#include "autograd/tensor.h"
#include "autograd/functional.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace autograd {

// ════════════════════════════════════════════════════════════════════════════
//  File-local helpers
// ════════════════════════════════════════════════════════════════════════════

namespace {

/// Post-order DFS topological sort on the Node DAG.
void topo_sort(const std::shared_ptr<Node>&       n,
               std::vector<std::shared_ptr<Node>>& order,
               std::unordered_set<Node*>&          visited)
{
    if (!n || visited.count(n.get())) return;
    visited.insert(n.get());
    for (const auto& inp : n->inputs)
        topo_sort(inp, order, visited);
    order.push_back(n);
}

void release_graph(const std::vector<std::shared_ptr<Node>>& order) {
    for (const auto& nd : order) {
        if (!nd || nd->is_leaf) continue;
        nd->inputs.clear();
        nd->backward_fn = nullptr;
        nd->vjp_fn = nullptr;
        nd->grad_tensor_erased.reset();
        std::vector<double>().swap(nd->grad);
    }
}

}  // anonymous namespace

std::vector<std::size_t> Tensor::make_contiguous_strides(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return {};
    std::vector<std::size_t> st(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        st[static_cast<std::size_t>(i)] = st[static_cast<std::size_t>(i + 1)] * shape[static_cast<std::size_t>(i + 1)];
    return st;
}

// ════════════════════════════════════════════════════════════════════════════
//  Constructors
// ════════════════════════════════════════════════════════════════════════════

Tensor::Tensor(std::vector<double>      data,
               std::vector<std::size_t> shape,
               bool                     requires_grad)
    : shape_(std::move(shape))
    , strides_(make_contiguous_strides(shape_))
    , data_(std::make_shared<AlignedBuffer>(std::move(data)))
    , requires_grad_(requires_grad)
{
    std::size_t expected = 1;
    for (std::size_t d : shape_) {
        if (d == 0)
            throw std::invalid_argument("Tensor: shape dimension cannot be 0");
        expected *= d;
    }
    if (data_->size() != expected)
        throw std::invalid_argument(
            "Tensor: data size (" + std::to_string(data_->size()) +
            ") does not match shape product (" + std::to_string(expected) + ")");

    if (requires_grad_) {
        node_          = std::make_shared<Node>(data_->size());
        node_->is_leaf = true;
    }
}

Tensor::Tensor(double value, bool requires_grad)
    : shape_{}
    , data_(std::make_shared<AlignedBuffer>(std::vector<double>{value}))
    , requires_grad_(requires_grad)
{
    if (requires_grad_) {
        node_          = std::make_shared<Node>(1);
        node_->is_leaf = true;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Accessors
// ════════════════════════════════════════════════════════════════════════════

std::size_t Tensor::numel() const noexcept {
    if (shape_.empty()) return 1;
    std::size_t n = 1;
    for (std::size_t d : shape_) n *= d;
    return n;
}

bool Tensor::is_scalar() const noexcept { return shape_.empty(); }

bool Tensor::is_contiguous() const noexcept {
    return strides_ == make_contiguous_strides(shape_);
}

double Tensor::value_flat(std::size_t flat) const {
    return data()[storage_index(flat)];
}

std::size_t Tensor::flat_to_storage(std::size_t flat) const {
    return storage_index(flat);
}

std::size_t Tensor::storage_index(std::size_t flat) const {
    if (shape_.empty()) return offset_;
    std::size_t idx = offset_;
    for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
        const std::size_t dim = static_cast<std::size_t>(d);
        const std::size_t coord = flat % shape_[dim];
        flat /= shape_[dim];
        idx += coord * strides_[dim];
    }
    return idx;
}

double Tensor::item() const {
    if (numel() != 1)
        throw std::runtime_error(
            "item(): tensor has " + std::to_string(numel()) +
            " elements; expected exactly 1");
    return value_flat(0);
}

const std::vector<double>& Tensor::grad() const {
    if (!node_)
        throw std::runtime_error(
            "grad(): called on a tensor that does not require grad");
    return node_->grad;
}

std::shared_ptr<Tensor> Tensor::grad_tensor() const {
    if (!node_ || !node_->grad_tensor_erased)
        return nullptr;
    return std::static_pointer_cast<Tensor>(node_->grad_tensor_erased);
}

void Tensor::zero_grad() {
    if (node_) node_->zero_grad();
}

// ════════════════════════════════════════════════════════════════════════════
//  Shape utilities
// ════════════════════════════════════════════════════════════════════════════

Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
    std::size_t n = 1;
    for (std::size_t d : new_shape) n *= d;
    if (n != numel())
        throw std::invalid_argument(
            "reshape: new shape product (" + std::to_string(n) +
            ") != numel (" + std::to_string(numel()) + ")");

    // Reshape is a view — shares data & node but with new shape.
    // Gradient flows back as-is (reshape is linear with identity gradient).
    if (!is_contiguous())
        throw std::invalid_argument("reshape: only contiguous tensors can be reshaped as a view");

    Tensor out = *this;
    out.shape_ = std::move(new_shape);
    out.strides_ = make_contiguous_strides(out.shape_);
    out.set_requires_grad(false);
    out.set_node(nullptr);
    if (!requires_grad_) return out;

    auto out_node  = std::make_shared<Node>(numel());
    auto self_node = node_;

    out_node->is_leaf = false;
    out_node->inputs.push_back(self_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, self_node]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
#pragma omp simd
        for (std::size_t i = 0; i < out_locked->grad.size(); ++i)
            self_node->grad[i] += out_locked->grad[i];
    };

    // VJP for reshape: gradient must be reshaped back to the INPUT shape
    // before being returned.  Passing gup as-is propagates the wrong shape
    // (the output shape) to any op consuming self_node — e.g. tanh/sigmoid
    // VJPs that call mul(gup, dt) where dt has shape == input shape.
    out_node->vjp_fn = [self_node, in_shape = shape_](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        // Reshape gradient from output shape back to input shape.
        Tensor g_reshaped = gup.reshape(in_shape);
        std::unordered_map<Node*, Tensor> res;
        res.emplace(self_node.get(), g_reshaped);
        return std::make_any<std::unordered_map<Node*, Tensor>>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

Tensor Tensor::flatten() const {
    return reshape({numel()});
}

Tensor Tensor::transpose_view(std::size_t dim0, std::size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim())
        throw std::invalid_argument("transpose_view: dimension out of range");
    Tensor out = *this;
    std::swap(out.shape_[dim0], out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

Tensor Tensor::slice_view(std::size_t dim, std::size_t start, std::size_t end, std::size_t step) const {
    if (dim >= ndim()) throw std::invalid_argument("slice_view: dimension out of range");
    if (step == 0) throw std::invalid_argument("slice_view: step must be > 0");
    if (start > end || end > shape_[dim]) throw std::invalid_argument("slice_view: invalid range");

    Tensor out = *this;
    out.offset_ += start * out.strides_[dim];
    out.shape_[dim] = (end - start + step - 1) / step;
    out.strides_[dim] *= step;
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  Backward (fast path – raw double arithmetic)
// ════════════════════════════════════════════════════════════════════════════

void Tensor::backward(bool retain_graph, bool create_graph) {
    if (!requires_grad_)
        throw std::runtime_error(
            "backward(): called on a tensor with requires_grad=false");
    require_graph_preserving("backward()");
    if (numel() != 1)
        throw std::runtime_error(
            "backward(): only supported for scalar (1-element) output tensors "
            "(numel=" + std::to_string(numel()) + ")");

    if (create_graph) {
        // Delegate to VJP-based backward in functional.cpp via grad().
        // The result (Tensor-valued gradient) is stored on leaf nodes.
        functional::grad(*this, *this,
                         /*retain_graph=*/true,
                         /*create_graph=*/true);
        return;
    }

    //   Fast path                               
    // Step 1: seed d(self)/d(self) = 1
    node_->grad[0] = 1.0;

    // Step 2: topological sort (post-order → leaves first)
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*>          visited;
    topo_sort(node_, order, visited);

    // Step 3: reverse traversal (output → leaves)
    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
        const auto& nd = order[static_cast<std::size_t>(i)];
        if (nd->backward_fn) nd->backward_fn();
    }

    if (!retain_graph) release_graph(order);
}

// ════════════════════════════════════════════════════════════════════════════
//  Tensor × Tensor operators  (delegate to free functions in ops.cpp)
// ════════════════════════════════════════════════════════════════════════════

Tensor Tensor::operator+(const Tensor& rhs) const { return add(*this, rhs); }
Tensor Tensor::operator-(const Tensor& rhs) const { return sub(*this, rhs); }
Tensor Tensor::operator*(const Tensor& rhs) const { return mul(*this, rhs); }

// ════════════════════════════════════════════════════════════════════════════
//  Tensor × scalar operators
// ════════════════════════════════════════════════════════════════════════════

//   operator+(double s)                            
//  Forward:   out[i] = self[i] + s
//  Backward:  grad_self[i] += grad_out[i]   (d/dx(x+s) = 1)

Tensor Tensor::operator+(double s) const {
    std::vector<double> out_data(numel());
    for (std::size_t i = 0; i < numel(); ++i) out_data[i] = value_flat(i) + s;
    Tensor out(std::move(out_data), shape_, false);
    if (!requires_grad_) return out;

    auto out_node  = std::make_shared<Node>(out.numel());
    auto self_node = node_;
    out_node->is_leaf = false;
    out_node->inputs.push_back(self_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, self_node]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
#pragma omp simd
        for (std::size_t i = 0; i < out_locked->grad.size(); ++i)
            self_node->grad[i] += out_locked->grad[i];
    };

    out_node->vjp_fn = [self_node](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        std::unordered_map<Node*, Tensor> res;
        res.emplace(self_node.get(), gup);
        return std::make_any<std::unordered_map<Node*, Tensor>>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

//   operator-(double s)                            

Tensor Tensor::operator-(double s) const { return (*this) + (-s); }

//   operator*(double s)                            
//  Forward:   out[i] = self[i] * s
//  Backward:  grad_self[i] += grad_out[i] * s

Tensor Tensor::operator*(double s) const {
    std::vector<double> out_data(numel());
    for (std::size_t i = 0; i < numel(); ++i) out_data[i] = value_flat(i) * s;
    Tensor out(std::move(out_data), shape_, false);
    if (!requires_grad_) return out;

    auto out_node  = std::make_shared<Node>(out.numel());
    auto self_node = node_;
    out_node->is_leaf = false;
    out_node->inputs.push_back(self_node);

    std::weak_ptr<Node> out_weak = out_node;
    out_node->backward_fn = [out_weak, self_node, s]() {
        auto out_locked = out_weak.lock();
        if (!out_locked) return;
        for (std::size_t i = 0; i < out_locked->grad.size(); ++i)
            self_node->grad[i] += out_locked->grad[i] * s;
    };

    out_node->vjp_fn = [self_node, s](const std::any& g) -> std::any {
        const Tensor& gup = std::any_cast<const Tensor&>(g);
        std::unordered_map<Node*, Tensor> res;
        res.emplace(self_node.get(), gup * s);
        return std::make_any<std::unordered_map<Node*, Tensor>>(std::move(res));
    };

    out.set_node(out_node);
    out.set_requires_grad(true);
    return out;
}

//   operator/(double s)                            

Tensor Tensor::operator/(double s) const {
    if (s == 0.0)
        throw std::invalid_argument("Tensor::operator/: division by zero");
    return (*this) * (1.0 / s);
}

//   operator-()  (unary)                            

Tensor Tensor::operator-() const { return (*this) * (-1.0); }

}  // namespace autograd
