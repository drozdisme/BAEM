// functional.cpp — Higher-level autograd utilities.
//
//  grad()           VJP-based backward that builds a new computation graph,
//                   enabling second-order (and higher) differentiation.
//
//  numerical_grad() Central finite differences for gradient checking.
//
//  max_rel_error()  Relative error metric used in gradcheck.
//
//   How create_graph backward works                      
//
//  Normal backward (create_graph=false):
//    Traverse the Node DAG in reverse topological order.
//    Each backward_fn accumulates raw doubles into node->grad.
//    Cheap and allocation-free.
//
//  VJP backward (create_graph=true):
//    Instead of raw doubles, we propagate *Tensor* objects as gradients.
//    Each node's vjp_fn:
//      • receives the upstream gradient as a Tensor
//      • returns a GradMap: Node* → Tensor (contribution to each input)
//    Because vjp_fn uses Tensor arithmetic (mul, add, pow …), those
//    operations build a NEW computation graph as a side-effect.
//    After the pass, every leaf node's grad_tensor_erased holds a
//    shared_ptr<Tensor> with requires_grad=true.
//    The caller can then call grad() again on that Tensor to get d²f/dx².

#include "autograd/functional.h"
#include "autograd/tensor.h"
#include "autograd/node.h"

#include <any>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace autograd {
namespace functional {

//   Types                                   

using GradMap = std::unordered_map<Node*, Tensor>;

//   Topological sort (same helper used in tensor.cpp)             

namespace {

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

// ════════════════════════════════════════════════════════════════════════════
//  grad()
// ════════════════════════════════════════════════════════════════════════════

Tensor grad(const Tensor& output,
            const Tensor& input,
            bool          retain_graph,
            bool          create_graph)
{
    output.require_graph_preserving("grad(output, input)");
    input.require_graph_preserving("grad(output, input)");
    if (!output.requires_grad())
        throw std::runtime_error("grad(): output does not require grad");
    if (!input.requires_grad())
        throw std::runtime_error("grad(): input does not require grad");
    if (output.numel() != 1)
        throw std::runtime_error("grad(): output must be a scalar (numel==1)");

    //   Topological sort                           
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*>          visited;
    topo_sort(output.node(), order, visited);

    if (!create_graph) {
        //   Fast path (same as Tensor::backward)              
        output.node()->grad[0] = 1.0;
        for (int i = (int)order.size() - 1; i >= 0; --i) {
            const auto& nd = order[i];
            if (nd->backward_fn) nd->backward_fn();
        }
        if (!retain_graph) release_graph(order);
        // Build result Tensor from raw grad.
        return Tensor(input.node()->grad, input.shape(), false);
    }

    //   VJP path (create_graph=true)                     
    //
    //  We propagate Tensor-valued gradients through the graph.
    //  Each node accumulates its gradient in a GradMap keyed by Node*.
    //  After the pass, leaf gradients (type Tensor) are stored on the node.

    // Map from Node* → accumulated Tensor gradient.
    std::unordered_map<Node*, Tensor> tensor_grads;

    // Seed: d(output)/d(output) = 1.
    //
    // When create_graph=true the seed itself must have requires_grad=true so
    // that every Tensor op called inside a vjp_fn (mul, add, reshape, …)
    // records itself in the new computation graph.  Without this, vjp_fn
    // lambdas whose only differentiable dependency is the upstream gradient
    // (e.g. reshape's  gup.reshape(in_shape) ) return dead-graph tensors,
    // silently breaking second-order derivatives.
    //
    // When create_graph=false this code path is unreachable (the fast path
    // above returns early), so passing create_graph here is always true.
    tensor_grads.emplace(output.node().get(), Tensor(1.0, create_graph));

    // Reverse traversal.
    for (int i = (int)order.size() - 1; i >= 0; --i) {
        const auto& nd = order[i];
        auto it = tensor_grads.find(nd.get());
        if (it == tensor_grads.end()) continue;  // unreachable from output

        const Tensor& node_grad = it->second;

        if (!nd->vjp_fn) continue;  // leaf: no VJP to propagate

        // Call the VJP closure.
        std::any result = nd->vjp_fn(std::make_any<const Tensor&>(node_grad));

        // Use the non-throwing pointer form of any_cast so a mismatch
        // (wrong return type from a vjp_fn) produces a clear runtime_error
        // instead of an unhandled std::bad_any_cast or — if the pointer were
        // dereferenced blindly — undefined behaviour.
        auto* contrib_map_ptr = std::any_cast<GradMap>(&result);
        if (!contrib_map_ptr)
            throw std::runtime_error(
                "vjp_fn returned unexpected type; expected GradMap "
                "(unordered_map<Node*, Tensor>)");

        // Accumulate contributions into tensor_grads.
        for (auto& [inp_ptr, contrib] : *contrib_map_ptr) {
            auto jt = tensor_grads.find(inp_ptr);
            if (jt == tensor_grads.end()) {
                tensor_grads.emplace(inp_ptr, contrib);
            } else {
                jt->second = jt->second + contrib;
            }
        }
    }

    // Store Tensor-valued gradients on leaf nodes so callers can access them.
    for (auto& [node_ptr, tg] : tensor_grads) {
        if (node_ptr->is_leaf) {
            // Enable grad on the stored Tensor so 2nd-order works.
            auto gt = std::make_shared<Tensor>(tg.data(), tg.shape(),
                                               create_graph);
            node_ptr->grad_tensor_erased =
                std::static_pointer_cast<void>(gt);
        }
    }

    // Return gradient w.r.t. `input`.
    auto it = tensor_grads.find(input.node().get());
    if (it == tensor_grads.end())
        throw std::runtime_error(
            "grad(): input is not connected to output in the graph");

    Tensor result = it->second;
    // If create_graph, the result should have requires_grad so callers can
    // differentiate through it a second time.
    if (create_graph) result.set_requires_grad(true);
    (void)retain_graph;
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
//  numerical_grad()  — central finite differences
// ════════════════════════════════════════════════════════════════════════════

Tensor numerical_grad(std::function<Tensor(const Tensor&)> f,
                      const Tensor& x,
                      double        epsilon)
{
    const std::size_t N = x.numel();
    std::vector<double> grad_data(N);

    for (std::size_t i = 0; i < N; ++i) {
        // x + eps*e_i
        std::vector<double> xp = x.data();
        xp[i] += epsilon;
        Tensor xp_t(xp, x.shape(), false);
        double fp = f(xp_t).item();

        // x - eps*e_i
        std::vector<double> xm = x.data();
        xm[i] -= epsilon;
        Tensor xm_t(xm, x.shape(), false);
        double fm = f(xm_t).item();

        grad_data[i] = (fp - fm) / (2.0 * epsilon);
    }
    return Tensor(grad_data, x.shape(), false);
}

// ════════════════════════════════════════════════════════════════════════════
//  max_rel_error()
// ════════════════════════════════════════════════════════════════════════════

double max_rel_error(const Tensor& a, const Tensor& b)
{
    if (a.numel() != b.numel())
        throw std::invalid_argument("max_rel_error: tensors have different sizes");
    double max_err = 0.0;
    for (std::size_t i = 0; i < a.numel(); ++i) {
        double ai  = a.data()[i];
        double bi  = b.data()[i];
        double denom = std::max({1.0, std::abs(ai), std::abs(bi)});
        double err   = std::abs(ai - bi) / denom;
        max_err = std::max(max_err, err);
    }
    return max_err;
}

}  // namespace functional
}  // namespace autograd
