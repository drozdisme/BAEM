// node.h — Computational-graph node for reverse-mode autodiff.
//
// Two backward mechanisms are stored per node:
//
//   backward_fn  – fast path: operates on raw double vectors.
//                  Used by Tensor::backward() (normal mode).
//
//   vjp_fn       – VJP path: operates on Tensor values so that the
//                  backward pass itself builds a new computation graph.
//                  Used by backward(create_graph=true) for higher-order
//                  derivatives.  The Tensor type is forward-declared here
//                  and the function signature is erased through std::any.
//
// Lifetime:
//   Nodes are kept alive by the Tensor::node_ shared_ptr AND by the
//   `inputs` vectors of downstream nodes.  The full sub-graph rooted at
//   the output tensor therefore stays alive for the duration of backward()
//   even if intermediate Tensor objects go out of scope.

#pragma once

#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace autograd {

enum class GraphIntegrity {
    Preserved,
    Bypassed,
};

struct Node {
    //   Fast-path gradient (raw doubles)                  
    // Accumulated dL/d(this), same numel as the owning Tensor.
    std::vector<double> grad;

    //   Graph connectivity                         
    // Input nodes from the forward pass.
    // Deduplicated (see mul / add) — the backward_fn handles repeated inputs
    // through explicit += accumulation.
    std::vector<std::shared_ptr<Node>> inputs;

    //   Fast-path backward closure                     
    // Reads this->grad and adds chain-rule contributions to each input node.
    // Leaf nodes have no backward_fn.
    std::function<void()> backward_fn;

    //   VJP closure (create_graph mode)                  
    // Receives: std::any wrapping const Tensor& (upstream gradient)
    // Returns:  std::any wrapping GradMap = unordered_map<Node*, Tensor>
    //           mapping each input node pointer to its gradient contribution.
    // Using std::any avoids a circular include between node.h and tensor.h.
    std::function<std::any(const std::any&)> vjp_fn;

    //   Type-erased Tensor-valued gradient                 
    // Set by backward(create_graph=true) on leaf nodes.
    // Holds a shared_ptr<Tensor>; erased to shared_ptr<void> to avoid
    // including tensor.h from this header.
    std::shared_ptr<void> grad_tensor_erased;

    //   Metadata                              
    bool is_leaf{true};
    GraphIntegrity graph_integrity{GraphIntegrity::Preserved};
    const char* graph_origin{"autograd"};
    const char* graph_warning{nullptr};

    //   Construction                            
    explicit Node(std::size_t numel) : grad(numel, 0.0) {}

    //   Utilities                              
    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0);
        grad_tensor_erased.reset();
    }

    void mark_graph_bypass(const char* origin, const char* warning) noexcept {
        graph_integrity = GraphIntegrity::Bypassed;
        graph_origin = origin;
        graph_warning = warning;
    }

    void require_graph_integrity(const char* op) const {
        if (graph_integrity == GraphIntegrity::Bypassed) {
            throw std::logic_error(
                std::string(op) + ": graph semantics were bypassed by " +
                (graph_origin ? graph_origin : "unknown") +
                (graph_warning ? std::string(" (") + graph_warning + ")" : std::string()));
        }
    }
};

}  // namespace autograd
