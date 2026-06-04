#pragma once
// pinn::activations redirects to core:: implementations which include full VJP support
// required for forward-mode second-order differentiation in PDE residuals.
#include "autograd/autograd.h"
#include "core/activations.hpp"
namespace pinn {
    using core::exp_act;
    using core::tanh_act;
    using core::sigmoid_act;
    using core::relu_act;
} // namespace pinn
