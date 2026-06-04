// pinn_model.h — High-level Physics-Informed Neural Network model.
//
// PINNModel wraps a NeuralNetwork and a PDE, and orchestrates:
//   1. Collocation point sampling
//   2. Boundary condition enforcement
//   3. PDE residual computation
//   4. Full loss assembly: L = λ_pde * L_pde + λ_bc * L_bc + λ_data * L_data
//   5. Backward pass and optimizer step
//
// Usage pattern:
//   PINNModel model(net, pde, optimizer);
//   model.set_collocation_points(x_coll);
//   model.set_bc_points(x_bc, u_bc);
//   for (int epoch = 0; epoch < N; ++epoch) {
//       double loss = model.train_step();
//   }

#pragma once

#include "models/pinn/neural_network.hpp"
#include "models/pinn/optimizers.hpp"
#include "models/pinn/pde.hpp"
#include "models/pinn/loss.hpp"
#include "ucao/engine_policy.hpp"

#include <memory>
#include <vector>

namespace pinn {

class PinnFastNet;

struct CollocationPoint {
    std::vector<double> coords;
};

struct BCPoint {
    std::vector<double> coords;
    double              u_value;
};

struct LossWeights {
    double pde  = 1.0;
    double bc   = 10.0;
    double data = 1.0;
};

class PINNModel {
public:
    PINNModel(NeuralNetwork& net, const PDE& pde, Adam& optimizer);
    ~PINNModel();

    void set_collocation_points(std::vector<CollocationPoint> pts);
    void set_bc_points(std::vector<BCPoint> pts);
    void set_data_points(std::vector<BCPoint> pts);
    void set_loss_weights(LossWeights w);
    const LossWeights& loss_weights() const noexcept { return weights_; }

    double train_step();

    std::vector<double> predict(const std::vector<std::vector<double>>& points) const;

    double last_pde_loss()   const noexcept { return last_pde_loss_;  }
    double last_bc_loss()    const noexcept { return last_bc_loss_;   }
    double last_data_loss()  const noexcept { return last_data_loss_; }
    double last_total_loss() const noexcept { return last_total_loss_;}


    struct StepProfile {
        double forward_ms  = 0.0;
        double loss_ms     = 0.0;
        double backward_ms = 0.0;
        double total_ms    = 0.0;
    };
    const StepProfile& last_step_profile() const noexcept { return profile_; }
    bool using_fast_backend() const noexcept { return static_cast<bool>(fast_net_); }
    bool using_ucao_engine() const noexcept { return ucao::engine::select_runtime(model_family()).selected && ucao_engine_.enabled; }
    ucao::engine::EngineDescriptor engine_descriptor() const noexcept { return ucao_engine_; }
    ucao::engine::SelectionResult engine_selection() const noexcept { return ucao::engine::select_runtime(model_family()); }
    static constexpr ucao::engine::ModelFamily model_family() noexcept { return ucao::engine::ModelFamily::Pinn; }

private:
    NeuralNetwork&                net_;
    const PDE&                    pde_;
    Adam&                         optimizer_;
    LossWeights                   weights_;
    std::vector<CollocationPoint> coll_pts_;
    std::vector<BCPoint>          bc_pts_;
    std::vector<BCPoint>          data_pts_;

    double last_pde_loss_  {0.0};
    double last_bc_loss_   {0.0};
    double last_data_loss_ {0.0};
    double last_total_loss_{0.0};


    void try_enable_fast_backend();
    void rebuild_fast_cache_if_needed();

    std::unique_ptr<PinnFastNet> fast_net_;
    ucao::engine::EngineDescriptor ucao_engine_{};
    std::vector<double>          fast_coll_x_;
    std::vector<double>          fast_coll_f_;
    std::vector<double>          fast_bc_x_;
    std::vector<double>          fast_bc_u_;
    double                       x_shift_{0.0};
    double                       x_scale_{1.0};
    bool                         cache_dirty_{true};

    StepProfile profile_;
};

}  // namespace pinn