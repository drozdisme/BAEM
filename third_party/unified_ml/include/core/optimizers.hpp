#pragma once
// optimizers.hpp — Unified SGD and Adam optimizers.
//
// Supersedes the near-identical implementations in mlp, deep_onet, and pinn.
// All three model namespaces expose these as type aliases.

#include "autograd/tensor.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace core {

struct SGDState {
    double lr = 0.0;
    double momentum = 0.0;
    double dampening = 0.0;
    double weight_decay = 0.0;
    bool decoupled_weight_decay = false;
    bool first_step = true;
    std::vector<std::vector<double>> velocity;
};

struct AdamState {
    double lr = 0.0;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double eps = 0.0;
    double weight_decay = 0.0;
    bool decoupled_weight_decay = true;
    int timestep = 0;
    std::vector<std::vector<double>> m;
    std::vector<std::vector<double>> v;
};

struct RMSPropState {
    double lr = 0.0;
    double alpha = 0.0;
    double eps = 0.0;
    double weight_decay = 0.0;
    bool decoupled_weight_decay = true;
    std::vector<std::vector<double>> v;
};

struct NAdamState {
    double lr = 0.0;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double eps = 0.0;
    double weight_decay = 0.0;
    bool decoupled_weight_decay = true;
    int timestep = 0;
    std::vector<std::vector<double>> m;
    std::vector<std::vector<double>> v;
};

class SGD {
public:
    SGD(std::vector<autograd::Tensor*> params,
        double lr,
        double momentum     = 0.0,
        double dampening    = 0.0,
        double weight_decay = 0.0,
        bool decoupled_weight_decay = false);

    void step();
    void zero_grad();
    void clip_grad_norm(double max_norm);

    double learning_rate() const noexcept { return lr_; }
    void   set_learning_rate(double lr)   { lr_ = lr; }
    [[nodiscard]] SGDState save_state() const;
    void load_state(const SGDState& state);

private:
    std::vector<autograd::Tensor*>   params_;
    double lr_, momentum_, dampening_, weight_decay_;
    bool decoupled_weight_decay_{false};
    std::vector<std::vector<double>> velocity_;
    bool first_step_{true};
};

class Adam {
public:
    Adam(std::vector<autograd::Tensor*> params,
         double lr           = 1e-3,
         double beta1        = 0.9,
         double beta2        = 0.999,
         double eps          = 1e-8,
         double weight_decay = 0.0,
         bool decoupled_weight_decay = true);

    void step();
    void zero_grad();
    void clip_grad_norm(double max_norm);

    double learning_rate() const noexcept { return lr_; }
    void   set_learning_rate(double lr)   { lr_ = lr; }
    int    timestep()       const noexcept { return t_; }
    [[nodiscard]] AdamState save_state() const;
    void load_state(const AdamState& state);

private:
    std::vector<autograd::Tensor*>   params_;
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    bool   decoupled_weight_decay_{true};
    int    t_{0};
    std::vector<std::vector<double>> m_, v_;
};

class RMSProp {
public:
    RMSProp(std::vector<autograd::Tensor*> params,
            double lr           = 1e-3,
            double alpha        = 0.99,
            double eps          = 1e-8,
            double weight_decay = 0.0,
            bool decoupled_weight_decay = true);

    void step();
    void zero_grad();
    void clip_grad_norm(double max_norm);

    double learning_rate() const noexcept { return lr_; }
    void   set_learning_rate(double lr)   { lr_ = lr; }
    [[nodiscard]] RMSPropState save_state() const;
    void load_state(const RMSPropState& state);

private:
    std::vector<autograd::Tensor*>   params_;
    double lr_, alpha_, eps_, weight_decay_;
    bool decoupled_weight_decay_{true};
    std::vector<std::vector<double>> v_;
};

class NAdam {
public:
    NAdam(std::vector<autograd::Tensor*> params,
          double lr           = 1e-3,
          double beta1        = 0.9,
          double beta2        = 0.999,
          double eps          = 1e-8,
          double weight_decay = 0.0,
          bool decoupled_weight_decay = true);

    void step();
    void zero_grad();
    void clip_grad_norm(double max_norm);

    double learning_rate() const noexcept { return lr_; }
    void   set_learning_rate(double lr)   { lr_ = lr; }
    int    timestep()       const noexcept { return t_; }
    [[nodiscard]] NAdamState save_state() const;
    void load_state(const NAdamState& state);

private:
    std::vector<autograd::Tensor*>   params_;
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    bool   decoupled_weight_decay_{true};
    int    t_{0};
    std::vector<std::vector<double>> m_, v_;
};

class CosineLRScheduler {
public:
    CosineLRScheduler(std::function<void(double)> lr_setter,
                      double base_lr,
                      double min_lr,
                      int max_steps);
    void step();
private:
    std::function<void(double)> set_lr_;
    double base_lr_, min_lr_;
    int max_steps_, step_{0};
};

class OneCycleLRScheduler {
public:
    OneCycleLRScheduler(std::function<void(double)> lr_setter,
                        double max_lr,
                        int total_steps,
                        double pct_start = 0.3,
                        double div_factor = 25.0,
                        double final_div_factor = 1e4);
    void step();
private:
    std::function<void(double)> set_lr_;
    double initial_lr_, max_lr_, final_lr_, pct_start_;
    int total_steps_, step_{0};
};

} // namespace core
