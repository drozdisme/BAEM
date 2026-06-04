#include "core/optimizers.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace core {

// ============================================================================
// SGD
// ============================================================================
SGD::SGD(std::vector<autograd::Tensor*> params,
   double lr, double momentum, double dampening, double weight_decay, bool decoupled_weight_decay)
  : params_(std::move(params))
  , lr_(lr), momentum_(momentum), dampening_(dampening), weight_decay_(weight_decay)
  , decoupled_weight_decay_(decoupled_weight_decay)
{
  if (lr <= 0.0) throw std::invalid_argument("SGD: lr must be > 0");
  if (momentum < 0.0) throw std::invalid_argument("SGD: momentum must be >= 0");
  velocity_.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i)
    velocity_[i].assign(params_[i]->numel(), 0.0);
}

void SGD::step()
{
#pragma omp parallel for if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    auto* p = params_[i];
    if (!p->requires_grad()) continue;

    const auto& g = p->grad();
    auto   d = p->data();
    const std::size_t N = d.size();

    // Fast path: pure SGD, no momentum, no weight decay    
    // Reduces to d -= lr * g  → pure AXPY, fully vectorizable.
    if (momentum_ == 0.0 && weight_decay_ == 0.0) {
    const double lr = lr_;
    const double* gp  = g.data();
    double*   dp  = d.data();
#pragma omp simd
    for (std::size_t j = 0; j < N; ++j)
      dp[j] -= lr * gp[j];
    continue;
    }

    // Full path: momentum / weight-decay       
    auto& v = velocity_[i];
    for (std::size_t j = 0; j < N; ++j) {
    if (decoupled_weight_decay_ && weight_decay_ > 0.0)
      d[j] -= lr_ * weight_decay_ * d[j];
    double gj = g[j] + ((decoupled_weight_decay_) ? 0.0 : weight_decay_ * d[j]);
    if (momentum_ != 0.0) {
      v[j] = first_step_ ? gj
             : momentum_ * v[j] + (1.0 - dampening_) * gj;
      gj = v[j];
    }
    d[j] -= lr_ * gj;
    }
  }
  first_step_ = false;
}

void SGD::clip_grad_norm(double max_norm)
{
  if (max_norm <= 0.0) return;
  double norm_sq = 0.0;
#pragma omp parallel for reduction(+ : norm_sq) if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const auto* p = params_[i];
    if (!p || !p->requires_grad()) continue;
    for (double gi : p->grad()) norm_sq += gi * gi;
  }
  const double norm = std::sqrt(norm_sq);
  if (norm <= max_norm || norm == 0.0) return;
  const double scale = max_norm / (norm + 1e-12);
  for (auto* p : params_) {
    if (!p || !p->requires_grad()) continue;
    auto* node = p->node().get();
    if (!node) continue;
    for (double& gi : node->grad) gi *= scale;
  }
}

void SGD::zero_grad()
{
  // Standard optimizer semantics: zero_grad only clears accumulated grads.
  // The training loop contract is:
  //   zero_grad() -> forward -> backward -> step()
  for (auto* p : params_) if (p) p->zero_grad();
}

SGDState SGD::save_state() const {
  return SGDState{lr_, momentum_, dampening_, weight_decay_, decoupled_weight_decay_, first_step_, velocity_};
}

void SGD::load_state(const SGDState& state) {
  lr_ = state.lr;
  momentum_ = state.momentum;
  dampening_ = state.dampening;
  weight_decay_ = state.weight_decay;
  decoupled_weight_decay_ = state.decoupled_weight_decay;
  first_step_ = state.first_step;
  velocity_ = state.velocity;
}

// ============================================================================
// Adam
// ============================================================================
Adam::Adam(std::vector<autograd::Tensor*> params,
     double lr, double beta1, double beta2, double eps, double weight_decay, bool decoupled_weight_decay)
  : params_(std::move(params))
  , lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
  , decoupled_weight_decay_(decoupled_weight_decay)
{
  if (lr <= 0.0) throw std::invalid_argument("Adam: lr must be > 0");
  if (beta1 < 0.0 || beta1 >= 1.0) throw std::invalid_argument("Adam: beta1 must be in [0,1)");
  if (beta2 < 0.0 || beta2 >= 1.0) throw std::invalid_argument("Adam: beta2 must be in [0,1)");
  if (eps  <= 0.0) throw std::invalid_argument("Adam: eps must be > 0");
  m_.resize(params_.size());
  v_.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) {
    m_[i].assign(params_[i]->numel(), 0.0);
    v_[i].assign(params_[i]->numel(), 0.0);
  }
}

void Adam::step()
{
  ++t_;
  const double bc1 = 1.0 - std::pow(beta1_, t_);
  const double bc2 = 1.0 - std::pow(beta2_, t_);
  // Standard Adam bias-corrected step size (Kingma & Ba, eq. 3):
  // alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
  // Then the update is:  theta -= alpha_t * m / (sqrt(v) + eps)
  // This avoids mixing bias-correction into both lr and the denominator,
  // which causes eps to be scaled incorrectly in the fused form.
  const double alpha_t = lr_ * std::sqrt(bc2) / bc1;

#pragma omp parallel for if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    auto* p = params_[i];
    if (!p->requires_grad()) continue;

    const auto& g  = p->grad();
    auto   d  = p->data();
    auto&   m  = m_[i];
    auto&   v  = v_[i];
    const std::size_t N = d.size();

    const double b1  = beta1_, b2  = beta2_;
    const double eps = eps_, wd  = weight_decay_;
    const double* gp = g.data();
    double* dp = d.data();
    double* mp = m.data();
    double* vp = v.data();

    // Loop is written for auto-vectorisation: no branches, no function calls
    // inside. Compiler sees: mp/vp/dp updates as independent SIMD lanes.
    for (std::size_t j = 0; j < N; ++j) {
    if (decoupled_weight_decay_ && wd > 0.0)
      dp[j] -= lr_ * wd * dp[j];
    double gj  = gp[j] + (decoupled_weight_decay_ ? 0.0 : wd * dp[j]);
    double mj  = b1 * mp[j] + (1.0 - b1) * gj;
    double vj  = b2 * vp[j] + (1.0 - b2) * gj * gj;
    mp[j]  = mj;
    vp[j]  = vj;
    // Standard Adam: theta -= alpha_t * m / (sqrt(v) + eps)
    dp[j] -= alpha_t * mj / (std::sqrt(vj) + eps);
    }
  }
}

void Adam::clip_grad_norm(double max_norm)
{
  if (max_norm <= 0.0) return;
  double norm_sq = 0.0;
#pragma omp parallel for reduction(+ : norm_sq) if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const auto* p = params_[i];
    if (!p || !p->requires_grad()) continue;
    for (double gi : p->grad()) norm_sq += gi * gi;
  }
  const double norm = std::sqrt(norm_sq);
  if (norm <= max_norm || norm == 0.0) return;
  const double scale = max_norm / (norm + 1e-12);
  for (auto* p : params_) {
    if (!p || !p->requires_grad()) continue;
    auto* node = p->node().get();
    if (!node) continue;
    for (double& gi : node->grad) gi *= scale;
  }
}

void Adam::zero_grad()
{
  // Standard optimizer semantics: zero_grad only clears accumulated grads.
  // The training loop contract is:
  //   zero_grad() -> forward -> backward -> step()
  for (auto* p : params_) if (p) p->zero_grad();
}

AdamState Adam::save_state() const {
  return AdamState{lr_, beta1_, beta2_, eps_, weight_decay_, decoupled_weight_decay_, t_, m_, v_};
}

void Adam::load_state(const AdamState& state) {
  lr_ = state.lr;
  beta1_ = state.beta1;
  beta2_ = state.beta2;
  eps_ = state.eps;
  weight_decay_ = state.weight_decay;
  decoupled_weight_decay_ = state.decoupled_weight_decay;
  t_ = state.timestep;
  m_ = state.m;
  v_ = state.v;
}

RMSProp::RMSProp(std::vector<autograd::Tensor*> params,
                 double lr,
                 double alpha,
                 double eps,
                 double weight_decay,
                 bool decoupled_weight_decay)
  : params_(std::move(params))
  , lr_(lr), alpha_(alpha), eps_(eps), weight_decay_(weight_decay)
  , decoupled_weight_decay_(decoupled_weight_decay)
{
  if (lr <= 0.0) throw std::invalid_argument("RMSProp: lr must be > 0");
  if (alpha < 0.0 || alpha >= 1.0) throw std::invalid_argument("RMSProp: alpha must be in [0,1)");
  if (eps <= 0.0) throw std::invalid_argument("RMSProp: eps must be > 0");
  v_.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i)
    v_[i].assign(params_[i]->numel(), 0.0);
}

void RMSProp::step()
{
  const double alpha = alpha_;
  const double eps = eps_;
  const double wd = weight_decay_;
#pragma omp parallel for if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    auto* p = params_[i];
    if (!p->requires_grad()) continue;

    const auto& g = p->grad();
    auto d = p->data();
    auto& v = v_[i];
    const std::size_t N = d.size();
    const double* gp = g.data();
    double* dp = d.data();
    double* vp = v.data();

    for (std::size_t j = 0; j < N; ++j) {
      if (decoupled_weight_decay_ && wd > 0.0)
        dp[j] -= lr_ * wd * dp[j];
      const double gj = gp[j] + (decoupled_weight_decay_ ? 0.0 : wd * dp[j]);
      const double vj = alpha * vp[j] + (1.0 - alpha) * gj * gj;
      vp[j] = vj;
      dp[j] -= lr_ * gj / (std::sqrt(vj) + eps);
    }
  }
}

void RMSProp::clip_grad_norm(double max_norm)
{
  if (max_norm <= 0.0) return;
  double norm_sq = 0.0;
#pragma omp parallel for reduction(+ : norm_sq) if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const auto* p = params_[i];
    if (!p || !p->requires_grad()) continue;
    for (double gi : p->grad()) norm_sq += gi * gi;
  }
  const double norm = std::sqrt(norm_sq);
  if (norm <= max_norm || norm == 0.0) return;
  const double scale = max_norm / (norm + 1e-12);
  for (auto* p : params_) {
    if (!p || !p->requires_grad()) continue;
    auto* node = p->node().get();
    if (!node) continue;
    for (double& gi : node->grad) gi *= scale;
  }
}

void RMSProp::zero_grad()
{
  for (auto* p : params_) if (p) p->zero_grad();
}

RMSPropState RMSProp::save_state() const {
  return RMSPropState{lr_, alpha_, eps_, weight_decay_, decoupled_weight_decay_, v_};
}

void RMSProp::load_state(const RMSPropState& state) {
  lr_ = state.lr;
  alpha_ = state.alpha;
  eps_ = state.eps;
  weight_decay_ = state.weight_decay;
  decoupled_weight_decay_ = state.decoupled_weight_decay;
  v_ = state.v;
}

NAdam::NAdam(std::vector<autograd::Tensor*> params,
             double lr,
             double beta1,
             double beta2,
             double eps,
             double weight_decay,
             bool decoupled_weight_decay)
  : params_(std::move(params))
  , lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
  , decoupled_weight_decay_(decoupled_weight_decay)
{
  if (lr <= 0.0) throw std::invalid_argument("NAdam: lr must be > 0");
  if (beta1 < 0.0 || beta1 >= 1.0) throw std::invalid_argument("NAdam: beta1 must be in [0,1)");
  if (beta2 < 0.0 || beta2 >= 1.0) throw std::invalid_argument("NAdam: beta2 must be in [0,1)");
  if (eps <= 0.0) throw std::invalid_argument("NAdam: eps must be > 0");

  m_.resize(params_.size());
  v_.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) {
    m_[i].assign(params_[i]->numel(), 0.0);
    v_[i].assign(params_[i]->numel(), 0.0);
  }
}

void NAdam::step()
{
  ++t_;
  const double bc1 = 1.0 - std::pow(beta1_, t_);
  const double bc2 = 1.0 - std::pow(beta2_, t_);
  const double b1_t = std::pow(beta1_, t_);
#pragma omp parallel for if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    auto* p = params_[i];
    if (!p->requires_grad()) continue;

    const auto& g = p->grad();
    auto d = p->data();
    auto& m = m_[i];
    auto& v = v_[i];
    const std::size_t N = d.size();
    const double* gp = g.data();
    double* dp = d.data();
    double* mp = m.data();
    double* vp = v.data();

    for (std::size_t j = 0; j < N; ++j) {
      if (decoupled_weight_decay_ && weight_decay_ > 0.0)
        dp[j] -= lr_ * weight_decay_ * dp[j];

      const double gj = gp[j] + (decoupled_weight_decay_ ? 0.0 : weight_decay_ * dp[j]);
      const double mj = beta1_ * mp[j] + (1.0 - beta1_) * gj;
      const double vj = beta2_ * vp[j] + (1.0 - beta2_) * gj * gj;
      mp[j] = mj;
      vp[j] = vj;

      const double m_hat = mj / bc1;
      const double nesterov = beta1_ * m_hat + ((1.0 - beta1_) * gj) / (1.0 - b1_t);
      const double v_hat = vj / bc2;
      dp[j] -= lr_ * nesterov / (std::sqrt(v_hat) + eps_);
    }
  }
}

void NAdam::clip_grad_norm(double max_norm)
{
  if (max_norm <= 0.0) return;
  double norm_sq = 0.0;
#pragma omp parallel for reduction(+ : norm_sq) if(params_.size() > 4) schedule(static)
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const auto* p = params_[i];
    if (!p || !p->requires_grad()) continue;
    for (double gi : p->grad()) norm_sq += gi * gi;
  }
  const double norm = std::sqrt(norm_sq);
  if (norm <= max_norm || norm == 0.0) return;
  const double scale = max_norm / (norm + 1e-12);
  for (auto* p : params_) {
    if (!p || !p->requires_grad()) continue;
    auto* node = p->node().get();
    if (!node) continue;
    for (double& gi : node->grad) gi *= scale;
  }
}

void NAdam::zero_grad()
{
  for (auto* p : params_) if (p) p->zero_grad();
}

NAdamState NAdam::save_state() const {
  return NAdamState{lr_, beta1_, beta2_, eps_, weight_decay_, decoupled_weight_decay_, t_, m_, v_};
}

void NAdam::load_state(const NAdamState& state) {
  lr_ = state.lr;
  beta1_ = state.beta1;
  beta2_ = state.beta2;
  eps_ = state.eps;
  weight_decay_ = state.weight_decay;
  decoupled_weight_decay_ = state.decoupled_weight_decay;
  t_ = state.timestep;
  m_ = state.m;
  v_ = state.v;
}

CosineLRScheduler::CosineLRScheduler(std::function<void(double)> lr_setter,
                                     double base_lr,
                                     double min_lr,
                                     int max_steps)
  : set_lr_(std::move(lr_setter))
  , base_lr_(base_lr)
  , min_lr_(min_lr)
  , max_steps_(std::max(1, max_steps))
{
  set_lr_(base_lr_);
}

void CosineLRScheduler::step() {
  ++step_;
  const double t = std::min(1.0, static_cast<double>(step_) / static_cast<double>(max_steps_));
  constexpr double kPi = 3.14159265358979323846;
  const double lr = min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + std::cos(kPi * t));
  set_lr_(lr);
}

OneCycleLRScheduler::OneCycleLRScheduler(std::function<void(double)> lr_setter,
                                         double max_lr,
                                         int total_steps,
                                         double pct_start,
                                         double div_factor,
                                         double final_div_factor)
  : set_lr_(std::move(lr_setter))
  , initial_lr_(max_lr / div_factor)
  , max_lr_(max_lr)
  , final_lr_(max_lr / final_div_factor)
  , pct_start_(pct_start)
  , total_steps_(std::max(1, total_steps))
{
  set_lr_(initial_lr_);
}

void OneCycleLRScheduler::step() {
  ++step_;
  const double t = std::min(1.0, static_cast<double>(step_) / static_cast<double>(total_steps_));
  const double up_end = std::clamp(pct_start_, 0.05, 0.95);
  double lr = final_lr_;
  if (t <= up_end) {
    const double u = t / up_end;
    lr = initial_lr_ + u * (max_lr_ - initial_lr_);
  } else {
    const double d = (t - up_end) / (1.0 - up_end);
    lr = max_lr_ - d * (max_lr_ - final_lr_);
  }
  set_lr_(lr);
}

} // namespace core
