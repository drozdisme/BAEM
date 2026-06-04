#include "models/pinn/pinn_model.hpp"

#include "models/pinn/pinn_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace pinn {
namespace {
using Clock = std::chrono::steady_clock;
inline double ms_since(const Clock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
} // namespace

PINNModel::PINNModel(NeuralNetwork& net, const PDE& pde, Adam& optimizer)
  : net_(net), pde_(pde), optimizer_(optimizer)
{
  try_enable_fast_backend();
}

PINNModel::~PINNModel() = default;

void PINNModel::set_collocation_points(std::vector<CollocationPoint> pts) {
  coll_pts_ = std::move(pts);
  cache_dirty_ = true;
}

void PINNModel::set_bc_points(std::vector<BCPoint> pts) {
  bc_pts_ = std::move(pts);
  cache_dirty_ = true;
}

void PINNModel::set_data_points(std::vector<BCPoint> pts) {
  data_pts_ = std::move(pts);
  cache_dirty_ = true;
}

void PINNModel::set_loss_weights(LossWeights w) {
  weights_ = w;
  cache_dirty_ = true;
}

void PINNModel::try_enable_fast_backend() {
  const bool supported_pde = (dynamic_cast<const Poisson1D*>(&pde_) != nullptr);
  const bool supported_arch = (net_.activation() == Activation::Tanh) &&
                              (net_.layer_sizes().size() >= 2) &&
                              (net_.layer_sizes().front() == 1) &&
                              (net_.layer_sizes().back() == 1);
  const auto selection = ucao::engine::select_runtime(ucao::engine::ModelFamily::Pinn);
  if (supported_pde && supported_arch && selection.selected) {
    fast_net_.reset(new PinnFastNet(net_.layer_sizes(), 42u, optimizer_.learning_rate()));
    ucao_engine_ = selection.descriptor;
  }
}

void PINNModel::rebuild_fast_cache_if_needed() {
  if (!fast_net_ || !cache_dirty_) return;

  fast_coll_x_.clear();
  fast_coll_f_.clear();
  fast_bc_x_.clear();
  fast_bc_u_.clear();

  if (coll_pts_.empty()) {
    x_shift_ = 0.0;
    x_scale_ = 1.0;
    cache_dirty_ = false;
    return;
  }

  double xmin = coll_pts_.front().coords[0];
  double xmax = xmin;
  for (const auto& pt : coll_pts_) {
    if (pt.coords.size() != 1) {
      throw std::invalid_argument("PINN fast backend supports only 1D collocation points");
    }
    xmin = std::min(xmin, pt.coords[0]);
    xmax = std::max(xmax, pt.coords[0]);
  }

  x_shift_ = 0.5 * (xmax + xmin);
  x_scale_ = std::max(0.5 * (xmax - xmin), 1e-8);
  const double inv_scale = 1.0 / x_scale_;
  const double d2_scale = x_scale_ * x_scale_;

  const auto* pde1d = dynamic_cast<const Poisson1D*>(&pde_);
  fast_coll_x_.reserve(coll_pts_.size());
  fast_coll_f_.reserve(coll_pts_.size());
  for (const auto& pt : coll_pts_) {
    const double x = pt.coords[0];
    const double xn = (x - x_shift_) * inv_scale;
    fast_coll_x_.push_back(xn);
    fast_coll_f_.push_back(d2_scale * pde1d->source(x));
  }

  fast_bc_x_.reserve(bc_pts_.size());
  fast_bc_u_.reserve(bc_pts_.size());
  for (const auto& bc : bc_pts_) {
    if (bc.coords.size() != 1) {
      throw std::invalid_argument("PINN fast backend supports only 1D BC points");
    }
    fast_bc_x_.push_back((bc.coords[0] - x_shift_) * inv_scale);
    fast_bc_u_.push_back(bc.u_value);
  }

  cache_dirty_ = false;
}

double PINNModel::train_step()
{
  optimizer_.zero_grad();
  profile_ = StepProfile{};

  // Fast-path for 1D Poisson PINN: analytical gradients, cached normalized inputs.
  if (fast_net_) {
    rebuild_fast_cache_if_needed();

    const auto t0 = Clock::now();
    const auto t_loss_start = Clock::now();
    const double n_coll = static_cast<double>(std::max<std::size_t>(1, fast_coll_x_.size()));
    const double n_bc = static_cast<double>(std::max<std::size_t>(1, fast_bc_x_.size()));
    const double balanced_bc = weights_.bc * std::sqrt(n_coll / n_bc);
    profile_.loss_ms += ms_since(t_loss_start);

    const auto t_bw_start = Clock::now();
    last_total_loss_ = fast_net_->train_step_poisson1d(
      fast_coll_x_, fast_coll_f_, fast_bc_x_, fast_bc_u_, weights_.pde, balanced_bc);
    profile_.backward_ms += ms_since(t_bw_start);

    // Cheap scalar diagnostics for comparability (computed from fast backend outputs).
    const auto t_fw_start = Clock::now();
    last_pde_loss_ = 0.0; // Not explicitly tracked in fast core path yet.

    last_bc_loss_ = 0.0;
    if (!bc_pts_.empty()) {
      std::vector<double> pred_bc = fast_net_->predict(fast_bc_x_);
      for (std::size_t i = 0; i < pred_bc.size(); ++i) {
        const double diff = pred_bc[i] - fast_bc_u_[i];
        last_bc_loss_ += diff * diff;
      }
      last_bc_loss_ /= static_cast<double>(bc_pts_.size());
    }
    last_data_loss_ = 0.0;
    profile_.forward_ms += ms_since(t_fw_start);
    profile_.total_ms = ms_since(t0);

    return last_total_loss_;
  }

  const auto t0 = Clock::now();
  const std::size_t input_dim = pde_.input_dim();

  autograd::Tensor pde_loss(0.0, false);
  std::size_t n_residuals = 0;
  {
    const auto t_loss = Clock::now();
    for (const auto& pt : coll_pts_) {
      if (pt.coords.size() != input_dim)
        throw std::invalid_argument("CollocationPoint: coords size != input_dim");
      autograd::Tensor r = pde_.residual(pt.coords, net_);
      autograd::Tensor sq = autograd::mul(r, r);
      if (n_residuals == 0) pde_loss = sq;
      else pde_loss = pde_loss + sq;
      ++n_residuals;
    }
    if (n_residuals > 0)
      pde_loss = pde_loss * (1.0 / static_cast<double>(n_residuals));
    profile_.loss_ms += ms_since(t_loss);
  }
  last_pde_loss_ = (n_residuals == 0) ? 0.0 : pde_loss.item();

  autograd::Tensor bc_loss(0.0, false);
  if (!bc_pts_.empty()) {
    const auto t_fw = Clock::now();
    autograd::Tensor sum_sq(0.0, false);
    bool first = true;
    for (const auto& bc : bc_pts_) {
      autograd::Tensor u_pred;
      if (input_dim == 1) {
        auto d = net_.forward_derivs_1d(bc.coords[0]);
        u_pred = d.u;
      } else {
        auto d = net_.forward_derivs_2d(bc.coords[0], bc.coords[1]);
        u_pred = d.u;
      }
      autograd::Tensor diff = u_pred - bc.u_value;
      autograd::Tensor sq = autograd::mul(diff, diff);
      if (first) { sum_sq = sq; first = false; }
      else sum_sq = sum_sq + sq;
    }
    bc_loss = sum_sq * (1.0 / static_cast<double>(bc_pts_.size()));
    profile_.forward_ms += ms_since(t_fw);
  }
  last_bc_loss_ = bc_pts_.empty() ? 0.0 : bc_loss.item();

  autograd::Tensor data_loss(0.0, false);
  if (!data_pts_.empty()) {
    const auto t_fw = Clock::now();
    autograd::Tensor sum_sq(0.0, false);
    bool first = true;
    for (const auto& dp : data_pts_) {
      autograd::Tensor u_pred;
      if (input_dim == 1) {
        auto d = net_.forward_derivs_1d(dp.coords[0]);
        u_pred = d.u;
      } else {
        auto d = net_.forward_derivs_2d(dp.coords[0], dp.coords[1]);
        u_pred = d.u;
      }
      autograd::Tensor diff = u_pred - dp.u_value;
      autograd::Tensor sq = autograd::mul(diff, diff);
      if (first) { sum_sq = sq; first = false; }
      else sum_sq = sum_sq + sq;
    }
    data_loss = sum_sq * (1.0 / static_cast<double>(data_pts_.size()));
    profile_.forward_ms += ms_since(t_fw);
  }
  last_data_loss_ = data_pts_.empty() ? 0.0 : data_loss.item();

  bool has_pde  = (n_residuals > 0);
  bool has_bc = !bc_pts_.empty();
  bool has_data = !data_pts_.empty();

  if (!has_pde && !has_bc && !has_data) {
    last_total_loss_ = 0.0;
    profile_.total_ms = ms_since(t0);
    return 0.0;
  }

  autograd::Tensor total_loss(0.0, false);
  bool first = true;
  auto accumulate = [&](bool has, autograd::Tensor& term, double weight) {
    if (!has) return;
    autograd::Tensor weighted = term * weight;
    if (first) { total_loss = weighted; first = false; }
    else total_loss = total_loss + weighted;
  };

  const double n_coll = static_cast<double>(std::max<std::size_t>(1, coll_pts_.size()));
  const double n_bc = static_cast<double>(std::max<std::size_t>(1, bc_pts_.size()));
  const double balanced_bc = weights_.bc * std::sqrt(n_coll / n_bc);

  accumulate(has_pde, pde_loss, weights_.pde);
  accumulate(has_bc, bc_loss, balanced_bc);
  accumulate(has_data, data_loss, weights_.data);

  last_total_loss_ = total_loss.item();

  {
    const auto t_bw = Clock::now();
    total_loss.backward();
    optimizer_.clip_grad_norm(1.0);
    optimizer_.step();
    profile_.backward_ms += ms_since(t_bw);
  }

  profile_.total_ms = ms_since(t0);
  return last_total_loss_;
}

std::vector<double> PINNModel::predict(
  const std::vector<std::vector<double>>& points) const
{
  std::vector<double> results;
  results.reserve(points.size());
  const std::size_t input_dim = pde_.input_dim();

  if (fast_net_) {
    std::vector<double> x_flat;
    x_flat.reserve(points.size());
    const double inv_scale = 1.0 / std::max(1e-8, x_scale_);
    for (const auto& coords : points) {
      x_flat.push_back((coords[0] - x_shift_) * inv_scale);
    }
    return fast_net_->predict(x_flat);
  }

  for (const auto& coords : points) {
    if (input_dim == 1) {
      auto d = net_.forward_derivs_1d(coords[0]);
      results.push_back(d.u.item());
    } else {
      auto d = net_.forward_derivs_2d(coords[0], coords[1]);
      results.push_back(d.u.item());
    }
  }
  return results;
}

}  // namespace pinn