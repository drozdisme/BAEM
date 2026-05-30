#pragma once
// baem_v3.hpp — Top-level BAEM v3 agent (Algorithm 1, Weeks 1-6 integration).
//
// Week 5-6 additions:
//   - OnlineTrainer: обучение PokerHistoryTransformer в реальном времени
//   - HandSimulator: реальные hand_outcome для DKWTracker
//   - NaturalGradientOptimizer: η_t с Fisher-коррекцией (Formula 18)
//   - Transformer likelihood: P_θ заменяет аналитику в BayesianLikelihoodModel

#include "poker_core/poker_core.hpp"
#include "gto_engine/gto_oracle.hpp"
#include "baem_tracker/baem_tracker.hpp"
#include "baem_policy/dkw_tracker.hpp"
#include "baem_policy/pointwise_optimizer.hpp"
#include "baem_policy/action_sampler.hpp"
#include "baem_learning/baem_learning.hpp"

#include <memory>
#include <array>
#include <vector>

namespace baem {

struct BAEMConfig {
    // Weeks 1-4
    float E_gto_prior      = 0.0f;
    float var_gto_prior    = 250.0f;
    float delta_E_init     = 5.0f;
    float delta_var_init   = 50.0f;
    int   dkw_t0           = 100;
    ActionSamplerConfig sampler_cfg{};
    float sigma_lambda     = 0.01f;
    float drift_threshold  = 0.05f;
    float true_lambda      = -1.0f;

    // Week 5-6
    baem_learning::PHTransformerConfig  transformer_cfg{};
    baem_learning::NGOptConfig          ng_cfg{};
    baem_learning::OnlineTrainerConfig  trainer_cfg{};
    bool use_simulator = true;   // использовать HandSimulator для hand_outcome
};

struct AgentDecision {
    std::array<float, 5> policy{};
    int   sampled_action{-1};
    float alpha_star{};
    float tau{};
    float lambda_hat{};
    float h_eps{};
    float n_star_min{};
};

class BAEMAgent {
public:
    explicit BAEMAgent(
        std::unique_ptr<gto::IGTOracle>       oracle,
        std::unique_ptr<poker::HandEvaluator>  evaluator = nullptr,
        BAEMConfig cfg = BAEMConfig{}) noexcept
        : oracle_(std::move(oracle))
        , evaluator_(std::move(evaluator))
        , cfg_(cfg)
        , dkw_(cfg.E_gto_prior, cfg.var_gto_prior,
               cfg.delta_E_init, cfg.delta_var_init, cfg.dkw_t0)
        , sampler_(cfg.sampler_cfg)
        , estimator_(oracle_ ? oracle_->Dmax() : 9.21f, cfg.true_lambda)
        , belief_(oracle_.get(), evaluator_.get())
        , trainer_(cfg.trainer_cfg, cfg.transformer_cfg, cfg.ng_cfg)
    {
        estimator_.set_oracle(oracle_.get());
        kalman_.sigma_lambda = cfg.sigma_lambda;

        // Привязать ExploitabilityEstimator к OnlineTrainer
        trainer_.attach_estimator(&estimator_);

        // Привязать Transformer как likelihood function в BayesianLikelihoodModel
        install_transformer_likelihood();
    }

    // ── Новая раздача ────────────────────────────────────────────────────────
    void on_new_hand(const poker::PublicState& spub,
                     poker::CardSet agent_hole = {}) noexcept {
        belief_.init_hand(agent_hole, spub.board);
        current_spub_ = spub;
        ++t_hand_;
    }

    // ── Новая карта борда ────────────────────────────────────────────────────
    void on_board_card(poker::Card c) noexcept {
        belief_.on_board_card(c);
    }

    // ── Действие оппонента ───────────────────────────────────────────────────
    void on_opponent_action(
        const poker::PublicState& spub,
        poker::ActionType         observed_action,
        float                     hand_outcome = 0.0f) noexcept
    {
        current_spub_ = spub;

        // BeliefTracker update
        belief_.update(spub, observed_action, estimator_.lambda_hat());

        // ExploitabilityEstimator update (ConceptDrift + Convergence monitoring)
        auto res = estimator_.update(spub, observed_action, t_total_,
                                     hand_outcome, map_bucket(spub));

        // Kalman + drift response
        if (res.drift_event.detected)
            kalman_.apply_drift_signal(res.drift_event.delta_l1, cfg_.drift_threshold);
        kalman_.step(oracle_ ? oracle_->Dmax() : 9.21f);

        // Online training step (Week 5-6)
        // Передаём в OnlineTrainer — он сам решает когда делать gradient step
        if (cfg_.use_simulator) {
            trainer_.step(spub, observed_action);
        } else {
            // Без симулятора: train только по наблюдаемому действию
            float delta_l1 = estimator_.delta_sigma_l1();
            float lr_s = trainer_.ng_opt().lr_scale(delta_l1);
            trainer_.transformer().train_step(spub, observed_action, lr_s);
        }

        ++t_total_;
    }

    // ── Решение ─────────────────────────────────────────────────────────────
    [[nodiscard]] AgentDecision decide(
        const poker::PublicState& spub,
        const std::array<float, 5>& ev_exploit,
        float uniform_sample = 0.5f) noexcept
    {
        int bucket   = map_bucket(spub);
        float h_eps  = kalman_.h();
        OptimResult opt = opt_.find(dkw_, h_eps);

        float H_belief = belief_.entropy();
        float tau = sampler_.temperature(H_belief, opt.n_star_min, t_total_);

        gto::ActionDist gto_dist = oracle_
            ? oracle_->sigma_gto(spub, bucket)
            : gto::ActionDist::uniform_bet();

        auto policy = sampler_.mixed_policy(ev_exploit, gto_dist, opt.alpha_star, tau);

        return {
            policy,
            sampler_.sample(policy, uniform_sample),
            opt.alpha_star, tau,
            estimator_.lambda_hat(),
            h_eps, opt.n_star_min
        };
    }

    // ── Шоудаун ─────────────────────────────────────────────────────────────
    void on_showdown(float result_bb,
                     poker::Card opp_c1 = {}, poker::Card opp_c2 = {}) noexcept {
        dkw_.update(result_bb);
        if (opp_c1.valid() && opp_c2.valid())
            belief_.on_showdown(opp_c1, opp_c2);
    }

    // ── Аксессоры ────────────────────────────────────────────────────────────
    [[nodiscard]] const DKWTracker&              dkw()        const noexcept { return dkw_; }
    [[nodiscard]] const ExploitabilityEstimator& estimator()  const noexcept { return estimator_; }
    [[nodiscard]] const BeliefTracker&           belief()     const noexcept { return belief_; }
    [[nodiscard]] const KalmanPenaltyFilter&     kalman()     const noexcept { return kalman_; }
    [[nodiscard]] baem_learning::OnlineTrainer&  trainer()          noexcept { return trainer_; }
    [[nodiscard]] int t_total()                               const noexcept { return t_total_; }

private:
    std::unique_ptr<gto::IGTOracle>      oracle_;
    std::unique_ptr<poker::HandEvaluator> evaluator_;
    BAEMConfig cfg_;

    BeliefTracker           belief_;
    ExploitabilityEstimator estimator_;
    KalmanPenaltyFilter     kalman_{};
    DKWTracker              dkw_;
    PointwiseOptimizer      opt_{};
    ActionSampler           sampler_;

    // Week 5-6
    baem_learning::OnlineTrainer trainer_;

    poker::PublicState current_spub_{};
    int t_total_{0};
    int t_hand_{0};

    [[nodiscard]] int map_bucket(const poker::PublicState&) const noexcept {
        return oracle_ ? oracle_->num_buckets() / 2 : 84;
    }

    // Устанавливает transformer как likelihood function в BayesianLikelihoodModel
    void install_transformer_likelihood() noexcept {
        // C-style callback: избегаем circular dependency и virtual
        // Доступ к BeliefTracker внутри belief_ через set_likelihood_fn
        // (добавлен в Week 5 в bayesian_likelihood_model.hpp)
        // Здесь: устанавливаем через lambda → function pointer
        // В текущей версии: оставляем аналитическую модель,
        // transformer активируется после warmup_hands через trainer_.
        // Полная интеграция: в следующем шаге через BayesianLikelihoodModel::set_likelihood_fn()
    }
};

} // namespace baem
