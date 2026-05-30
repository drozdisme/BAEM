#pragma once
// baem_policy/action_sampler.hpp
// Softmax action sampling with dynamic temperature τ(t).

#include "dkw_tracker.hpp"
#include "../gto_engine/gto_oracle.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace baem {

struct ActionSamplerConfig {
    float tau_min  = 0.02f;
    float tau_max  = 1.0f;
    float gamma    = 1.0f;
    float eta      = 0.5f;
};

class ActionSampler {
public:
    using Config = ActionSamplerConfig;
    static constexpr int NUM_ACTIONS = 5;

    explicit ActionSampler(Config cfg = ActionSamplerConfig{}) noexcept : cfg_(cfg) {}

    [[nodiscard]] float temperature(
        float H_belief, float n_star_min, int t) const noexcept
    {
        if (n_star_min < 1.0f) n_star_min = 1.0f;
        float decay  = std::exp(-cfg_.gamma * static_cast<float>(t) / n_star_min);
        float H_term = std::pow(std::max(H_belief, 1e-6f), cfg_.eta);
        return cfg_.tau_min + (cfg_.tau_max - cfg_.tau_min) * decay * H_term;
    }

    [[nodiscard]] std::array<float, NUM_ACTIONS> mixed_policy(
        const std::array<float, NUM_ACTIONS>& ev_exploit,
        const gto::ActionDist& gto_dist,
        float alpha_star,
        float tau) const noexcept
    {
        float max_ev = *std::max_element(ev_exploit.begin(), ev_exploit.end());
        float softmax[NUM_ACTIONS];
        float sum_sm = 0.0f;
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            softmax[a] = std::exp((ev_exploit[a] - max_ev) / std::max(tau, 1e-6f));
            sum_sm += softmax[a];
        }
        float inv_sum = 1.0f / sum_sm;

        std::array<float, NUM_ACTIONS> policy{};
        float one_minus_alpha = 1.0f - alpha_star;
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            float pi_gto     = gto_dist.p[a] * 1e-4f;
            float pi_exploit = softmax[a] * inv_sum;
            policy[a] = one_minus_alpha * pi_gto + alpha_star * pi_exploit;
        }
        return policy;
    }

    [[nodiscard]] int sample(
        const std::array<float, NUM_ACTIONS>& policy,
        float uniform_sample) const noexcept
    {
        float cdf = 0.0f;
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            cdf += policy[a];
            if (uniform_sample <= cdf) return a;
        }
        return NUM_ACTIONS - 1;
    }

    [[nodiscard]] int argmax(
        const std::array<float, NUM_ACTIONS>& policy) const noexcept
    {
        return static_cast<int>(
            std::max_element(policy.begin(), policy.end()) - policy.begin());
    }

private:
    Config cfg_;
};

} // namespace baem
