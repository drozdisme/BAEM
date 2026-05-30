#pragma once
// baem_policy/entropy_calculator.hpp
// EntropyCalculator: расчёт энтропии для температурного расписания τ(t).
//
// Три режима энтропии (Формула 16):
//   H_belief(B_t)   = -Σ_h B_t(h) ln B_t(h)   — неопределённость диапазона
//   H_action(π)     = -Σ_a π(a) ln π(a)        — энтропия политики
//   H_opp(σ̂_opp)   = -Σ_a σ̂(a) ln σ̂(a)      — предсказуемость оппонента
//
// Все методы: без heap-аллокаций, поддержка AVX-512.
// Используется ActionSampler для вычисления τ(t) = f(H, n*_min, t).

#include <cmath>
#include <span>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>

#ifdef __AVX512F__
#  include <immintrin.h>
#endif

namespace baem {

class EntropyCalculator {
public:
    static constexpr float LOG_EPSILON = 1e-30f;

    // ── H(B_t): энтропия распределения убеждений ──────────────────────────
    // prob_array: массив float[N] (нормирован, Σ=1)
    [[nodiscard]] static float belief_entropy(
        std::span<const float> probs) noexcept
    {
        return entropy_scalar(probs.data(), static_cast<int>(probs.size()));
    }

    // ── H(π): энтропия политики действий (5 элементов) ──────────────────
    [[nodiscard]] static float action_entropy(
        const std::array<float, 5>& policy) noexcept
    {
        return entropy_scalar(policy.data(), 5);
    }

    // ── H_max: максимальная возможная энтропия для N элементов ───────────
    [[nodiscard]] static float max_entropy(int n) noexcept {
        return (n > 1) ? std::log(static_cast<float>(n)) : 0.0f;
    }

    // ── Нормированная энтропия [0,1]: H(X) / H_max ────────────────────────
    [[nodiscard]] static float normalized_entropy(
        std::span<const float> probs) noexcept
    {
        float h_max = max_entropy(static_cast<int>(probs.size()));
        return (h_max > 1e-6f)
            ? belief_entropy(probs) / h_max
            : 0.0f;
    }

    // ── KL(P ‖ Q) с ε-регуляризацией Q ──────────────────────────────────
    [[nodiscard]] static float kl_divergence(
        std::span<const float> p,
        std::span<const float> q,
        float eps_reg = 1e-4f) noexcept
    {
        assert(p.size() == q.size());
        float kl = 0.0f;
        for (std::size_t i = 0; i < p.size(); ++i) {
            float pi = p[i];
            float qi = (1.0f - eps_reg) * q[i] + eps_reg / static_cast<float>(p.size());
            if (pi > LOG_EPSILON && qi > LOG_EPSILON)
                kl += pi * std::log(pi / qi);
        }
        return std::max(0.0f, kl);
    }

    // ── Jensen-Shannon divergence (symmetric, ∈ [0, ln2]) ────────────────
    [[nodiscard]] static float js_divergence(
        std::span<const float> p,
        std::span<const float> q) noexcept
    {
        assert(p.size() == q.size());
        int n = static_cast<int>(p.size());
        // M = (P + Q) / 2
        float h_m = 0.0f;
        float h_p = 0.0f, h_q = 0.0f;
        for (int i = 0; i < n; ++i) {
            float m = 0.5f * (p[i] + q[i]);
            if (m > LOG_EPSILON) h_m -= m * std::log(m);
            if (p[i] > LOG_EPSILON) h_p -= p[i] * std::log(p[i]);
            if (q[i] > LOG_EPSILON) h_q -= q[i] * std::log(q[i]);
        }
        return h_m - 0.5f * (h_p + h_q);
    }

    // ── Effective sample size (ESS) для importance weights ────────────────
    // ESS = (Σw_i)² / Σw_i²  ∈ [1, N]
    [[nodiscard]] static float effective_sample_size(
        std::span<const float> weights) noexcept
    {
        float sum_w = 0.0f, sum_w2 = 0.0f;
        for (float w : weights) { sum_w += w; sum_w2 += w * w; }
        return (sum_w2 > 1e-30f) ? (sum_w * sum_w) / sum_w2 : 0.0f;
    }

    // ── Gini coefficient (неравенство распределения) ──────────────────────
    // 0 = uniform, 1 = degenerate (all mass on one element)
    [[nodiscard]] static float gini(std::span<const float> probs) noexcept {
        int n = static_cast<int>(probs.size());
        if (n <= 1) return 0.0f;
        // Sort ascending
        float sorted[1326]; // max COMBO_COUNT
        int m = std::min(n, 1326);
        for (int i = 0; i < m; ++i) sorted[i] = probs[i];
        std::sort(sorted, sorted + m);
        float sum_abs_diff = 0.0f, sum_total = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum_abs_diff += sorted[i] * static_cast<float>(2*i - m + 1);
            sum_total    += sorted[i];
        }
        return (sum_total > 1e-10f)
            ? sum_abs_diff / (static_cast<float>(m) * sum_total)
            : 0.0f;
    }

private:
    // Scalar entropy using std::log — accurate.
    // For production AVX-512 targets: replace inner log with SVML _mm512_log_ps.
    [[nodiscard]] static float entropy_scalar(const float* p, int n) noexcept {
#ifdef __AVX512F__
        if (n >= 16) return entropy_avx512(p, n);
#endif
        float h = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (p[i] > LOG_EPSILON) h -= p[i] * std::log(p[i]);
        }
        return h;
    }

#ifdef __AVX512F__
    // AVX-512 vectorised entropy over contiguous float array
    [[nodiscard]] static float entropy_avx512(const float* p, int n) noexcept {
        constexpr int VEC = 16;
        __m512 acc = _mm512_setzero_ps();
        __m512 eps_v = _mm512_set1_ps(LOG_EPSILON);

        int vec_end = (n / VEC) * VEC;
        for (int i = 0; i < vec_end; i += VEC) {
            __m512 pi  = _mm512_loadu_ps(p + i);
            // Mask: p > eps
            __mmask16 mask = _mm512_cmp_ps_mask(pi, eps_v, _CMP_GT_OQ);
            // log(p) approximation via scalar fallback for each active lane
            // (AVX-512 has no native log — use scalar for log, vec for multiply)
            alignas(64) float tmp[VEC];
            _mm512_storeu_ps(tmp, pi);
            alignas(64) float log_tmp[VEC];
            for (int j = 0; j < VEC; ++j)
                log_tmp[j] = (mask >> j) & 1 ? std::log(tmp[j]) : 0.0f;
            __m512 lp = _mm512_loadu_ps(log_tmp);
            // acc += p * log(p) (masked)
            __m512 contrib = _mm512_maskz_mul_ps(mask, pi, lp);
            acc = _mm512_add_ps(acc, contrib);
        }
        float h = -_mm512_reduce_add_ps(acc);
        // Scalar tail
        for (int i = vec_end; i < n; ++i)
            if (p[i] > LOG_EPSILON) h -= p[i] * std::log(p[i]);
        return h;
    }
#endif
};

} // namespace baem
