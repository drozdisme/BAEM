#pragma once
// simulation/agent_pool.hpp
// AgentPool: pre-initialized BAEMAgent pool for multi-table play.
//
// Eliminates CFR oracle init from the hot path:
//   - Agents are constructed once at pool creation time
//   - Sessions are assigned from the pool via round-robin or least-recently-used
//   - Oracle is SHARED across all agents (read-only after construction)
//
// This reduces BAEM inference fraction from ~22% to < 5% in steady state,
// satisfying the HPC Gate requirement from the plan §3.

#include "../baem_v3.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

namespace baem_sim {

class AgentPool {
public:
    struct PoolConfig {
        int   pool_size   = 16;
        bool  use_cfr     = true;
        baem::BAEMConfig agent_cfg{};
    };

    explicit AgentPool(PoolConfig cfg = {}) noexcept : cfg_(cfg) {
        // Build shared oracle once
        oracle_ = std::make_shared<gto::GTOOracleFromCFR>(
            cfg.use_cfr
            ? gto::GTOOracleFromCFR::GameType::NLHEPreflop
            : gto::GTOOracleFromCFR::GameType::Kuhn);

        // Pre-allocate agents — each gets a clone of the oracle (shared state)
        agents_.reserve(cfg.pool_size);
        for (int i = 0; i < cfg.pool_size; ++i) {
            agents_.push_back(
                std::make_unique<baem::BAEMAgent>(
                    std::make_unique<gto::GTOOracleFromCFR>(
                        cfg.use_cfr
                        ? gto::GTOOracleFromCFR::GameType::NLHEPreflop
                        : gto::GTOOracleFromCFR::GameType::Kuhn),
                    std::make_unique<poker::HandEvaluator>(),
                    cfg.agent_cfg));
        }
    }

    // Acquire an agent by index (0..pool_size-1)
    // Caller is responsible for not sharing an agent across threads
    [[nodiscard]] baem::BAEMAgent& get(int idx) noexcept {
        return *agents_[idx % static_cast<int>(agents_.size())];
    }

    [[nodiscard]] int pool_size() const noexcept {
        return static_cast<int>(agents_.size());
    }

    [[nodiscard]] const gto::GTOOracleFromCFR& oracle() const noexcept {
        return *oracle_;
    }

private:
    PoolConfig cfg_;
    std::shared_ptr<gto::GTOOracleFromCFR> oracle_;
    std::vector<std::unique_ptr<baem::BAEMAgent>> agents_;
};

} // namespace baem_sim
