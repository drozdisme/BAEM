#pragma once

#include "core/hugepage_buffer.hpp"
#include "sle/topology_cache.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace core {

using JitCompileOptions = ::sle::JitCompileOptions;
using CachedJitProgram = ::sle::CachedTopologyProgram;

struct JitCacheMetadata {
    std::uint32_t format_version = 1;
    std::uint32_t engine_version = 1;
    std::uint64_t gate_budget = 0;
    double grid_density = 0.0;
    std::string pde_signature;
    std::string sle_signature;
};

class JitCache {
public:
    static JitCache& instance() {
        static JitCache cache;
        return cache;
    }

    [[nodiscard]] CachedJitProgram get_or_compile(const ::sle::BooleanCascade& cascade) {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.get_or_compile(cascade);
    }

    [[nodiscard]] CachedJitProgram get(const std::string& signature) {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.get(signature);
    }

    [[nodiscard]] bool contains(const std::string& signature) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.contains(signature);
    }

    [[nodiscard]] std::string signature(const ::sle::BooleanCascade& cascade) const {
        return ::sle::topology_signature(cascade);
    }

    [[nodiscard]] std::string signature(const ::sle::BooleanCascade& cascade,
                                        std::string_view pde_signature,
                                        double grid_density,
                                        std::uint64_t gate_budget,
                                        std::string_view sle_signature) const {
        return ::sle::topology_signature(cascade) + "|pde:" + std::string(pde_signature) +
               "|grid:" + std::to_string(grid_density) +
               "|budget:" + std::to_string(gate_budget) +
               "|sle:" + std::string(sle_signature);
    }

    void save_to_disk(const std::string& path, const ::sle::BooleanCascade& cascade, const JitCacheMetadata& metadata) const;
    [[nodiscard]] ::sle::BooleanCascade load_from_disk(const std::string& path, JitCacheMetadata* metadata = nullptr) const;
    [[nodiscard]] HugePageBuffer mmap_cache_file(const std::string& path, bool prefer_huge_pages = true) const;

private:
    JitCache() = default;

    mutable std::mutex mutex_;
    ::sle::CompilationCache cache_{};
};

} // namespace core
