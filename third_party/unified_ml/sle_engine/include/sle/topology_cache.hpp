#pragma once

#include "sle/engine.hpp"
#include "sle/jit.hpp"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>

namespace sle {

struct CachedTopologyProgram {
    std::string signature;
    std::shared_ptr<CompiledBooleanCascade> compiled;
    std::size_t hits = 0;
};

struct CompilationCacheEntry {
    std::shared_ptr<CompiledBooleanCascade> compiled;
    std::list<std::string>::iterator lru_it;
};

[[nodiscard]] std::string topology_signature(const BooleanCascade& cascade);

class CompilationCache {
public:
    explicit CompilationCache(JitCompileOptions options = {}, std::size_t max_entries = 256)
        : options_(options), max_entries_(max_entries) {}

    [[nodiscard]] bool contains(const std::string& signature) const;
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::size_t misses() const noexcept { return misses_; }
    [[nodiscard]] std::size_t max_entries() const noexcept { return max_entries_; }

    [[nodiscard]] CachedTopologyProgram get_or_compile(const BooleanCascade& cascade);
    [[nodiscard]] CachedTopologyProgram get(const std::string& signature);

private:
    void touch(const std::string& signature);
    void evict_if_needed();

    std::unordered_map<std::string, CompilationCacheEntry> entries_;
    std::list<std::string> lru_;
    JitCompileOptions options_{};
    std::size_t max_entries_ = 256;
    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
};

} // namespace sle
