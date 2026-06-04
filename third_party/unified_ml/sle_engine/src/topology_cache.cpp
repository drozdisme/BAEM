#include "sle/topology_cache.hpp"

#include <sstream>
#include <stdexcept>

namespace sle {

std::string topology_signature(const BooleanCascade& cascade) {
    std::ostringstream oss;
    oss << "i" << cascade.input_count() << ";g" << cascade.gate_count();
    for (const auto& gate : cascade.gates()) {
        oss << '|' << gate.a << ',' << gate.b << ',' << gate.c;
    }
    return oss.str();
}

bool CompilationCache::contains(const std::string& signature) const {
    return entries_.find(signature) != entries_.end();
}

void CompilationCache::touch(const std::string& signature) {
    auto it = entries_.find(signature);
    if (it == entries_.end()) return;
    lru_.erase(it->second.lru_it);
    lru_.push_front(signature);
    it->second.lru_it = lru_.begin();
}

void CompilationCache::evict_if_needed() {
    while (entries_.size() > max_entries_ && !lru_.empty()) {
        const auto& victim = lru_.back();
        entries_.erase(victim);
        lru_.pop_back();
    }
}

CachedTopologyProgram CompilationCache::get(const std::string& signature) {
    const auto it = entries_.find(signature);
    if (it == entries_.end()) throw std::out_of_range("topology signature not in cache");
    ++hits_;
    touch(signature);
    return CachedTopologyProgram{signature, it->second.compiled, 1};
}

CachedTopologyProgram CompilationCache::get_or_compile(const BooleanCascade& cascade) {
    const auto signature = topology_signature(cascade);
    const auto it = entries_.find(signature);
    if (it != entries_.end()) {
        ++hits_;
        touch(signature);
        return CachedTopologyProgram{signature, it->second.compiled, 1};
    }

    auto compiled = std::make_shared<CompiledBooleanCascade>(compile_boolean_cascade(cascade, options_));
    lru_.push_front(signature);
    entries_.emplace(signature, CompilationCacheEntry{compiled, lru_.begin()});
    ++misses_;
    evict_if_needed();
    return CachedTopologyProgram{signature, std::move(compiled), 0};
}

} // namespace sle
