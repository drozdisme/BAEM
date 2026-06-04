#include "core/jit_cache.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace core {
namespace {

struct alignas(64) CacheHeader {
    std::uint32_t magic = 0x554D4A43; // UMJC
    std::uint32_t format_version = 1;
    std::uint32_t engine_version = 1;
    std::uint32_t reserved = 0;
    std::uint64_t gate_budget = 0;
    double grid_density = 0.0;
    std::uint64_t pde_sig_size = 0;
    std::uint64_t sle_sig_size = 0;
    std::uint64_t input_count = 0;
    std::uint64_t gate_count = 0;
};

} // namespace

void JitCache::save_to_disk(const std::string& path,
                            const ::sle::BooleanCascade& cascade,
                            const JitCacheMetadata& metadata) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("JitCache::save_to_disk: failed to open file");

    CacheHeader header;
    header.format_version = metadata.format_version;
    header.engine_version = metadata.engine_version;
    header.gate_budget = metadata.gate_budget;
    header.grid_density = metadata.grid_density;
    header.pde_sig_size = metadata.pde_signature.size();
    header.sle_sig_size = metadata.sle_signature.size();
    header.input_count = cascade.input_count();
    header.gate_count = cascade.gate_count();

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(metadata.pde_signature.data(), static_cast<std::streamsize>(metadata.pde_signature.size()));
    out.write(metadata.sle_signature.data(), static_cast<std::streamsize>(metadata.sle_signature.size()));

    for (const auto& gate : cascade.gates()) {
        out.write(reinterpret_cast<const char*>(&gate.a), sizeof(gate.a));
        out.write(reinterpret_cast<const char*>(&gate.b), sizeof(gate.b));
        out.write(reinterpret_cast<const char*>(&gate.c), sizeof(gate.c));
        out.write(reinterpret_cast<const char*>(&gate.mask), sizeof(gate.mask));
    }
}

::sle::BooleanCascade JitCache::load_from_disk(const std::string& path, JitCacheMetadata* metadata) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("JitCache::load_from_disk: failed to open file");

    CacheHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != 0x554D4A43) {
        throw std::runtime_error("JitCache::load_from_disk: invalid cache header");
    }

    std::string pde_sig(header.pde_sig_size, '\0');
    std::string sle_sig(header.sle_sig_size, '\0');
    in.read(pde_sig.data(), static_cast<std::streamsize>(pde_sig.size()));
    in.read(sle_sig.data(), static_cast<std::streamsize>(sle_sig.size()));

    if (metadata) {
        metadata->format_version = header.format_version;
        metadata->engine_version = header.engine_version;
        metadata->gate_budget = header.gate_budget;
        metadata->grid_density = header.grid_density;
        metadata->pde_signature = std::move(pde_sig);
        metadata->sle_signature = std::move(sle_sig);
    }

    ::sle::BooleanCascade cascade(header.input_count);
    for (std::uint64_t i = 0; i < header.gate_count; ++i) {
        ::sle::TernaryGate gate{};
        in.read(reinterpret_cast<char*>(&gate.a), sizeof(gate.a));
        in.read(reinterpret_cast<char*>(&gate.b), sizeof(gate.b));
        in.read(reinterpret_cast<char*>(&gate.c), sizeof(gate.c));
        in.read(reinterpret_cast<char*>(&gate.mask), sizeof(gate.mask));
        cascade.add_gate(gate);
    }
    return cascade;
}

HugePageBuffer JitCache::mmap_cache_file(const std::string& path, bool prefer_huge_pages) const {
#if defined(__linux__)
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("JitCache::mmap_cache_file: open failed");
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("JitCache::mmap_cache_file: fstat failed");
    }
    HugePageBuffer buffer(static_cast<std::size_t>(st.st_size), prefer_huge_pages);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        close(fd);
        throw std::runtime_error("JitCache::mmap_cache_file: input stream failed");
    }
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    close(fd);
    return buffer;
#else
    (void)prefer_huge_pages;
    throw std::runtime_error("JitCache::mmap_cache_file: mmap backend is only implemented on Linux");
#endif
}

} // namespace core
