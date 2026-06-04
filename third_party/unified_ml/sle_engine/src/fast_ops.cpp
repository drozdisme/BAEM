#include "sle/fast_ops.hpp"
#include "sle/ternary.hpp"

namespace sle {

namespace {

#if SLE_HAS_AVX512
template <std::uint8_t Mask>
[[gnu::target("avx512f,avx512vl")]]
BitVector ternary_apply_avx512_impl(const sle::BitVector& a,
                                    const sle::BitVector& b,
                                    const sle::BitVector& c) {
    sle::BitVector out(a.size());
    constexpr std::size_t lanes_per_reg = 8;
    const auto blocks = a.word_count() / lanes_per_reg;
    const auto* aw = a.words().data();
    const auto* bw = b.words().data();
    const auto* cw = c.words().data();
    auto* ow = out.words().data();
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto* pa = reinterpret_cast<const long long*>(aw + block * lanes_per_reg);
        const auto* pb = reinterpret_cast<const long long*>(bw + block * lanes_per_reg);
        const auto* pc = reinterpret_cast<const long long*>(cw + block * lanes_per_reg);
        auto* po = reinterpret_cast<long long*>(ow + block * lanes_per_reg);
        const __m512i va = _mm512_loadu_si512(pa);
        const __m512i vb = _mm512_loadu_si512(pb);
        const __m512i vc = _mm512_loadu_si512(pc);
        const __m512i vr = _mm512_ternarylogic_epi64(va, vb, vc, Mask);
        _mm512_storeu_si512(po, vr);
    }
    for (std::size_t i = blocks * lanes_per_reg; i < a.word_count(); ++i) {
        std::uint64_t result = 0;
        for (unsigned bit = 0; bit < 64U; ++bit) {
            const bool abit = ((aw[i] >> bit) & 1ULL) != 0ULL;
            const bool bbit = ((bw[i] >> bit) & 1ULL) != 0ULL;
            const bool cbit = ((cw[i] >> bit) & 1ULL) != 0ULL;
            if (sle::ternary_truth(Mask, abit, bbit, cbit)) result |= (1ULL << bit);
        }
        ow[i] = result;
    }
    return out;
}

#define SLE_TERNARY_CASE(IMM) case IMM: return ternary_apply_avx512_impl<static_cast<std::uint8_t>(IMM)>(a, b, c)
#endif

} // namespace

BitVector ternary_apply_fast(const BitVector& a,
                             const BitVector& b,
                             const BitVector& c,
                             std::uint8_t mask) {
    if (a.size() != b.size() || a.size() != c.size()) {
        throw std::invalid_argument("ternary_apply_fast size mismatch");
    }

#if SLE_HAS_AVX512
    if (simd::runtime_has_avx512()) {
        switch (mask) {
            SLE_TERNARY_CASE(0x00); SLE_TERNARY_CASE(0x01); SLE_TERNARY_CASE(0x02); SLE_TERNARY_CASE(0x03);
            SLE_TERNARY_CASE(0x04); SLE_TERNARY_CASE(0x05); SLE_TERNARY_CASE(0x06); SLE_TERNARY_CASE(0x07);
            SLE_TERNARY_CASE(0x08); SLE_TERNARY_CASE(0x09); SLE_TERNARY_CASE(0x0A); SLE_TERNARY_CASE(0x0B);
            SLE_TERNARY_CASE(0x0C); SLE_TERNARY_CASE(0x0D); SLE_TERNARY_CASE(0x0E); SLE_TERNARY_CASE(0x0F);
            SLE_TERNARY_CASE(0x10); SLE_TERNARY_CASE(0x11); SLE_TERNARY_CASE(0x12); SLE_TERNARY_CASE(0x13);
            SLE_TERNARY_CASE(0x14); SLE_TERNARY_CASE(0x15); SLE_TERNARY_CASE(0x16); SLE_TERNARY_CASE(0x17);
            SLE_TERNARY_CASE(0x18); SLE_TERNARY_CASE(0x19); SLE_TERNARY_CASE(0x1A); SLE_TERNARY_CASE(0x1B);
            SLE_TERNARY_CASE(0x1C); SLE_TERNARY_CASE(0x1D); SLE_TERNARY_CASE(0x1E); SLE_TERNARY_CASE(0x1F);
            SLE_TERNARY_CASE(0x20); SLE_TERNARY_CASE(0x21); SLE_TERNARY_CASE(0x22); SLE_TERNARY_CASE(0x23);
            SLE_TERNARY_CASE(0x24); SLE_TERNARY_CASE(0x25); SLE_TERNARY_CASE(0x26); SLE_TERNARY_CASE(0x27);
            SLE_TERNARY_CASE(0x28); SLE_TERNARY_CASE(0x29); SLE_TERNARY_CASE(0x2A); SLE_TERNARY_CASE(0x2B);
            SLE_TERNARY_CASE(0x2C); SLE_TERNARY_CASE(0x2D); SLE_TERNARY_CASE(0x2E); SLE_TERNARY_CASE(0x2F);
            SLE_TERNARY_CASE(0x30); SLE_TERNARY_CASE(0x31); SLE_TERNARY_CASE(0x32); SLE_TERNARY_CASE(0x33);
            SLE_TERNARY_CASE(0x34); SLE_TERNARY_CASE(0x35); SLE_TERNARY_CASE(0x36); SLE_TERNARY_CASE(0x37);
            SLE_TERNARY_CASE(0x38); SLE_TERNARY_CASE(0x39); SLE_TERNARY_CASE(0x3A); SLE_TERNARY_CASE(0x3B);
            SLE_TERNARY_CASE(0x3C); SLE_TERNARY_CASE(0x3D); SLE_TERNARY_CASE(0x3E); SLE_TERNARY_CASE(0x3F);
            SLE_TERNARY_CASE(0x40); SLE_TERNARY_CASE(0x41); SLE_TERNARY_CASE(0x42); SLE_TERNARY_CASE(0x43);
            SLE_TERNARY_CASE(0x44); SLE_TERNARY_CASE(0x45); SLE_TERNARY_CASE(0x46); SLE_TERNARY_CASE(0x47);
            SLE_TERNARY_CASE(0x48); SLE_TERNARY_CASE(0x49); SLE_TERNARY_CASE(0x4A); SLE_TERNARY_CASE(0x4B);
            SLE_TERNARY_CASE(0x4C); SLE_TERNARY_CASE(0x4D); SLE_TERNARY_CASE(0x4E); SLE_TERNARY_CASE(0x4F);
            SLE_TERNARY_CASE(0x50); SLE_TERNARY_CASE(0x51); SLE_TERNARY_CASE(0x52); SLE_TERNARY_CASE(0x53);
            SLE_TERNARY_CASE(0x54); SLE_TERNARY_CASE(0x55); SLE_TERNARY_CASE(0x56); SLE_TERNARY_CASE(0x57);
            SLE_TERNARY_CASE(0x58); SLE_TERNARY_CASE(0x59); SLE_TERNARY_CASE(0x5A); SLE_TERNARY_CASE(0x5B);
            SLE_TERNARY_CASE(0x5C); SLE_TERNARY_CASE(0x5D); SLE_TERNARY_CASE(0x5E); SLE_TERNARY_CASE(0x5F);
            SLE_TERNARY_CASE(0x60); SLE_TERNARY_CASE(0x61); SLE_TERNARY_CASE(0x62); SLE_TERNARY_CASE(0x63);
            SLE_TERNARY_CASE(0x64); SLE_TERNARY_CASE(0x65); SLE_TERNARY_CASE(0x66); SLE_TERNARY_CASE(0x67);
            SLE_TERNARY_CASE(0x68); SLE_TERNARY_CASE(0x69); SLE_TERNARY_CASE(0x6A); SLE_TERNARY_CASE(0x6B);
            SLE_TERNARY_CASE(0x6C); SLE_TERNARY_CASE(0x6D); SLE_TERNARY_CASE(0x6E); SLE_TERNARY_CASE(0x6F);
            SLE_TERNARY_CASE(0x70); SLE_TERNARY_CASE(0x71); SLE_TERNARY_CASE(0x72); SLE_TERNARY_CASE(0x73);
            SLE_TERNARY_CASE(0x74); SLE_TERNARY_CASE(0x75); SLE_TERNARY_CASE(0x76); SLE_TERNARY_CASE(0x77);
            SLE_TERNARY_CASE(0x78); SLE_TERNARY_CASE(0x79); SLE_TERNARY_CASE(0x7A); SLE_TERNARY_CASE(0x7B);
            SLE_TERNARY_CASE(0x7C); SLE_TERNARY_CASE(0x7D); SLE_TERNARY_CASE(0x7E); SLE_TERNARY_CASE(0x7F);
            SLE_TERNARY_CASE(0x80); SLE_TERNARY_CASE(0x81); SLE_TERNARY_CASE(0x82); SLE_TERNARY_CASE(0x83);
            SLE_TERNARY_CASE(0x84); SLE_TERNARY_CASE(0x85); SLE_TERNARY_CASE(0x86); SLE_TERNARY_CASE(0x87);
            SLE_TERNARY_CASE(0x88); SLE_TERNARY_CASE(0x89); SLE_TERNARY_CASE(0x8A); SLE_TERNARY_CASE(0x8B);
            SLE_TERNARY_CASE(0x8C); SLE_TERNARY_CASE(0x8D); SLE_TERNARY_CASE(0x8E); SLE_TERNARY_CASE(0x8F);
            SLE_TERNARY_CASE(0x90); SLE_TERNARY_CASE(0x91); SLE_TERNARY_CASE(0x92); SLE_TERNARY_CASE(0x93);
            SLE_TERNARY_CASE(0x94); SLE_TERNARY_CASE(0x95); SLE_TERNARY_CASE(0x96); SLE_TERNARY_CASE(0x97);
            SLE_TERNARY_CASE(0x98); SLE_TERNARY_CASE(0x99); SLE_TERNARY_CASE(0x9A); SLE_TERNARY_CASE(0x9B);
            SLE_TERNARY_CASE(0x9C); SLE_TERNARY_CASE(0x9D); SLE_TERNARY_CASE(0x9E); SLE_TERNARY_CASE(0x9F);
            SLE_TERNARY_CASE(0xA0); SLE_TERNARY_CASE(0xA1); SLE_TERNARY_CASE(0xA2); SLE_TERNARY_CASE(0xA3);
            SLE_TERNARY_CASE(0xA4); SLE_TERNARY_CASE(0xA5); SLE_TERNARY_CASE(0xA6); SLE_TERNARY_CASE(0xA7);
            SLE_TERNARY_CASE(0xA8); SLE_TERNARY_CASE(0xA9); SLE_TERNARY_CASE(0xAA); SLE_TERNARY_CASE(0xAB);
            SLE_TERNARY_CASE(0xAC); SLE_TERNARY_CASE(0xAD); SLE_TERNARY_CASE(0xAE); SLE_TERNARY_CASE(0xAF);
            SLE_TERNARY_CASE(0xB0); SLE_TERNARY_CASE(0xB1); SLE_TERNARY_CASE(0xB2); SLE_TERNARY_CASE(0xB3);
            SLE_TERNARY_CASE(0xB4); SLE_TERNARY_CASE(0xB5); SLE_TERNARY_CASE(0xB6); SLE_TERNARY_CASE(0xB7);
            SLE_TERNARY_CASE(0xB8); SLE_TERNARY_CASE(0xB9); SLE_TERNARY_CASE(0xBA); SLE_TERNARY_CASE(0xBB);
            SLE_TERNARY_CASE(0xBC); SLE_TERNARY_CASE(0xBD); SLE_TERNARY_CASE(0xBE); SLE_TERNARY_CASE(0xBF);
            SLE_TERNARY_CASE(0xC0); SLE_TERNARY_CASE(0xC1); SLE_TERNARY_CASE(0xC2); SLE_TERNARY_CASE(0xC3);
            SLE_TERNARY_CASE(0xC4); SLE_TERNARY_CASE(0xC5); SLE_TERNARY_CASE(0xC6); SLE_TERNARY_CASE(0xC7);
            SLE_TERNARY_CASE(0xC8); SLE_TERNARY_CASE(0xC9); SLE_TERNARY_CASE(0xCA); SLE_TERNARY_CASE(0xCB);
            SLE_TERNARY_CASE(0xCC); SLE_TERNARY_CASE(0xCD); SLE_TERNARY_CASE(0xCE); SLE_TERNARY_CASE(0xCF);
            SLE_TERNARY_CASE(0xD0); SLE_TERNARY_CASE(0xD1); SLE_TERNARY_CASE(0xD2); SLE_TERNARY_CASE(0xD3);
            SLE_TERNARY_CASE(0xD4); SLE_TERNARY_CASE(0xD5); SLE_TERNARY_CASE(0xD6); SLE_TERNARY_CASE(0xD7);
            SLE_TERNARY_CASE(0xD8); SLE_TERNARY_CASE(0xD9); SLE_TERNARY_CASE(0xDA); SLE_TERNARY_CASE(0xDB);
            SLE_TERNARY_CASE(0xDC); SLE_TERNARY_CASE(0xDD); SLE_TERNARY_CASE(0xDE); SLE_TERNARY_CASE(0xDF);
            SLE_TERNARY_CASE(0xE0); SLE_TERNARY_CASE(0xE1); SLE_TERNARY_CASE(0xE2); SLE_TERNARY_CASE(0xE3);
            SLE_TERNARY_CASE(0xE4); SLE_TERNARY_CASE(0xE5); SLE_TERNARY_CASE(0xE6); SLE_TERNARY_CASE(0xE7);
            SLE_TERNARY_CASE(0xE8); SLE_TERNARY_CASE(0xE9); SLE_TERNARY_CASE(0xEA); SLE_TERNARY_CASE(0xEB);
            SLE_TERNARY_CASE(0xEC); SLE_TERNARY_CASE(0xED); SLE_TERNARY_CASE(0xEE); SLE_TERNARY_CASE(0xEF);
            SLE_TERNARY_CASE(0xF0); SLE_TERNARY_CASE(0xF1); SLE_TERNARY_CASE(0xF2); SLE_TERNARY_CASE(0xF3);
            SLE_TERNARY_CASE(0xF4); SLE_TERNARY_CASE(0xF5); SLE_TERNARY_CASE(0xF6); SLE_TERNARY_CASE(0xF7);
            SLE_TERNARY_CASE(0xF8); SLE_TERNARY_CASE(0xF9); SLE_TERNARY_CASE(0xFA); SLE_TERNARY_CASE(0xFB);
            SLE_TERNARY_CASE(0xFC); SLE_TERNARY_CASE(0xFD); SLE_TERNARY_CASE(0xFE); SLE_TERNARY_CASE(0xFF);
        }
    }
#endif
    return ternary_apply(a, b, c, mask);
}

std::array<std::uint8_t, 256> build_byte_lut(const std::array<std::uint8_t, 8>& rows) noexcept {
    std::array<std::uint8_t, 256> lut{};
    for (unsigned value = 0; value < 256U; ++value) {
        std::uint8_t out = 0;
        for (std::uint8_t row = 0; row < 8; ++row) {
            const auto bits = static_cast<std::uint8_t>(rows[row] & static_cast<std::uint8_t>(value));
            const auto parity = static_cast<std::uint8_t>(std::popcount(bits) & 0x1U);
            out |= static_cast<std::uint8_t>(parity << row);
        }
        lut[value] = out;
    }
    return lut;
}

BitVector apply_byte_map_fast(const BitVector& input,
                              const std::array<std::uint8_t, 256>& lut) {
    if ((input.size() % 8U) != 0U) throw std::invalid_argument("apply_byte_map_fast requires bit length divisible by 8");
    BitVector out(input.size());
    for (std::size_t byte_index = 0; byte_index < input.size() / 8U; ++byte_index) {
        std::uint8_t value = 0;
        for (std::uint8_t bit = 0; bit < 8; ++bit) {
            if (input.get(byte_index * 8U + bit)) value |= static_cast<std::uint8_t>(1U << bit);
        }
        const auto mapped = lut[value];
        for (std::uint8_t bit = 0; bit < 8; ++bit) {
            out.set(byte_index * 8U + bit, ((mapped >> bit) & 0x1U) != 0U);
        }
    }
    return out;
}

} // namespace sle
