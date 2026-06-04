#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace sle {

class BitVector {
public:
    BitVector() = default;
    explicit BitVector(std::size_t bit_count, bool value = false)
        : bit_count_(bit_count), words_(word_count_for(bit_count), value ? ~std::uint64_t{0} : 0) {
        trim_tail();
    }

    [[nodiscard]] std::size_t size() const noexcept { return bit_count_; }
    [[nodiscard]] bool empty() const noexcept { return bit_count_ == 0; }
    [[nodiscard]] std::size_t word_count() const noexcept { return words_.size(); }

    [[nodiscard]] bool get(std::size_t index) const {
        if (index >= bit_count_) throw std::out_of_range("BitVector::get");
        return (words_[index / 64] >> (index % 64)) & 1ULL;
    }

    void set(std::size_t index, bool value) {
        if (index >= bit_count_) throw std::out_of_range("BitVector::set");
        auto& word = words_[index / 64];
        const auto mask = 1ULL << (index % 64);
        if (value) word |= mask;
        else word &= ~mask;
    }

    void fill(bool value) {
        std::fill(words_.begin(), words_.end(), value ? ~std::uint64_t{0} : 0);
        trim_tail();
    }

    [[nodiscard]] std::size_t popcount() const noexcept {
        std::size_t count = 0;
#if defined(__AVX512F__)
        constexpr std::size_t lanes = 8;
        const std::size_t simd_words = (words_.size() / lanes) * lanes;
        alignas(64) std::uint64_t tmp[lanes];
        for (std::size_t i = 0; i < simd_words; i += lanes) {
            const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(words_.data() + i));
            _mm512_store_si512(reinterpret_cast<void*>(tmp), v);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                count += static_cast<std::size_t>(std::popcount(tmp[lane]));
            }
        }
        for (std::size_t i = simd_words; i < words_.size(); ++i) {
            count += static_cast<std::size_t>(std::popcount(words_[i]));
        }
#else
        for (auto word : words_) count += static_cast<std::size_t>(std::popcount(word));
#endif
        return count;
    }

    [[nodiscard]] double density() const noexcept {
        return bit_count_ == 0 ? 0.0 : static_cast<double>(popcount()) / static_cast<double>(bit_count_);
    }

    [[nodiscard]] std::size_t hamming_distance(const BitVector& other) const {
        ensure_same_size(other);
        std::size_t count = 0;
        for (std::size_t i = 0; i < words_.size(); ++i) {
            count += static_cast<std::size_t>(std::popcount(words_[i] ^ other.words_[i]));
        }
        return count;
    }

    [[nodiscard]] BitVector bit_and(const BitVector& other) const {
        ensure_same_size(other);
        BitVector out(bit_count_);
#if defined(__AVX512F__)
        constexpr std::size_t lanes = 8;
        const std::size_t simd_words = (words_.size() / lanes) * lanes;
        for (std::size_t i = 0; i < simd_words; i += lanes) {
            const __m512i lhs = _mm512_loadu_si512(reinterpret_cast<const void*>(words_.data() + i));
            const __m512i rhs = _mm512_loadu_si512(reinterpret_cast<const void*>(other.words_.data() + i));
            const __m512i res = _mm512_and_si512(lhs, rhs);
            _mm512_storeu_si512(reinterpret_cast<void*>(out.words_.data() + i), res);
        }
        for (std::size_t i = simd_words; i < words_.size(); ++i) out.words_[i] = words_[i] & other.words_[i];
#else
        for (std::size_t i = 0; i < words_.size(); ++i) out.words_[i] = words_[i] & other.words_[i];
#endif
        return out;
    }

    [[nodiscard]] BitVector bit_or(const BitVector& other) const {
        ensure_same_size(other);
        BitVector out(bit_count_);
        for (std::size_t i = 0; i < words_.size(); ++i) out.words_[i] = words_[i] | other.words_[i];
        return out;
    }

    [[nodiscard]] BitVector bit_xor(const BitVector& other) const {
        ensure_same_size(other);
        BitVector out(bit_count_);
#if defined(__AVX512F__)
        constexpr std::size_t lanes = 8;
        const std::size_t simd_words = (words_.size() / lanes) * lanes;
        for (std::size_t i = 0; i < simd_words; i += lanes) {
            const __m512i lhs = _mm512_loadu_si512(reinterpret_cast<const void*>(words_.data() + i));
            const __m512i rhs = _mm512_loadu_si512(reinterpret_cast<const void*>(other.words_.data() + i));
            const __m512i res = _mm512_xor_si512(lhs, rhs);
            _mm512_storeu_si512(reinterpret_cast<void*>(out.words_.data() + i), res);
        }
        for (std::size_t i = simd_words; i < words_.size(); ++i) out.words_[i] = words_[i] ^ other.words_[i];
#else
        for (std::size_t i = 0; i < words_.size(); ++i) out.words_[i] = words_[i] ^ other.words_[i];
#endif
        return out;
    }

    [[nodiscard]] BitVector bit_not() const {
        BitVector out(bit_count_);
        for (std::size_t i = 0; i < words_.size(); ++i) out.words_[i] = ~words_[i];
        out.trim_tail();
        return out;
    }

    [[nodiscard]] const std::vector<std::uint64_t>& words() const noexcept { return words_; }
    [[nodiscard]] std::vector<std::uint64_t>& words() noexcept { return words_; }

    [[nodiscard]] std::string to_string() const {
        std::string out;
        out.reserve(bit_count_);
        for (std::size_t i = 0; i < bit_count_; ++i) out.push_back(get(i) ? '1' : '0');
        return out;
    }

private:
    std::size_t bit_count_ = 0;
    std::vector<std::uint64_t> words_;

    static constexpr std::size_t word_count_for(std::size_t bits) noexcept {
        return (bits + 63U) / 64U;
    }

    void trim_tail() noexcept {
        if (words_.empty() || (bit_count_ % 64U) == 0U) return;
        const auto tail_bits = static_cast<unsigned>(bit_count_ % 64U);
        const auto mask = (tail_bits == 64U) ? ~std::uint64_t{0} : ((1ULL << tail_bits) - 1ULL);
        words_.back() &= mask;
    }

    void ensure_same_size(const BitVector& other) const {
        if (bit_count_ != other.bit_count_) throw std::invalid_argument("BitVector size mismatch");
    }
};

} // namespace sle
