#pragma once
//              
//  xoshiro256** — fast, high-quality 64-bit PRNG
//  Reference: David Blackman & Sebastiano Vigna (2019)
//  Period: 2^256 - 1  Statistical quality: excellent (passes PractRand)
//
//  Usage:
//  Xoshiro256 rng(seed);
//  uint64_t  u = rng.next();   // uniform 64-bit
//  double  d = rng.next_double(); // uniform [0,1)
//  uint64_t  b = rng.next_range(n); // uniform [0, n)
//              
#include <cstdint>
#include <array>
#include <cstring>

namespace xgb {

class Xoshiro256 {
public:
  explicit Xoshiro256(uint64_t seed = 0x12345678ABCDEF01ULL) {
    // SplitMix64 initialiser — guarantees non-zero state
    auto splitmix64 = [](uint64_t& x) -> uint64_t {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
    };
    s_[0] = splitmix64(seed);
    s_[1] = splitmix64(seed);
    s_[2] = splitmix64(seed);
    s_[3] = splitmix64(seed);
  }

  // Generate next 64-bit integer
  uint64_t next() noexcept {
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t  = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3]  = rotl(s_[3], 45);
    return result;
  }

  // Uniform double in [0, 1)
  double next_double() noexcept {
    // Use upper 53 bits for mantissa
    return (next() >> 11) * (1.0 / (UINT64_C(1) << 53));
  }

  // Uniform float in [0, 1)
  float next_float() noexcept {
    return static_cast<float>(next_double());
  }

  // Uniform integer in [0, n)  — uses rejection sampling for uniformity
  uint64_t next_range(uint64_t n) noexcept {
    if (n == 0) return 0;
    const uint64_t threshold = (~n + 1) % n;  // 2^64 mod n
    uint64_t r;
    do { r = next(); } while (r < threshold);
    return r % n;
  }

  // 32-bit convenience (upper half)
  uint32_t next32() noexcept {
    return static_cast<uint32_t>(next() >> 32);
  }

  // Expose state for serialisation
  std::array<uint64_t, 4> state() const { return {s_[0], s_[1], s_[2], s_[3]}; }
  void set_state(std::array<uint64_t, 4> st) { s_ = st; }

private:
  std::array<uint64_t, 4> s_{};

  static uint64_t rotl(uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }
};

} // namespace xgb
