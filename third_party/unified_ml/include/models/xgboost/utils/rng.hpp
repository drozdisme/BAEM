#pragma once
//              
//  RNG utilities — xoshiro256** + std::mt19937 wrappers
//  P4-6: Replaces the LCG in GradientBooster with deterministic, high-quality RNG
//              
#include <cstdint>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

namespace xgb {

// xoshiro256** (Blackman & Vigna, 2019)       
// Period: 2^256 - 1, passes all statistical tests, very fast
class Xoshiro256 {
public:
  using result_type = uint64_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }

  explicit Xoshiro256(uint64_t seed = 0x123456789ABCDEF0ULL) {
    // SplitMix64 seeding — ensures non-zero state
    uint64_t s = seed;
    for (int i = 0; i < 4; ++i) {
    s += 0x9e3779b97f4a7c15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    state_[i] = z ^ (z >> 31);
    }
  }

  result_type operator()() {
    const uint64_t result = rotl(state_[1] * 5, 7) * 9;
    const uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
  }

  // Jump: equivalent to 2^128 calls (for parallel streams)
  void jump() {
    static const uint64_t JUMP[] = {
    0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
    0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; ++i)
    for (int b = 0; b < 64; ++b) {
      if (JUMP[i] & (1ULL << b)) {
        s0 ^= state_[0]; s1 ^= state_[1];
        s2 ^= state_[2]; s3 ^= state_[3];
      }
      (*this)();
    }
    state_[0] = s0; state_[1] = s1;
    state_[2] = s2; state_[3] = s3;
  }

  // Uniform float in [0, 1)
  float uniform_float() {
    return static_cast<float>((*this)() >> 40) * (1.f / (1ULL << 24));
  }

  // Uniform int in [0, n)
  uint64_t uniform_int(uint64_t n) {
    return (*this)() % n;
  }

private:
  uint64_t state_[4];
  static uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
};

// Convenience: Fisher-Yates shuffle using xoshiro256**    
template<typename T>
inline void shuffle_indices(std::vector<T>& v, Xoshiro256& rng) {
  for (std::size_t i = v.size(); i > 1; --i) {
    std::size_t j = static_cast<std::size_t>(rng.uniform_int(i));
    std::swap(v[i - 1], v[j]);
  }
}

// Subsample using xoshiro256**        
inline std::vector<uint32_t> subsample(uint32_t n_total, float rate, Xoshiro256& rng) {
  std::vector<uint32_t> idx(n_total);
  std::iota(idx.begin(), idx.end(), 0u);
  if (rate >= 1.0f) return idx;
  uint32_t n_keep = std::max(1u, static_cast<uint32_t>(n_total * rate));
  shuffle_indices(idx, rng);
  idx.resize(n_keep);
  return idx;
}

} // namespace xgb
