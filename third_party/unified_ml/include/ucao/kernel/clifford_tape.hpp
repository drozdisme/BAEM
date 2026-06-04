#pragma once

#include "ucao/kernel/blade_kernel.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace ucao::kernel {

class TLSArena {
    struct Block {
        static constexpr std::size_t BLOCK_SZ = 1u << 19;
        alignas(64) char data[BLOCK_SZ];
        std::size_t used = 0;
        Block* next = nullptr;

        [[nodiscard]] void* try_alloc(std::size_t sz, std::size_t align) noexcept {
            const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(data);
            const std::uintptr_t ptr = base + used;
            const std::uintptr_t aligned = (ptr + align - 1) & ~(static_cast<std::uintptr_t>(align) - 1u);
            const std::size_t new_used = static_cast<std::size_t>(aligned - base + sz);
            if (new_used > BLOCK_SZ) {
                return nullptr;
            }
            used = new_used;
            return reinterpret_cast<void*>(aligned);
        }
    };

    Block* head_ = nullptr;
    Block* curr_ = nullptr;

public:
    TLSArena() = default;

    ~TLSArena() {
        Block* node = head_;
        while (node != nullptr) {
            Block* next = node->next;
            delete node;
            node = next;
        }
    }

    TLSArena(const TLSArena&) = delete;

    [[nodiscard]] void* alloc(std::size_t sz, std::size_t align = 64) {
        if (curr_ == nullptr) {
            head_ = new Block{};
            curr_ = head_;
        }
        if (void* p = curr_->try_alloc(sz, align); p != nullptr) {
            return p;
        }
        if (curr_->next == nullptr) {
            curr_->next = new Block{};
        }
        curr_ = curr_->next;
        return curr_->try_alloc(sz, align);
    }

    void reset() noexcept {
        for (Block* node = head_; node != nullptr; node = node->next) {
            node->used = 0;
        }
        curr_ = head_;
    }
};

struct alignas(64) TLSArenaSlot {
    TLSArena arena;
    char pad[(64 - (sizeof(TLSArena) % 64)) % 64]{};
};

inline thread_local TLSArenaSlot tls_arena_slot;
inline thread_local TLSArena& tls_arena = tls_arena_slot.arena;

struct TapeNode {
    void (*bwd)(TapeNode*) noexcept = nullptr;
    TapeNode* prev = nullptr;
};

class TLSTape {
public:
    TapeNode* tail = nullptr;

    template <typename T, typename... Args>
    [[nodiscard]] T* record(Args&&... args) {
        void* mem = tls_arena.alloc(sizeof(T), alignof(T));
        T* node = new (mem) T(std::forward<Args>(args)...);
        node->prev = tail;
        tail = node;
        return node;
    }

    void backward() noexcept {
        for (TapeNode* node = tail; node != nullptr; node = node->prev) {
            if (node->bwd != nullptr) {
                node->bwd(node);
            }
        }
    }

    void reset() noexcept {
        tail = nullptr;
        tls_arena.reset();
    }
};

struct alignas(64) TLSTapeSlot {
    TLSTape tape;
    char pad[(64 - (sizeof(TLSTape) % 64)) % 64]{};
};

inline thread_local TLSTapeSlot tls_tape_slot;
inline thread_local TLSTape& tls_tape = tls_tape_slot.tape;

namespace detail {

template <int D>
[[nodiscard]] constexpr int infer_n() noexcept {
    if constexpr (D == 2) return 1;
    if constexpr (D == 4) return 2;
    if constexpr (D == 8) return 3;
    if constexpr (D == 16) return 4;
    return 5;
}

template <int N>
[[nodiscard]] constexpr int rev_sign_tape(int blade) noexcept {
    int g = 0;
    for (int i = 0; i < N; ++i) {
        g += (blade >> i) & 1;
    }
    return ((g * (g - 1) / 2) & 1) ? -1 : 1;
}

template <int D>
inline void gp_dispatch(float* out, const float* a, const float* b) noexcept {
    constexpr int N = infer_n<D>();
    if constexpr (D == 2) {
        gp_single<1, 1, 0>(out, a, b);
    } else if constexpr (D == 4) {
        gp_single<2, 2, 0>(out, a, b);
    } else if constexpr (D == 8) {
        gp_single<3, 3, 0>(out, a, b);
    } else if constexpr (D == 16) {
        gp_avx512<4, 3, 1>(out, a, b);
    } else {
        gp_single<N, N, 0>(out, a, b);
    }
}

} // namespace detail

template <int D>
struct GPTapeNode : TapeNode {
    const float* A;
    float* gA;
    const float* B;
    float* gB;
    const float* gC;

    explicit GPTapeNode(const float* A_, float* gA_,
                        const float* B_, float* gB_,
                        const float* gC_) noexcept
        : A(A_), gA(gA_), B(B_), gB(gB_), gC(gC_) {
        this->bwd = &GPTapeNode::bwd_impl;
    }

private:
    static void bwd_impl(TapeNode* base) noexcept {
        auto* self = static_cast<GPTapeNode*>(base);
        constexpr int N = detail::infer_n<D>();
        alignas(64) float rev_A[D]{};
        alignas(64) float rev_B[D]{};
        alignas(64) float tmp[D]{};
        for (int i = 0; i < D; ++i) {
            const float s = static_cast<float>(detail::rev_sign_tape<N>(i));
            rev_B[i] = s * self->B[i];
            rev_A[i] = s * self->A[i];
        }
        detail::gp_dispatch<D>(tmp, self->gC, rev_B);
        for (int i = 0; i < D; ++i) {
            self->gA[i] += tmp[i];
        }
        detail::gp_dispatch<D>(tmp, rev_A, self->gC);
        for (int i = 0; i < D; ++i) {
            self->gB[i] += tmp[i];
        }
    }
};

template <int D, int MaxThreads = 32>
struct GradientReducer {
    static constexpr int CacheLineFloats = 16;
    static constexpr int Stride = ((D + CacheLineFloats - 1) / CacheLineFloats) * CacheLineFloats;
    alignas(64) float local[MaxThreads][Stride]{};

    [[nodiscard]] float* thread_slot(int tid) noexcept { return local[tid]; }
    [[nodiscard]] const float* thread_slot(int tid) const noexcept { return local[tid]; }

    void reduce_into(float* out, int n_threads) noexcept {
        for (int t = 0; t < n_threads; ++t) {
            for (int i = 0; i < D; ++i) {
                out[i] += local[t][i];
            }
        }
    }
};

} // namespace ucao::kernel
