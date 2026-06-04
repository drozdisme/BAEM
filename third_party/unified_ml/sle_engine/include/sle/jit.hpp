#pragma once

#include "sle/bit_vector.hpp"
#include "sle/engine.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sle {

enum class JitBackend {
    ScalarGpr,
    Avx512Ternlog,
};

enum class TargetInstructionSet {
    ScalarX86_64,
    Avx512F,
};

struct JitCompileOptions {
    std::size_t word_count = 1;
    JitBackend preferred_backend = JitBackend::Avx512Ternlog;
};

struct GatePatchPoint {
    std::size_t gate_index = 0;
    std::size_t immediate_offset = 0;
};

class ExecutableBuffer {
public:
    ExecutableBuffer() = default;
    explicit ExecutableBuffer(std::size_t size);
    ExecutableBuffer(const ExecutableBuffer&) = delete;
    ExecutableBuffer& operator=(const ExecutableBuffer&) = delete;
    ExecutableBuffer(ExecutableBuffer&& other) noexcept;
    ExecutableBuffer& operator=(ExecutableBuffer&& other) noexcept;
    ~ExecutableBuffer();

    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    void release() noexcept;

    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

class CompiledBooleanCascade {
public:
    CompiledBooleanCascade() = default;
    CompiledBooleanCascade(const CompiledBooleanCascade&) = delete;
    CompiledBooleanCascade& operator=(const CompiledBooleanCascade&) = delete;
    CompiledBooleanCascade(CompiledBooleanCascade&&) noexcept = default;
    CompiledBooleanCascade& operator=(CompiledBooleanCascade&&) noexcept = default;

    [[nodiscard]] BitVector evaluate(const std::vector<BitVector>& inputs) const;
    void run_words(const std::uint64_t* const* inputs, std::uint64_t* workspace, std::size_t words) const;
    [[nodiscard]] std::uint8_t gate_mask(std::size_t gate_index) const;
    void patch_gate_mask(std::size_t gate_index, std::uint8_t new_mask, bool synchronize = true);
    void patch_all_masks(const BooleanCascade& cascade, bool synchronize = true);
    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(buffer_); }
    [[nodiscard]] std::uint8_t* raw_code() noexcept { return buffer_.data(); }
    [[nodiscard]] const std::uint8_t* raw_code() const noexcept { return buffer_.data(); }
    [[nodiscard]] std::size_t code_size() const noexcept { return buffer_.size(); }
    [[nodiscard]] std::size_t input_count() const noexcept { return input_count_; }
    [[nodiscard]] std::size_t gate_count() const noexcept { return gate_count_; }
    [[nodiscard]] std::size_t workspace_word_count() const noexcept { return workspace_word_count_; }
    [[nodiscard]] JitBackend backend() const noexcept { return backend_; }
    [[nodiscard]] const std::vector<GatePatchPoint>& patch_points() const noexcept { return patch_points_; }

private:
    using KernelFn = void (*)(const std::uint64_t* const* inputs, std::uint64_t* workspace, std::size_t words);

    friend CompiledBooleanCascade compile_boolean_cascade(const BooleanCascade& cascade, JitCompileOptions options);
    friend std::shared_ptr<CompiledBooleanCascade> rehydrate_compiled_boolean_cascade(
        std::size_t input_count,
        std::size_t gate_count,
        std::size_t workspace_word_count,
        JitBackend backend,
        std::vector<GatePatchPoint> patch_points,
        const std::vector<std::uint8_t>& code_bytes);

    ExecutableBuffer buffer_;
    KernelFn kernel_ = nullptr;
    std::size_t input_count_ = 0;
    std::size_t gate_count_ = 0;
    std::size_t workspace_word_count_ = 0;
    JitBackend backend_ = JitBackend::ScalarGpr;
    std::vector<GatePatchPoint> patch_points_;
};

[[nodiscard]] bool runtime_supports_jit_backend(JitBackend backend) noexcept;
[[nodiscard]] TargetInstructionSet target_instruction_set_for_backend(JitBackend backend) noexcept;
[[nodiscard]] bool runtime_supports_instruction_set(TargetInstructionSet isa) noexcept;
[[nodiscard]] CompiledBooleanCascade compile_boolean_cascade(const BooleanCascade& cascade,
                                                             JitCompileOptions options = {});
[[nodiscard]] std::shared_ptr<CompiledBooleanCascade> rehydrate_compiled_boolean_cascade(
    std::size_t input_count,
    std::size_t gate_count,
    std::size_t workspace_word_count,
    JitBackend backend,
    std::vector<GatePatchPoint> patch_points,
    const std::vector<std::uint8_t>& code_bytes);

} // namespace sle
