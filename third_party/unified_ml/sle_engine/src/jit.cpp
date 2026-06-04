#include "sle/jit.hpp"

#include "sle/simd.hpp"
#include "sle/ternary.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <utility>
#include <vector>

namespace sle {
namespace {

constexpr std::uint8_t kRax = 0;
constexpr std::uint8_t kRcx = 1;
constexpr std::uint8_t kRdx = 2;
constexpr std::uint8_t kRbx = 3;
constexpr std::uint8_t kRsp = 4;
constexpr std::uint8_t kRbp = 5;
constexpr std::uint8_t kRsi = 6;
constexpr std::uint8_t kRdi = 7;
constexpr std::uint8_t kR8 = 8;
constexpr std::uint8_t kR9 = 9;
constexpr std::uint8_t kR10 = 10;
constexpr std::uint8_t kR11 = 11;
constexpr std::uint8_t kR12 = 12;
constexpr std::uint8_t kR13 = 13;
constexpr std::uint8_t kR14 = 14;
constexpr std::uint8_t kR15 = 15;

struct CodeEmitter {
    std::vector<std::uint8_t> bytes;

    void emit8(std::uint8_t value) { bytes.push_back(value); }
    void emit32(std::uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) emit8(static_cast<std::uint8_t>((value >> (i * 8U)) & 0xFFU));
    }
    void emit64(std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) emit8(static_cast<std::uint8_t>((value >> (i * 8U)) & 0xFFU));
    }

    void rex(bool w, std::uint8_t reg, std::uint8_t rm) {
        emit8(static_cast<std::uint8_t>(0x40U |
                                        (w ? 0x08U : 0x00U) |
                                        (((reg >> 3U) & 0x1U) << 2U) |
                                        ((rm >> 3U) & 0x1U)));
    }

    void modrm(std::uint8_t mod, std::uint8_t reg, std::uint8_t rm) {
        emit8(static_cast<std::uint8_t>((mod << 6U) | ((reg & 0x7U) << 3U) | (rm & 0x7U)));
    }

    void vex3(std::uint8_t r, std::uint8_t x, std::uint8_t b, std::uint8_t mmmmm, std::uint8_t w, std::uint8_t vvvv, std::uint8_t l, std::uint8_t pp) {
        emit8(0xC4U);
        emit8(static_cast<std::uint8_t>(((r & 0x1U) << 7U) | ((x & 0x1U) << 6U) | ((b & 0x1U) << 5U) | (mmmmm & 0x1FU)));
        emit8(static_cast<std::uint8_t>(((w & 0x1U) << 7U) | ((vvvv & 0xFU) << 3U) | ((l & 0x3U) << 2U) | (pp & 0x3U)));
    }

    void push(std::uint8_t reg) {
        if (reg < 8U) emit8(static_cast<std::uint8_t>(0x50U + reg));
        else {
            emit8(0x41U);
            emit8(static_cast<std::uint8_t>(0x50U + (reg & 0x7U)));
        }
    }

    void pop(std::uint8_t reg) {
        if (reg < 8U) emit8(static_cast<std::uint8_t>(0x58U + reg));
        else {
            emit8(0x41U);
            emit8(static_cast<std::uint8_t>(0x58U + (reg & 0x7U)));
        }
    }

    void mov_rr(std::uint8_t dst, std::uint8_t src) {
        rex(true, src, dst);
        emit8(0x89U);
        modrm(0x3U, src, dst);
    }

    void mov_ri(std::uint8_t dst, std::uint64_t imm) {
        emit8(static_cast<std::uint8_t>(0x48U | ((dst >> 3U) & 0x1U)));
        emit8(static_cast<std::uint8_t>(0xB8U + (dst & 0x7U)));
        emit64(imm);
    }

    void mov_rm_disp32(std::uint8_t dst, std::uint8_t base, std::int32_t disp) {
        rex(true, dst, base);
        emit8(0x8BU);
        modrm(0x2U, dst, base & 0x7U);
        if ((base & 0x7U) == kRsp) emit8(0x24U);
        emit32(static_cast<std::uint32_t>(disp));
    }

    void mov_mr_disp32(std::uint8_t base, std::int32_t disp, std::uint8_t src) {
        rex(true, src, base);
        emit8(0x89U);
        modrm(0x2U, src, base & 0x7U);
        if ((base & 0x7U) == kRsp) emit8(0x24U);
        emit32(static_cast<std::uint32_t>(disp));
    }

    void and_rr(std::uint8_t dst, std::uint8_t src) {
        rex(true, src, dst);
        emit8(0x21U);
        modrm(0x3U, src, dst);
    }

    void or_rr(std::uint8_t dst, std::uint8_t src) {
        rex(true, src, dst);
        emit8(0x09U);
        modrm(0x3U, src, dst);
    }

    void not_r(std::uint8_t reg) {
        rex(true, 2U, reg);
        emit8(0xF7U);
        modrm(0x3U, 2U, reg);
    }

    void add_ri32(std::uint8_t reg, std::uint32_t imm) {
        rex(true, 0U, reg);
        emit8(0x81U);
        modrm(0x3U, 0U, reg);
        emit32(imm);
    }

    void dec_r(std::uint8_t reg) {
        rex(true, 1U, reg);
        emit8(0xFFU);
        modrm(0x3U, 1U, reg);
    }

    void test_rr(std::uint8_t a, std::uint8_t b) {
        rex(true, b, a);
        emit8(0x85U);
        modrm(0x3U, b, a);
    }

    void vmovdqu64_load(std::uint8_t zmm_dst, std::uint8_t base, std::int32_t disp) {
        vex3(static_cast<std::uint8_t>((zmm_dst >> 3U) ^ 0x1U), 1U, static_cast<std::uint8_t>((base >> 3U) ^ 0x1U), 0x02U, 1U, 0xFU, 0x2U, 0x1U);
        emit8(0x6FU);
        modrm(0x2U, zmm_dst, base & 0x7U);
        if ((base & 0x7U) == kRsp) emit8(0x24U);
        emit32(static_cast<std::uint32_t>(disp));
    }

    void vmovdqu64_store(std::uint8_t base, std::int32_t disp, std::uint8_t zmm_src) {
        vex3(static_cast<std::uint8_t>((zmm_src >> 3U) ^ 0x1U), 1U, static_cast<std::uint8_t>((base >> 3U) ^ 0x1U), 0x02U, 1U, 0xFU, 0x2U, 0x1U);
        emit8(0x7FU);
        modrm(0x2U, zmm_src, base & 0x7U);
        if ((base & 0x7U) == kRsp) emit8(0x24U);
        emit32(static_cast<std::uint32_t>(disp));
    }

    std::size_t vpternlogq(std::uint8_t dst, std::uint8_t src1, std::uint8_t src2, std::uint8_t imm) {
        vex3(static_cast<std::uint8_t>((dst >> 3U) ^ 0x1U), static_cast<std::uint8_t>((src2 >> 3U) ^ 0x1U), static_cast<std::uint8_t>((src1 >> 3U) ^ 0x1U), 0x03U, 1U,
             static_cast<std::uint8_t>((~src1) & 0xFU), 0x2U, 0x1U);
        emit8(0x25U);
        modrm(0x3U, dst & 0x7U, src2 & 0x7U);
        emit8(imm);
        return bytes.size() - 1U;
    }

    std::size_t jz_placeholder() {
        emit8(0x0FU);
        emit8(0x84U);
        const auto pos = bytes.size();
        emit32(0);
        return pos;
    }

    std::size_t jnz_placeholder() {
        emit8(0x0FU);
        emit8(0x85U);
        const auto pos = bytes.size();
        emit32(0);
        return pos;
    }

    void patch_rel32(std::size_t imm_pos, std::size_t target_pos) {
        const auto next = imm_pos + 4U;
        const auto rel = static_cast<std::int64_t>(target_pos) - static_cast<std::int64_t>(next);
        const auto rel32 = static_cast<std::int32_t>(rel);
        for (unsigned i = 0; i < 4; ++i) bytes[imm_pos + i] = static_cast<std::uint8_t>((static_cast<std::uint32_t>(rel32) >> (i * 8U)) & 0xFFU);
    }

    void ret() { emit8(0xC3U); }
};

struct TermSpec {
    bool invert_a = false;
    bool invert_b = false;
    bool invert_c = false;
};

std::vector<TermSpec> decode_terms(std::uint8_t mask) {
    std::vector<TermSpec> terms;
    for (unsigned index = 0; index < 8U; ++index) {
        if (((mask >> index) & 0x1U) == 0U) continue;
        terms.push_back(TermSpec{
            (index & 0x4U) == 0U,
            (index & 0x2U) == 0U,
            (index & 0x1U) == 0U,
        });
    }
    return terms;
}

std::size_t node_offset_words(std::size_t node_index, std::size_t word_count) {
    return node_index * word_count;
}

void emit_load_node_word(CodeEmitter& code,
                         std::uint8_t dst,
                         std::uint8_t workspace_base,
                         std::size_t node_index,
                         std::size_t word_count) {
    const auto disp = static_cast<std::int32_t>(node_offset_words(node_index, word_count) * sizeof(std::uint64_t));
    code.mov_rm_disp32(dst, workspace_base, disp);
}

void emit_store_node_word(CodeEmitter& code,
                          std::uint8_t workspace_base,
                          std::size_t node_index,
                          std::size_t word_count,
                          std::uint8_t src) {
    const auto disp = static_cast<std::int32_t>(node_offset_words(node_index, word_count) * sizeof(std::uint64_t));
    code.mov_mr_disp32(workspace_base, disp, src);
}

void emit_materialize_gate_scalar(CodeEmitter& code,
                                  const TernaryGate& gate,
                                  std::size_t output_node,
                                  std::size_t word_count) {
    auto terms = decode_terms(gate.mask);
    if (terms.empty()) {
        code.mov_ri(kR11, 0);
        emit_store_node_word(code, kR13, output_node, word_count, kR11);
        return;
    }

    if (terms.size() == 8U) {
        code.mov_ri(kR11, ~std::uint64_t{0});
        emit_store_node_word(code, kR13, output_node, word_count, kR11);
        return;
    }

    bool first = true;
    for (const auto& term : terms) {
        emit_load_node_word(code, kRax, kR13, gate.a, word_count);
        emit_load_node_word(code, kRcx, kR13, gate.b, word_count);
        emit_load_node_word(code, kRdx, kR13, gate.c, word_count);
        if (term.invert_a) code.not_r(kRax);
        if (term.invert_b) code.not_r(kRcx);
        if (term.invert_c) code.not_r(kRdx);
        code.and_rr(kRax, kRcx);
        code.and_rr(kRax, kRdx);
        if (first) {
            code.mov_rr(kR11, kRax);
            first = false;
        } else {
            code.or_rr(kR11, kRax);
        }
    }
    emit_store_node_word(code, kR13, output_node, word_count, kR11);
}

CodeEmitter build_scalar_kernel(const BooleanCascade& cascade, std::size_t word_count) {
    CodeEmitter code;

    code.push(kR12);
    code.push(kR13);
    code.push(kR14);
    code.push(kR15);

    code.mov_rr(kR12, kRdi);
    code.mov_rr(kR13, kRsi);
    code.mov_rr(kR14, kRdx);

    code.test_rr(kR14, kR14);
    const auto done_jump = code.jz_placeholder();

    const auto loop_start = code.bytes.size();

    for (std::size_t input = 0; input < cascade.input_count(); ++input) {
        code.mov_rm_disp32(kR15, kR12, static_cast<std::int32_t>(input * static_cast<std::size_t>(sizeof(void*))));
        code.mov_rm_disp32(kRax, kR15, 0);
        emit_store_node_word(code, kR13, input, word_count, kRax);
        code.add_ri32(kR15, 8U);
    }

    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        emit_materialize_gate_scalar(code, cascade.gates()[gate_index], cascade.input_count() + gate_index, word_count);
    }

    code.add_ri32(kR12, 8U);
    code.add_ri32(kR13, static_cast<std::uint32_t>(((cascade.input_count() + cascade.gate_count()) * word_count) * sizeof(std::uint64_t)));
    code.dec_r(kR14);
    const auto loop_jump = code.jnz_placeholder();
    const auto done_pos = code.bytes.size();

    code.pop(kR15);
    code.pop(kR14);
    code.pop(kR13);
    code.pop(kR12);
    code.ret();

    code.patch_rel32(done_jump, done_pos);
    code.patch_rel32(loop_jump, loop_start);
    return code;
}

CodeEmitter build_avx512_kernel(const BooleanCascade& cascade,
                                std::size_t word_count,
                                std::vector<GatePatchPoint>& patch_points) {
    if (word_count < 8U) throw std::invalid_argument("AVX-512 JIT requires word_count >= 8");
    if ((word_count % 8U) != 0U) throw std::invalid_argument("AVX-512 JIT requires word_count multiple of 8");

    CodeEmitter code;

    code.push(kR12);
    code.push(kR13);
    code.push(kR14);
    code.push(kR15);

    code.mov_rr(kR12, kRdi);
    code.mov_rr(kR13, kRsi);
    code.mov_rr(kR14, kRdx);

    code.test_rr(kR14, kR14);
    const auto done_jump = code.jz_placeholder();
    const auto loop_start = code.bytes.size();

    for (std::size_t input = 0; input < cascade.input_count(); ++input) {
        code.mov_rm_disp32(kR15, kR12, static_cast<std::int32_t>(input * static_cast<std::size_t>(sizeof(void*))));
        code.vmovdqu64_load(static_cast<std::uint8_t>(input), kR15, 0);
        const auto dst_disp = static_cast<std::int32_t>(node_offset_words(input, word_count) * sizeof(std::uint64_t));
        code.vmovdqu64_store(kR13, dst_disp, static_cast<std::uint8_t>(input));
        code.add_ri32(kR15, 64U);
    }

    const std::uint8_t output_base = static_cast<std::uint8_t>(cascade.input_count());
    for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
        const auto& gate = cascade.gates()[gate_index];
        const auto out_reg = static_cast<std::uint8_t>(output_base + gate_index);
        const auto imm_pos = code.vpternlogq(out_reg,
                                             static_cast<std::uint8_t>(gate.a),
                                             static_cast<std::uint8_t>(gate.b),
                                             gate.mask);
        patch_points.push_back(GatePatchPoint{gate_index, imm_pos});
        const auto dst_disp = static_cast<std::int32_t>(node_offset_words(cascade.input_count() + gate_index, word_count) * sizeof(std::uint64_t));
        code.vmovdqu64_store(kR13, dst_disp, out_reg);
    }

    code.add_ri32(kR12, 64U);
    code.add_ri32(kR13, static_cast<std::uint32_t>(((cascade.input_count() + cascade.gate_count()) * word_count) * sizeof(std::uint64_t)));
    code.dec_r(kR14);
    const auto loop_jump = code.jnz_placeholder();
    const auto done_pos = code.bytes.size();

    code.pop(kR15);
    code.pop(kR14);
    code.pop(kR13);
    code.pop(kR12);
    code.ret();

    code.patch_rel32(done_jump, done_pos);
    code.patch_rel32(loop_jump, loop_start);
    return code;
}

bool can_use_backend(JitBackend backend, const BooleanCascade& cascade, std::size_t word_count) noexcept {
    switch (backend) {
        case JitBackend::ScalarGpr:
            return true;
        case JitBackend::Avx512Ternlog:
            return simd::runtime_has_avx512() && word_count >= 8U && (word_count % 8U) == 0U &&
                   cascade.input_count() + cascade.gate_count() <= 16U;
    }
    return false;
}

} // namespace

bool runtime_supports_jit_backend(JitBackend backend) noexcept {
    switch (backend) {
        case JitBackend::ScalarGpr:
            return true;
        case JitBackend::Avx512Ternlog:
            return simd::runtime_has_avx512();
    }
    return false;
}

TargetInstructionSet target_instruction_set_for_backend(JitBackend backend) noexcept {
    switch (backend) {
        case JitBackend::ScalarGpr:
            return TargetInstructionSet::ScalarX86_64;
        case JitBackend::Avx512Ternlog:
            return TargetInstructionSet::Avx512F;
    }
    return TargetInstructionSet::ScalarX86_64;
}

bool runtime_supports_instruction_set(TargetInstructionSet isa) noexcept {
    switch (isa) {
        case TargetInstructionSet::ScalarX86_64:
            return true;
        case TargetInstructionSet::Avx512F:
            return simd::runtime_has_avx512();
    }
    return false;
}

ExecutableBuffer::ExecutableBuffer(std::size_t size) : size_(size) {
    if (size == 0) return;
    void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) throw std::runtime_error("mmap executable buffer failed");
    data_ = static_cast<std::uint8_t*>(ptr);
}

ExecutableBuffer::ExecutableBuffer(ExecutableBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

ExecutableBuffer& ExecutableBuffer::operator=(ExecutableBuffer&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

ExecutableBuffer::~ExecutableBuffer() { release(); }

void ExecutableBuffer::release() noexcept {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
}

CompiledBooleanCascade compile_boolean_cascade(const BooleanCascade& cascade, JitCompileOptions options) {
    if (cascade.input_count() == 0) throw std::invalid_argument("compile_boolean_cascade requires at least one input");
    if (cascade.gate_count() == 0) throw std::invalid_argument("compile_boolean_cascade requires at least one gate");
    if (options.word_count == 0) throw std::invalid_argument("compile_boolean_cascade requires word_count > 0");

    CompiledBooleanCascade compiled;
    compiled.input_count_ = cascade.input_count();
    compiled.gate_count_ = cascade.gate_count();
    compiled.workspace_word_count_ = (cascade.input_count() + cascade.gate_count()) * options.word_count;

    CodeEmitter code;
    if (can_use_backend(options.preferred_backend, cascade, options.word_count)) {
        compiled.backend_ = options.preferred_backend;
    } else {
        compiled.backend_ = JitBackend::ScalarGpr;
    }

    if (compiled.backend_ == JitBackend::Avx512Ternlog) {
        code = build_avx512_kernel(cascade, options.word_count, compiled.patch_points_);
    } else {
        code = build_scalar_kernel(cascade, options.word_count);
        for (std::size_t gate_index = 0; gate_index < cascade.gate_count(); ++gate_index) {
            compiled.patch_points_.push_back(GatePatchPoint{gate_index, 0});
        }
    }

    compiled.buffer_ = ExecutableBuffer(code.bytes.size());
    std::memcpy(compiled.buffer_.data(), code.bytes.data(), code.bytes.size());
    compiled.kernel_ = reinterpret_cast<CompiledBooleanCascade::KernelFn>(compiled.buffer_.data());
    return compiled;
}

std::shared_ptr<CompiledBooleanCascade> rehydrate_compiled_boolean_cascade(
    std::size_t input_count,
    std::size_t gate_count,
    std::size_t workspace_word_count,
    JitBackend backend,
    std::vector<GatePatchPoint> patch_points,
    const std::vector<std::uint8_t>& code_bytes) {
    if (code_bytes.empty()) throw std::invalid_argument("rehydrate_compiled_boolean_cascade requires code bytes");
    if (!runtime_supports_instruction_set(target_instruction_set_for_backend(backend))) {
        throw std::runtime_error("instruction set mismatch during JIT rehydration");
    }

    auto compiled = std::make_shared<CompiledBooleanCascade>();
    compiled->buffer_ = ExecutableBuffer(code_bytes.size());
    std::memcpy(compiled->buffer_.data(), code_bytes.data(), code_bytes.size());
    __builtin___clear_cache(reinterpret_cast<char*>(compiled->buffer_.data()),
                            reinterpret_cast<char*>(compiled->buffer_.data() + compiled->buffer_.size()));
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("cpuid" : : "a"(0) : "rbx", "rcx", "rdx", "memory");
#endif
    compiled->kernel_ = reinterpret_cast<CompiledBooleanCascade::KernelFn>(compiled->buffer_.data());
    compiled->input_count_ = input_count;
    compiled->gate_count_ = gate_count;
    compiled->workspace_word_count_ = workspace_word_count;
    compiled->backend_ = backend;
    compiled->patch_points_ = std::move(patch_points);
    return compiled;
}

void CompiledBooleanCascade::run_words(const std::uint64_t* const* inputs, std::uint64_t* workspace, std::size_t words) const {
    if (!kernel_) throw std::logic_error("CompiledBooleanCascade is empty");
    kernel_(inputs, workspace, words);
}

std::uint8_t CompiledBooleanCascade::gate_mask(std::size_t gate_index) const {
    if (!valid()) throw std::logic_error("CompiledBooleanCascade is empty");
    if (backend_ != JitBackend::Avx512Ternlog) throw std::invalid_argument("gate_mask is available only for patchable AVX-512 backend");
    const auto it = std::find_if(patch_points_.begin(), patch_points_.end(), [gate_index](const GatePatchPoint& point) {
        return point.gate_index == gate_index;
    });
    if (it == patch_points_.end()) throw std::invalid_argument("gate has no patch point in current backend");
    return buffer_.data()[it->immediate_offset];
}

void CompiledBooleanCascade::patch_gate_mask(std::size_t gate_index, std::uint8_t new_mask, bool synchronize) {
    if (!valid()) throw std::logic_error("CompiledBooleanCascade is empty");
    if (backend_ != JitBackend::Avx512Ternlog) throw std::invalid_argument("patch_gate_mask is available only for patchable AVX-512 backend");
    const auto it = std::find_if(patch_points_.begin(), patch_points_.end(), [gate_index](const GatePatchPoint& point) {
        return point.gate_index == gate_index;
    });
    if (it == patch_points_.end()) throw std::invalid_argument("gate has no patch point in current backend");
    buffer_.data()[it->immediate_offset] = new_mask;
    if (synchronize) {
        __builtin___clear_cache(reinterpret_cast<char*>(buffer_.data()),
                                reinterpret_cast<char*>(buffer_.data() + buffer_.size()));
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("cpuid" : : "a"(0) : "rbx", "rcx", "rdx", "memory");
#endif
    }
}

void CompiledBooleanCascade::patch_all_masks(const BooleanCascade& cascade, bool synchronize) {
    if (!valid()) throw std::logic_error("CompiledBooleanCascade is empty");
    if (cascade.gate_count() != gate_count_) throw std::invalid_argument("patch_all_masks gate count mismatch");
    if (backend_ != JitBackend::Avx512Ternlog) return;
    for (std::size_t gate_index = 0; gate_index < gate_count_; ++gate_index) {
        patch_gate_mask(gate_index, cascade.gates()[gate_index].mask, false);
    }
    if (synchronize) {
        __builtin___clear_cache(reinterpret_cast<char*>(buffer_.data()),
                                reinterpret_cast<char*>(buffer_.data() + buffer_.size()));
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("cpuid" : : "a"(0) : "rbx", "rcx", "rdx", "memory");
#endif
    }
}

BitVector CompiledBooleanCascade::evaluate(const std::vector<BitVector>& inputs) const {
    if (!kernel_) throw std::logic_error("CompiledBooleanCascade is empty");
    if (inputs.size() != input_count_) throw std::invalid_argument("CompiledBooleanCascade input count mismatch");
    if (inputs.empty()) throw std::invalid_argument("CompiledBooleanCascade requires at least one input");

    const auto bit_count = inputs.front().size();
    const auto word_count = inputs.front().word_count();
    for (const auto& input : inputs) {
        if (input.size() != bit_count) throw std::invalid_argument("CompiledBooleanCascade bit size mismatch");
    }

    std::vector<const std::uint64_t*> ptrs;
    ptrs.reserve(inputs.size());
    for (const auto& input : inputs) ptrs.push_back(input.words().data());

    std::vector<std::uint64_t> workspace(workspace_word_count_, 0);
    kernel_(ptrs.data(), workspace.data(), word_count);

    BitVector out(bit_count);
    const auto output_node = input_count_ + gate_count_ - 1;
    const auto base = output_node * word_count;
    for (std::size_t i = 0; i < word_count; ++i) out.words()[i] = workspace[base + i];
    if ((bit_count % 64U) != 0U) {
        const auto tail_bits = static_cast<unsigned>(bit_count % 64U);
        const auto mask = (1ULL << tail_bits) - 1ULL;
        out.words().back() &= mask;
    }
    return out;
}

} // namespace sle
