#pragma once

#include "autograd/tensor.h"
#include "core/activations.hpp"
#include "ucao/engine_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace transformer {

struct TransformerSamplingConfig {
    std::size_t top_k = 0;
    double temperature = 1.0;
    std::uint64_t seed = 42;
    std::size_t eos_token = static_cast<std::size_t>(-1);
};

struct TransformerConfig {
    std::size_t embed_dim = 64;
    std::size_t ff_hidden_dim = 256;
    std::size_t num_heads = 8;
    std::size_t num_layers = 1;
    std::size_t vocab_size = 0;
    std::size_t max_seq_len = 512;
    std::size_t num_classes = 0;
    bool causal = false;
    bool use_positional_encoding = true;
    bool return_attention = false;
    bool tie_lm_head = true;
};

class TransformerBlock : public ucao::engine::PolicyBound<ucao::engine::ModelFamily::Transformer> {
public:
    TransformerBlock(std::size_t embed_dim,
                     std::size_t ff_hidden_dim,
                     std::size_t num_heads = 1,
                     bool causal = false);

    autograd::Tensor forward(const autograd::Tensor& x) const;
    autograd::Tensor forward(const autograd::Tensor& x,
                             const autograd::Tensor* attention_mask) const;
    autograd::Tensor cross_attention(const autograd::Tensor& query,
                                     const autograd::Tensor& memory,
                                     const autograd::Tensor* memory_mask = nullptr) const;
    autograd::Tensor decoder_step(const autograd::Tensor& x,
                                  const autograd::Tensor& memory,
                                  const autograd::Tensor* self_mask = nullptr,
                                  const autograd::Tensor* memory_mask = nullptr) const;
    autograd::Tensor attention_weights(const autograd::Tensor& x,
                                       const autograd::Tensor* attention_mask = nullptr) const;
    static constexpr bool preserves_graph_contract() noexcept { return true; }
    std::vector<autograd::Tensor*> parameters();

    std::size_t embed_dim() const noexcept { return embed_dim_; }
    std::size_t ff_hidden_dim() const noexcept { return ff_hidden_dim_; }
    std::size_t num_heads() const noexcept { return num_heads_; }
    bool causal() const noexcept { return causal_; }

private:
    autograd::Tensor linear_lastdim(const autograd::Tensor& x,
                                    const autograd::Tensor& weight,
                                    const autograd::Tensor& bias) const;
    autograd::Tensor compute_attention(const autograd::Tensor& query,
                                       const autograd::Tensor& key_value,
                                       const autograd::Tensor* attention_mask,
                                       autograd::Tensor* weights_out,
                                       bool apply_causal_mask) const;

    std::size_t embed_dim_;
    std::size_t ff_hidden_dim_;
    std::size_t num_heads_;
    bool causal_;

    autograd::Tensor w_q_;
    autograd::Tensor b_q_;
    autograd::Tensor w_k_;
    autograd::Tensor b_k_;
    autograd::Tensor w_v_;
    autograd::Tensor b_v_;
    autograd::Tensor w_o_;
    autograd::Tensor b_o_;

    autograd::Tensor ln1_gamma_;
    autograd::Tensor ln1_beta_;
    autograd::Tensor ln2_gamma_;
    autograd::Tensor ln2_beta_;

    autograd::Tensor ff1_w_;
    autograd::Tensor ff1_b_;
    autograd::Tensor ff2_w_;
    autograd::Tensor ff2_b_;
};

class TransformerEncoder : public ucao::engine::PolicyBound<ucao::engine::ModelFamily::Transformer> {
public:
    explicit TransformerEncoder(const TransformerConfig& config);

    autograd::Tensor forward(const autograd::Tensor& x) const;
    autograd::Tensor forward(const autograd::Tensor& x,
                             const autograd::Tensor* attention_mask) const;
    autograd::Tensor forward_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const;
    autograd::Tensor pooled_output(const autograd::Tensor& x,
                                   const autograd::Tensor* attention_mask = nullptr) const;
    autograd::Tensor classify(const autograd::Tensor& x,
                              const autograd::Tensor* attention_mask = nullptr) const;
    autograd::Tensor classify_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const;
    autograd::Tensor language_model_logits(const autograd::Tensor& x,
                                           const autograd::Tensor* attention_mask = nullptr) const;
    autograd::Tensor language_model_logits_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const;
    double next_token_loss(const std::vector<std::vector<std::size_t>>& token_ids) const;
    std::vector<std::size_t> greedy_decode(const std::vector<std::size_t>& prompt,
                                           std::size_t max_new_tokens) const;
    std::vector<std::size_t> top_k_decode(const std::vector<std::size_t>& prompt,
                                          std::size_t max_new_tokens,
                                          std::size_t k = 3) const;
    std::vector<std::size_t> beam_search_decode(const std::vector<std::size_t>& prompt,
                                                std::size_t max_new_tokens,
                                                std::size_t beam_width = 3) const;
    std::vector<std::size_t> incremental_decode(const std::vector<std::size_t>& prompt,
                                                std::size_t max_new_tokens) const;
    std::vector<std::size_t> sample_decode(const std::vector<std::size_t>& prompt,
                                           std::size_t max_new_tokens,
                                           const TransformerSamplingConfig& sampling = {}) const;
    std::vector<std::vector<std::size_t>> batch_generate(const std::vector<std::vector<std::size_t>>& prompts,
                                                         std::size_t max_new_tokens,
                                                         const TransformerSamplingConfig* sampling = nullptr,
                                                         std::size_t beam_width = 0,
                                                         bool incremental = false) const;
    std::vector<autograd::Tensor> collect_attention_maps(const autograd::Tensor& x,
                                                         const autograd::Tensor* attention_mask = nullptr) const;
    std::vector<autograd::Tensor*> parameters();

    const TransformerConfig& config() const noexcept { return config_; }
    std::size_t num_layers() const noexcept { return blocks_.size(); }
    autograd::Tensor lm_projection(const autograd::Tensor& encoded) const;
    static constexpr bool uses_ucao_engine() noexcept { return true; }
    static constexpr ucao::engine::EngineDescriptor engine_descriptor() noexcept {
        return ucao::engine::sequence_geometry;
    }

public:
    autograd::Tensor add_positional_encoding(const autograd::Tensor& x) const;
    autograd::Tensor embed_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const;

protected:
    TransformerConfig config_;
    std::vector<TransformerBlock> blocks_;
    autograd::Tensor token_embedding_;
    autograd::Tensor classifier_w_;
    autograd::Tensor classifier_b_;
    autograd::Tensor lm_head_w_;
    autograd::Tensor lm_head_b_;
};

class TransformerSeq2Seq {
public:
    explicit TransformerSeq2Seq(const TransformerConfig& config);

    autograd::Tensor encode_tokens(const std::vector<std::vector<std::size_t>>& source_tokens) const;
    autograd::Tensor decode_tokens(const std::vector<std::vector<std::size_t>>& target_tokens,
                                   const autograd::Tensor& memory,
                                   const autograd::Tensor* target_mask = nullptr,
                                   const autograd::Tensor* memory_mask = nullptr) const;
    autograd::Tensor forward_tokens(const std::vector<std::vector<std::size_t>>& source_tokens,
                                    const std::vector<std::vector<std::size_t>>& target_tokens,
                                    const autograd::Tensor* source_mask = nullptr,
                                    const autograd::Tensor* target_mask = nullptr) const;
    double teacher_forcing_loss(const std::vector<std::vector<std::size_t>>& source_tokens,
                                const std::vector<std::vector<std::size_t>>& target_tokens) const;
    std::vector<autograd::Tensor> collect_cross_attention_maps(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                               const std::vector<std::vector<std::size_t>>& target_tokens) const;
    std::vector<std::size_t> greedy_decode(const std::vector<std::size_t>& source_tokens,
                                           const std::vector<std::size_t>& prompt,
                                           std::size_t max_new_tokens,
                                           const std::vector<double>* source_mask = nullptr) const;
    std::vector<std::size_t> top_k_decode(const std::vector<std::size_t>& source_tokens,
                                          const std::vector<std::size_t>& prompt,
                                          std::size_t max_new_tokens,
                                          std::size_t k = 3,
                                          const std::vector<double>* source_mask = nullptr) const;
    std::vector<std::size_t> beam_search_decode(const std::vector<std::size_t>& source_tokens,
                                                const std::vector<std::size_t>& prompt,
                                                std::size_t max_new_tokens,
                                                std::size_t beam_width = 3,
                                                const std::vector<double>* source_mask = nullptr) const;
    std::vector<std::size_t> incremental_decode(const std::vector<std::size_t>& source_tokens,
                                                const std::vector<std::size_t>& prompt,
                                                std::size_t max_new_tokens,
                                                const std::vector<double>* source_mask = nullptr) const;
    std::vector<std::size_t> sample_decode(const std::vector<std::size_t>& source_tokens,
                                           const std::vector<std::size_t>& prompt,
                                           std::size_t max_new_tokens,
                                           const TransformerSamplingConfig& sampling = {},
                                           const std::vector<double>* source_mask = nullptr) const;
    std::vector<std::vector<std::size_t>> batch_generate(const std::vector<std::vector<std::size_t>>& source_tokens_batch,
                                                         const std::vector<std::vector<std::size_t>>& prompts,
                                                         std::size_t max_new_tokens,
                                                         const TransformerSamplingConfig* sampling = nullptr,
                                                         std::size_t beam_width = 0,
                                                         bool incremental = false,
                                                         const std::vector<std::vector<double>>* source_masks = nullptr) const;
    std::vector<autograd::Tensor*> parameters();

private:
    autograd::Tensor make_batch_mask(const std::vector<double>* mask, std::size_t seq_len) const;

    TransformerConfig config_;
    TransformerEncoder encoder_;
    TransformerEncoder decoder_;
    std::vector<TransformerBlock> cross_blocks_;
};

class TransformerSystem {
public:
    explicit TransformerSystem(const TransformerConfig& config);

    autograd::Tensor encode_tokens(const std::vector<std::vector<std::size_t>>& token_ids,
                                   const autograd::Tensor* attention_mask = nullptr) const;
    autograd::Tensor classify_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const;
    autograd::Tensor language_model_logits_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const;
    autograd::Tensor seq2seq_logits(const std::vector<std::vector<std::size_t>>& source_tokens,
                                    const std::vector<std::vector<std::size_t>>& target_tokens,
                                    const autograd::Tensor* source_mask = nullptr,
                                    const autograd::Tensor* target_mask = nullptr) const;
    double next_token_loss(const std::vector<std::vector<std::size_t>>& token_ids) const;
    double teacher_forcing_loss(const std::vector<std::vector<std::size_t>>& source_tokens,
                                const std::vector<std::vector<std::size_t>>& target_tokens) const;

    std::vector<std::size_t> generate_causal(const std::vector<std::size_t>& prompt,
                                             std::size_t max_new_tokens,
                                             const TransformerSamplingConfig* sampling = nullptr,
                                             std::size_t beam_width = 0,
                                             bool incremental = false) const;
    std::vector<std::vector<std::size_t>> generate_causal_batch(const std::vector<std::vector<std::size_t>>& prompts,
                                                                std::size_t max_new_tokens,
                                                                const TransformerSamplingConfig* sampling = nullptr,
                                                                std::size_t beam_width = 0,
                                                                bool incremental = false) const;
    std::vector<std::size_t> generate_seq2seq(const std::vector<std::size_t>& source_tokens,
                                              const std::vector<std::size_t>& prompt,
                                              std::size_t max_new_tokens,
                                              const TransformerSamplingConfig* sampling = nullptr,
                                              std::size_t beam_width = 0,
                                              bool incremental = false,
                                              const std::vector<double>* source_mask = nullptr) const;
    std::vector<std::vector<std::size_t>> generate_seq2seq_batch(const std::vector<std::vector<std::size_t>>& source_tokens_batch,
                                                                 const std::vector<std::vector<std::size_t>>& prompts,
                                                                 std::size_t max_new_tokens,
                                                                 const TransformerSamplingConfig* sampling = nullptr,
                                                                 std::size_t beam_width = 0,
                                                                 bool incremental = false,
                                                                 const std::vector<std::vector<double>>* source_masks = nullptr) const;

    std::vector<autograd::Tensor> encoder_attention_maps(const autograd::Tensor& x,
                                                         const autograd::Tensor* attention_mask = nullptr) const;
    std::vector<autograd::Tensor> seq2seq_cross_attention_maps(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                               const std::vector<std::vector<std::size_t>>& target_tokens) const;
    std::vector<autograd::Tensor*> parameters();

    const TransformerEncoder& encoder() const noexcept { return encoder_; }
    const TransformerEncoder& classifier() const noexcept { return classifier_; }
    const TransformerEncoder& language_model() const noexcept { return language_model_; }
    const TransformerSeq2Seq& seq2seq() const noexcept { return seq2seq_; }

private:
    TransformerConfig config_;
    TransformerEncoder encoder_;
    TransformerEncoder classifier_;
    TransformerEncoder language_model_;
    TransformerSeq2Seq seq2seq_;
};

}  // namespace transformer
