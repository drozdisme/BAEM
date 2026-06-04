#include "models/transformer/transformer_block.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace transformer {

namespace {

std::size_t select_sampled_token(const std::vector<double>& logits,
                                 const TransformerSamplingConfig& sampling,
                                 std::mt19937_64& rng) {
    if (logits.empty()) throw std::invalid_argument("select_sampled_token: logits must be non-empty");
    const std::size_t k = sampling.top_k == 0 ? logits.size() : std::min(sampling.top_k, logits.size());
    std::vector<std::pair<double, std::size_t>> ranked;
    ranked.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) ranked.emplace_back(logits[i], i);
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    double temperature = sampling.temperature <= 1e-8 ? 1e-8 : sampling.temperature;
    double max_logit = ranked.front().first;
    std::vector<double> probs(k, 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        probs[i] = std::exp((ranked[i].first - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0) return ranked.front().second;
    std::uniform_real_distribution<double> dist(0.0, sum);
    double draw = dist(rng);
    double accum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        accum += probs[i];
        if (draw <= accum) return ranked[i].second;
    }
    return ranked.front().second;
}

autograd::Tensor make_contiguous_tensor(const autograd::Tensor& x,
                                        const std::vector<std::size_t>& shape) {
    std::vector<double> data(x.numel());
    for (std::size_t i = 0; i < x.numel(); ++i) data[i] = x.value_flat(i);
    return autograd::Tensor(std::move(data), shape, x.requires_grad());
}

autograd::Tensor init_weight(std::size_t out_dim, std::size_t in_dim) {
    std::mt19937 rng(42 + static_cast<unsigned>(out_dim * 31 + in_dim));
    const double limit = std::sqrt(6.0 / static_cast<double>(in_dim + out_dim));
    std::uniform_real_distribution<double> dist(-limit, limit);
    std::vector<double> data(out_dim * in_dim);
    for (double& v : data) v = dist(rng);
    return autograd::Tensor(std::move(data), {out_dim, in_dim}, true);
}

autograd::Tensor init_bias(std::size_t dim) {
    return autograd::Tensor(std::vector<double>(dim, 0.0), {dim}, true);
}

autograd::Tensor split_heads(const autograd::Tensor& x,
                             std::size_t batch,
                             std::size_t seq,
                             std::size_t num_heads,
                             std::size_t head_dim) {
    auto reshaped = x.reshape({batch, seq, num_heads, head_dim});
    auto transposed = reshaped.transpose_view(1, 2);
    return make_contiguous_tensor(transposed, {batch, num_heads, seq, head_dim});
}

autograd::Tensor merge_heads(const autograd::Tensor& x,
                             std::size_t batch,
                             std::size_t seq,
                             std::size_t embed) {
    auto transposed = x.transpose_view(1, 2);
    auto contiguous = make_contiguous_tensor(transposed, {batch, seq, x.shape()[1], x.shape()[3]});
    return contiguous.reshape({batch, seq, embed});
}

}  // namespace

TransformerBlock::TransformerBlock(std::size_t embed_dim,
                                   std::size_t ff_hidden_dim,
                                   std::size_t num_heads,
                                   bool causal)
    : embed_dim_(embed_dim)
    , ff_hidden_dim_(ff_hidden_dim)
    , num_heads_(num_heads)
    , causal_(causal)
    , w_q_(init_weight(embed_dim, embed_dim))
    , b_q_(init_bias(embed_dim))
    , w_k_(init_weight(embed_dim, embed_dim))
    , b_k_(init_bias(embed_dim))
    , w_v_(init_weight(embed_dim, embed_dim))
    , b_v_(init_bias(embed_dim))
    , w_o_(init_weight(embed_dim, embed_dim))
    , b_o_(init_bias(embed_dim))
    , ln1_gamma_(autograd::Tensor(std::vector<double>(embed_dim, 1.0), {embed_dim}, true))
    , ln1_beta_(init_bias(embed_dim))
    , ln2_gamma_(autograd::Tensor(std::vector<double>(embed_dim, 1.0), {embed_dim}, true))
    , ln2_beta_(init_bias(embed_dim))
    , ff1_w_(init_weight(ff_hidden_dim, embed_dim))
    , ff1_b_(init_bias(ff_hidden_dim))
    , ff2_w_(init_weight(embed_dim, ff_hidden_dim))
    , ff2_b_(init_bias(embed_dim)) {
    if (embed_dim == 0 || ff_hidden_dim == 0) throw std::invalid_argument("TransformerBlock: dims must be > 0");
    if (num_heads == 0 || embed_dim % num_heads != 0) throw std::invalid_argument("TransformerBlock: embed_dim must be divisible by num_heads");
}

autograd::Tensor TransformerBlock::linear_lastdim(const autograd::Tensor& x,
                                                  const autograd::Tensor& weight,
                                                  const autograd::Tensor& bias) const {
    if (x.ndim() < 2) throw std::invalid_argument("linear_lastdim: expected rank >= 2");
    if (weight.ndim() != 2 || bias.ndim() != 1) throw std::invalid_argument("linear_lastdim: expected weight [out,in] and bias [out]");
    const std::size_t in_dim = x.shape().back();
    if (weight.shape().back() != in_dim || bias.shape().front() != weight.shape().front()) throw std::invalid_argument("linear_lastdim: parameter shape mismatch");
    const std::size_t out_dim = weight.shape().front();
    std::size_t rows = x.numel() / in_dim;
    autograd::Tensor x2 = x.reshape({rows, in_dim});
    autograd::Tensor y2 = autograd::matmul(x2, autograd::transpose(weight));
    y2 = y2 + bias;
    auto out_shape = x.shape();
    out_shape.back() = out_dim;
    return y2.reshape(out_shape);
}

autograd::Tensor TransformerBlock::compute_attention(const autograd::Tensor& query,
                                                     const autograd::Tensor& key_value,
                                                     const autograd::Tensor* attention_mask,
                                                     autograd::Tensor* weights_out,
                                                     bool apply_causal_mask) const {
    const auto qshape = query.shape();
    const auto kvshape = key_value.shape();
    if (qshape.size() != 3 || kvshape.size() != 3) throw std::invalid_argument("TransformerBlock::compute_attention: expected [batch, seq, embed]");
    if (qshape[0] != kvshape[0] || qshape[2] != kvshape[2] || qshape[2] != embed_dim_) throw std::invalid_argument("TransformerBlock::compute_attention: batch/embed mismatch");

    const std::size_t batch = qshape[0];
    const std::size_t qseq = qshape[1];
    const std::size_t kvseq = kvshape[1];
    const std::size_t head_dim = embed_dim_ / num_heads_;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    auto q = split_heads(linear_lastdim(query, w_q_, b_q_), batch, qseq, num_heads_, head_dim);
    auto k = split_heads(linear_lastdim(key_value, w_k_, b_k_), batch, kvseq, num_heads_, head_dim);
    auto v = split_heads(linear_lastdim(key_value, w_v_, b_v_), batch, kvseq, num_heads_, head_dim);

    auto scores = autograd::matmul(q, k.transpose_view(2, 3)) * scale;
    std::vector<double> raw(scores.numel());
    for (std::size_t i = 0; i < scores.numel(); ++i) raw[i] = scores.value_flat(i);
    const auto sshape = scores.shape();
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < num_heads_; ++h) {
            for (std::size_t i = 0; i < qseq; ++i) {
                for (std::size_t j = 0; j < kvseq; ++j) {
                    const std::size_t idx = ((b * num_heads_ + h) * qseq + i) * kvseq + j;
                    bool masked = apply_causal_mask && j > i;
                    if (!masked && attention_mask != nullptr) {
                        if (attention_mask->ndim() != 2 || attention_mask->shape()[0] != batch || attention_mask->shape()[1] != kvseq) {
                            throw std::invalid_argument("TransformerBlock::compute_attention: attention_mask must have shape [batch, seq]");
                        }
                        masked = attention_mask->value_flat(b * kvseq + j) <= 0.0;
                    }
                    if (masked) raw[idx] = -1e9;
                }
            }
        }
    }
    scores = autograd::Tensor(std::move(raw), sshape, scores.requires_grad());
    auto attn = autograd::softmax(scores);
    if (weights_out != nullptr) *weights_out = attn;
    auto ctx = autograd::matmul(attn, v);
    return linear_lastdim(merge_heads(ctx, batch, qseq, embed_dim_), w_o_, b_o_);
}

autograd::Tensor TransformerBlock::attention_weights(const autograd::Tensor& x,
                                                     const autograd::Tensor* attention_mask) const {
    autograd::Tensor weights;
    (void)compute_attention(x, x, attention_mask, &weights, causal_);
    return weights;
}

autograd::Tensor TransformerBlock::forward(const autograd::Tensor& x) const {
    return forward(x, nullptr);
}

autograd::Tensor TransformerBlock::forward(const autograd::Tensor& x,
                                           const autograd::Tensor* attention_mask) const {
    if (x.ndim() != 3) throw std::invalid_argument("TransformerBlock::forward: expected input shape [batch, seq, embed]");
    if (x.shape().back() != embed_dim_) throw std::invalid_argument("TransformerBlock::forward: input embed dimension mismatch");
    if (x.requires_grad()) x.require_graph_preserving("TransformerBlock::forward");
    auto attn = compute_attention(x, x, attention_mask, nullptr, causal_);
    auto x1 = autograd::layer_norm(x + attn, ln1_gamma_, ln1_beta_);
    auto ff = linear_lastdim(x1, ff1_w_, ff1_b_);
    ff = core::gelu_act(ff);
    ff = linear_lastdim(ff, ff2_w_, ff2_b_);
    return autograd::layer_norm(x1 + ff, ln2_gamma_, ln2_beta_);
}

autograd::Tensor TransformerBlock::cross_attention(const autograd::Tensor& query,
                                                   const autograd::Tensor& memory,
                                                   const autograd::Tensor* memory_mask) const {
    if (query.ndim() != 3 || memory.ndim() != 3) throw std::invalid_argument("TransformerBlock::cross_attention: expected rank-3 tensors");
    auto attn = compute_attention(query, memory, memory_mask, nullptr, false);
    return autograd::layer_norm(query + attn, ln1_gamma_, ln1_beta_);
}

autograd::Tensor TransformerBlock::decoder_step(const autograd::Tensor& x,
                                                const autograd::Tensor& memory,
                                                const autograd::Tensor* self_mask,
                                                const autograd::Tensor* memory_mask) const {
    auto self_out = forward(x, self_mask);
    auto cross_out = cross_attention(self_out, memory, memory_mask);
    auto ff = linear_lastdim(cross_out, ff1_w_, ff1_b_);
    ff = core::gelu_act(ff);
    ff = linear_lastdim(ff, ff2_w_, ff2_b_);
    return autograd::layer_norm(cross_out + ff, ln2_gamma_, ln2_beta_);
}

std::vector<autograd::Tensor*> TransformerBlock::parameters() {
    return {&w_q_, &b_q_, &w_k_, &b_k_, &w_v_, &b_v_, &w_o_, &b_o_,
            &ln1_gamma_, &ln1_beta_, &ln2_gamma_, &ln2_beta_,
            &ff1_w_, &ff1_b_, &ff2_w_, &ff2_b_};
}

TransformerEncoder::TransformerEncoder(const TransformerConfig& config)
    : config_(config)
    , token_embedding_(config.vocab_size > 0 ? init_weight(config.vocab_size, config.embed_dim) : autograd::Tensor())
    , classifier_w_(config.num_classes > 0 ? init_weight(config.num_classes, config.embed_dim) : autograd::Tensor())
    , classifier_b_(config.num_classes > 0 ? init_bias(config.num_classes) : autograd::Tensor())
    , lm_head_w_((config.vocab_size > 0 && !config.tie_lm_head) ? init_weight(config.vocab_size, config.embed_dim) : autograd::Tensor())
    , lm_head_b_(config.vocab_size > 0 ? init_bias(config.vocab_size) : autograd::Tensor()) {
    if (config_.embed_dim == 0 || config_.ff_hidden_dim == 0 || config_.num_heads == 0 || config_.num_layers == 0) throw std::invalid_argument("TransformerEncoder: invalid zero dimension in config");
    if (config_.embed_dim % config_.num_heads != 0) throw std::invalid_argument("TransformerEncoder: embed_dim must be divisible by num_heads");
    blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) blocks_.emplace_back(config_.embed_dim, config_.ff_hidden_dim, config_.num_heads, config_.causal);
}

autograd::Tensor TransformerEncoder::add_positional_encoding(const autograd::Tensor& x) const {
    if (!config_.use_positional_encoding) return x;
    const auto shape = x.shape();
    std::vector<double> data(x.numel());
    for (std::size_t i = 0; i < x.numel(); ++i) data[i] = x.value_flat(i);
    const std::size_t batch = shape[0], seq = shape[1], embed = shape[2];
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t pos = 0; pos < seq; ++pos) {
            for (std::size_t e = 0; e < embed; ++e) {
                const double denom = std::pow(10000.0, static_cast<double>(2 * (e / 2)) / static_cast<double>(embed));
                const double pe = (e % 2 == 0) ? std::sin(static_cast<double>(pos) / denom) : std::cos(static_cast<double>(pos) / denom);
                data[(b * seq + pos) * embed + e] += pe;
            }
        }
    }
    return autograd::Tensor(std::move(data), shape, x.requires_grad());
}

autograd::Tensor TransformerEncoder::embed_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const {
    if (config_.vocab_size == 0) throw std::runtime_error("TransformerEncoder::embed_tokens: vocab_size must be > 0");
    if (token_embedding_.ndim() != 2) throw std::runtime_error("TransformerEncoder::embed_tokens: token embedding matrix missing");
    if (token_ids.empty()) throw std::invalid_argument("TransformerEncoder::embed_tokens: token_ids must be non-empty");
    const std::size_t batch = token_ids.size();
    const std::size_t seq = token_ids.front().size();
    for (const auto& row : token_ids) if (row.size() != seq) throw std::invalid_argument("TransformerEncoder::embed_tokens: ragged token matrix");
    std::vector<double> data(batch * seq * config_.embed_dim, 0.0);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            const std::size_t token = token_ids[b][s];
            if (token >= config_.vocab_size) throw std::out_of_range("TransformerEncoder::embed_tokens: token id out of range");
            for (std::size_t e = 0; e < config_.embed_dim; ++e) data[(b * seq + s) * config_.embed_dim + e] = token_embedding_.value_flat(token * config_.embed_dim + e);
        }
    }
    return autograd::Tensor(std::move(data), {batch, seq, config_.embed_dim}, true);
}

autograd::Tensor TransformerEncoder::forward(const autograd::Tensor& x) const {
    return forward(x, nullptr);
}

autograd::Tensor TransformerEncoder::forward(const autograd::Tensor& x,
                                             const autograd::Tensor* attention_mask) const {
    if (x.ndim() != 3) throw std::invalid_argument("TransformerEncoder::forward: expected [batch, seq, embed]");
    if (x.shape()[2] != config_.embed_dim) throw std::invalid_argument("TransformerEncoder::forward: embed dim mismatch");
    auto out = add_positional_encoding(x);
    for (const auto& block : blocks_) out = block.forward(out, attention_mask);
    return out;
}

autograd::Tensor TransformerEncoder::forward_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const {
    auto embedded = embed_tokens(token_ids);
    return forward(embedded, nullptr);
}

autograd::Tensor TransformerEncoder::pooled_output(const autograd::Tensor& x,
                                                   const autograd::Tensor* attention_mask) const {
    auto encoded = forward(x, attention_mask);
    const auto shape = encoded.shape();
    const std::size_t batch = shape[0], seq = shape[1], embed = shape[2];
    std::vector<double> pooled(batch * embed, 0.0);
    for (std::size_t b = 0; b < batch; ++b) {
        double denom = 0.0;
        for (std::size_t s = 0; s < seq; ++s) {
            const double mask_val = attention_mask ? std::max(0.0, attention_mask->value_flat(b * seq + s)) : 1.0;
            denom += mask_val;
            for (std::size_t e = 0; e < embed; ++e) pooled[b * embed + e] += mask_val * encoded.value_flat((b * seq + s) * embed + e);
        }
        if (denom <= 0.0) denom = 1.0;
        for (std::size_t e = 0; e < embed; ++e) pooled[b * embed + e] /= denom;
    }
    return autograd::Tensor(std::move(pooled), {batch, embed}, true);
}

autograd::Tensor TransformerEncoder::classify(const autograd::Tensor& x,
                                              const autograd::Tensor* attention_mask) const {
    if (config_.num_classes == 0 || classifier_w_.ndim() != 2) throw std::runtime_error("TransformerEncoder::classify: classifier head is not configured");
    auto pooled = pooled_output(x, attention_mask);
    return autograd::matmul(pooled, autograd::transpose(classifier_w_)) + classifier_b_;
}

autograd::Tensor TransformerEncoder::classify_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const {
    auto embedded = embed_tokens(token_ids);
    return classify(embedded, nullptr);
}

autograd::Tensor TransformerEncoder::lm_projection(const autograd::Tensor& encoded) const {
    if (config_.vocab_size == 0) throw std::runtime_error("TransformerEncoder::lm_projection: vocab_size must be > 0");
    const autograd::Tensor& lm_w = (config_.tie_lm_head ? token_embedding_ : lm_head_w_);
    if (lm_w.ndim() != 2) throw std::runtime_error("TransformerEncoder::lm_projection: LM head is not configured");
    auto logits = autograd::matmul(encoded.reshape({encoded.shape()[0] * encoded.shape()[1], encoded.shape()[2]}), autograd::transpose(lm_w));
    logits = logits + lm_head_b_;
    return logits.reshape({encoded.shape()[0], encoded.shape()[1], config_.vocab_size});
}

autograd::Tensor TransformerEncoder::language_model_logits(const autograd::Tensor& x,
                                                           const autograd::Tensor* attention_mask) const {
    auto encoded = forward(x, attention_mask);
    return lm_projection(encoded);
}

autograd::Tensor TransformerEncoder::language_model_logits_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const {
    auto embedded = embed_tokens(token_ids);
    return language_model_logits(embedded, nullptr);
}


double TransformerEncoder::next_token_loss(const std::vector<std::vector<std::size_t>>& token_ids) const {
    if (!config_.causal) throw std::runtime_error("TransformerEncoder::next_token_loss: requires causal transformer config");
    if (token_ids.empty() || token_ids.front().size() < 2) throw std::invalid_argument("TransformerEncoder::next_token_loss: need sequences of length >= 2");
    auto logits = language_model_logits_tokens(token_ids);
    const auto shape = logits.shape();
    double loss = 0.0;
    double count = 0.0;
    for (std::size_t b = 0; b < shape[0]; ++b) {
        for (std::size_t t = 0; t + 1 < shape[1]; ++t) {
            double max_logit = -1e18;
            for (std::size_t v = 0; v < shape[2]; ++v) max_logit = std::max(max_logit, logits.value_flat((b * shape[1] + t) * shape[2] + v));
            double sum_exp = 0.0;
            for (std::size_t v = 0; v < shape[2]; ++v) sum_exp += std::exp(logits.value_flat((b * shape[1] + t) * shape[2] + v) - max_logit);
            const std::size_t target = token_ids[b][t + 1];
            const double target_logit = logits.value_flat((b * shape[1] + t) * shape[2] + target);
            loss += -(target_logit - max_logit - std::log(sum_exp));
            count += 1.0;
        }
    }
    return loss / std::max(1.0, count);
}

std::vector<std::size_t> TransformerEncoder::greedy_decode(const std::vector<std::size_t>& prompt,
                                                           std::size_t max_new_tokens) const {
    if (config_.vocab_size == 0) throw std::runtime_error("TransformerEncoder::greedy_decode: vocab_size must be > 0");
    if (!config_.causal) throw std::runtime_error("TransformerEncoder::greedy_decode: requires causal transformer config");
    if (prompt.empty()) throw std::invalid_argument("TransformerEncoder::greedy_decode: prompt must be non-empty");
    std::vector<std::size_t> seq = prompt;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = language_model_logits_tokens({seq});
        const auto shape = logits.shape();
        const std::size_t last_pos = shape[1] - 1;
        std::size_t best_token = 0;
        double best_score = logits.value_flat((last_pos) * shape[2]);
        for (std::size_t v = 1; v < shape[2]; ++v) {
            const double score = logits.value_flat(last_pos * shape[2] + v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }
        seq.push_back(best_token);
        if (seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}


std::vector<std::size_t> TransformerEncoder::top_k_decode(const std::vector<std::size_t>& prompt,
                                                          std::size_t max_new_tokens,
                                                          std::size_t k) const {
    if (k == 0) throw std::invalid_argument("TransformerEncoder::top_k_decode: k must be > 0");
    std::vector<std::size_t> seq = prompt;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = language_model_logits_tokens({seq});
        const auto shape = logits.shape();
        const std::size_t last_pos = shape[1] - 1;
        std::vector<std::pair<double, std::size_t>> scores;
        scores.reserve(shape[2]);
        for (std::size_t v = 0; v < shape[2]; ++v) scores.emplace_back(logits.value_flat(last_pos * shape[2] + v), v);
        std::partial_sort(scores.begin(), scores.begin() + std::min(k, scores.size()), scores.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        seq.push_back(scores.front().second);
        if (seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}


std::vector<std::size_t> TransformerEncoder::beam_search_decode(const std::vector<std::size_t>& prompt,
                                                                std::size_t max_new_tokens,
                                                                std::size_t beam_width) const {
    if (beam_width == 0) throw std::invalid_argument("TransformerEncoder::beam_search_decode: beam_width must be > 0");
    std::vector<std::pair<std::vector<std::size_t>, double>> beams{{prompt, 0.0}};
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<std::pair<std::vector<std::size_t>, double>> next_beams;
        for (const auto& beam : beams) {
            auto logits = language_model_logits_tokens({beam.first});
            const auto shape = logits.shape();
            const std::size_t last_pos = shape[1] - 1;
            std::vector<std::pair<double, std::size_t>> scores;
            scores.reserve(shape[2]);
            for (std::size_t v = 0; v < shape[2]; ++v) scores.emplace_back(logits.value_flat(last_pos * shape[2] + v), v);
            std::partial_sort(scores.begin(), scores.begin() + std::min(beam_width, scores.size()), scores.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t i = 0; i < std::min(beam_width, scores.size()); ++i) {
                auto seq = beam.first;
                seq.push_back(scores[i].second);
                next_beams.emplace_back(std::move(seq), beam.second + scores[i].first);
            }
        }
        std::partial_sort(next_beams.begin(), next_beams.begin() + std::min(beam_width, next_beams.size()), next_beams.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        beams.assign(next_beams.begin(), next_beams.begin() + std::min(beam_width, next_beams.size()));
        if (beams.front().first.size() >= config_.max_seq_len) break;
    }
    return beams.front().first;
}


std::vector<std::size_t> TransformerEncoder::incremental_decode(const std::vector<std::size_t>& prompt,
                                                                std::size_t max_new_tokens) const {
    if (!config_.causal) throw std::runtime_error("TransformerEncoder::incremental_decode: requires causal transformer config");
    if (prompt.empty()) throw std::invalid_argument("TransformerEncoder::incremental_decode: prompt must be non-empty");
    std::vector<std::size_t> seq = prompt;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const std::vector<std::vector<std::size_t>> tail{{seq.back()}};
        auto tail_embed = embed_tokens(tail);
        auto logits = language_model_logits(tail_embed, nullptr);
        const auto shape = logits.shape();
        std::size_t best_token = 0;
        double best_score = logits.value_flat(shape[2] - 1);
        for (std::size_t v = 1; v < shape[2]; ++v) {
            const double score = logits.value_flat(v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }
        seq.push_back(best_token);
        if (seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}

std::vector<std::size_t> TransformerEncoder::sample_decode(const std::vector<std::size_t>& prompt,
                                                           std::size_t max_new_tokens,
                                                           const TransformerSamplingConfig& sampling) const {
    if (config_.vocab_size == 0) throw std::runtime_error("TransformerEncoder::sample_decode: vocab_size must be > 0");
    if (!config_.causal) throw std::runtime_error("TransformerEncoder::sample_decode: requires causal transformer config");
    if (prompt.empty()) throw std::invalid_argument("TransformerEncoder::sample_decode: prompt must be non-empty");
    std::mt19937_64 rng(sampling.seed);
    std::vector<std::size_t> seq = prompt;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = language_model_logits_tokens({seq});
        const auto shape = logits.shape();
        const std::size_t last_pos = shape[1] - 1;
        std::vector<double> row(shape[2], 0.0);
        for (std::size_t v = 0; v < shape[2]; ++v) row[v] = logits.value_flat(last_pos * shape[2] + v);
        const auto token = select_sampled_token(row, sampling, rng);
        seq.push_back(token);
        if (token == sampling.eos_token || seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}

std::vector<std::vector<std::size_t>> TransformerEncoder::batch_generate(const std::vector<std::vector<std::size_t>>& prompts,
                                                                         std::size_t max_new_tokens,
                                                                         const TransformerSamplingConfig* sampling,
                                                                         std::size_t beam_width,
                                                                         bool incremental) const {
    std::vector<std::vector<std::size_t>> out;
    out.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        if (sampling != nullptr) out.push_back(sample_decode(prompt, max_new_tokens, *sampling));
        else if (beam_width > 1) out.push_back(beam_search_decode(prompt, max_new_tokens, beam_width));
        else if (incremental) out.push_back(incremental_decode(prompt, max_new_tokens));
        else out.push_back(greedy_decode(prompt, max_new_tokens));
    }
    return out;
}

std::vector<autograd::Tensor> TransformerEncoder::collect_attention_maps(const autograd::Tensor& x,
                                                                         const autograd::Tensor* attention_mask) const {
    auto out = add_positional_encoding(x);
    std::vector<autograd::Tensor> maps;
    maps.reserve(blocks_.size());
    for (const auto& block : blocks_) {
        maps.push_back(block.attention_weights(out, attention_mask));
        out = block.forward(out, attention_mask);
    }
    return maps;
}

std::vector<autograd::Tensor*> TransformerEncoder::parameters() {
    std::vector<autograd::Tensor*> params;
    if (token_embedding_.ndim() == 2) params.push_back(&token_embedding_);
    if (lm_head_w_.ndim() == 2) params.push_back(&lm_head_w_);
    if (lm_head_b_.ndim() == 1) params.push_back(&lm_head_b_);
    if (classifier_w_.ndim() == 2) {
        params.push_back(&classifier_w_);
        params.push_back(&classifier_b_);
    }
    for (auto& block : blocks_) {
        auto block_params = block.parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }
    return params;
}

TransformerSeq2Seq::TransformerSeq2Seq(const TransformerConfig& config)
    : config_(config)
    , encoder_([&config]() {
        TransformerConfig c = config;
        c.causal = false;
        c.num_classes = 0;
        return c;
    }())
    , decoder_([&config]() {
        TransformerConfig c = config;
        c.causal = true;
        c.num_classes = 0;
        return c;
    }()) {
    cross_blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) cross_blocks_.emplace_back(config_.embed_dim, config_.ff_hidden_dim, config_.num_heads, true);
}

autograd::Tensor TransformerSeq2Seq::encode_tokens(const std::vector<std::vector<std::size_t>>& source_tokens) const {
    return encoder_.forward_tokens(source_tokens);
}

autograd::Tensor TransformerSeq2Seq::decode_tokens(const std::vector<std::vector<std::size_t>>& target_tokens,
                                                   const autograd::Tensor& memory,
                                                   const autograd::Tensor* target_mask,
                                                   const autograd::Tensor* memory_mask) const {
    auto decoded = decoder_.forward_tokens(target_tokens);
    for (const auto& block : cross_blocks_) decoded = block.decoder_step(decoded, memory, target_mask, memory_mask);
    return decoded;
}

autograd::Tensor TransformerSeq2Seq::forward_tokens(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                    const std::vector<std::vector<std::size_t>>& target_tokens,
                                                    const autograd::Tensor* source_mask,
                                                    const autograd::Tensor* target_mask) const {
    auto memory = encoder_.forward_tokens(source_tokens);
    auto fused = decode_tokens(target_tokens, memory, target_mask, source_mask);
    return decoder_.lm_projection(fused);
}


double TransformerSeq2Seq::teacher_forcing_loss(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                const std::vector<std::vector<std::size_t>>& target_tokens) const {
    if (target_tokens.empty() || target_tokens.front().size() < 2) throw std::invalid_argument("TransformerSeq2Seq::teacher_forcing_loss: need target length >= 2");
    auto logits = forward_tokens(source_tokens, target_tokens, nullptr, nullptr);
    const auto shape = logits.shape();
    double loss = 0.0;
    double count = 0.0;
    for (std::size_t b = 0; b < shape[0]; ++b) {
        for (std::size_t t = 0; t + 1 < shape[1]; ++t) {
            double max_logit = -1e18;
            for (std::size_t v = 0; v < shape[2]; ++v) max_logit = std::max(max_logit, logits.value_flat((b * shape[1] + t) * shape[2] + v));
            double sum_exp = 0.0;
            for (std::size_t v = 0; v < shape[2]; ++v) sum_exp += std::exp(logits.value_flat((b * shape[1] + t) * shape[2] + v) - max_logit);
            const std::size_t target = target_tokens[b][t + 1];
            const double target_logit = logits.value_flat((b * shape[1] + t) * shape[2] + target);
            loss += -(target_logit - max_logit - std::log(sum_exp));
            count += 1.0;
        }
    }
    return loss / std::max(1.0, count);
}

std::vector<autograd::Tensor> TransformerSeq2Seq::collect_cross_attention_maps(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                                               const std::vector<std::vector<std::size_t>>& target_tokens) const {
    auto memory = encoder_.forward_tokens(source_tokens);
    auto decoded = decoder_.forward_tokens(target_tokens);
    std::vector<autograd::Tensor> maps;
    maps.reserve(cross_blocks_.size());
    for (const auto& block : cross_blocks_) {
        maps.push_back(block.attention_weights(decoded, nullptr));
        decoded = block.decoder_step(decoded, memory, nullptr, nullptr);
    }
    return maps;
}

std::vector<std::size_t> TransformerSeq2Seq::greedy_decode(const std::vector<std::size_t>& source_tokens,
                                                           const std::vector<std::size_t>& prompt,
                                                           std::size_t max_new_tokens,
                                                           const std::vector<double>* source_mask) const {
    std::vector<std::size_t> seq = prompt;
    auto memory_mask = make_batch_mask(source_mask, source_tokens.size());
    const autograd::Tensor* memory_mask_ptr = memory_mask.ndim() == 2 ? &memory_mask : nullptr;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = forward_tokens({source_tokens}, {seq}, memory_mask_ptr, nullptr);
        const auto shape = logits.shape();
        const std::size_t last_pos = shape[1] - 1;
        std::size_t best_token = 0;
        double best_score = logits.value_flat(last_pos * shape[2]);
        for (std::size_t v = 1; v < shape[2]; ++v) {
            const double score = logits.value_flat(last_pos * shape[2] + v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }
        seq.push_back(best_token);
        if (seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}


std::vector<std::size_t> TransformerSeq2Seq::top_k_decode(const std::vector<std::size_t>& source_tokens,
                                                          const std::vector<std::size_t>& prompt,
                                                          std::size_t max_new_tokens,
                                                          std::size_t k,
                                                          const std::vector<double>* source_mask) const {
    if (k == 0) throw std::invalid_argument("TransformerSeq2Seq::top_k_decode: k must be > 0");
    std::vector<std::size_t> seq = prompt;
    auto memory_mask = make_batch_mask(source_mask, source_tokens.size());
    const autograd::Tensor* memory_mask_ptr = memory_mask.ndim() == 2 ? &memory_mask : nullptr;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = forward_tokens({source_tokens}, {seq}, memory_mask_ptr, nullptr);
        const auto shape = logits.shape();
        const std::size_t last_pos = shape[1] - 1;
        std::vector<std::pair<double, std::size_t>> scores;
        scores.reserve(shape[2]);
        for (std::size_t v = 0; v < shape[2]; ++v) scores.emplace_back(logits.value_flat(last_pos * shape[2] + v), v);
        std::partial_sort(scores.begin(), scores.begin() + std::min(k, scores.size()), scores.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        seq.push_back(scores.front().second);
        if (seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}


std::vector<std::size_t> TransformerSeq2Seq::beam_search_decode(const std::vector<std::size_t>& source_tokens,
                                                                const std::vector<std::size_t>& prompt,
                                                                std::size_t max_new_tokens,
                                                                std::size_t beam_width,
                                                                const std::vector<double>* source_mask) const {
    if (beam_width == 0) throw std::invalid_argument("TransformerSeq2Seq::beam_search_decode: beam_width must be > 0");
    auto memory_mask = make_batch_mask(source_mask, source_tokens.size());
    const autograd::Tensor* memory_mask_ptr = memory_mask.ndim() == 2 ? &memory_mask : nullptr;
    std::vector<std::pair<std::vector<std::size_t>, double>> beams{{prompt, 0.0}};
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<std::pair<std::vector<std::size_t>, double>> next_beams;
        for (const auto& beam : beams) {
            auto logits = forward_tokens({source_tokens}, {beam.first}, memory_mask_ptr, nullptr);
            const auto shape = logits.shape();
            const std::size_t last_pos = shape[1] - 1;
            std::vector<std::pair<double, std::size_t>> scores;
            scores.reserve(shape[2]);
            for (std::size_t v = 0; v < shape[2]; ++v) scores.emplace_back(logits.value_flat(last_pos * shape[2] + v), v);
            std::partial_sort(scores.begin(), scores.begin() + std::min(beam_width, scores.size()), scores.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t i = 0; i < std::min(beam_width, scores.size()); ++i) {
                auto seq = beam.first;
                seq.push_back(scores[i].second);
                next_beams.emplace_back(std::move(seq), beam.second + scores[i].first);
            }
        }
        std::partial_sort(next_beams.begin(), next_beams.begin() + std::min(beam_width, next_beams.size()), next_beams.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        beams.assign(next_beams.begin(), next_beams.begin() + std::min(beam_width, next_beams.size()));
        if (beams.front().first.size() >= config_.max_seq_len) break;
    }
    return beams.front().first;
}


std::vector<std::size_t> TransformerSeq2Seq::incremental_decode(const std::vector<std::size_t>& source_tokens,
                                                                const std::vector<std::size_t>& prompt,
                                                                std::size_t max_new_tokens,
                                                                const std::vector<double>* source_mask) const {
    std::vector<std::size_t> seq = prompt;
    auto memory_mask = make_batch_mask(source_mask, source_tokens.size());
    const autograd::Tensor* memory_mask_ptr = memory_mask.ndim() == 2 ? &memory_mask : nullptr;
    auto memory = encoder_.forward_tokens({source_tokens});
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto fused = decode_tokens({{seq.back()}}, memory, nullptr, memory_mask_ptr);
        auto logits = decoder_.lm_projection(fused);
        const auto shape = logits.shape();
        std::size_t best_token = 0;
        double best_score = logits.value_flat(shape[2] - 1);
        for (std::size_t v = 1; v < shape[2]; ++v) {
            const double score = logits.value_flat(v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }
        seq.push_back(best_token);
        if (seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}

autograd::Tensor TransformerSeq2Seq::make_batch_mask(const std::vector<double>* mask, std::size_t seq_len) const {
    if (mask == nullptr) return autograd::Tensor();
    if (mask->size() != seq_len) throw std::invalid_argument("TransformerSeq2Seq::make_batch_mask: mask size mismatch");
    return autograd::Tensor(*mask, {1, seq_len}, false);
}

std::vector<std::size_t> TransformerSeq2Seq::sample_decode(const std::vector<std::size_t>& source_tokens,
                                                           const std::vector<std::size_t>& prompt,
                                                           std::size_t max_new_tokens,
                                                           const TransformerSamplingConfig& sampling,
                                                           const std::vector<double>* source_mask) const {
    std::mt19937_64 rng(sampling.seed);
    std::vector<std::size_t> seq = prompt;
    auto memory_mask = make_batch_mask(source_mask, source_tokens.size());
    const autograd::Tensor* memory_mask_ptr = memory_mask.ndim() == 2 ? &memory_mask : nullptr;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = forward_tokens({source_tokens}, {seq}, memory_mask_ptr, nullptr);
        const auto shape = logits.shape();
        const std::size_t last_pos = shape[1] - 1;
        std::vector<double> row(shape[2], 0.0);
        for (std::size_t v = 0; v < shape[2]; ++v) row[v] = logits.value_flat(last_pos * shape[2] + v);
        const auto token = select_sampled_token(row, sampling, rng);
        seq.push_back(token);
        if (token == sampling.eos_token || seq.size() >= config_.max_seq_len) break;
    }
    return seq;
}

std::vector<std::vector<std::size_t>> TransformerSeq2Seq::batch_generate(const std::vector<std::vector<std::size_t>>& source_tokens_batch,
                                                                         const std::vector<std::vector<std::size_t>>& prompts,
                                                                         std::size_t max_new_tokens,
                                                                         const TransformerSamplingConfig* sampling,
                                                                         std::size_t beam_width,
                                                                         bool incremental,
                                                                         const std::vector<std::vector<double>>* source_masks) const {
    if (source_tokens_batch.size() != prompts.size()) throw std::invalid_argument("TransformerSeq2Seq::batch_generate: batch size mismatch");
    if (source_masks != nullptr && source_masks->size() != prompts.size()) throw std::invalid_argument("TransformerSeq2Seq::batch_generate: mask batch size mismatch");
    std::vector<std::vector<std::size_t>> out;
    out.reserve(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const std::vector<double>* mask = source_masks ? &((*source_masks)[i]) : nullptr;
        if (sampling != nullptr) out.push_back(sample_decode(source_tokens_batch[i], prompts[i], max_new_tokens, *sampling, mask));
        else if (beam_width > 1) out.push_back(beam_search_decode(source_tokens_batch[i], prompts[i], max_new_tokens, beam_width, mask));
        else if (incremental) out.push_back(incremental_decode(source_tokens_batch[i], prompts[i], max_new_tokens, mask));
        else out.push_back(greedy_decode(source_tokens_batch[i], prompts[i], max_new_tokens, mask));
    }
    return out;
}

std::vector<autograd::Tensor*> TransformerSeq2Seq::parameters() {
    auto enc = encoder_.parameters();
    auto dec = decoder_.parameters();
    enc.insert(enc.end(), dec.begin(), dec.end());
    for (auto& block : cross_blocks_) {
        auto block_params = block.parameters();
        enc.insert(enc.end(), block_params.begin(), block_params.end());
    }
    return enc;
}

TransformerSystem::TransformerSystem(const TransformerConfig& config)
    : config_(config)
    , encoder_([&config]() {
        TransformerConfig c = config;
        c.causal = false;
        c.num_classes = 0;
        return c;
    }())
    , classifier_([&config]() {
        TransformerConfig c = config;
        c.causal = false;
        if (c.num_classes == 0) c.num_classes = 2;
        return c;
    }())
    , language_model_([&config]() {
        TransformerConfig c = config;
        c.causal = true;
        c.num_classes = 0;
        return c;
    }())
    , seq2seq_(config) {}

autograd::Tensor TransformerSystem::encode_tokens(const std::vector<std::vector<std::size_t>>& token_ids,
                                                  const autograd::Tensor* attention_mask) const {
    if (attention_mask == nullptr) return encoder_.forward_tokens(token_ids);
    return encoder_.forward(encoder_.embed_tokens(token_ids), attention_mask);
}

autograd::Tensor TransformerSystem::classify_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const {
    return classifier_.classify_tokens(token_ids);
}

autograd::Tensor TransformerSystem::language_model_logits_tokens(const std::vector<std::vector<std::size_t>>& token_ids) const {
    return language_model_.language_model_logits_tokens(token_ids);
}

autograd::Tensor TransformerSystem::seq2seq_logits(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                   const std::vector<std::vector<std::size_t>>& target_tokens,
                                                   const autograd::Tensor* source_mask,
                                                   const autograd::Tensor* target_mask) const {
    return seq2seq_.forward_tokens(source_tokens, target_tokens, source_mask, target_mask);
}

double TransformerSystem::next_token_loss(const std::vector<std::vector<std::size_t>>& token_ids) const {
    return language_model_.next_token_loss(token_ids);
}

double TransformerSystem::teacher_forcing_loss(const std::vector<std::vector<std::size_t>>& source_tokens,
                                               const std::vector<std::vector<std::size_t>>& target_tokens) const {
    return seq2seq_.teacher_forcing_loss(source_tokens, target_tokens);
}

std::vector<std::size_t> TransformerSystem::generate_causal(const std::vector<std::size_t>& prompt,
                                                            std::size_t max_new_tokens,
                                                            const TransformerSamplingConfig* sampling,
                                                            std::size_t beam_width,
                                                            bool incremental) const {
    if (sampling != nullptr) return language_model_.sample_decode(prompt, max_new_tokens, *sampling);
    if (beam_width > 1) return language_model_.beam_search_decode(prompt, max_new_tokens, beam_width);
    if (incremental) return language_model_.incremental_decode(prompt, max_new_tokens);
    return language_model_.greedy_decode(prompt, max_new_tokens);
}

std::vector<std::size_t> TransformerSystem::generate_seq2seq(const std::vector<std::size_t>& source_tokens,
                                                             const std::vector<std::size_t>& prompt,
                                                             std::size_t max_new_tokens,
                                                             const TransformerSamplingConfig* sampling,
                                                             std::size_t beam_width,
                                                             bool incremental,
                                                             const std::vector<double>* source_mask) const {
    if (sampling != nullptr) return seq2seq_.sample_decode(source_tokens, prompt, max_new_tokens, *sampling, source_mask);
    if (beam_width > 1) return seq2seq_.beam_search_decode(source_tokens, prompt, max_new_tokens, beam_width, source_mask);
    if (incremental) return seq2seq_.incremental_decode(source_tokens, prompt, max_new_tokens, source_mask);
    return seq2seq_.greedy_decode(source_tokens, prompt, max_new_tokens, source_mask);
}

std::vector<autograd::Tensor> TransformerSystem::encoder_attention_maps(const autograd::Tensor& x,
                                                                        const autograd::Tensor* attention_mask) const {
    return encoder_.collect_attention_maps(x, attention_mask);
}

std::vector<autograd::Tensor> TransformerSystem::seq2seq_cross_attention_maps(const std::vector<std::vector<std::size_t>>& source_tokens,
                                                                              const std::vector<std::vector<std::size_t>>& target_tokens) const {
    return seq2seq_.collect_cross_attention_maps(source_tokens, target_tokens);
}

std::vector<std::vector<std::size_t>> TransformerSystem::generate_causal_batch(const std::vector<std::vector<std::size_t>>& prompts,
                                                                               std::size_t max_new_tokens,
                                                                               const TransformerSamplingConfig* sampling,
                                                                               std::size_t beam_width,
                                                                               bool incremental) const {
    return language_model_.batch_generate(prompts, max_new_tokens, sampling, beam_width, incremental);
}

std::vector<std::vector<std::size_t>> TransformerSystem::generate_seq2seq_batch(const std::vector<std::vector<std::size_t>>& source_tokens_batch,
                                                                                const std::vector<std::vector<std::size_t>>& prompts,
                                                                                std::size_t max_new_tokens,
                                                                                const TransformerSamplingConfig* sampling,
                                                                                std::size_t beam_width,
                                                                                bool incremental,
                                                                                const std::vector<std::vector<double>>* source_masks) const {
    return seq2seq_.batch_generate(source_tokens_batch, prompts, max_new_tokens, sampling, beam_width, incremental, source_masks);
}

std::vector<autograd::Tensor*> TransformerSystem::parameters() {
    auto params = encoder_.parameters();
    auto cls = classifier_.parameters();
    auto lm = language_model_.parameters();
    auto s2s = seq2seq_.parameters();
    params.insert(params.end(), cls.begin(), cls.end());
    params.insert(params.end(), lm.begin(), lm.end());
    params.insert(params.end(), s2s.begin(), s2s.end());
    return params;
}

}  // namespace transformer
