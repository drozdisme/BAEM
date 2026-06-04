#include "models/pideeponet/pideeponet.hpp"

#include "models/deep_onet/loss.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace pideeponet {

namespace {

constexpr char kMagic[8] = {'U','M','L','P','I','D','1','\0'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
void write_raw(std::vector<char>& out, const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T read_raw(const std::vector<char>& in, std::size_t& off) {
    if (off + sizeof(T) > in.size()) throw std::runtime_error("PIDeepONet deserialize: truncated payload");
    T v{};
    std::memcpy(&v, in.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
}

std::uint64_t fnv1a64(const std::vector<char>& data) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : data) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

void write_vector_size(std::vector<char>& out, const std::vector<std::size_t>& values) {
    write_raw(out, values.size());
    for (std::size_t v : values) write_raw(out, v);
}

std::vector<std::size_t> read_vector_size(const std::vector<char>& in, std::size_t& off) {
    const std::size_t n = read_raw<std::size_t>(in, off);
    std::vector<std::size_t> values(n);
    for (std::size_t& v : values) v = read_raw<std::size_t>(in, off);
    return values;
}

}  // namespace

PIDeepONet::PIDeepONet(const PIDeepONetConfig& config)
    : config_(config)
    , model_(config.branch_input_dim, config.branch_hidden_dims,
             config.trunk_input_dim, config.trunk_hidden_dims,
             config.latent_dim, config.branch_act, config.trunk_act) {
    if (config_.branch_input_dim == 0 || config_.trunk_input_dim == 0 || config_.latent_dim == 0) {
        throw std::invalid_argument("PIDeepONet: input dimensions and latent_dim must be > 0");
    }
}

PIDeepONet::PIDeepONet(std::size_t branch_input_dim,
                       const std::vector<std::size_t>& branch_hidden_dims,
                       std::size_t trunk_input_dim,
                       const std::vector<std::size_t>& trunk_hidden_dims,
                       std::size_t latent_dim,
                       deep_onet::Activation branch_act,
                       deep_onet::Activation trunk_act)
    : PIDeepONet(PIDeepONetConfig{branch_input_dim, branch_hidden_dims,
                                  trunk_input_dim, trunk_hidden_dims,
                                  latent_dim, branch_act, trunk_act}) {}

autograd::Tensor PIDeepONet::forward(const autograd::Tensor& u_batch,
                                     const autograd::Tensor& y_batch) const {
    if (u_batch.ndim() != 2 || y_batch.ndim() != 2) {
        throw std::invalid_argument("PIDeepONet::forward: expected rank-2 branch and trunk batches");
    }
    if (u_batch.shape().front() != y_batch.shape().front()) {
        throw std::invalid_argument("PIDeepONet::forward: branch and trunk batch sizes must match");
    }
    if (u_batch.shape().back() != config_.branch_input_dim) {
        throw std::invalid_argument("PIDeepONet::forward: branch feature dimension mismatch");
    }
    if (y_batch.shape().back() != config_.trunk_input_dim) {
        throw std::invalid_argument("PIDeepONet::forward: trunk feature dimension mismatch");
    }
    return model_.forward(u_batch, y_batch);
}

LossBreakdown PIDeepONet::loss(const autograd::Tensor& u_batch,
                               const autograd::Tensor& y_batch,
                               const autograd::Tensor& target,
                               const ResidualFn& residual_fn,
                               double physics_weight,
                               double data_weight) const {
    if (!residual_fn) {
        throw std::invalid_argument("PIDeepONet::loss: residual_fn must be provided");
    }
    if (physics_weight < 0.0 || data_weight < 0.0) {
        throw std::invalid_argument("PIDeepONet::loss: loss weights must be >= 0");
    }
    auto pred = forward(u_batch, y_batch);
    if (target.shape() != pred.shape()) {
        throw std::invalid_argument("PIDeepONet::loss: target shape must match model output shape");
    }
    auto data_loss = deep_onet::mse_loss(pred, target);
    auto residual = residual_fn(pred, y_batch);
    if (residual.numel() == 0) {
        throw std::invalid_argument("PIDeepONet::loss: residual_fn must return a non-empty tensor");
    }
    auto physics_loss = autograd::mean(residual * residual);
    auto total = data_loss * data_weight + physics_loss * physics_weight;
    return {total, data_loss, physics_loss};
}

std::vector<autograd::Tensor*> PIDeepONet::parameters() {
    return model_.parameters();
}

void PIDeepONet::zero_grad() {
    model_.zero_grad();
}

void PIDeepONet::save(const std::string& filepath) const {
    auto params = const_cast<PIDeepONet*>(this)->parameters();
    std::vector<char> payload;

    write_raw(payload, config_.branch_input_dim);
    write_vector_size(payload, config_.branch_hidden_dims);
    write_raw(payload, config_.trunk_input_dim);
    write_vector_size(payload, config_.trunk_hidden_dims);
    write_raw(payload, config_.latent_dim);
    write_raw(payload, static_cast<int>(config_.branch_act));
    write_raw(payload, static_cast<int>(config_.trunk_act));

    write_raw(payload, params.size());
    for (const auto* p : params) {
        const auto& shape = p->shape();
        write_raw(payload, shape.size());
        for (std::size_t dim : shape) write_raw(payload, dim);
        for (std::size_t i = 0; i < p->numel(); ++i) write_raw(payload, p->value_flat(i));
    }

    const std::uint64_t checksum = fnv1a64(payload);
    const std::uint64_t payload_size = static_cast<std::uint64_t>(payload.size());

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("PIDeepONet::save: cannot open file");
    ofs.write(kMagic, sizeof(kMagic));
    ofs.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    ofs.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
    ofs.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

PIDeepONet PIDeepONet::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) throw std::runtime_error("PIDeepONet::load: cannot open file");

    char magic[8]{};
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0, checksum = 0;
    ifs.read(magic, sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
    ifs.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        throw std::runtime_error("PIDeepONet::load: invalid magic");
    if (version != kVersion)
        throw std::runtime_error("PIDeepONet::load: unsupported version");

    std::vector<char> payload(static_cast<std::size_t>(payload_size));
    ifs.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (fnv1a64(payload) != checksum)
        throw std::runtime_error("PIDeepONet::load: checksum mismatch");

    std::size_t off = 0;
    PIDeepONetConfig config;
    config.branch_input_dim = read_raw<std::size_t>(payload, off);
    config.branch_hidden_dims = read_vector_size(payload, off);
    config.trunk_input_dim = read_raw<std::size_t>(payload, off);
    config.trunk_hidden_dims = read_vector_size(payload, off);
    config.latent_dim = read_raw<std::size_t>(payload, off);
    config.branch_act = static_cast<deep_onet::Activation>(read_raw<int>(payload, off));
    config.trunk_act = static_cast<deep_onet::Activation>(read_raw<int>(payload, off));

    PIDeepONet model(config);
    auto params = model.parameters();
    const std::size_t param_count = read_raw<std::size_t>(payload, off);
    if (param_count != params.size()) {
        throw std::runtime_error("PIDeepONet::load: parameter count mismatch");
    }

    for (std::size_t idx = 0; idx < params.size(); ++idx) {
        auto* p = params[idx];
        const std::size_t rank = read_raw<std::size_t>(payload, off);
        std::vector<std::size_t> shape(rank);
        for (std::size_t& dim : shape) dim = read_raw<std::size_t>(payload, off);
        if (shape != p->shape()) {
            throw std::runtime_error("PIDeepONet::load: parameter shape mismatch");
        }
        std::vector<double> values(p->numel());
        for (double& v : values) v = read_raw<double>(payload, off);
        *p = autograd::Tensor(std::move(values), shape, true);
    }
    return model;
}

}  // namespace pideeponet
