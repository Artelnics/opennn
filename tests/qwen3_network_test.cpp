#include "pch.h"

#include <random>
#include <vector>

#include "opennn/tensor_types.h"
#include "opennn/standard_networks.h"
#include "opennn/neural_network.h"
#include "opennn/configuration.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/device_backend.h"
#endif

using namespace opennn;

namespace
{

struct Dims
{
    Index seq, vocab, hidden, layers, q_heads, kv_heads, head_dim, intermediate;
    Index prompt1, decodes, prompt2;
};

constexpr Dims TINY { 16, 50, 32, 2, 4, 2, 8, 64, 5, 2, 8 };

void fill_parameters(NeuralNetwork& network)
{
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (auto& layer : network.get_layers())
        for (auto& view : layer->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] = nd(rng);
}

void run(NeuralNetwork& network, ForwardPropagation& forward_propagation,
         std::vector<float>& window, const std::vector<Index>& ids, Index past)
{
    const Index count = Index(ids.size());
    for (Index i = 0; i < count; ++i) window[size_t(i)] = float(ids[size_t(i)]);
    forward_propagation.past_length = past;
    forward_propagation.set_active_sequence_length(count);
    vector<TensorView> inputs = { TensorView(window.data(), {1, count}) };
    network.forward_propagate(inputs, forward_propagation, false);
}

std::vector<float> logits_row(const ForwardPropagation& forward_propagation, Index pos)
{
    const TensorView output = forward_propagation.get_outputs();
    const Index vocabulary = output.shape.back();
    std::vector<float> row(size_t(vocabulary), 0.0f);

    const Index elem = Index(type_bytes(output.type));
    std::vector<char> host(size_t(vocabulary) * size_t(elem));
    const char* src = static_cast<const char*>(output.data) + size_t(pos) * vocabulary * elem;

#ifdef OPENNN_HAS_CUDA
    if (output.is_cuda())
    {
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(host.data(), src, Index(host.size()), Device::CUDA, Device::CPU, stream);
        device::synchronize(stream);
    }
    else
#endif
        std::memcpy(host.data(), src, host.size());

    if (output.is_fp32())
        std::memcpy(row.data(), host.data(), size_t(vocabulary) * sizeof(float));
    else
    {
        const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(host.data());
        for (Index i = 0; i < vocabulary; ++i)
        {
            const uint32_t bits = uint32_t(bf16[size_t(i)]) << 16;
            std::memcpy(&row[size_t(i)], &bits, sizeof(float));
        }
    }
    return row;
}

// The chat flow: turn-1 prefill + decodes, then a turn-2 prefill with past = 0
// must give the same logits as a network that never ran turn 1. bf16_upload
// mimics the example's upload_parameters_bf16_inference().
float multi_turn_max_logit_diff(const Dims& d, bool bf16_upload = false)
{
    Qwen3 used(d.seq, d.vocab, d.hidden, d.layers, d.q_heads, d.kv_heads, d.head_dim, d.intermediate, 1000000.0f, 1.0e-6f);
    Qwen3 fresh(d.seq, d.vocab, d.hidden, d.layers, d.q_heads, d.kv_heads, d.head_dim, d.intermediate, 1000000.0f, 1.0e-6f);
    fill_parameters(used);
    fill_parameters(fresh);

#ifdef OPENNN_HAS_CUDA
    if (bf16_upload)
    {
        used.upload_parameters_bf16_inference();
        fresh.upload_parameters_bf16_inference();
    }
#else
    (void)bf16_upload;
#endif

    std::vector<float> window(size_t(d.seq), 0.0f);

    std::mt19937 id_rng(3);
    auto random_ids = [&](Index count) {
        std::vector<Index> ids(size_t(count), Index(0));
        for (auto& id : ids) id = 1 + Index(id_rng() % uint32_t(d.vocab - 1));
        return ids;
    };
    const std::vector<Index> prompt1 = random_ids(d.prompt1);
    const std::vector<Index> prompt2 = random_ids(d.prompt2);

    ForwardPropagation fp_used(1, &used);
    run(used, fp_used, window, prompt1, 0);
    for (Index i = 0; i < d.decodes; ++i)
        run(used, fp_used, window, { 1 + Index(id_rng() % uint32_t(d.vocab - 1)) }, d.prompt1 + i);

    run(used, fp_used, window, prompt2, 0);
    const std::vector<float> got = logits_row(fp_used, d.prompt2 - 1);

    ForwardPropagation fp_fresh(1, &fresh);
    run(fresh, fp_fresh, window, prompt2, 0);
    const std::vector<float> expected = logits_row(fp_fresh, d.prompt2 - 1);

    float max_diff = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i)
        max_diff = std::max(max_diff, std::abs(got[i] - expected[i]));
    return max_diff;
}

}


TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheCpu)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    EXPECT_LT(multi_turn_max_logit_diff(TINY), 1.0e-4f);
    Configuration::instance().set();
}


#ifdef OPENNN_HAS_CUDA
TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheGpu)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    // Both sides run BF16, so they must agree tightly unless cross-turn state leaks.
    EXPECT_LT(multi_turn_max_logit_diff(TINY), 1.0e-2f);
    Configuration::instance().set();
}


// Same, but through the example's large-model upload path
// (upload_parameters_bf16_inference: bf16 mirror + compact fp32, master released).
TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheGpuBf16Upload)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(multi_turn_max_logit_diff(TINY, /*bf16_upload*/ true), 1.0e-2f);
    Configuration::instance().set();
}


// Regression for the stale cuDNN tensor-descriptor bug: descriptors cached on
// arena views froze the first pass's sequence length, so a later LONGER prefill
// only added the first rows of the residual adds (multi-turn chat answered every
// turn like the first). Wide hidden with a growing second prefill reproduces it.
TEST(Qwen3NetworkTest, MultiTurnGrowingPrefillGpu)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    const Dims d { 64, 50, 2560, 2, 4, 2, 8, 64, 17, 0, 48 };
    EXPECT_LT(multi_turn_max_logit_diff(d, false), 1.0e-3f);
    Configuration::instance().set();
}


// Regression for decode graphs capturing the thread-global cuBLASLt scratch:
// a later suffix prefill can introduce a larger GEMM plan and move that global
// buffer. Decode must instead keep using its ForwardPropagation-owned stable
// workspaces, without recapturing between chat turns.
TEST(Qwen3NetworkTest, DecodeGraphSurvivesFiveSuffixPrefillsGpu)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    const Dims d { 64, 50, 32, 2, 4, 2, 8, 64, 4, 0, 0 };
    Qwen3 network(d.seq, d.vocab, d.hidden, d.layers,
                  d.q_heads, d.kv_heads, d.head_dim, d.intermediate,
                  1000000.0f, 1.0e-6f);
    fill_parameters(network);
    network.upload_parameters_bf16_inference();

    ForwardPropagation prefill(
        1, &network, ForwardPropagationMode::Inference);
    ForwardPropagation decode;
    decode.set(1, &network, &prefill.data, ForwardPropagationMode::Inference);
    decode.set_active_sequence_length(1);
    decode.set_cuda_graph(true);

    Buffer token_device{Device::CUDA};
    token_device.resize_bytes(Index(sizeof(float)), Device::CUDA);
    const vector<TensorView> decode_inputs = {
        TensorView(token_device.data, {1, 1}, Type::FP32, Device::CUDA)
    };

    vector<float> window(size_t(d.seq), 0.0f);
    run(network, prefill, window, {3, 7, 11, 13}, 0);
    Index position = 4;

    const auto decode_token = [&](Index token)
    {
        const float token_value = float(token);
        device::copy_async(token_device.data, &token_value, Index(sizeof(float)),
                           device::CopyKind::HostToDevice,
                           device::get_compute_stream());
        device::synchronize(device::get_compute_stream());
        decode.past_length = position++;
        return network.calculate_outputs_resident(decode_inputs, decode, false);
    };

    decode_token(17);  // measured warmup 1
    decode_token(19);  // measured warmup 2 + capture
    ASSERT_TRUE(static_cast<bool>(decode.inference_graph_exec));
    ASSERT_FALSE(decode.cuda_graph_workspaces_need_growth());
    auto* const graph_identity = decode.inference_graph_exec.get();

    const vector<vector<Index>> suffixes = {
        {2, 5},
        {23, 29, 31, 37, 41},
        {43},
        {3, 5, 7, 11, 13, 17, 19},
        {23, 31, 47}
    };

    for (size_t turn = 0; turn < suffixes.size(); ++turn)
    {
        run(network, prefill, window, suffixes[turn], position);
        position += Index(suffixes[turn].size());

        const Index token = 1 + Index((turn * 7 + 3) % size_t(d.vocab - 1));
        const TensorView graph_view = decode_token(token);
        const vector<float> graph_logits = logits_row(decode, 0);
        ASSERT_EQ(graph_view.data, decode.get_outputs().data);

        // Re-run the same position eagerly. It overwrites the same KV row with
        // the same values and gives a direct numerical reference while leaving
        // the instantiated graph intact.
        --position;
        decode.past_length = position;
        network.forward_propagate(decode_inputs, decode, false);
        ++position;
        const vector<float> eager_logits = logits_row(decode, 0);

        ASSERT_EQ(graph_logits.size(), eager_logits.size());
        EXPECT_EQ(distance(graph_logits.begin(),
                           max_element(graph_logits.begin(), graph_logits.end())),
                  distance(eager_logits.begin(),
                           max_element(eager_logits.begin(), eager_logits.end())))
            << "turn=" << turn;
        for (size_t i = 0; i < graph_logits.size(); ++i)
            ASSERT_NEAR(graph_logits[i], eager_logits[i], 1.0e-2f)
                << "turn=" << turn << " logit=" << i;

        EXPECT_EQ(decode.inference_graph_exec.get(), graph_identity);
        EXPECT_FALSE(decode.cuda_graph_workspaces_need_growth());
    }

    Configuration::instance().set();
}
#endif
