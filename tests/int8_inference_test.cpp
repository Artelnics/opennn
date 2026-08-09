#include "pch.h"

#include <filesystem>
#include <random>
#include <vector>

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/configuration.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/device_backend.h"
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
constexpr Dims WIDE { 32, 50, 32, 2, 4, 2, 8, 64, 20, 2, 24 };

unique_ptr<Qwen3> make_qwen(const Dims& d)
{
    return make_unique<Qwen3>(
        d.seq, d.vocab, d.hidden, d.layers, d.q_heads, d.kv_heads,
        d.head_dim, d.intermediate, 1000000.0f, 1.0e-6f);
}

void fill_parameters(NeuralNetwork& network)
{
    mt19937 rng(21);
    normal_distribution<float> nd(0.0f, 0.05f);
    for (auto& layer : network.get_layers())
        for (auto& view : layer->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] = nd(rng);
}

void run(NeuralNetwork& network, ForwardPropagation& forward_propagation,
         vector<float>& window, const vector<Index>& ids, Index past)
{
    const Index count = Index(ids.size());
    for (Index i = 0; i < count; ++i) window[size_t(i)] = float(ids[size_t(i)]);
    forward_propagation.past_length = past;
    forward_propagation.set_active_sequence_length(count);
    vector<TensorView> inputs = { TensorView(window.data(), {1, count}) };
    network.forward_propagate(inputs, forward_propagation, false);
}

vector<float> logits_row(const ForwardPropagation& forward_propagation, Index pos)
{
    const TensorView output = forward_propagation.get_outputs();
    const Index vocabulary = output.shape.back();
    vector<float> row(size_t(vocabulary), 0.0f);

    const Index elem = Index(type_bytes(output.type));
    vector<char> host(size_t(vocabulary) * size_t(elem));
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
        memcpy(host.data(), src, host.size());

    if (output.is_fp32())
        memcpy(row.data(), host.data(), size_t(vocabulary) * sizeof(float));
    else
    {
        const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(host.data());
        for (Index i = 0; i < vocabulary; ++i)
        {
            const uint32_t bits = uint32_t(bf16[size_t(i)]) << 16;
            memcpy(&row[size_t(i)], &bits, sizeof(float));
        }
    }
    return row;
}

float max_difference(const vector<float>& a,
                     const vector<float>& b)
{
    EXPECT_EQ(a.size(), b.size());
    float result = 0.0f;
    for (size_t i = 0; i < min(a.size(), b.size()); ++i)
        result = max(result, abs(a[i] - b[i]));
    return result;
}

uint16_t to_bfloat16(const float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

float from_bfloat16(const uint16_t value)
{
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

void round_parameters_to_bf16(NeuralNetwork& network)
{
    for (auto& layer : network.get_layers())
        for (TensorView& view : layer->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] =
                    from_bfloat16(to_bfloat16(view.as<float>()[i]));
}

void write_logical_bf16_parameters(
    const NeuralNetwork& network,
    const filesystem::path& path)
{
    filesystem::path fp32_path = path;
    fp32_path += ".fp32";
    network.save_parameters_binary(fp32_path);

    ifstream input(fp32_path, ios::binary | ios::ate);
    ASSERT_TRUE(input.is_open());
    const streamoff bytes = input.tellg();
    ASSERT_GE(bytes, 0);
    ASSERT_EQ(bytes % streamoff(sizeof(float)), 0);
    input.seekg(0);

    vector<float> fp32(size_t(bytes / streamoff(sizeof(float))));
    input.read(reinterpret_cast<char*>(fp32.data()), bytes);
    ASSERT_TRUE(input.good());

    vector<uint16_t> bf16(fp32.size());
    transform(fp32.begin(), fp32.end(), bf16.begin(), to_bfloat16);
    ofstream output(path, ios::binary | ios::trunc);
    ASSERT_TRUE(output.is_open());
    output.write(reinterpret_cast<const char*>(bf16.data()),
                 streamsize(bf16.size() * sizeof(uint16_t)));
    ASSERT_TRUE(output.good());

    input.close();
    error_code remove_error;
    filesystem::remove(fp32_path, remove_error);
}

void fake_quantize_parameters(NeuralNetwork& network)
{
    for (auto& layer : network.get_layers())
    {
        const auto specs = layer->get_parameter_specs();
        const auto quantization = layer->get_parameter_quantization();
        const Layer::TiedWeight tie = layer->get_tied_weight();
        auto& views = layer->get_parameter_views();

        for (size_t i = 0; i < views.size() && i < specs.size(); ++i)
        {
            if (specs[i].dtype == Type::FP32) continue;
            if (i >= quantization.size() || quantization[i].channels <= 0) continue;
            if (tie.source && i == tie.spec_index) continue;

            TensorView& view = views[i];
            const Index size = view.size();
            const Index channels = quantization[i].channels;
            const Index row_length = size / channels;
            const int axis = quantization[i].axis;
            float* values = view.as<float>();

            vector<float> scales(size_t(channels), 0.0f);
            for (Index j = 0; j < size; ++j)
            {
                const Index channel = axis == 0 ? j / row_length : j % channels;
                scales[size_t(channel)] = max(scales[size_t(channel)], abs(values[j]));
            }
            for (float& scale : scales) scale = scale > 0.0f ? scale / 127.0f : 1.0f;

            for (Index j = 0; j < size; ++j)
            {
                const Index channel = axis == 0 ? j / row_length : j % channels;
                const float scale = scales[size_t(channel)];
                values[j] = clamp(roundf(values[j] / scale), -127.0f, 127.0f) * scale;
            }
        }
    }
}

}

TEST(Int8InferenceTest, Int8CpuConfigurationThrows)
{
    Configuration::instance().set(Device::CPU, Type::INT8);
    EXPECT_THROW(make_qwen(TINY), exception);
    Configuration::instance().set();
}

#ifdef OPENNN_HAS_CUDA

TEST(Int8InferenceTest, Int8TrainingThrowsGpu)
{
    Configuration::instance().set(Device::CUDA, Type::INT8);
    unique_ptr<Qwen3> network = make_qwen(TINY);
    Loss loss(network.get());
    EXPECT_THROW(BackPropagation(1, &loss), exception);
    Configuration::instance().set();
}

TEST(Int8InferenceTest, DirectLogicalBf16WeightsMatchUploadGpuInt8)
{
    Configuration::instance().set(Device::CUDA, Type::INT8);
    unique_ptr<Qwen3> uploaded = make_qwen(TINY);
    unique_ptr<Qwen3> direct = make_qwen(TINY);
    fill_parameters(*uploaded);
    round_parameters_to_bf16(*uploaded);

    const filesystem::path path =
        filesystem::temp_directory_path()
        / "opennn_qwen3_logical_bf16_int8.bin";
    write_logical_bf16_parameters(*uploaded, path);
    direct->load_parameters_bf16_inference_binary(path);
    uploaded->upload_parameters_int8_inference();

    vector<float> uploaded_window(size_t(TINY.seq), 0.0f);
    vector<float> direct_window(size_t(TINY.seq), 0.0f);
    const vector<Index> ids = {2, 3, 5, 7, 11};
    ForwardPropagation uploaded_fp(1, uploaded.get());
    ForwardPropagation direct_fp(1, direct.get());
    run(*uploaded, uploaded_fp, uploaded_window, ids, 0);
    run(*direct, direct_fp, direct_window, ids, 0);

    EXPECT_LT(max_difference(logits_row(uploaded_fp, ssize(ids) - 1),
                             logits_row(direct_fp, ssize(ids) - 1)),
              1.0e-6f);
    filesystem::remove(path);
    Configuration::instance().set();
}

TEST(Int8InferenceTest, Int8MultiTurnPrefillRestartsCacheGpu)
{
    Configuration::instance().set(Device::CUDA, Type::INT8);
    unique_ptr<Qwen3> used = make_qwen(TINY);
    unique_ptr<Qwen3> fresh = make_qwen(TINY);
    fill_parameters(*used);
    fill_parameters(*fresh);
    used->upload_parameters_int8_inference();
    fresh->upload_parameters_int8_inference();

    vector<float> window(size_t(TINY.seq), 0.0f);
    mt19937 id_rng(3);
    auto random_ids = [&](Index count) {
        vector<Index> ids(size_t(count), Index(0));
        for (auto& id : ids) id = 1 + Index(id_rng() % uint32_t(TINY.vocab - 1));
        return ids;
    };
    const vector<Index> prompt1 = random_ids(TINY.prompt1);
    const vector<Index> prompt2 = random_ids(TINY.prompt2);

    ForwardPropagation fp_used(1, used.get());
    run(*used, fp_used, window, prompt1, 0);
    for (Index i = 0; i < TINY.decodes; ++i)
        run(*used, fp_used, window,
            { 1 + Index(id_rng() % uint32_t(TINY.vocab - 1)) }, TINY.prompt1 + i);
    run(*used, fp_used, window, prompt2, 0);
    const vector<float> got = logits_row(fp_used, TINY.prompt2 - 1);

    ForwardPropagation fp_fresh(1, fresh.get());
    run(*fresh, fp_fresh, window, prompt2, 0);
    const vector<float> expected = logits_row(fp_fresh, TINY.prompt2 - 1);

    EXPECT_LT(max_difference(got, expected), 1.0e-2f);
    Configuration::instance().set();
}

TEST(Int8InferenceTest, Int8ChunkedPrefillAndDecodeEqualFullPassGpu)
{
    Configuration::instance().set(Device::CUDA, Type::INT8);
    unique_ptr<Qwen3> full_network = make_qwen(WIDE);
    unique_ptr<Qwen3> chunked_network = make_qwen(WIDE);
    fill_parameters(*full_network);
    fill_parameters(*chunked_network);
    full_network->upload_parameters_int8_inference();
    chunked_network->upload_parameters_int8_inference();

    vector<Index> ids(size_t(WIDE.prompt2));
    for (Index i = 0; i < WIDE.prompt2; ++i)
        ids[size_t(i)] = 1 + (i * 7) % (WIDE.vocab - 1);
    vector<float> full_window(size_t(WIDE.seq), 0.0f);
    vector<float> chunk_window(size_t(WIDE.seq), 0.0f);

    ForwardPropagation full_prefill(
        1, full_network.get(), ForwardPropagationMode::Inference);
    run(*full_network, full_prefill, full_window, ids, 0);
    const vector<float> full_last = logits_row(full_prefill, WIDE.prompt2 - 1);

    const Index block = 3;
    ForwardPropagation chunked_prefill(
        1, chunked_network.get(), ForwardPropagationMode::Inference,
        {block, 1});
    for (Index offset = 0; offset < WIDE.prompt2; offset += block)
    {
        const Index count = min(block, WIDE.prompt2 - offset);
        vector<Index> part(ids.begin() + offset, ids.begin() + offset + count);
        run(*chunked_network, chunked_prefill, chunk_window, part, offset);
        chunked_prefill.set_output_sequence_window(count - 1, 1);
    }
    const vector<float> chunked_last = logits_row(chunked_prefill, 0);

    const Index decode_id = WIDE.vocab - 1;
    ForwardPropagation full_decode(
        1, full_network.get(), ForwardPropagationMode::Inference, {1, 1});
    ForwardPropagation chunked_decode(
        1, chunked_network.get(), ForwardPropagationMode::Inference, {1, 1});
    run(*full_network, full_decode, full_window, {decode_id}, WIDE.prompt2);
    run(*chunked_network, chunked_decode, chunk_window, {decode_id}, WIDE.prompt2);

    EXPECT_LT(max_difference(full_last, chunked_last), 1.0e-2f);
    EXPECT_LT(max_difference(logits_row(full_decode, 0),
                             logits_row(chunked_decode, 0)),
              1.0e-2f);
    Configuration::instance().set();
}

TEST(Int8InferenceTest, Int8MatchesFakeQuantBf16ReferenceGpu)
{
    Configuration::instance().set(Device::CUDA, Type::INT8);
    unique_ptr<Qwen3> quantized = make_qwen(TINY);
    fill_parameters(*quantized);
    quantized->upload_parameters_int8_inference();

    Configuration::instance().set(Device::CUDA, Type::BF16);
    unique_ptr<Qwen3> reference = make_qwen(TINY);
    fill_parameters(*reference);
    fake_quantize_parameters(*reference);
    reference->upload_parameters_bf16_inference();

    vector<float> quantized_window(size_t(TINY.seq), 0.0f);
    vector<float> reference_window(size_t(TINY.seq), 0.0f);
    const vector<Index> ids = {2, 3, 5, 7, 11};
    ForwardPropagation quantized_fp(1, quantized.get());
    ForwardPropagation reference_fp(1, reference.get());
    run(*quantized, quantized_fp, quantized_window, ids, 0);
    run(*reference, reference_fp, reference_window, ids, 0);

    EXPECT_LT(max_difference(logits_row(quantized_fp, ssize(ids) - 1),
                             logits_row(reference_fp, ssize(ids) - 1)),
              5.0e-2f);
    Configuration::instance().set();
}

#endif
