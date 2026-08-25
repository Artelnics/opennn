#include "tests/pch.h"

#include "tests/neural_network/llm_test_helpers.h"

#include <filesystem>
#include <random>
#include <vector>

#include "opennn/core/tensor_types.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/configuration.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/device_backend.h"

#endif

using namespace opennn;
using namespace opennn_test;

namespace
{

void write_logical_bf16_parameters(
    const NeuralNetwork& network,
    const filesystem::path& path)
{
    ASSERT_EQ(network.get_parameters_device(), Device::CPU);
    const Index parameters_number = network.get_parameters_buffer_size();
    const float* fp32 = network.get_parameters_data();

    vector<uint16_t> bf16(static_cast<size_t>(parameters_number));
    transform(fp32, fp32 + parameters_number, bf16.begin(), float_to_bfloat16_host);
    ofstream output(path, ios::binary | ios::trunc);
    ASSERT_TRUE(output.is_open());
    output.write(reinterpret_cast<const char*>(bf16.data()),
                 streamsize(bf16.size() * sizeof(uint16_t)));
    ASSERT_TRUE(output.good());
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
}

#ifdef OPENNN_HAS_CUDA

TEST(Int8InferenceTest, Int8TrainingThrowsGpu)
{
    Configuration::instance().set(Device::CUDA, Type::INT8);
    unique_ptr<Qwen3> network = make_qwen(TINY);
    Loss loss(network.get());
    EXPECT_THROW(BackPropagation(1, loss), exception);
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
}

#endif
