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

float max_difference(const std::vector<float>& a,
                     const std::vector<float>& b)
{
    EXPECT_EQ(a.size(), b.size());
    float result = 0.0f;
    for (size_t i = 0; i < min(a.size(), b.size()); ++i)
        result = max(result, abs(a[i] - b[i]));
    return result;
}

unique_ptr<Qwen3> make_tiny_qwen(const Dims& d)
{
    auto network = make_unique<Qwen3>(
        d.seq, d.vocab, d.hidden, d.layers, d.q_heads, d.kv_heads,
        d.head_dim, d.intermediate, 1000000.0f, 1.0e-6f);
    fill_parameters(*network);
    return network;
}

float compact_last_row_max_diff(const Dims& d, bool bf16_upload)
{
    unique_ptr<Qwen3> network = make_tiny_qwen(d);
#ifdef OPENNN_HAS_CUDA
    if (bf16_upload) network->upload_parameters_bf16_inference();
#else
    (void)bf16_upload;
#endif

    vector<float> window(size_t(d.seq), 0.0f);
    vector<Index> ids(size_t(d.prompt2));
    for (Index i = 0; i < d.prompt2; ++i)
        ids[size_t(i)] = i + 1;

    ForwardPropagation full(
        1, network.get(), ForwardPropagationMode::Inference);
    run(*network, full, window, ids, 0);
    const vector<float> expected =
        logits_row(full, d.prompt2 - 1);

    ForwardPropagation compact(
        1, network.get(), ForwardPropagationMode::Inference,
        {d.prompt2, 1});
    run(*network, compact, window, ids, 0);

    EXPECT_EQ(compact.get_outputs().shape[1], 1);
    EXPECT_LT(compact.data.bytes, full.data.bytes);
    return max_difference(expected, logits_row(compact, 0));
}

float chunked_prefill_and_decode_max_diff(const Dims& d,
                                          Index block,
                                          bool bf16_upload)
{
    unique_ptr<Qwen3> full_network = make_tiny_qwen(d);
    unique_ptr<Qwen3> chunked_network = make_tiny_qwen(d);
#ifdef OPENNN_HAS_CUDA
    if (bf16_upload)
    {
        full_network->upload_parameters_bf16_inference();
        chunked_network->upload_parameters_bf16_inference();
    }
#else
    (void)bf16_upload;
#endif

    vector<Index> ids(size_t(d.prompt2));
    for (Index i = 0; i < d.prompt2; ++i)
        ids[size_t(i)] = 1 + (i * 7) % (d.vocab - 1);
    vector<float> full_window(size_t(d.seq), 0.0f);
    vector<float> chunk_window(size_t(d.seq), 0.0f);

    ForwardPropagation full_prefill(
        1, full_network.get(), ForwardPropagationMode::Inference);
    run(*full_network, full_prefill, full_window, ids, 0);
    const vector<float> full_last =
        logits_row(full_prefill, d.prompt2 - 1);

    ForwardPropagation chunked_prefill(
        1, chunked_network.get(), ForwardPropagationMode::Inference,
        {block, 1});
    for (Index offset = 0; offset < d.prompt2; offset += block)
    {
        const Index count = min(block, d.prompt2 - offset);
        vector<Index> part(ids.begin() + offset,
                           ids.begin() + offset + count);
        run(*chunked_network, chunked_prefill,
            chunk_window, part, offset);
        chunked_prefill.set_output_sequence_window(count - 1, 1);
    }
    const vector<float> chunked_last =
        logits_row(chunked_prefill, 0);

    const Index decode_id = d.vocab - 1;
    ForwardPropagation full_decode(
        1, full_network.get(), ForwardPropagationMode::Inference, {1, 1});
    ForwardPropagation chunked_decode(
        1, chunked_network.get(), ForwardPropagationMode::Inference, {1, 1});
    run(*full_network, full_decode, full_window,
        {decode_id}, d.prompt2);
    run(*chunked_network, chunked_decode, chunk_window,
        {decode_id}, d.prompt2);

    return max(max_difference(full_last, chunked_last),
               max_difference(logits_row(full_decode, 0),
                              logits_row(chunked_decode, 0)));
}

uint16_t to_bfloat16(const float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

float from_bfloat16(const uint16_t value)
{
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
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

}

TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheCpu)
{
    EXPECT_LT(multi_turn_max_logit_diff(TINY), 1.0e-4f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, CompactLogitsEqualFullLastRowCpu)
{
    EXPECT_LT(compact_last_row_max_diff(TINY, false), 1.0e-4f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, ChunkedPrefillAndDecodeEqualFullPassCpu)
{
    EXPECT_LT(chunked_prefill_and_decode_max_diff(TINY, 3, false),
              1.0e-4f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, CompactPoolDependsOnBlockNotModelContext)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    Dims short_dims = TINY;
    Dims long_dims = TINY;
    short_dims.seq = 16;
    long_dims.seq = 64;

    unique_ptr<Qwen3> short_network = make_tiny_qwen(short_dims);
    unique_ptr<Qwen3> long_network = make_tiny_qwen(long_dims);
    ForwardPropagation short_compact(
        1, short_network.get(), ForwardPropagationMode::Inference, {4, 1});
    ForwardPropagation long_compact(
        1, long_network.get(), ForwardPropagationMode::Inference, {4, 1});

    EXPECT_EQ(short_compact.data.bytes, long_compact.data.bytes);
    EXPECT_EQ(short_compact.get_sequence_capacity(), 4);
    EXPECT_EQ(long_compact.get_sequence_capacity(), 4);
    EXPECT_EQ(short_compact.get_final_output_capacity(), 1);
    EXPECT_EQ(long_compact.get_final_output_capacity(), 1);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, CompactOutputWindowMatchesSelectedFullRowsCpu)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    unique_ptr<Qwen3> network = make_tiny_qwen(TINY);
    vector<float> window(size_t(TINY.seq), 0.0f);
    const vector<Index> ids = {2, 3, 5, 7, 11, 13};

    ForwardPropagation full(
        1, network.get(), ForwardPropagationMode::Inference);
    run(*network, full, window, ids, 0);

    ForwardPropagation selected(
        1, network.get(), ForwardPropagationMode::Inference, {6, 4});
    run(*network, selected, window, ids, 0);
    selected.set_output_sequence_window(1, 4);
    vector<TensorView> inputs = {
        TensorView(window.data(), {1, Index(ids.size())})
    };
    network->forward_propagate(inputs, selected, false);

    ASSERT_EQ(selected.get_outputs().shape[1], 4);
    for (Index row = 0; row < 4; ++row)
        EXPECT_LT(max_difference(logits_row(full, row + 1),
                                 logits_row(selected, row)),
                  1.0e-4f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, DirectLogicalBf16WeightsMatchRoundedCpu)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    Qwen3 expected(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    Qwen3 loaded(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    fill_parameters(expected);
    round_parameters_to_bf16(expected);

    const filesystem::path path =
        filesystem::temp_directory_path()
        / "opennn_qwen3_logical_bf16_cpu.bin";
    write_logical_bf16_parameters(expected, path);
    loaded.load_parameters_bf16_inference_binary(path);

    vector<float> expected_window(size_t(TINY.seq), 0.0f);
    vector<float> loaded_window(size_t(TINY.seq), 0.0f);
    const vector<Index> ids = {2, 3, 5, 7, 11};
    ForwardPropagation expected_fp(1, &expected);
    ForwardPropagation loaded_fp(1, &loaded);
    run(expected, expected_fp, expected_window, ids, 0);
    run(loaded, loaded_fp, loaded_window, ids, 0);

    EXPECT_LT(max_difference(logits_row(expected_fp, ssize(ids) - 1),
                             logits_row(loaded_fp, ssize(ids) - 1)),
              1.0e-6f);
    filesystem::remove(path);
    Configuration::instance().set();
}

#ifdef OPENNN_HAS_CUDA
TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheGpu)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    EXPECT_LT(multi_turn_max_logit_diff(TINY), 1.0e-2f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheGpuBf16Upload)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(multi_turn_max_logit_diff(TINY,  true), 1.0e-2f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, DirectLogicalBf16WeightsMatchUploadGpu)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    Qwen3 uploaded(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    Qwen3 direct(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    fill_parameters(uploaded);
    round_parameters_to_bf16(uploaded);

    const filesystem::path path =
        filesystem::temp_directory_path()
        / "opennn_qwen3_logical_bf16_gpu.bin";
    write_logical_bf16_parameters(uploaded, path);
    direct.load_parameters_bf16_inference_binary(path);
    uploaded.upload_parameters_bf16_inference();

    vector<float> uploaded_window(size_t(TINY.seq), 0.0f);
    vector<float> direct_window(size_t(TINY.seq), 0.0f);
    const vector<Index> ids = {2, 3, 5, 7, 11};
    ForwardPropagation uploaded_fp(1, &uploaded);
    ForwardPropagation direct_fp(1, &direct);
    run(uploaded, uploaded_fp, uploaded_window, ids, 0);
    run(direct, direct_fp, direct_window, ids, 0);

    EXPECT_LT(max_difference(logits_row(uploaded_fp, ssize(ids) - 1),
                             logits_row(direct_fp, ssize(ids) - 1)),
              1.0e-2f);
    filesystem::remove(path);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, CompactLogitsEqualFullLastRowGpuBf16)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(compact_last_row_max_diff(TINY, true), 1.0e-2f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, ChunkedPrefillAndDecodeEqualFullPassGpuBf16)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(chunked_prefill_and_decode_max_diff(TINY, 3, true),
              1.0e-2f);
    Configuration::instance().set();
}

TEST(Qwen3NetworkTest, MultiTurnGrowingPrefillGpu)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    const Dims d { 64, 50, 2560, 2, 4, 2, 8, 64, 17, 0, 48 };
    EXPECT_LT(multi_turn_max_logit_diff(d, false), 1.0e-3f);
    Configuration::instance().set();
}

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

    decode_token(17);
    decode_token(19);
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
