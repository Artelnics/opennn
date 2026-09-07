#include "tests/pch.h"

#include "tests/neural_network/llm_test_helpers.h"

#include <fstream>
#include <iterator>
#include <random>
#include <vector>

#include "opennn/core/tensor_types.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/layers/grouped_query_attention_layer.h"
#include "opennn/core/configuration.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/device_backend.h"

#endif

using namespace opennn;
using namespace opennn_test;

TEST(Qwen3NetworkTest, CompactScratchIsLimitedToBf16CudaInference)
{
#ifdef OPENNN_HAS_CUDA
    if (!device::has_cuda_device()) GTEST_SKIP();
    Configuration::instance().set(Device::CUDA, Type::BF16);
    Qwen3 network(64, 96, 32, 1, 4, 2, 8, 64);

    ForwardPropagation compact(1, &network, ForwardPropagationMode::Inference);
    ForwardPropagation training(1, &network, ForwardPropagationMode::Training);
    ForwardPropagation batched(2, &network, ForwardPropagationMode::Inference);
    bool found = false;
    for (size_t i = 0; i < network.get_layers().size(); ++i)
    {
        const auto* attention = dynamic_cast<const GroupedQueryAttention*>(network.get_layers()[i].get());
        if (!attention) continue;
        found = true;
        EXPECT_TRUE(attention->uses_compact_inference());
        for (const size_t slot : {size_t(3), size_t(5)})
        {
            EXPECT_EQ(compact.slots[i][slot].size(), 0);
            EXPECT_GT(training.slots[i][slot].size(), 0);
            EXPECT_GT(batched.slots[i][slot].size(), 0);
            EXPECT_FALSE(attention->is_forward_slot_inference_elidable(slot, Device::CPU, 1));
        }
    }
    EXPECT_TRUE(found);
    GroupedQueryAttention generic({64, 32}, 4, 2, 8);
    EXPECT_FALSE(generic.uses_compact_inference());
    EXPECT_FALSE(generic.is_forward_slot_inference_elidable(3, Device::CUDA, 1));
    Configuration::instance().set(Device::CUDA, Type::INT8);
    Qwen3 quantized(64, 96, 32, 1, 4, 2, 8, 64);
    ForwardPropagation int8_inference(1, &quantized, ForwardPropagationMode::Inference);
    EXPECT_FALSE(int8_inference.reserve_kv_cache(32, 0));
    for (size_t i = 0; i < quantized.get_layers().size(); ++i)
        if (dynamic_cast<GroupedQueryAttention*>(quantized.get_layers()[i].get()))
        {
            EXPECT_GT(int8_inference.slots[i][3].size(), 0);
            EXPECT_GT(int8_inference.slots[i][5].size(), 0);
        }
    Configuration::instance().set(Device::CUDA, Type::BF16);
#else
    GTEST_SKIP() << "Requires CUDA.";
#endif
}

namespace
{

#ifdef OPENNN_HAS_CUDA
TEST(Qwen3NetworkTest, KvBucketsPreserveBothHalvesAndFailedGrowth)
{
    if (!device::has_cuda_device()) GTEST_SKIP();
    Configuration::instance().set(Device::CUDA, Type::BF16);
    Qwen3 network(2304, 96, 32, 2, 4, 2, 8, 64);
    ForwardPropagation propagation(1, &network, ForwardPropagationMode::Inference, {256, 1});
    ASSERT_TRUE(propagation.reserve_kv_cache(128, 0));
    vector<size_t> attention_layers;
    vector<vector<uint16_t>> originals;
    for (size_t i = 0; i < network.get_layers().size(); ++i)
    {
        if (!dynamic_cast<GroupedQueryAttention*>(network.get_layers()[i].get())) continue;
        attention_layers.push_back(i);
        Buffer& buffer = (*propagation.layer_session_state_storage)[i];
        ASSERT_EQ(buffer.byte_size(), 2 * 256 * 16 * Index(sizeof(uint16_t)));
        originals.emplace_back(size_t(buffer.byte_size() / 2));
        for (size_t j = 0; j < originals.back().size(); ++j)
            originals.back()[j] = uint16_t(j + i);
        device::copy_async(buffer.data(), originals.back().data(), buffer.byte_size(),
                           device::CopyKind::HostToDevice, device::get_compute_stream());
    }
    device::synchronize(device::get_compute_stream());
    const void* old_pointer = (*propagation.layer_session_state_storage)[attention_layers[0]].data();
    EXPECT_FALSE(propagation.reserve_kv_cache(256, 200));
    EXPECT_EQ((*propagation.layer_session_state_storage)[attention_layers[0]].data(), old_pointer);
    {
        const device::CudaBlockCacheBypass bypass;
        const device::CudaAllocationGrowthGuard no_growth(true, false);
        EXPECT_THROW(propagation.reserve_kv_cache(257, 200), runtime_error);
    }
    EXPECT_EQ((*propagation.layer_session_state_storage)[attention_layers[0]].data(), old_pointer);
    {
        // Fail on the second layer, after the first replacement was copied.
        // The failure must not commit a partially grown multi-layer cache.
        Buffer& second = (*propagation.layer_session_state_storage)[attention_layers[1]];
        Buffer original;
        original.swap(second);
        second.set_view(original.data(), original.byte_size() - 2, Device::CUDA);
        EXPECT_THROW(propagation.reserve_kv_cache(257, 200), runtime_error);
        EXPECT_EQ((*propagation.layer_session_state_storage)[attention_layers[0]].data(), old_pointer);
        second.swap(original);
    }
    ASSERT_TRUE(propagation.reserve_kv_cache(257, 200));
    for (size_t n = 0; n < attention_layers.size(); ++n)
    {
        Buffer& buffer = (*propagation.layer_session_state_storage)[attention_layers[n]];
        ASSERT_EQ(buffer.byte_size(), 2 * 512 * 16 * Index(sizeof(uint16_t)));
        vector<uint16_t> actual(size_t(buffer.byte_size() / 2));
        device::copy_async(actual.data(), buffer.data(), buffer.byte_size(),
                           device::CopyKind::DeviceToHost, device::get_compute_stream());
        device::synchronize(device::get_compute_stream());
        for (size_t j = 0; j < 200 * 16; ++j)
        {
            ASSERT_EQ(actual[j], originals[n][j]);
            ASSERT_EQ(actual[512 * 16 + j], originals[n][256 * 16 + j]);
        }
    }
    EXPECT_TRUE(propagation.reserve_kv_cache(2304, 200));
    EXPECT_EQ((*propagation.layer_session_state_storage)[attention_layers[0]].byte_size(),
              2 * 2304 * 16 * Index(sizeof(uint16_t)));
    EXPECT_THROW(propagation.reserve_kv_cache(2305, 200), runtime_error);
}
#endif

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

    vector<float> window(size_t(d.seq), 0.0f);

    mt19937 id_rng(3);
    auto random_ids = [&](Index count) {
        vector<Index> ids(size_t(count), Index(0));
        for (auto& id : ids) id = 1 + Index(id_rng() % uint32_t(d.vocab - 1));
        return ids;
    };
    const vector<Index> prompt1 = random_ids(d.prompt1);
    const vector<Index> prompt2 = random_ids(d.prompt2);

    ForwardPropagation fp_used(1, &used);
    run(used, fp_used, window, prompt1, 0);
    for (Index i = 0; i < d.decodes; ++i)
        run(used, fp_used, window, { 1 + Index(id_rng() % uint32_t(d.vocab - 1)) }, d.prompt1 + i);

    run(used, fp_used, window, prompt2, 0);
    const vector<float> got = logits_row(fp_used, d.prompt2 - 1);

    ForwardPropagation fp_fresh(1, &fresh);
    run(fresh, fp_fresh, window, prompt2, 0);
    const vector<float> expected = logits_row(fp_fresh, d.prompt2 - 1);

    float max_diff = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i)
        max_diff = max(max_diff, abs(got[i] - expected[i]));
    return max_diff;
}

float compact_last_row_max_diff(const Dims& d, bool bf16_upload)
{
    unique_ptr<Qwen3> network = make_filled_qwen(d);
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

    EXPECT_EQ(compact.get_outputs().get_shape()[1], 1);
    EXPECT_LT(compact.arena.byte_size(), full.arena.byte_size());
    return max_difference(expected, logits_row(compact, 0));
}

float chunked_prefill_and_decode_max_diff(const Dims& d,
                                          Index block,
                                          bool bf16_upload)
{
    unique_ptr<Qwen3> full_network = make_filled_qwen(d);
    unique_ptr<Qwen3> chunked_network = make_filled_qwen(d);
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
    full_decode.share_session_state_from(full_prefill);
    chunked_decode.share_session_state_from(chunked_prefill);
    run(*full_network, full_decode, full_window,
        {decode_id}, d.prompt2);
    run(*chunked_network, chunked_decode, chunk_window,
        {decode_id}, d.prompt2);

    return max(max_difference(full_last, chunked_last),
               max_difference(logits_row(full_decode, 0),
                              logits_row(chunked_decode, 0)));
}

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

}

TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheCpu)
{
    EXPECT_LT(multi_turn_max_logit_diff(TINY), 1.0e-4f);
}

TEST(Qwen3NetworkTest, CompactLogitsEqualFullLastRowCpu)
{
    EXPECT_LT(compact_last_row_max_diff(TINY, false), 1.0e-4f);
}

TEST(Qwen3NetworkTest, ChunkedPrefillAndDecodeEqualFullPassCpu)
{
    EXPECT_LT(chunked_prefill_and_decode_max_diff(TINY, 3, false),
              1.0e-4f);
}

TEST(Qwen3NetworkTest, CompactPoolDependsOnBlockNotModelContext)
{
    Dims short_dims = TINY;
    Dims long_dims = TINY;
    short_dims.seq = 16;
    long_dims.seq = 64;

    unique_ptr<Qwen3> short_network = make_filled_qwen(short_dims);
    unique_ptr<Qwen3> long_network = make_filled_qwen(long_dims);
    ForwardPropagation short_compact(
        1, short_network.get(), ForwardPropagationMode::Inference, {4, 1});
    ForwardPropagation long_compact(
        1, long_network.get(), ForwardPropagationMode::Inference, {4, 1});

    EXPECT_EQ(short_compact.arena.byte_size(), long_compact.arena.byte_size());
    EXPECT_EQ(short_compact.get_sequence_capacity(), 4);
    EXPECT_EQ(long_compact.get_sequence_capacity(), 4);
    EXPECT_EQ(short_compact.get_final_output_capacity(), 1);
    EXPECT_EQ(long_compact.get_final_output_capacity(), 1);
}

TEST(Qwen3NetworkTest, CompactOutputWindowMatchesSelectedFullRowsCpu)
{
    unique_ptr<Qwen3> network = make_filled_qwen(TINY);
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
    network->forward_propagate(inputs, selected, ForwardPropagationMode::Inference);

    ASSERT_EQ(selected.get_outputs().get_shape()[1], 4);
    for (Index row = 0; row < 4; ++row)
        EXPECT_LT(max_difference(logits_row(full, row + 1),
                                 logits_row(selected, row)),
                  1.0e-4f);
}

TEST(Qwen3NetworkTest, DirectLogicalBf16WeightsMatchRoundedCpu)
{
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
}

#ifdef OPENNN_HAS_CUDA
TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheGpu)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    EXPECT_LT(multi_turn_max_logit_diff(TINY), 1.0e-2f);
}

TEST(Qwen3NetworkTest, MultiTurnPrefillRestartsCacheGpuBf16Upload)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(multi_turn_max_logit_diff(TINY,  true), 1.0e-2f);
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
}

TEST(Qwen3NetworkTest, CompactLogitsEqualFullLastRowGpuBf16)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(compact_last_row_max_diff(TINY, true), 1.0e-2f);
}

TEST(Qwen3NetworkTest, ChunkedPrefillAndDecodeEqualFullPassGpuBf16)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    EXPECT_LT(chunked_prefill_and_decode_max_diff(TINY, 3, true),
              1.0e-2f);
}

TEST(Qwen3NetworkTest, MultiTurnGrowingPrefillGpu)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    const Dims d { 64, 50, 2560, 2, 4, 2, 8, 64, 17, 0, 48 };
    EXPECT_LT(multi_turn_max_logit_diff(d, false), 1.0e-3f);
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
    decode.set(1, &network, &prefill.arena, ForwardPropagationMode::Inference);
    decode.share_session_state_from(prefill);
    decode.set_active_sequence_length(1);
    decode.set_cuda_graph(true);

    Buffer token_device{Device::CUDA};
    token_device.resize_bytes(Index(sizeof(float)), Device::CUDA);
    const vector<TensorView> decode_inputs = {
        TensorView(token_device.data(), {1, 1}, Type::FP32, Device::CUDA)
    };

    vector<float> window(size_t(d.seq), 0.0f);
    run(network, prefill, window, {3, 7, 11, 13}, 0);
    Index position = 4;

    const auto decode_token = [&](Index token)
    {
        const float token_value = float(token);
        device::copy_async(token_device.data(), &token_value, Index(sizeof(float)),
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
        ASSERT_EQ(graph_view.get_data(), decode.get_outputs().get_data());

        --position;
        decode.past_length = position;
        network.forward_propagate(decode_inputs, decode, ForwardPropagationMode::Inference);
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
}
#endif

namespace
{

unique_ptr<Qwen3> qwen_from_binary(const filesystem::path& path)
{
    return Qwen3::from_binary(
        path, TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
}

filesystem::path write_tiny_bf16_binary(const string& name)
{
    Qwen3 source(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    fill_parameters(source);
    round_parameters_to_bf16(source);

    const filesystem::path path = filesystem::temp_directory_path() / name;
    write_logical_bf16_parameters(source, path);
    return path;
}

float last_logits_max_difference(NeuralNetwork& a, NeuralNetwork& b)
{
    vector<float> a_window(size_t(TINY.seq), 0.0f);
    vector<float> b_window(size_t(TINY.seq), 0.0f);
    const vector<Index> ids = {2, 3, 5, 7, 11};
    ForwardPropagation a_fp(1, &a);
    ForwardPropagation b_fp(1, &b);
    run(a, a_fp, a_window, ids, 0);
    run(b, b_fp, b_window, ids, 0);
    return max_difference(logits_row(a_fp, ssize(ids) - 1),
                          logits_row(b_fp, ssize(ids) - 1));
}

}

// The factory route compiles without the fp32 master; on a host
// configuration the loader materializes it, so the two networks must be
// bitwise the same object as far as parameters and logits go.
TEST(Qwen3NetworkTest, FromBinaryMatchesConstructorLoadCpu)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    const filesystem::path path = write_tiny_bf16_binary("opennn_qwen3_from_binary_cpu.bin");

    Qwen3 loaded(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    loaded.load_parameters_bf16_inference_binary(path);
    const unique_ptr<Qwen3> built = qwen_from_binary(path);

    ASSERT_EQ(built->get_parameters_buffer_size(), loaded.get_parameters_buffer_size());
    EXPECT_EQ(built->get_parameters_device(), Device::CPU);
    EXPECT_TRUE(equal(loaded.get_parameters_data(),
                      loaded.get_parameters_data() + loaded.get_parameters_buffer_size(),
                      built->get_parameters_data()));
    EXPECT_EQ(last_logits_max_difference(loaded, *built), 0.0f);
    filesystem::remove(path);
}

TEST(Qwen3NetworkTest, FromBinaryRejectsTruncatedFileAndRecovers)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    const filesystem::path good = write_tiny_bf16_binary("opennn_qwen3_from_binary_good.bin");
    const filesystem::path truncated =
        filesystem::temp_directory_path() / "opennn_qwen3_from_binary_truncated.bin";
    {
        ifstream input(good, ios::binary);
        vector<char> bytes((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
        bytes.resize(bytes.size() / 2);
        ofstream output(truncated, ios::binary | ios::trunc);
        output.write(bytes.data(), streamsize(bytes.size()));
    }

    EXPECT_THROW(qwen_from_binary(truncated), runtime_error);
    EXPECT_NO_THROW(qwen_from_binary(good));

    filesystem::remove(good);
    filesystem::remove(truncated);
}

#ifdef OPENNN_HAS_CUDA
TEST(Qwen3NetworkTest, FromBinaryMatchesDirectLoadGpuBf16)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    const filesystem::path path = write_tiny_bf16_binary("opennn_qwen3_from_binary_gpu.bin");

    Qwen3 direct(
        TINY.seq, TINY.vocab, TINY.hidden, TINY.layers,
        TINY.q_heads, TINY.kv_heads, TINY.head_dim, TINY.intermediate,
        1000000.0f, 1.0e-6f);
    direct.load_parameters_bf16_inference_binary(path);
    const unique_ptr<Qwen3> built = qwen_from_binary(path);

    EXPECT_TRUE(direct.fp32_master_released());
    EXPECT_TRUE(built->fp32_master_released());
    EXPECT_EQ(built->get_parameters_buffer_size(), direct.get_parameters_buffer_size());
    EXPECT_EQ(last_logits_max_difference(direct, *built), 0.0f);
    filesystem::remove(path);
}

TEST(Qwen3NetworkTest, FromBinaryRejectsTruncatedFileAndRecoversGpu)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    const filesystem::path good = write_tiny_bf16_binary("opennn_qwen3_from_binary_good_gpu.bin");
    const filesystem::path truncated =
        filesystem::temp_directory_path() / "opennn_qwen3_from_binary_truncated_gpu.bin";
    {
        ifstream input(good, ios::binary);
        vector<char> bytes((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
        bytes.resize(bytes.size() / 2);
        ofstream output(truncated, ios::binary | ios::trunc);
        output.write(bytes.data(), streamsize(bytes.size()));
    }

    EXPECT_THROW(qwen_from_binary(truncated), runtime_error);
    EXPECT_NO_THROW(qwen_from_binary(good));

    filesystem::remove(good);
    filesystem::remove(truncated);
}
#endif
