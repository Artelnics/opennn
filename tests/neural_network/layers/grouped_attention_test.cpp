#include "tests/pch.h"

#include <cmath>
#include <future>
#include <limits>
#include <vector>

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/layers/grouped_query_attention_layer.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#endif

using namespace opennn;

namespace
{

vector<float> run_grouped_attention(Index batch, Index sequence,
                                    Index query_heads, Index key_value_heads,
                                    Index head_dimension)
{
    const Index query_width = query_heads * head_dimension;
    const Index key_value_width = key_value_heads * head_dimension;

    vector<float> query(size_t(batch * sequence * query_width));
    vector<float> key(size_t(batch * sequence * key_value_width));
    vector<float> value(size_t(batch * sequence * key_value_width));
    vector<float> output(query.size());

    for (size_t i = 0; i < query.size(); ++i)
        query[i] = sin(0.017f * float(i)) + 0.2f;
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = cos(0.013f * float(i));
    for (size_t i = 0; i < value.size(); ++i)
        value[i] = sin(0.009f * float(i)) - 0.1f;

    TensorView query_view(query.data(), {batch, sequence, query_width});
    TensorView key_view(key.data(), {batch, sequence, key_value_width});
    TensorView value_view(value.data(), {batch, sequence, key_value_width});
    TensorView output_view(output.data(), {batch, sequence, query_width});

    grouped_attention_forward(query_view, key_view, value_view, output_view,
                              query_heads, key_value_heads, head_dimension,
                              true, 1.0f / sqrt(float(head_dimension)));
    return output;
}

float maximum_difference(const vector<float>& left, const vector<float>& right)
{
    if (left.size() != right.size()) return numeric_limits<float>::infinity();

    float difference = 0.0f;
    for (size_t i = 0; i < left.size(); ++i)
        difference = max(difference, abs(left[i] - right[i]));
    return difference;
}

}

TEST(GroupedAttentionTest, CausalFirstPositionEqualsValue)
{
    const Index batch = 1, seq = 3, q_heads = 1, kv_heads = 1, head_dim = 4;
    const float scale = 0.5f;

    vector<float> query(size_t(seq * head_dim));
    vector<float> key(size_t(seq * head_dim));
    vector<float> value(size_t(seq * head_dim));
    vector<float> output(size_t(seq * head_dim), 0.0f);
    for (size_t i = 0; i < query.size(); ++i)
    {
        query[i] = std::sin(0.1f * float(i));
        key[i]   = std::cos(0.2f * float(i));
        value[i] = 0.5f - 0.03f * float(i);
    }

    TensorView q(query.data(), {batch, seq, q_heads * head_dim});
    TensorView k(key.data(),   {batch, seq, kv_heads * head_dim});
    TensorView v(value.data(), {batch, seq, kv_heads * head_dim});
    TensorView o(output.data(),{batch, seq, q_heads * head_dim});

    grouped_attention_forward(q, k, v, o, q_heads, kv_heads, head_dim,  true, scale);

    for (Index d = 0; d < head_dim; ++d)
        EXPECT_NEAR(output[size_t(d)], value[size_t(d)], 1.0e-6f);
}

TEST(GroupedAttentionTest, GroupedHeadsShareKeyValue)
{
    const Index batch = 1, seq = 2, q_heads = 2, kv_heads = 1, head_dim = 3;
    const float scale = 0.7f;

    const Index q_model = q_heads * head_dim;
    const Index kv_model = kv_heads * head_dim;

    vector<float> query(size_t(seq * q_model));
    vector<float> key(size_t(seq * kv_model));
    vector<float> value(size_t(seq * kv_model));
    vector<float> output(size_t(seq * q_model), 0.0f);

    for (size_t i = 0; i < key.size(); ++i)   key[i]   = std::sin(0.3f * float(i) + 0.1f);
    for (size_t i = 0; i < value.size(); ++i) value[i] = std::cos(0.15f * float(i));
    for (Index t = 0; t < seq; ++t)
        for (Index d = 0; d < head_dim; ++d)
        {
            const float val = 0.2f + 0.1f * float(t) - 0.05f * float(d);
            query[size_t((t * q_heads + 0) * head_dim + d)] = val;
            query[size_t((t * q_heads + 1) * head_dim + d)] = val;
        }

    TensorView q(query.data(), {batch, seq, q_model});
    TensorView k(key.data(),   {batch, seq, kv_model});
    TensorView v(value.data(), {batch, seq, kv_model});
    TensorView o(output.data(),{batch, seq, q_model});

    grouped_attention_forward(q, k, v, o, q_heads, kv_heads, head_dim,  true, scale);

    for (Index t = 0; t < seq; ++t)
        for (Index d = 0; d < head_dim; ++d)
        {
            const float head0 = output[size_t((t * q_heads + 0) * head_dim + d)];
            const float head1 = output[size_t((t * q_heads + 1) * head_dim + d)];
            EXPECT_NEAR(head0, head1, 1.0e-6f);
        }
}

TEST(GroupedAttentionTest, ConcurrentMixedShapesMatchSequentialResults)
{
    const vector<float> expected_small = run_grouped_attention(1, 5, 4, 2, 8);
    const vector<float> expected_large = run_grouped_attention(2, 11, 8, 2, 16);

    future<vector<float>> small = async(launch::async, []
    {
        return run_grouped_attention(1, 5, 4, 2, 8);
    });
    future<vector<float>> large = async(launch::async, []
    {
        return run_grouped_attention(2, 11, 8, 2, 16);
    });

    EXPECT_LT(maximum_difference(small.get(), expected_small), 1.0e-6f);
    EXPECT_LT(maximum_difference(large.get(), expected_large), 1.0e-6f);
}

#ifdef OPENNN_HAS_CUDA

TEST(GroupedAttentionTest, GpuMatchesCpu)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    const Index batch = 2, seq = 6, q_heads = 8, kv_heads = 2, head_dim = 64;
    const float scale = 1.0f / std::sqrt(float(head_dim));
    const Index q_model = q_heads * head_dim, kv_model = kv_heads * head_dim;

    vector<float> Q(size_t(batch * seq * q_model));
    vector<float> K(size_t(batch * seq * kv_model));
    vector<float> V(size_t(batch * seq * kv_model));
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = std::sin(0.017f * float(i)) + 0.2f;
    for (size_t i = 0; i < K.size(); ++i) K[i] = std::cos(0.013f * float(i));
    for (size_t i = 0; i < V.size(); ++i) V[i] = std::sin(0.009f * float(i)) - 0.1f;

    vector<float> out_cpu(Q.size(), 0.0f);
    {
        TensorView q(Q.data(), {batch, seq, q_model}), k(K.data(), {batch, seq, kv_model}),
                   v(V.data(), {batch, seq, kv_model}), o(out_cpu.data(), {batch, seq, q_model});
        grouped_attention_forward(q, k, v, o, q_heads, kv_heads, head_dim, true, scale);
    }

    cudaStream_t stream = device::get_compute_stream();
    auto to_device = [&](const vector<float>& h, Buffer& buf) {
        buf.resize_bytes(Index(h.size()) * Index(sizeof(float)), Device::CPU);
        memcpy(buf.data(), h.data(), h.size() * sizeof(float));
        buf.migrate_to(Device::CUDA, stream);
    };
    Buffer qb(Device::CPU), kb(Device::CPU), vb(Device::CPU), ob(Device::CUDA);
    to_device(Q, qb); to_device(K, kb); to_device(V, vb);
    ob.resize_bytes(Index(Q.size()) * Index(sizeof(float)), Device::CUDA);

    {
        TensorView q(qb.as<float>(), {batch, seq, q_model}, Type::FP32, Device::CUDA);
        TensorView k(kb.as<float>(), {batch, seq, kv_model}, Type::FP32, Device::CUDA);
        TensorView v(vb.as<float>(), {batch, seq, kv_model}, Type::FP32, Device::CUDA);
        TensorView o(ob.as<float>(), {batch, seq, q_model}, Type::FP32, Device::CUDA);
        grouped_attention_forward(q, k, v, o, q_heads, kv_heads, head_dim, true, scale);
    }
    device::synchronize(stream);
    ob.migrate_to(Device::CPU, stream);

    const float* out_gpu = ob.as<float>();
    double max_abs = 0.0;
    for (size_t i = 0; i < Q.size(); ++i)
        max_abs = max(max_abs, abs(double(out_gpu[i]) - double(out_cpu[i])));
    EXPECT_LT(max_abs, 1.0e-4);

}
#endif
