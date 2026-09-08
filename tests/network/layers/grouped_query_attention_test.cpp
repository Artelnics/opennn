#include "tests/pch.h"

#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/network/layers/grouped_query_attention_layer.h"
#include "opennn/network/network.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"

using namespace opennn;

TEST(GroupedQueryAttentionTest, GeneralConstructor)
{
    GroupedQueryAttention attention({16, 32}, 4, 2, 8, 1000000.0f, 1.0e-6f, true, "attn");

    EXPECT_EQ(attention.get_name(), "GroupedQueryAttention");
    EXPECT_EQ(attention.get_label(), "attn");
    EXPECT_EQ(attention.get_sequence_length(), 16);
    EXPECT_EQ(attention.get_hidden(), 32);
    EXPECT_EQ(attention.get_q_heads(), 4);
    EXPECT_EQ(attention.get_kv_heads(), 2);
    EXPECT_EQ(attention.get_head_dim(), 8);
    EXPECT_TRUE(attention.get_use_qk_norm());
    EXPECT_EQ(attention.get_output_shape()[0], 16);
    EXPECT_EQ(attention.get_output_shape()[1], 32);
}

namespace
{

float layer_vs_recipe_max_diff(bool use_qk_norm)
{
    const Index batch = 1, seq = 6, hidden = 32;
    const Index q_heads = 4, kv_heads = 2, head_dim = 16;
    const float theta = 1000000.0f, eps = 1.0e-6f;
    const Index qd = q_heads * head_dim, kd = kv_heads * head_dim;

    Network network;
    network.add_layer(make_unique<GroupedQueryAttention>(
        Shape{seq, hidden}, q_heads, kv_heads, head_dim, theta, eps, use_qk_norm, "attn"));
    network.compile();
    network.set_parameters_random();

    mt19937 rng(123);
    normal_distribution<float> nd(0.0f, 0.05f);
    auto fill = [&](vector<float>& v, size_t n) { v.resize(n); for (auto& x : v) x = nd(rng); };

    vector<float> wq, wk, wv, wo, nq, nk;
    fill(wq, size_t(qd) * hidden);
    fill(wk, size_t(kd) * hidden);
    fill(wv, size_t(kd) * hidden);
    fill(wo, size_t(hidden) * qd);
    nq.resize(head_dim); nk.resize(head_dim);
    for (auto& x : nq) x = 1.0f + nd(rng);
    for (auto& x : nk) x = 1.0f + nd(rng);

    auto& views = network.get_layer(Index(0))->get_parameter_views();
    EXPECT_EQ(views.size(), use_qk_norm ? size_t(6) : size_t(4));
    auto put = [&](TensorView& tv, const vector<float>& s) { copy(s.begin(), s.end(), tv.as<float>()); };
    put(views[0], wq); put(views[1], wk); put(views[2], wv); put(views[3], wo);
    if (use_qk_norm) { put(views[4], nq); put(views[5], nk); }

    vector<float> x(size_t(batch) * seq * hidden);
    for (auto& e : x) e = nd(rng);

    ForwardPropagation forward_propagation(batch, &network);
    vector<TensorView> inputs = { TensorView(x.data(), {batch, seq, hidden}) };
    network.forward_propagate(inputs, forward_propagation, ForwardPropagationMode::Inference);
    const TensorView output = forward_propagation.get_outputs();
    const float* got = output.as<float>();

    vector<float> cos(size_t(seq) * head_dim), sin(size_t(seq) * head_dim);
    TensorView cv(cos.data(), {seq, head_dim}), sv(sin.data(), {seq, head_dim});
    rotary_build_tables(cv, sv, seq, head_dim, theta);

    vector<float> q(size_t(seq) * qd), k(size_t(seq) * kd), v(size_t(seq) * kd);
    vector<float> qr(size_t(seq) * qd), kr(size_t(seq) * kd), attn(size_t(seq) * qd), ref(size_t(seq) * hidden);

    TensorView xv(x.data(), {1, seq, hidden});
    TensorView Wq(wq.data(), {qd, hidden}), Wk(wk.data(), {kd, hidden}), Wv(wv.data(), {kd, hidden}), Wo(wo.data(), {hidden, qd});
    TensorView Nq(nq.data(), {head_dim}), Nk(nk.data(), {head_dim});
    TensorView qv(q.data(), {1, seq, qd}), kv(k.data(), {1, seq, kd}), vv(v.data(), {1, seq, kd});

    linear_forward_transposed(xv, Wq, qv);
    linear_forward_transposed(xv, Wk, kv);
    linear_forward_transposed(xv, Wv, vv);
    if (use_qk_norm)
    {
        qk_norm_forward(qv, Nq, qv, head_dim, eps);
        qk_norm_forward(kv, Nk, kv, head_dim, eps);
    }

    TensorView qrv(qr.data(), {1, seq, qd}), krv(kr.data(), {1, seq, kd});
    rotary_forward(qv, cv, sv, qrv, head_dim, head_dim, 0);
    rotary_forward(kv, cv, sv, krv, head_dim, head_dim, 0);

    TensorView av(attn.data(), {1, seq, qd});
    grouped_attention_forward(qrv, krv, vv, av, q_heads, kv_heads, head_dim, true, 1.0f / std::sqrt(float(head_dim)), 0);

    TensorView rv(ref.data(), {1, seq, hidden});
    linear_forward_transposed(av, Wo, rv);

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = max(max_diff, abs(got[i] - ref[i]));

    return max_diff;
}

}

TEST(GroupedQueryAttentionTest, ForwardMatchesFreeOpRecipe)
{
    EXPECT_LT(layer_vs_recipe_max_diff( true), 1.0e-5f);
}

TEST(GroupedQueryAttentionTest, ForwardWithoutQKNormMatchesFreeOpRecipe)
{
    EXPECT_LT(layer_vs_recipe_max_diff( false), 1.0e-5f);
}

TEST(GroupedQueryAttentionTest, PrefillAfterDecodeRestartsCache)
{
    const Index max_seq = 8, hidden = 16;
    const Index q_heads = 2, kv_heads = 1, head_dim = 8;

    mt19937 rng(11);
    normal_distribution<float> nd(0.0f, 0.1f);

    auto build = [&](Network& net) {
        net.add_layer(make_unique<GroupedQueryAttention>(
            Shape{max_seq, hidden}, q_heads, kv_heads, head_dim, 1000000.0f, 1.0e-6f, true, "attn"));
        net.compile();
        net.set_parameters_random();
    };
    auto fill_parameters = [&](Network& net) {
        mt19937 weight_rng(7);
        normal_distribution<float> wd(0.0f, 0.1f);
        for (auto& view : net.get_layer(Index(0))->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] = wd(weight_rng);
    };

    Network used, fresh;
    build(used);
    build(fresh);
    fill_parameters(used);
    fill_parameters(fresh);

    ForwardPropagation fp_used(1, &used);

    vector<float> tokens(size_t(max_seq) * hidden);
    for (auto& v : tokens) v = nd(rng);

    auto run = [&](Network& net, ForwardPropagation& fp, float* data, Index count, Index past) {
        fp.past_length = past;
        fp.set_active_sequence_length(count);
        vector<TensorView> inputs = { TensorView(data, {1, count, hidden}) };
        net.forward_propagate(inputs, fp, ForwardPropagationMode::Inference);
    };

    run(used, fp_used, tokens.data(), 4, 0);
    run(used, fp_used, tokens.data() + 4 * hidden, 1, 4);
    run(used, fp_used, tokens.data() + 5 * hidden, 1, 5);

    vector<float> prompt2(size_t(6) * hidden);
    for (auto& v : prompt2) v = nd(rng);
    run(used, fp_used, prompt2.data(), 6, 0);
    const TensorView out_used = fp_used.get_outputs();
    const vector<float> got(out_used.as<float>(), out_used.as<float>() + out_used.size());

    ForwardPropagation fp_fresh(1, &fresh);
    run(fresh, fp_fresh, prompt2.data(), 6, 0);
    const TensorView out_fresh = fp_fresh.get_outputs();

    ASSERT_EQ(out_used.size(), out_fresh.size());
    for (Index i = 0; i < out_fresh.size(); ++i)
        EXPECT_NEAR(got[size_t(i)], out_fresh.as<float>()[i], 1.0e-5f) << "at " << i;
}

TEST(GroupedQueryAttentionTest, KvCacheIsIsolatedUntilExplicitlyShared)
{
    constexpr Index max_sequence = 8;
    constexpr Index hidden = 16;

    Network network;
    network.add_layer(make_unique<GroupedQueryAttention>(
        Shape{max_sequence, hidden}, 2, 1, 8,
        1000000.0f, 1.0e-6f, true, "attn"));
    network.compile();
    network.set_parameters_random();

    ForwardPropagation first(
        1, &network, ForwardPropagationMode::Inference);
    ForwardPropagation second(
        1, &network, ForwardPropagationMode::Inference);
    ForwardPropagation shared(
        1, &network, ForwardPropagationMode::Inference);
    shared.share_session_state_from(first);

    vector<float> first_input(size_t(2 * hidden), 0.25f);
    vector<float> second_input(size_t(2 * hidden), -0.5f);

    network.forward_propagate(
        {TensorView(first_input.data(), {1, 2, hidden})}, first, ForwardPropagationMode::Inference);
    network.forward_propagate(
        {TensorView(second_input.data(), {1, 2, hidden})}, second, ForwardPropagationMode::Inference);

    ASSERT_FALSE((*first.layer_session_state_storage)[0].empty());
    ASSERT_FALSE((*second.layer_session_state_storage)[0].empty());
    EXPECT_NE((*first.layer_session_state_storage)[0].data(),
              (*second.layer_session_state_storage)[0].data());

    EXPECT_EQ(shared.layer_session_state_storage,
              first.layer_session_state_storage);
}

// The batched cuDNN SDPA graph -- the one grouped_attention_sdpa_gpu builds --
// is reached only with a batch above one, which skips the KV-cache path, and
// only in BF16, the sole dtype it accepts. Nothing exercised it: a probe on
// that call site counted zero executions across the whole suite, while the
// prefill graph ran 74 times.
TEST(GroupedQueryAttentionTest, Bf16BatchedAttentionMatchesCpu)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    const Index batch = 2, seq = 6, hidden = 32;
    const Index q_heads = 4, kv_heads = 2, head_dim = 16;
    const Index qd = q_heads * head_dim, kd = kv_heads * head_dim;

    const auto build = [&]
    {
        auto network = make_unique<Network>();
        network->add_layer(make_unique<GroupedQueryAttention>(
            Shape{seq, hidden}, q_heads, kv_heads, head_dim,
            1000000.0f, 1.0e-6f, false, "attn"));
        network->compile();
        return network;
    };

    mt19937 rng(4242);
    normal_distribution<float> distribution(0.0f, 0.15f);
    const auto sample = [&](size_t count)
    {
        vector<float> values(count);
        for (auto& value : values) value = distribution(rng);
        return values;
    };

    const vector<float> wq = sample(size_t(qd) * hidden);
    const vector<float> wk = sample(size_t(kd) * hidden);
    const vector<float> wv = sample(size_t(kd) * hidden);
    const vector<float> wo = sample(size_t(hidden) * qd);

    Configuration::instance().set(Device::CPU, Type::FP32);

    const auto cpu_network = build();

    // GroupedQueryAttention is inference-only: set_parameters_random sets the
    // QK-norm vectors and leaves the four projections at zero, which would make
    // this compare zeros against zeros. The weights have to be written in.
    {
        auto& views = cpu_network->get_layer(Index(0))->get_parameter_views();
        ASSERT_EQ(views.size(), size_t(4));
        const auto put = [](TensorView& view, const vector<float>& source)
            { copy(source.begin(), source.end(), view.as<float>()); };
        put(views[0], wq); put(views[1], wk); put(views[2], wv); put(views[3], wo);
    }

    const VectorR parameters = cpu_network->get_parameters_map();

    Tensor3 inputs(batch, seq, hidden);
    for (Index i = 0; i < inputs.size(); ++i)
        inputs.data()[i] = distribution(rng);

    const MatrixR cpu_outputs = cpu_network->calculate_outputs(inputs);

    // Guards the comparison below against being vacuously satisfied.
    ASSERT_GT(cpu_outputs.array().abs().maxCoeff(), 1.0e-6f);

    Configuration::instance().set(Device::CUDA, Type::BF16);

    const auto gpu_network = build();
    gpu_network->set_parameters(parameters);
    const MatrixR gpu_outputs = gpu_network->calculate_outputs(inputs);

    Configuration::instance().set(Device::CPU, Type::FP32);

    ASSERT_EQ(cpu_outputs.rows(), gpu_outputs.rows());
    ASSERT_EQ(cpu_outputs.cols(), gpu_outputs.cols());
    ASSERT_GT(gpu_outputs.array().abs().maxCoeff(), 1.0e-6f);

    const float max_difference =
        (cpu_outputs - gpu_outputs).array().abs().maxCoeff();

    // Measured 1.9e-3 on an RTX 3060; the bound leaves about five times that.
    EXPECT_LT(max_difference, 1.0e-2f)
        << "Max FP32 CPU vs BF16 GPU forward output difference: "
        << max_difference;
}

namespace
{

const GroupedQueryAttentionOperator& attention_operator(const Network& network)
{
    return *static_cast<const GroupedQueryAttentionOperator*>(
        network.get_layer(Index(0))->get_operators().front());
}

// A one-layer network whose first forward pass makes the operator fetch its
// RoPE tables from the shared registry.
struct RopeProbe
{
    Index seq, hidden, head_dim;
    float theta;
    unique_ptr<Network> network;
    MatrixR outputs;

    RopeProbe(Index new_seq, Index new_head_dim, float new_theta)
        : seq(new_seq), hidden(2 * new_head_dim), head_dim(new_head_dim), theta(new_theta),
          network(make_unique<Network>())
    {
        network->add_layer(make_unique<GroupedQueryAttention>(
            Shape{seq, hidden}, 2, 1, head_dim, theta, 1.0e-6f, true, "attn"));
        network->compile();
        network->set_parameters_random();
    }

    void forward()
    {
        Tensor3 inputs(1, seq, hidden);
        for (Index i = 0; i < inputs.size(); ++i)
            inputs.data()[i] = 0.01f * float(i % 7);
        outputs = network->calculate_outputs(inputs);
    }

    const Buffer* tables() const { return attention_operator(*network).get_rope_tables(); }
};

}

TEST(GroupedQueryAttentionTest, RopeTablesAreSharedByEqualConfigurations)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    RopeProbe first(16, 8, 1000000.0f);
    RopeProbe second(16, 8, 1000000.0f);
    RopeProbe longer(32, 8, 1000000.0f);
    RopeProbe other_theta(16, 8, 10000.0f);

    EXPECT_EQ(first.tables(), nullptr);

    first.forward();
    second.forward();
    longer.forward();
    other_theta.forward();

    ASSERT_NE(first.tables(), nullptr);
    EXPECT_EQ(first.tables(), second.tables());
    EXPECT_EQ(first.tables()->get_device(), Device::CPU);
    EXPECT_EQ(first.tables()->byte_size(), 2 * 16 * 8 * Index(sizeof(float)));
    EXPECT_NE(first.tables(), longer.tables());
    EXPECT_NE(first.tables(), other_theta.tables());
    EXPECT_EQ(longer.tables()->byte_size(), 2 * 32 * 8 * Index(sizeof(float)));
    EXPECT_EQ((first.outputs - second.outputs).array().abs().maxCoeff(), 0.0f);
}

TEST(GroupedQueryAttentionTest, RopeTablesOutliveTheirFirstOwner)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    auto first = make_unique<RopeProbe>(16, 8, 1000000.0f);
    RopeProbe second(16, 8, 1000000.0f);
    first->forward();
    second.forward();
    const Buffer* shared = second.tables();
    ASSERT_NE(shared, nullptr);
    ASSERT_EQ(first->tables(), shared);
    const MatrixR before = second.outputs;

    first.reset();

    second.forward();
    EXPECT_EQ(second.tables(), shared);
    EXPECT_EQ((before - second.outputs).array().abs().maxCoeff(), 0.0f);
}

TEST(GroupedQueryAttentionTest, RopeTablesFollowOperatorReconfiguration)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    RopeProbe probe(16, 8, 1000000.0f);
    RopeProbe keeper(16, 8, 1000000.0f);
    probe.forward();
    keeper.forward();
    const Buffer* shared = keeper.tables();
    ASSERT_EQ(probe.tables(), shared);

    auto& reconfigured = *static_cast<GroupedQueryAttentionOperator*>(
        probe.network->get_layer(Index(0))->get_operators().front());
    reconfigured.set(16, 16, 2, 1, 8, 1000000.0f, 1.0e-6f, true);
    EXPECT_EQ(probe.tables(), nullptr);

    probe.forward();
    EXPECT_EQ(probe.tables(), shared);
}

TEST(GroupedQueryAttentionTest, RopeTablesConcurrentConstructionYieldsOneTable)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    constexpr int probes_count = 8;
    vector<unique_ptr<RopeProbe>> probes;
    for (int i = 0; i < probes_count; ++i)
        probes.push_back(make_unique<RopeProbe>(48, 8, 1000000.0f));

    vector<thread> threads;
    for (auto& probe : probes)
        threads.emplace_back([&probe] { probe->forward(); });
    for (thread& worker : threads) worker.join();

    ASSERT_NE(probes.front()->tables(), nullptr);
    for (const auto& probe : probes)
    {
        EXPECT_EQ(probe->tables(), probes.front()->tables());
        EXPECT_EQ((probe->outputs - probes.front()->outputs).array().abs().maxCoeff(), 0.0f);
    }
}

TEST(GroupedQueryAttentionTest, RopeTablesAreSharedPerDeviceGpu)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::BF16);
    RopeProbe first(16, 16, 1000000.0f);
    RopeProbe second(16, 16, 1000000.0f);
    first.forward();
    second.forward();

    Configuration::instance().set(Device::CPU, Type::FP32);
    RopeProbe host(16, 16, 1000000.0f);
    host.forward();

    ASSERT_NE(first.tables(), nullptr);
    EXPECT_EQ(first.tables(), second.tables());
    EXPECT_EQ(first.tables()->get_device(), Device::CUDA);
    ASSERT_NE(host.tables(), nullptr);
    EXPECT_NE(first.tables(), host.tables());
    EXPECT_EQ(host.tables()->get_device(), Device::CPU);
}

// Every attention layer with the same prefill shape shares one built cuDNN
// plan: two networks of two layers each must cost one build, and the shared
// plan must give both the same answer.
TEST(GroupedQueryAttentionTest, PrefillSdpaPlanIsBuiltOncePerShapeGpu)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::BF16);

    // A shape no other test uses, so the registry cannot already hold it.
    const Index seq = 24, hidden = 128, q_heads = 2, kv_heads = 1, head_dim = 64;

    const auto build = [&]
    {
        auto network = make_unique<Network>();
        network->add_layer(make_unique<GroupedQueryAttention>(
            Shape{seq, hidden}, q_heads, kv_heads, head_dim, 1000000.0f, 1.0e-6f, true, "attn_0"));
        network->add_layer(make_unique<GroupedQueryAttention>(
            Shape{seq, hidden}, q_heads, kv_heads, head_dim, 1000000.0f, 1.0e-6f, true, "attn_1"), {0});
        network->compile();
        return network;
    };

    const auto first = build();
    const auto second = build();
    {
        mt19937 rng(77);
        normal_distribution<float> distribution(0.0f, 0.1f);
        VectorR parameters(first->get_parameters_number());
        for (Index i = 0; i < parameters.size(); ++i) parameters[i] = distribution(rng);
        first->set_parameters(parameters);
        second->set_parameters(parameters);
    }

    Tensor3 inputs(1, seq, hidden);
    for (Index i = 0; i < inputs.size(); ++i)
        inputs.data()[i] = 0.05f * float(i % 11) - 0.25f;

    const bool profiler_was_enabled = profiler::is_enabled();
    profiler::stats().clear();
    profiler::set_enabled(true);
    const MatrixR first_outputs = first->calculate_outputs(inputs);
    const MatrixR second_outputs = second->calculate_outputs(inputs);
    profiler::set_enabled(profiler_was_enabled);

    const long builds = profiler::stats().call_count("gqa:sdpa_build");
    const long prefills = profiler::stats().call_count("gqa:prefill_sdpa");
    profiler::stats().clear();

    Configuration::instance().set(Device::CPU, Type::FP32);

    EXPECT_EQ(prefills, 4);
    EXPECT_EQ(builds, 1);
    ASSERT_EQ(first_outputs.rows(), second_outputs.rows());
    ASSERT_GT(first_outputs.array().abs().maxCoeff(), 1.0e-6f);
    EXPECT_EQ((first_outputs - second_outputs).array().abs().maxCoeff(), 0.0f);
}
