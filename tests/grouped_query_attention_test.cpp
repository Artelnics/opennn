#include "pch.h"

#include <cmath>
#include <random>
#include <vector>

#include "opennn/tensor_types.h"
#include "opennn/tensor_operations.h"
#include "opennn/grouped_query_attention_layer.h"
#include "opennn/neural_network.h"

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

// Layer through the network's arena vs the same math via the free tensor ops.
// Dims deliberately decoupled: head_dim (16) != hidden/heads (8) and
// q_dim (64) != hidden (32), with rectangular GQA (q_heads=4 > kv_heads=2).
float layer_vs_recipe_max_diff(bool use_qk_norm)
{
    const Index batch = 1, seq = 6, hidden = 32;
    const Index q_heads = 4, kv_heads = 2, head_dim = 16;
    const float theta = 1000000.0f, eps = 1.0e-6f;
    const Index qd = q_heads * head_dim, kd = kv_heads * head_dim;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<GroupedQueryAttention>(
        Shape{seq, hidden}, q_heads, kv_heads, head_dim, theta, eps, use_qk_norm, "attn"));
    neural_network.compile();
    neural_network.set_parameters_random();

    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    auto fill = [&](std::vector<float>& v, size_t n) { v.resize(n); for (auto& x : v) x = nd(rng); };

    std::vector<float> wq, wk, wv, wo, nq, nk;
    fill(wq, size_t(qd) * hidden);
    fill(wk, size_t(kd) * hidden);
    fill(wv, size_t(kd) * hidden);
    fill(wo, size_t(hidden) * qd);
    nq.resize(head_dim); nk.resize(head_dim);
    for (auto& x : nq) x = 1.0f + nd(rng);
    for (auto& x : nk) x = 1.0f + nd(rng);

    // Parameter view order: q, k, v, o [, q_norm, k_norm].
    auto& views = neural_network.get_layer(Index(0))->get_parameter_views();
    EXPECT_EQ(views.size(), use_qk_norm ? size_t(6) : size_t(4));
    auto put = [&](TensorView& tv, const std::vector<float>& s) { std::copy(s.begin(), s.end(), tv.as<float>()); };
    put(views[0], wq); put(views[1], wk); put(views[2], wv); put(views[3], wo);
    if (use_qk_norm) { put(views[4], nq); put(views[5], nk); }

    std::vector<float> x(size_t(batch) * seq * hidden);
    for (auto& e : x) e = nd(rng);

    ForwardPropagation forward_propagation(batch, &neural_network);
    vector<TensorView> inputs = { TensorView(x.data(), {batch, seq, hidden}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);
    const TensorView output = forward_propagation.get_outputs();
    const float* got = output.as<float>();

    std::vector<float> cos(size_t(seq) * head_dim), sin(size_t(seq) * head_dim);
    TensorView cv(cos.data(), {seq, head_dim}), sv(sin.data(), {seq, head_dim});
    rotary_build_tables(cv, sv, seq, head_dim, theta);

    std::vector<float> q(size_t(seq) * qd), k(size_t(seq) * kd), v(size_t(seq) * kd);
    std::vector<float> qr(size_t(seq) * qd), kr(size_t(seq) * kd), attn(size_t(seq) * qd), ref(size_t(seq) * hidden);

    TensorView xv(x.data(), {1, seq, hidden});
    TensorView Wq(wq.data(), {qd, hidden}), Wk(wk.data(), {kd, hidden}), Wv(wv.data(), {kd, hidden}), Wo(wo.data(), {hidden, qd});
    TensorView Nq(nq.data(), {head_dim}), Nk(nk.data(), {head_dim});
    TensorView qv(q.data(), {1, seq, qd}), kv(k.data(), {1, seq, kd}), vv(v.data(), {1, seq, kd});

    tied_lm_head_forward(xv, Wq, qv);
    tied_lm_head_forward(xv, Wk, kv);
    tied_lm_head_forward(xv, Wv, vv);
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
    tied_lm_head_forward(av, Wo, rv);

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::abs(got[i] - ref[i]));

    return max_diff;
}

}


TEST(GroupedQueryAttentionTest, ForwardMatchesFreeOpRecipe)
{
    EXPECT_LT(layer_vs_recipe_max_diff(/*use_qk_norm*/ true), 1.0e-5f);
}


TEST(GroupedQueryAttentionTest, ForwardWithoutQKNormMatchesFreeOpRecipe)
{
    EXPECT_LT(layer_vs_recipe_max_diff(/*use_qk_norm*/ false), 1.0e-5f);
}


// Multi-turn chat contract: after a prefill + several single-token decode
// passes, a NEW prefill with past_length == 0 must restart the cache, giving
// the same output as a fresh network fed the second prompt directly.
TEST(GroupedQueryAttentionTest, PrefillAfterDecodeRestartsCache)
{
    const Index max_seq = 8, hidden = 16;
    const Index q_heads = 2, kv_heads = 1, head_dim = 8;

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 0.1f);

    auto build = [&](NeuralNetwork& net) {
        net.add_layer(make_unique<GroupedQueryAttention>(
            Shape{max_seq, hidden}, q_heads, kv_heads, head_dim, 1000000.0f, 1.0e-6f, true, "attn"));
        net.compile();
        net.set_parameters_random();
    };
    auto fill_parameters = [&](NeuralNetwork& net) {
        std::mt19937 weight_rng(7);
        std::normal_distribution<float> wd(0.0f, 0.1f);
        for (auto& view : net.get_layer(Index(0))->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] = wd(weight_rng);
    };

    NeuralNetwork used, fresh;
    build(used);
    build(fresh);
    fill_parameters(used);
    fill_parameters(fresh);

    ForwardPropagation fp_used(1, &used);

    std::vector<float> tokens(size_t(max_seq) * hidden);
    for (auto& v : tokens) v = nd(rng);

    auto run = [&](NeuralNetwork& net, ForwardPropagation& fp, float* data, Index count, Index past) {
        fp.past_length = past;
        fp.set_active_sequence_length(count);
        vector<TensorView> inputs = { TensorView(data, {1, count, hidden}) };
        net.forward_propagate(inputs, fp, false);
    };

    run(used, fp_used, tokens.data(), 4, 0);
    run(used, fp_used, tokens.data() + 4 * hidden, 1, 4);
    run(used, fp_used, tokens.data() + 5 * hidden, 1, 5);

    std::vector<float> prompt2(size_t(6) * hidden);
    for (auto& v : prompt2) v = nd(rng);
    run(used, fp_used, prompt2.data(), 6, 0);
    const TensorView out_used = fp_used.get_outputs();
    const std::vector<float> got(out_used.as<float>(), out_used.as<float>() + out_used.size());

    ForwardPropagation fp_fresh(1, &fresh);
    run(fresh, fp_fresh, prompt2.data(), 6, 0);
    const TensorView out_fresh = fp_fresh.get_outputs();

    ASSERT_EQ(out_used.size(), out_fresh.size());
    for (Index i = 0; i < out_fresh.size(); ++i)
        EXPECT_NEAR(got[size_t(i)], out_fresh.as<float>()[i], 1.0e-5f) << "at " << i;
}
