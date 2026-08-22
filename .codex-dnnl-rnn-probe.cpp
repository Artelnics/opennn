#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv)
{
    using namespace dnnl;
    using clock = std::chrono::steady_clock;
    const int N = argc > 1 ? std::atoi(argv[1]) : 512;
    constexpr int T = 168, IC = 15, HC = 64, L = 1, D = 1, G = 1;
    const memory::dims src_dims{T, N, IC};
    const memory::dims state_dims{L, D, N, HC};
    const memory::dims weights_layer_dims{L, D, IC, G, HC};
    const memory::dims weights_iter_dims{L, D, HC, G, HC};
    const memory::dims bias_dims{L, D, G, HC};
    const memory::dims dst_dims{T, N, HC};
    const auto f32 = memory::data_type::f32;
    const auto any = memory::format_tag::any;
    const auto eng = engine(engine::kind::cpu, 0);
    auto strm = stream(eng);

    const auto src_md = memory::desc(src_dims, f32, memory::format_tag::tnc);
    const auto state_md = memory::desc(state_dims, f32, memory::format_tag::ldnc);
    const auto wl_user_md = memory::desc(weights_layer_dims, f32, memory::format_tag::ldigo);
    const auto wi_user_md = memory::desc(weights_iter_dims, f32, memory::format_tag::ldigo);
    const auto bias_md = memory::desc(bias_dims, f32, memory::format_tag::ldgo);
    const auto dst_md = memory::desc(dst_dims, f32, memory::format_tag::tnc);
    const auto wl_any_md = memory::desc(weights_layer_dims, f32, any);
    const auto wi_any_md = memory::desc(weights_iter_dims, f32, any);

    const auto fwd_pd = vanilla_rnn_forward::primitive_desc(
        eng, prop_kind::forward_training, algorithm::eltwise_tanh,
        rnn_direction::unidirectional_left2right, src_md, state_md,
        wl_any_md, wi_any_md, bias_md, dst_md, state_md);
    const auto bwd_pd = vanilla_rnn_backward::primitive_desc(
        eng, prop_kind::backward, algorithm::eltwise_tanh,
        rnn_direction::unidirectional_left2right, src_md, state_md,
        wl_any_md, wi_any_md, bias_md, dst_md, state_md,
        src_md, state_md, wl_any_md, wi_any_md, bias_md, dst_md, state_md,
        fwd_pd);
    const auto fwd = vanilla_rnn_forward(fwd_pd);
    const auto bwd = vanilla_rnn_backward(bwd_pd);

    std::vector<float> src(size_t(T) * N * IC), state(size_t(N) * HC);
    std::vector<float> wl(size_t(IC) * HC), wi(size_t(HC) * HC), bias(HC);
    std::vector<float> dst(size_t(T) * N * HC), dst_state(size_t(N) * HC);
    std::vector<float> diff_src(src.size()), diff_state(state.size());
    std::vector<float> diff_wl(wl.size()), diff_wi(wi.size()), diff_bias(bias.size());
    std::vector<float> diff_dst(dst.size()), diff_dst_state(state.size());
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (auto* values : {&src, &wl, &wi, &bias, &diff_dst})
        std::generate(values->begin(), values->end(), [&] { return dist(rng); });

    auto src_mem = memory(src_md, eng, src.data());
    auto state_mem = memory(state_md, eng, state.data());
    auto wl_user = memory(wl_user_md, eng, wl.data());
    auto wi_user = memory(wi_user_md, eng, wi.data());
    auto wl_mem = memory(fwd_pd.weights_layer_desc(), eng);
    auto wi_mem = memory(fwd_pd.weights_iter_desc(), eng);
    auto bias_mem = memory(bias_md, eng, bias.data());
    auto dst_mem = memory(dst_md, eng, dst.data());
    auto dst_state_mem = memory(state_md, eng, dst_state.data());
    auto workspace = memory(fwd_pd.workspace_desc(), eng);
    auto diff_src_mem = memory(src_md, eng, diff_src.data());
    auto diff_state_mem = memory(state_md, eng, diff_state.data());
    auto diff_wl_mem = memory(bwd_pd.diff_weights_layer_desc(), eng);
    auto diff_wi_mem = memory(bwd_pd.diff_weights_iter_desc(), eng);
    auto diff_bias_mem = memory(bias_md, eng, diff_bias.data());
    auto diff_dst_mem = memory(dst_md, eng, diff_dst.data());
    auto diff_dst_state_mem = memory(state_md, eng, diff_dst_state.data());
    reorder(wl_user, wl_mem).execute(strm, wl_user, wl_mem);
    reorder(wi_user, wi_mem).execute(strm, wi_user, wi_mem);

    std::unordered_map<int, memory> fwd_args{
        {DNNL_ARG_SRC_LAYER, src_mem}, {DNNL_ARG_SRC_ITER, state_mem},
        {DNNL_ARG_WEIGHTS_LAYER, wl_mem}, {DNNL_ARG_WEIGHTS_ITER, wi_mem},
        {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST_LAYER, dst_mem},
        {DNNL_ARG_DST_ITER, dst_state_mem}, {DNNL_ARG_WORKSPACE, workspace}};
    std::unordered_map<int, memory> bwd_args{
        {DNNL_ARG_SRC_LAYER, src_mem}, {DNNL_ARG_SRC_ITER, state_mem},
        {DNNL_ARG_WEIGHTS_LAYER, wl_mem}, {DNNL_ARG_WEIGHTS_ITER, wi_mem},
        {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST_LAYER, dst_mem},
        {DNNL_ARG_DST_ITER, dst_state_mem}, {DNNL_ARG_DIFF_DST_LAYER, diff_dst_mem},
        {DNNL_ARG_DIFF_DST_ITER, diff_dst_state_mem},
        {DNNL_ARG_DIFF_SRC_LAYER, diff_src_mem}, {DNNL_ARG_DIFF_SRC_ITER, diff_state_mem},
        {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_wl_mem},
        {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_wi_mem}, {DNNL_ARG_DIFF_BIAS, diff_bias_mem},
        {DNNL_ARG_WORKSPACE, workspace}};

    fwd.execute(strm, fwd_args);
    bwd.execute(strm, bwd_args);
    strm.wait();
    constexpr int iterations = 20;
    const auto begin = clock::now();
    for (int i = 0; i < iterations; ++i) {
        reorder(wl_user, wl_mem).execute(strm, wl_user, wl_mem);
        reorder(wi_user, wi_mem).execute(strm, wi_user, wi_mem);
        fwd.execute(strm, fwd_args);
        bwd.execute(strm, bwd_args);
    }
    strm.wait();
    const double seconds = std::chrono::duration<double>(clock::now() - begin).count();
    std::cout << "batch=" << N << " milliseconds=" << seconds * 1000.0 / iterations
              << " samples_per_second=" << double(N) * iterations / seconds << '\n';
}
