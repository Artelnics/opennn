//   cuDNN fusion-pattern probe for ResNet-50 at CIFAR geometry.
//
//   The plan's Phase 3 (the MLPerf fusion architecture: batch-norm apply and
//   ReLU fused into the convolution's input, statistics into its epilogue,
//   dReLU + dBN-weight into the data gradient) rests on one premise: that
//   cuDNN's fused engines run at close to plain-convolution speed on our
//   shapes - 8x8 down to 1x1 spatial at batch 2048 - which NVIDIA's MLPerf
//   work at 224x224 never had to establish. This probe measures that premise
//   directly, per real ResNet-50/CIFAR shape, through the library's own
//   cudnn-frontend plumbing (same autotune, same workspace budget as
//   production):
//
//     fprop:  plain conv | conv + genstats | scale-bias-ReLU prologue + conv + genstats (SBRCS)
//     dgrad:  plain dgrad | dgrad + dReLU  | dgrad + dReLU + dbn_weight (DBAR)
//
//   usage: cudnn_fusion_probe [batch=2048] [bf16|fp32] [iterations=20]
//
//   Prints one line per (shape, pattern): ms per execute and the ratio to the
//   plain form. "n/a" means no engine for that pattern on this GPU/cuDNN.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/cuda/cudnn_frontend_utilities.h"

using namespace opennn;
namespace fe = ::cudnn_frontend;
namespace ocf = opennn::cudnn_frontend;

namespace
{

struct ConvShape
{
    const char* name;
    int64_t h, w, c, k, r, s, stride;
};

// ResNet-50 v1.5 on 32x32 inputs after the stem: stage spatial 8, 4, 2, 1.
const std::vector<ConvShape> shapes = {
    {"s0 conv1 1x1 256->64 @8x8",   8, 8, 256,  64, 1, 1, 1},
    {"s0 conv2 3x3 64->64 @8x8",    8, 8,  64,  64, 3, 3, 1},
    {"s0 conv3 1x1 64->256 @8x8",   8, 8,  64, 256, 1, 1, 1},
    {"s1 conv2 3x3 128->128 @4x4",  4, 4, 128, 128, 3, 3, 1},
    {"s1 conv3 1x1 128->512 @4x4",  4, 4, 128, 512, 1, 1, 1},
    {"s2 conv2 3x3 256->256 @2x2",  2, 2, 256, 256, 3, 3, 1},
    {"s2 conv3 1x1 256->1024 @2x2", 2, 2, 256, 1024, 1, 1, 1},
    {"s3 conv2 3x3 512->512 @1x1",  1, 1, 512, 512, 3, 3, 1},
    {"s3 conv3 1x1 512->2048 @1x1", 1, 1, 512, 2048, 1, 1, 1},
};

using TensorPtr = std::shared_ptr<fe::graph::Tensor_attributes>;
using ProbeTensorMap = std::unordered_map<TensorPtr, void*>;

TensorPtr nhwc(fe::graph::Graph& g, const char* name, int64_t n, int64_t c, int64_t h, int64_t w)
{
    return ocf::nhwc_tensor(g, name, n, c, h, w);
}

TensorPtr channel(fe::graph::Graph& g, const char* name, int64_t c, fe::DataType_t dtype = fe::DataType_t::FLOAT)
{
    return g.tensor(fe::graph::Tensor_attributes().set_name(name)
                    .set_dim({1, c, 1, 1}).set_stride({c, 1, c, c}).set_data_type(dtype));
}

TensorPtr krsc(fe::graph::Graph& g, const ConvShape& d)
{
    return g.tensor(fe::graph::Tensor_attributes().set_name("W")
                    .set_dim({d.k, d.c, d.r, d.s})
                    .set_stride({d.r * d.s * d.c, 1, d.s * d.c, d.c}));
}

template<typename Attributes>
Attributes conv_attributes(const ConvShape& d)
{
    const int64_t pad_h = (d.r - 1) / 2, pad_w = (d.s - 1) / 2;
    return Attributes().set_padding({pad_h, pad_w}).set_stride({d.stride, d.stride}).set_dilation({1, 1});
}

struct Timed { bool ok = false; double ms = 0.0; std::string why; };

// Builds, tunes and times one graph. `bind` fills the tensor map with device
// pointers once the graph exists.
template<typename Build, typename Bind>
Timed time_graph(const std::string& tag, Type dtype, int iterations, Build&& build, Bind&& bind)
{
    Timed result;
    try
    {
        auto graph = ocf::new_graph(dtype);
        auto handles = build(*graph);
        int64_t workspace_bytes = 0;
        bool pending = ocf::finalize(*graph, workspace_bytes, tag, device::conv_autotune_enabled());
        ProbeTensorMap tensors;
        bind(handles, tensors);
        ocf::autotune_now(pending, *graph, tensors, workspace_bytes, tag.c_str());

        cudaStream_t stream = Backend::get_compute_stream();
        for (int i = 0; i < 3; ++i)
            ocf::execute_graph(*graph, tensors, ocf::shared_workspace(workspace_bytes), tag, "");
        cudaStreamSynchronize(stream);

        cudaEvent_t begin, end;
        cudaEventCreate(&begin); cudaEventCreate(&end);
        cudaEventRecord(begin, stream);
        for (int i = 0; i < iterations; ++i)
            ocf::execute_graph(*graph, tensors, ocf::shared_workspace(workspace_bytes), tag, "");
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, begin, end);
        cudaEventDestroy(begin); cudaEventDestroy(end);
        result.ok = true;
        result.ms = double(ms) / iterations;
    }
    catch (const std::exception& e)
    {
        result.why = e.what();
        cudaGetLastError();
    }
    return result;
}

void fill(Buffer& buffer, Index bytes)
{
    buffer.resize_bytes(bytes, Device::CUDA);
    // 0x3f3f... is ~0.75 in fp32 and bf16 alike: keeps ReLU masks live.
    cudaMemset(buffer.data, 0x3f, size_t(bytes));
}

void print(const char* pattern, const Timed& t, const Timed& base)
{
    if (!t.ok) { std::printf("    %-34s n/a  (%s)\n", pattern, t.why.substr(0, 60).c_str()); return; }
    if (base.ok && base.ms > 0) std::printf("    %-34s %8.3f ms  %5.2fx plain\n", pattern, t.ms, t.ms / base.ms);
    else                        std::printf("    %-34s %8.3f ms\n", pattern, t.ms);
}

}

int main(int argc, char** argv)
{
    const int64_t n = argc > 1 ? std::atoll(argv[1]) : 2048;
    const std::string precision = argc > 2 ? argv[2] : "bf16";
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 20;
    const Type dtype = precision == "fp32" ? Type::FP32 : Type::BF16;
    const Index elem = dtype == Type::FP32 ? 4 : 2;

    Configuration::instance().set(Device::CUDA, dtype);
    device::set_conv_autotune(true);
    device::set_conv_workspace_cap(-1);

    std::printf("cuDNN %zu, batch %lld, %s, %d timed executes per graph, autotune top-%lld\n\n",
                size_t(cudnnGetVersion()), (long long)n, precision.c_str(), iterations,
                (long long)ocf::autotune_candidate_limit());

    for (const ConvShape& d : shapes)
    {
        const int64_t ho = (d.h + 2 * ((d.r - 1) / 2) - d.r) / d.stride + 1;
        const int64_t wo = (d.w + 2 * ((d.s - 1) / 2) - d.s) / d.stride + 1;
        const Index x_elems = n * d.h * d.w * d.c, y_elems = n * ho * wo * d.k, w_elems = d.k * d.c * d.r * d.s;

        Buffer X{Device::CUDA}, Y{Device::CUDA}, W{Device::CUDA}, DY{Device::CUDA}, DX{Device::CUDA}, Yref{Device::CUDA};
        Buffer scale{Device::CUDA}, bias{Device::CUDA}, mean{Device::CUDA}, invvar{Device::CUDA};
        Buffer sum{Device::CUDA}, sqsum{Device::CUDA}, dscale{Device::CUDA}, dbias{Device::CUDA};
        Buffer eq1{Device::CUDA}, eq2{Device::CUDA}, eq3{Device::CUDA};
        fill(X, x_elems * elem); fill(Y, y_elems * elem); fill(W, w_elems * elem);
        fill(DY, y_elems * elem); fill(DX, x_elems * elem); fill(Yref, x_elems * elem);
        for (Buffer* b : {&scale, &bias, &mean, &invvar, &dscale, &dbias, &eq1, &eq2, &eq3}) fill(*b, d.c * 4);
        fill(sum, d.k * 4); fill(sqsum, d.k * 4);

        std::printf("%s  (N=%lld)\n", d.name, (long long)n);

        // ---- forward ----
        struct F { TensorPtr X, W, Y, S, B, SUM, SQ; };

        const Timed f_plain = time_graph("probe fprop", dtype, iterations,
            [&](fe::graph::Graph& g) { F h; h.X = nhwc(g, "X", n, d.c, d.h, d.w); h.W = krsc(g, d);
                h.Y = g.conv_fprop(h.X, h.W, conv_attributes<fe::graph::Conv_fprop_attributes>(d));
                ocf::set_nhwc_output(h.Y, n, d.k, ho, wo); return h; },
            [&](const F& h, ProbeTensorMap& t) { t[h.X] = X.data; t[h.W] = W.data; t[h.Y] = Y.data; });

        const Timed f_genstats = time_graph("probe fprop+genstats", dtype, iterations,
            [&](fe::graph::Graph& g) { F h; h.X = nhwc(g, "X", n, d.c, d.h, d.w); h.W = krsc(g, d);
                h.Y = g.conv_fprop(h.X, h.W, conv_attributes<fe::graph::Conv_fprop_attributes>(d));
                ocf::set_nhwc_output(h.Y, n, d.k, ho, wo);
                auto [SUM, SQ] = g.genstats(h.Y, fe::graph::Genstats_attributes());
                SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, d.k, 1, 1}).set_stride({d.k, 1, d.k, d.k});
                SQ->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, d.k, 1, 1}).set_stride({d.k, 1, d.k, d.k});
                h.SUM = SUM; h.SQ = SQ; return h; },
            [&](const F& h, ProbeTensorMap& t) { t[h.X] = X.data; t[h.W] = W.data; t[h.Y] = Y.data; t[h.SUM] = sum.data; t[h.SQ] = sqsum.data; });

        const Timed f_sbrcs = time_graph("probe SBRCS", dtype, iterations,
            [&](fe::graph::Graph& g) { F h; h.X = nhwc(g, "X", n, d.c, d.h, d.w); h.W = krsc(g, d);
                h.S = channel(g, "S", d.c); h.B = channel(g, "B", d.c);
                auto scaled = g.pointwise(h.X, h.S, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
                auto shifted = g.pointwise(scaled, h.B, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
                auto relu = g.pointwise(shifted, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
                h.Y = g.conv_fprop(relu, h.W, conv_attributes<fe::graph::Conv_fprop_attributes>(d));
                ocf::set_nhwc_output(h.Y, n, d.k, ho, wo);
                auto [SUM, SQ] = g.genstats(h.Y, fe::graph::Genstats_attributes());
                SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, d.k, 1, 1}).set_stride({d.k, 1, d.k, d.k});
                SQ->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, d.k, 1, 1}).set_stride({d.k, 1, d.k, d.k});
                h.SUM = SUM; h.SQ = SQ; return h; },
            [&](const F& h, ProbeTensorMap& t) { t[h.X] = X.data; t[h.W] = W.data; t[h.S] = scale.data; t[h.B] = bias.data;
                t[h.Y] = Y.data; t[h.SUM] = sum.data; t[h.SQ] = sqsum.data; });

        print("fprop plain", f_plain, f_plain);
        print("fprop + genstats", f_genstats, f_plain);
        print("SBRCS (BN-apply+ReLU prologue+genstats)", f_sbrcs, f_plain);

        // ---- backward ----
        struct Bw { TensorPtr DY, W, DX, Yref, X, M, V, S, DS, DB, E1, E2, E3; };

        const Timed b_plain = time_graph("probe dgrad", dtype, iterations,
            [&](fe::graph::Graph& g) { Bw h; h.DY = nhwc(g, "DY", n, d.k, ho, wo); h.W = krsc(g, d);
                h.DX = g.conv_dgrad(h.DY, h.W, conv_attributes<fe::graph::Conv_dgrad_attributes>(d));
                ocf::set_nhwc_output(h.DX, n, d.c, d.h, d.w); return h; },
            [&](const Bw& h, ProbeTensorMap& t) { t[h.DY] = DY.data; t[h.W] = W.data; t[h.DX] = DX.data; });

        const Timed b_drelu = time_graph("probe dgrad+drelu", dtype, iterations,
            [&](fe::graph::Graph& g) { Bw h; h.DY = nhwc(g, "DY", n, d.k, ho, wo); h.W = krsc(g, d);
                auto dgrad = g.conv_dgrad(h.DY, h.W, conv_attributes<fe::graph::Conv_dgrad_attributes>(d));
                dgrad->set_dim({n, d.c, d.h, d.w}).set_stride(ocf::nhwc_strides(d.c, d.h, d.w));
                h.Yref = nhwc(g, "Yref", n, d.c, d.h, d.w);
                h.DX = g.pointwise(dgrad, h.Yref, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD));
                ocf::set_nhwc_output(h.DX, n, d.c, d.h, d.w); return h; },
            [&](const Bw& h, ProbeTensorMap& t) { t[h.DY] = DY.data; t[h.W] = W.data; t[h.Yref] = Yref.data; t[h.DX] = DX.data; });

        const Timed b_dbar = time_graph("probe DBAR", dtype, iterations,
            [&](fe::graph::Graph& g) { Bw h; h.DY = nhwc(g, "DY", n, d.k, ho, wo); h.W = krsc(g, d);
                auto dgrad = g.conv_dgrad(h.DY, h.W, conv_attributes<fe::graph::Conv_dgrad_attributes>(d));
                dgrad->set_dim({n, d.c, d.h, d.w}).set_stride(ocf::nhwc_strides(d.c, d.h, d.w));
                h.Yref = nhwc(g, "Yref", n, d.c, d.h, d.w);
                auto drelu = g.pointwise(dgrad, h.Yref, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD));
                drelu->set_dim({n, d.c, d.h, d.w}).set_stride(ocf::nhwc_strides(d.c, d.h, d.w));
                h.DX = drelu; h.DX->set_output(true);
                h.X = nhwc(g, "X", n, d.c, d.h, d.w);
                h.M = channel(g, "M", d.c); h.V = channel(g, "V", d.c); h.S = channel(g, "S", d.c);
                auto [ds, db, e1, e2, e3] = g.dbn_weight(drelu, h.X, h.M, h.V, h.S, fe::graph::DBN_weight_attributes());
                for (auto& o : {ds, db, e1, e2, e3})
                    o->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, d.c, 1, 1}).set_stride({d.c, 1, d.c, d.c});
                h.DS = ds; h.DB = db; h.E1 = e1; h.E2 = e2; h.E3 = e3; return h; },
            [&](const Bw& h, ProbeTensorMap& t) { t[h.DY] = DY.data; t[h.W] = W.data; t[h.Yref] = Yref.data; t[h.DX] = DX.data;
                t[h.X] = X.data; t[h.M] = mean.data; t[h.V] = invvar.data; t[h.S] = scale.data;
                t[h.DS] = dscale.data; t[h.DB] = dbias.data; t[h.E1] = eq1.data; t[h.E2] = eq2.data; t[h.E3] = eq3.data; });

        print("dgrad plain", b_plain, b_plain);
        print("dgrad + dReLU", b_drelu, b_plain);
        print("DBAR (dgrad+dReLU+dbn_weight)", b_dbar, b_plain);
        std::printf("\n");
    }
    return 0;
}
