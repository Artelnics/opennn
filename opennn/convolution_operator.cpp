//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef HAVE_CUDNN_FRONTEND
#include <cudnn_frontend.h>
#endif

#include "convolution_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"
#include "cudnn_frontend_utilities.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

namespace
{

void configure_convolution_descriptors(ConvolutionOp& op)
{
    op.planned_batch_size = 0;

    if (op.kernels_number <= 0) return;

    if (!op.kernel_descriptor)
    {
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&op.kernel_descriptor.handle));
        op.kernel_descriptor.deleter = &cudnnDestroyFilterDescriptor;
    }

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(op.kernel_descriptor,
                                           to_cudnn(op.compute_dtype),
                                           CUDNN_TENSOR_NHWC,
                                           to_int(op.kernels_number), to_int(op.kernel_channels),
                                           to_int(op.kernel_height),  to_int(op.kernel_width)));

    if (!op.convolution_descriptor)
    {
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&op.convolution_descriptor.handle));
        op.convolution_descriptor.deleter = &cudnnDestroyConvolutionDescriptor;
    }

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(op.convolution_descriptor,
                                                to_int(op.padding_height), to_int(op.padding_width),
                                                to_int(op.row_stride), to_int(op.column_stride),
                                                1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    CHECK_CUDNN(cudnnSetConvolutionMathType(op.convolution_descriptor,
                                            CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
}

}

#endif


struct ConvolutionOp::ConvGraphCache
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> fwd, wgrad, bgrad, dgrad;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_W, fwd_B, fwd_Y;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> wgrad_X, wgrad_DY, wgrad_DW;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bgrad_DY, bgrad_DB;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> dgrad_W, dgrad_DY, dgrad_DX;
        void* fwd_workspace = nullptr;
        void* wgrad_workspace = nullptr;
        void* bgrad_workspace = nullptr;
        void* dgrad_workspace = nullptr;
        bool fwd_autotune = false;
        bool wgrad_autotune = false;
        bool bgrad_autotune = false;
        bool dgrad_autotune = false;
    };

    map<Index, Entry> entries;

    ~ConvGraphCache()
    {
        for (auto& [_, entry] : entries)
        {
            device::deallocate(Device::CUDA, entry.fwd_workspace, 0);
            device::deallocate(Device::CUDA, entry.wgrad_workspace, 0);
            device::deallocate(Device::CUDA, entry.bgrad_workspace, 0);
            device::deallocate(Device::CUDA, entry.dgrad_workspace, 0);
        }
    }
#endif

    bool disabled = false;
};

ConvolutionOp::ConvolutionOp() = default;

ConvolutionOp::~ConvolutionOp() = default;

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

namespace cudnn_fe
{

struct Dims
{
    int64_t batch, channels, height, width;
    int64_t kernels, kernel_height, kernel_width;
    int64_t output_height, output_width;
    int64_t padding_height, padding_width;
    int64_t row_stride, column_stride;
};

Dims make_dims(const ConvolutionOp& op, int64_t batch)
{
    return {
        batch, op.kernel_channels, op.input_height, op.input_width,
        op.kernels_number, op.kernel_height, op.kernel_width,
        (op.input_height + 2 * op.padding_height - op.kernel_height) / op.row_stride + 1,
        (op.input_width + 2 * op.padding_width - op.kernel_width) / op.column_stride + 1,
        op.padding_height, op.padding_width,
        op.row_stride, op.column_stride
    };
}

vector<int64_t> krsc_strides(const Dims& d)
{
    return {d.kernel_height * d.kernel_width * d.channels, 1,
            d.kernel_width * d.channels, d.channels};
}

template<typename Attributes>
Attributes conv_attributes(const Dims& d)
{
    return Attributes()
           .set_padding({d.padding_height, d.padding_width})
           .set_stride({d.row_stride, d.column_stride})
           .set_dilation({1, 1});
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
krsc_tensor(cudnn_frontend::graph::Graph& graph, const char* name, const Dims& d)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({d.kernels, d.channels, d.kernel_height, d.kernel_width})
                        .set_stride(krsc_strides(d)));
}

void build_forward(ConvolutionOp::ConvGraphCache::Entry& entry, const Dims& d,
                   bool fuse_relu, bool use_bias)
{
    auto graph = new_graph();

    entry.fwd_X = nhwc_tensor(*graph, "X", d.batch, d.channels, d.height, d.width);
    entry.fwd_W = krsc_tensor(*graph, "W", d);

    entry.fwd_Y = graph->conv_fprop(entry.fwd_X, entry.fwd_W,
                                    conv_attributes<cudnn_frontend::graph::Conv_fprop_attributes>(d));

    if (use_bias)
    {
        entry.fwd_B = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                    .set_name("B")
                                    .set_dim({1, d.kernels, 1, 1})
                                    .set_stride({d.kernels, 1, d.kernels, d.kernels}));

        entry.fwd_Y = graph->pointwise(entry.fwd_Y, entry.fwd_B,
                                       cudnn_frontend::graph::Pointwise_attributes()
                                       .set_mode(cudnn_frontend::PointwiseMode_t::ADD));
    }

    if (fuse_relu)
        entry.fwd_Y = graph->pointwise(entry.fwd_Y,
                                       cudnn_frontend::graph::Pointwise_attributes()
                                       .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD));

    set_nhwc_output(entry.fwd_Y, d.batch, d.kernels, d.output_height, d.output_width);

    entry.fwd_autotune = finalize(*graph, entry.fwd_workspace, "forward", autotune_enabled());
    entry.fwd = graph;
}

void build_wgrad(ConvolutionOp::ConvGraphCache::Entry& entry, const Dims& d)
{
    auto graph = new_graph();

    entry.wgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);
    entry.wgrad_X  = nhwc_tensor(*graph, "X", d.batch, d.channels, d.height, d.width);

    entry.wgrad_DW = graph->conv_wgrad(entry.wgrad_DY, entry.wgrad_X,
                                       conv_attributes<cudnn_frontend::graph::Conv_wgrad_attributes>(d));
    entry.wgrad_DW->set_output(true)
                   .set_dim({d.kernels, d.channels, d.kernel_height, d.kernel_width})
                   .set_stride(krsc_strides(d));

    entry.wgrad_autotune = finalize(*graph, entry.wgrad_workspace, "wgrad", autotune_enabled());
    entry.wgrad = graph;
}

void build_bgrad(ConvolutionOp::ConvGraphCache::Entry& entry, const Dims& d)
{
    auto graph = new_graph();

    entry.bgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);

    entry.bgrad_DB = graph->reduction(entry.bgrad_DY,
                                      cudnn_frontend::graph::Reduction_attributes()
                                      .set_mode(cudnn_frontend::ReductionMode_t::ADD));
    entry.bgrad_DB->set_output(true)
                   .set_dim({1, d.kernels, 1, 1})
                   .set_stride({d.kernels, 1, d.kernels, d.kernels});

    entry.bgrad_autotune = finalize(*graph, entry.bgrad_workspace, "bgrad", autotune_enabled());
    entry.bgrad = graph;
}

void build_dgrad(ConvolutionOp::ConvGraphCache::Entry& entry, const Dims& d)
{
    auto graph = new_graph();

    entry.dgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);
    entry.dgrad_W  = krsc_tensor(*graph, "W", d);

    entry.dgrad_DX = graph->conv_dgrad(entry.dgrad_DY, entry.dgrad_W,
                                       conv_attributes<cudnn_frontend::graph::Conv_dgrad_attributes>(d));
    set_nhwc_output(entry.dgrad_DX, d.batch, d.channels, d.height, d.width);

    entry.dgrad_autotune = finalize(*graph, entry.dgrad_workspace, "dgrad", autotune_enabled());
    entry.dgrad = graph;
}

string timing_label(const ConvolutionOp& op, const char* kind)
{
    if (!graph_timing_enabled()) return {};
    return format("{} {}x{}x{} k{}x{}x{} s{}", kind,
                  op.input_height, op.input_width, op.kernel_channels,
                  op.kernel_height, op.kernel_width, op.kernels_number, op.row_stride);
}

}  // namespace cudnn_fe

#endif


void ConvolutionOp::set(Index new_input_h, Index new_input_w,
                      Index new_kernels_n, Index new_kernel_h, Index new_kernel_w, Index new_kernel_c,
                      Index new_row_stride, Index new_column_stride,
                      Index new_padding_h, Index new_padding_w,
                      Type new_compute_dtype)
{
    input_height     = new_input_h;
    input_width      = new_input_w;
    kernels_number   = new_kernels_n;
    kernel_height    = new_kernel_h;
    kernel_width     = new_kernel_w;
    kernel_channels  = new_kernel_c;
    row_stride       = new_row_stride;
    column_stride    = new_column_stride;
    padding_height   = new_padding_h;
    padding_width    = new_padding_w;
    compute_dtype = new_compute_dtype;

#ifdef OPENNN_HAS_CUDA
    configure_convolution_descriptors(*this);
#endif
}

vector<TensorSpec> ConvolutionOp::parameter_specs() const
{
    // The bias is redundant under batch normalization (its beta absorbs it).
    if (!use_bias)
        return {{{kernels_number, kernel_height, kernel_width, kernel_channels}, compute_dtype}};

    return {
        {{kernels_number}, compute_dtype},
        {{kernels_number, kernel_height, kernel_width, kernel_channels}, compute_dtype},
    };
}

void ConvolutionOp::link_parameters(span<const TensorView> views)
{
    if (views.empty()) return;
    bias    = use_bias ? views[0] : TensorView{};
    weights = views[use_bias ? 1 : 0];
}

void ConvolutionOp::link_gradients(span<const TensorView> views)
{
    if (views.empty()) return;
    bias_gradient   = use_bias ? views[0] : TensorView{};
    weight_gradient = views[use_bias ? 1 : 0];
}

void ConvolutionOp::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void ConvolutionOp::set_parameters_glorot()
{
    if (weights.empty()) return;
    const Index kernel_area = kernel_height * kernel_width;
    const float limit = glorot_limit(kernel_area * kernel_channels, kernel_area * kernels_number);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}


void ConvolutionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/)
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    apply(input, output);
}

void ConvolutionOp::apply_delta(const TensorView& input,
                                const TensorView& output_delta,
                                TensorView& input_delta) const
{
    if (output_delta.is_cuda())
    {
        apply_delta_gpu(input, output_delta, input_delta);
        return;
    }
    apply_delta_cpu(input, output_delta, input_delta);
}

void ConvolutionOp::apply(const TensorView& input, TensorView& output)
{
    if (input.is_cuda())
    {
        apply_gpu(input, output);
        return;
    }
    apply_cpu(input, output);
}

void ConvolutionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    auto& backward_slots = bp.backward_slots[layer];

    const TensorView& input        = get_input(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(backward_slots, input_delta_slots, 0, empty);

    apply_delta(input, output_delta, input_delta);
}

array<pair<Index, Index>, 4> ConvolutionOp::nhwc_padding() const
{
    return {
        make_pair(Index(0), Index(0)),
        make_pair(padding_height, padding_height),
        make_pair(padding_width,  padding_width),
        make_pair(Index(0), Index(0))
    };
}

void ConvolutionOp::apply_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = use_bias ? bias.as_vector() : VectorMap(nullptr, 0);

    const Index batch_size = inputs.dimension(0);

    const array<Index, 3> conv_dims({1, 2, 3});
    const array<Index, 3> out_slice_shape({batch_size, output.shape[1], output.shape[2]});

    const auto input_paddings = nhwc_padding();

    TensorMap4 outputs = output.as_tensor<4>();

    const bool needs_padding = padding_height > 0 || padding_width > 0;
    const Tensor4 padded_inputs_storage = needs_padding
        ? Tensor4(inputs.pad(input_paddings))
        : Tensor4();
    const TensorMap4 padded_inputs = needs_padding
        ? TensorMap4(const_cast<float*>(padded_inputs_storage.data()),
                     padded_inputs_storage.dimensions())
        : inputs;

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_map = weights.as_tensor<3>(kernel_index);
        const float bias_value = use_bias ? biases(kernel_index) : 0.0f;

        if (row_stride == 1 && column_stride == 1)
            outputs.chip(kernel_index, 3).device(get_device()) =
                padded_inputs.convolve(kernel_map, conv_dims).reshape(out_slice_shape) + bias_value;
        else
            outputs.chip(kernel_index, 3).device(get_device()) =
                padded_inputs.convolve(kernel_map, conv_dims)
                             .stride(array<Index, 4>({1, row_stride, column_stride, 1}))
                             .reshape(out_slice_shape) + bias_value;
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
void ConvolutionOp::apply_delta_cpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    const TensorMap4 inputs        = input.as_tensor<4>();
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();

    VectorMap bias_gradients = use_bias ? bias_gradient.as_vector() : VectorMap(nullptr, 0);
    TensorMap4 weight_gradients = weight_gradient.as_tensor<4>();
    const TensorMap4 kernels = weights.as_tensor<4>();

    if (use_bias) bias_gradients.setZero();
    weight_gradients.setZero();

    const bool write_input_delta = !input_delta.empty();
    float* const input_delta_data = write_input_delta ? input_delta.as<float>() : nullptr;
    if (write_input_delta)
        fill_n(input_delta_data, input_delta.size(), 0.0f);

    const Index batch_size = output_deltas.dimension(0);
    const Index output_height = output_deltas.dimension(1);
    const Index output_width = output_deltas.dimension(2);

    for (Index image_index = 0; image_index < batch_size; ++image_index)
    {
        for (Index output_row = 0; output_row < output_height; ++output_row)
        {
            for (Index output_column = 0; output_column < output_width; ++output_column)
            {
                for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
                {
                    const float delta = output_deltas(image_index, output_row, output_column, kernel_index);
                    if (use_bias) bias_gradients(kernel_index) += delta;

                    for (Index kernel_row = 0; kernel_row < kernel_height; ++kernel_row)
                    {
                        const Index input_row = output_row * row_stride + kernel_row - padding_height;
                        if (input_row < 0 || input_row >= input_height) continue;

                        for (Index kernel_column = 0; kernel_column < kernel_width; ++kernel_column)
                        {
                            const Index input_column = output_column * column_stride + kernel_column - padding_width;
                            if (input_column < 0 || input_column >= input_width) continue;

                            for (Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
                            {
                                weight_gradients(kernel_index, kernel_row, kernel_column, channel_index) +=
                                    inputs(image_index, input_row, input_column, channel_index) * delta;

                                if (write_input_delta)
                                {
                                    const Index input_index = ((image_index * input_height + input_row)
                                                              * input_width + input_column)
                                                            * kernel_channels + channel_index;
                                    input_delta_data[input_index] +=
                                        kernels(kernel_index, kernel_row, kernel_column, channel_index) * delta;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef OPENNN_HAS_CUDA

void ConvolutionOp::plan_convolution_algorithms(const TensorView& input, const TensorView& output)
{
    cudnnHandle_t handle = Backend::get_cudnn_handle();
    cudnnTensorDescriptor_t input_desc  = input.get_descriptor();
    cudnnTensorDescriptor_t output_desc = output.get_descriptor();

    auto pick_algo = [](auto* perfs, int count, auto fallback) {
        for (int i = 0; i < count; ++i)
            if (perfs[i].status == CUDNN_STATUS_SUCCESS)
                return perfs[i].algo;
        return fallback;
    };

    int count = 0;

    cudnnConvolutionFwdAlgoPerf_t fwd_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        handle, input_desc, kernel_descriptor, convolution_descriptor, output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &count, fwd_perf));
    algorithm_forward = pick_algo(fwd_perf, count, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        handle, kernel_descriptor, output_desc, convolution_descriptor, input_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &count, bwd_data_perf));
    algorithm_data = pick_algo(bwd_data_perf, count, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        handle, input_desc, output_desc, convolution_descriptor, kernel_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &count, bwd_filter_perf));
    algorithm_filter = pick_algo(bwd_filter_perf, count, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);

    size_t fwd_ws = 0, bwd_data_ws = 0, bwd_filter_ws = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, kernel_descriptor, convolution_descriptor, output_desc,
        algorithm_forward, &fwd_ws));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, kernel_descriptor, output_desc, convolution_descriptor, input_desc,
        algorithm_data, &bwd_data_ws));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, input_desc, output_desc, convolution_descriptor, kernel_descriptor,
        algorithm_filter, &bwd_filter_ws));

    cudnn_workspace_size_ = max({fwd_ws, bwd_data_ws, bwd_filter_ws});
    ensure_cudnn_conv_workspace(cudnn_workspace_size_);

    planned_batch_size = input.shape[0];
}


void ConvolutionOp::apply_gpu(const TensorView& input, TensorView& output)
{
    PROFILE_SCOPE("op:conv_fwd");

#ifdef HAVE_CUDNN_FRONTEND
    if (input.type == Type::FP32 && cudnn_fe::frontend_enabled()
        && cudnn_fe::run_frontend(conv_graph_cache, "ConvolutionOp", [&](ConvGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.fwd)
            // The only fused activation the layer requests is ReLU
            // (see Convolutional::update_convolution_operator).
            cudnn_fe::build_forward(entry, cudnn_fe::make_dims(*this, input.shape[0]),
                                    fused_activation != nullptr, use_bias);

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.fwd_X] = input.data;
        tensors[entry.fwd_W] = weights.data;
        if (use_bias) tensors[entry.fwd_B] = bias.data;
        tensors[entry.fwd_Y] = output.data;

        cudnn_fe::autotune_now(entry.fwd_autotune, *entry.fwd, tensors, entry.fwd_workspace);

        cudnn_fe::execute_graph(*entry.fwd, tensors, entry.fwd_workspace, "forward execute",
                                cudnn_fe::timing_label(*this, "conv_fwd"));
    }))
        return;
#endif

    if (input.shape[0] > planned_batch_size)
        plan_convolution_algorithms(input, output);

    void* workspace = ensure_cudnn_conv_workspace(cudnn_workspace_size_);

    if (fused_activation)
    {
        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            Backend::get_cudnn_handle(),
            &one,
            input.get_descriptor(),  input.data,
            kernel_descriptor,        weights.data,
            convolution_descriptor,
            algorithm_forward,
            workspace, cudnn_workspace_size_,
            &zero,
            output.get_descriptor(), output.data,
            bias.get_descriptor(),   bias.data,
            fused_activation,
            output.get_descriptor(), output.data));
        return;
    }

    CHECK_CUDNN(cudnnConvolutionForward(Backend::get_cudnn_handle(),
                                        &one,
                                        input.get_descriptor(),  input.data,
                                        kernel_descriptor,        weights.data,
                                        convolution_descriptor,
                                        algorithm_forward,
                                        workspace, cudnn_workspace_size_,
                                        &zero,
                                        output.get_descriptor(), output.data));

    if (use_bias)
        CHECK_CUDNN(cudnnAddTensor(Backend::get_cudnn_handle(),
                                   &one,
                                   bias.get_descriptor(), bias.data,
                                   &one,
                                   output.get_descriptor(), output.data));
}

void ConvolutionOp::apply_delta_gpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    PROFILE_SCOPE("op:conv_bwd");

    assert(output_delta.type == input.type);
    assert(weight_gradient.type == Type::FP32);

#ifdef HAVE_CUDNN_FRONTEND
    if (input.type == Type::FP32 && cudnn_fe::frontend_enabled()
        && cudnn_fe::run_frontend(conv_graph_cache, "ConvolutionOp", [&](ConvGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        const cudnn_fe::Dims dims = cudnn_fe::make_dims(*this, input.shape[0]);

        if (!entry.wgrad) cudnn_fe::build_wgrad(entry, dims);

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.wgrad_DY] = output_delta.data;
        tensors[entry.wgrad_X]  = input.data;
        tensors[entry.wgrad_DW] = weight_gradient.data;

        cudnn_fe::autotune_now(entry.wgrad_autotune, *entry.wgrad, tensors, entry.wgrad_workspace);

        cudnn_fe::execute_graph(*entry.wgrad, tensors, entry.wgrad_workspace, "wgrad execute",
                                cudnn_fe::timing_label(*this, "conv_wgrad"));

        if (use_bias)
        {
            // cudnnConvolutionBackwardBias is pathologically slow on NHWC
            // deltas (~2 ms for a 16-channel reduction); use a frontend
            // reduction graph instead.
            if (!entry.bgrad) cudnn_fe::build_bgrad(entry, dims);

            unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> bgrad_tensors;
            bgrad_tensors[entry.bgrad_DY] = output_delta.data;
            bgrad_tensors[entry.bgrad_DB] = bias_gradient.data;

            cudnn_fe::autotune_now(entry.bgrad_autotune, *entry.bgrad, bgrad_tensors, entry.bgrad_workspace);

            cudnn_fe::execute_graph(*entry.bgrad, bgrad_tensors, entry.bgrad_workspace, "bgrad execute",
                                    cudnn_fe::timing_label(*this, "conv_bgrad"));
        }

        if (input_delta.data && input_delta.size() != 0)
        {
            if (!entry.dgrad) cudnn_fe::build_dgrad(entry, dims);

            unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> dgrad_tensors;
            dgrad_tensors[entry.dgrad_DY] = output_delta.data;
            dgrad_tensors[entry.dgrad_W]  = weights.data;
            dgrad_tensors[entry.dgrad_DX] = input_delta.data;

            cudnn_fe::autotune_now(entry.dgrad_autotune, *entry.dgrad, dgrad_tensors, entry.dgrad_workspace);

            cudnn_fe::execute_graph(*entry.dgrad, dgrad_tensors, entry.dgrad_workspace, "dgrad execute",
                                    cudnn_fe::timing_label(*this, "conv_dgrad"));
        }
    }))
        return;
#endif

    // The legacy algorithms are planned in the forward pass only when the
    // frontend path is not in use; plan here if this is the first legacy call.
    if (input.shape[0] > planned_batch_size)
        const_cast<ConvolutionOp*>(this)->plan_convolution_algorithms(input, output_delta);

    const bool bf16 = (input.type == Type::BF16);

    void* weight_gradient_buffer = weight_gradient.data;
    bfloat16* weight_gradient_bf16_workspace = nullptr;

    if (bf16)
    {
        weight_gradient_bf16_workspace = ensure_bf16_gradient_workspace(weight_gradient.size());
        weight_gradient_buffer = weight_gradient_bf16_workspace;
    }

    void* workspace = ensure_cudnn_conv_workspace(cudnn_workspace_size_);

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(Backend::get_cudnn_handle(),
        &one,
        input.get_descriptor(),        input.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_filter,
        workspace, cudnn_workspace_size_,
        &zero,
        kernel_descriptor, weight_gradient_buffer));

    if (use_bias)
    {
        TensorView output_delta_for_bias = output_delta;

        if (bf16)
        {
            float* const output_delta_fp32 = ensure_bf16_to_fp32_workspace(output_delta.size());
            cast_bf16_to_fp32_cuda(output_delta.size(),
                                   output_delta.as<bfloat16>(),
                                   output_delta_fp32);

            output_delta_for_bias = TensorView(output_delta_fp32, output_delta.shape, Type::FP32, Device::CUDA);
        }

        CHECK_CUDNN(cudnnConvolutionBackwardBias(Backend::get_cudnn_handle(),
            &one,
            output_delta_for_bias.get_descriptor(), output_delta_for_bias.data,
            &zero,
            bias_gradient.get_descriptor(), bias_gradient.data));
    }

    if (bf16)
        cast_bf16_to_fp32_cuda(weight_gradient.size(), weight_gradient_bf16_workspace, weight_gradient.as_float());

    if (!input_delta.data || input_delta.size() == 0) return;

    CHECK_CUDNN(cudnnConvolutionBackwardData(Backend::get_cudnn_handle(),
        &one,
        kernel_descriptor, weights.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_data,
        workspace, cudnn_workspace_size_,
        &zero,
        input_delta.get_descriptor(), input_delta.data));
}

#else

void ConvolutionOp::apply_gpu(const TensorView&, TensorView&)                                       { throw runtime_error("Convolution::apply_gpu: CUDA support not compiled in."); }
void ConvolutionOp::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("Convolution::apply_delta_gpu: CUDA support not compiled in."); }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
