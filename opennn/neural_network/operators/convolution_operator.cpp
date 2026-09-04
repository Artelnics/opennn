//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/convolution_operator.h"

#include "opennn/core/cuda/cudnn_frontend_utilities.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/kernel_quantization.cuh"
#endif
#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"

namespace opennn
{

#ifndef OPENNN_HAS_CUDA

struct ConvolutionOperator::ConvGraphCache
{
    mutex access_mutex;
    bool disabled = false;
};
#endif

#ifdef OPENNN_HAS_CUDA

struct ConvolutionOperator::ConvGraphCache
{
    mutex access_mutex;

    struct Entry
    {
        cudnn_frontend::GraphSlot fwd, wgrad, bgrad, dgrad;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_W, fwd_B, fwd_R, fwd_Y;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> wgrad_X, wgrad_DY, wgrad_DW;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bgrad_DY, bgrad_DB;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> dgrad_W, dgrad_DY, dgrad_DX, dgrad_R;
        bool wgrad_fp32_output = false;
        bool dgrad_adds = false;
        device::CudaEvent fork_event, join_event;
    };

    unordered_map<Index, Entry> entries;
    bool disabled = false;
};

namespace cudnn_frontend
{
using namespace ::cudnn_frontend;

namespace
{

struct Dims
{
    int64_t batch, channels, height, width;
    int64_t kernels, kernel_height, kernel_width;
    int64_t output_height, output_width;
    int64_t padding_height, padding_width;
    int64_t row_stride, column_stride;
};

Dims make_dims(const ConvolutionOperator& op, int64_t batch)
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

shared_ptr<graph::Tensor_attributes>
krsc_tensor(graph::Graph& graph, const char* name, const Dims& d)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({d.kernels, d.channels, d.kernel_height, d.kernel_width})
                        .set_stride(krsc_strides(d)));
}

void build_forward(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d,
                   bool fuse_relu, bool use_bias, Type dtype, bool fuse_residual = false)
{
    auto graph = new_graph(dtype);

    entry.fwd_X = nhwc_tensor(*graph, "X", d.batch, d.channels, d.height, d.width);
    entry.fwd_W = krsc_tensor(*graph, "W", d);

    entry.fwd_Y = graph->conv_fprop(entry.fwd_X, entry.fwd_W,
                                    conv_attributes<graph::Conv_fprop_attributes>(d));

    if (use_bias)
    {
        entry.fwd_B = graph->tensor(graph::Tensor_attributes()
                                    .set_name("B")
                                    .set_dim({1, d.kernels, 1, 1})
                                    .set_stride({d.kernels, 1, d.kernels, d.kernels}));

        entry.fwd_Y = graph->pointwise(entry.fwd_Y, entry.fwd_B,
                                       graph::Pointwise_attributes()
                                       .set_mode(PointwiseMode_t::ADD));
    }

    // The residual add belongs in the graph, not after it. Left outside, it
    // was a separate streaming pass over the block output -- 26% of GPU time
    // once batch norm stopped being the headline -- reading two tensors and
    // writing a third that the convolution had just written.
    if (fuse_residual)
    {
        entry.fwd_R = nhwc_tensor(*graph, "R", d.batch, d.kernels,
                                  d.output_height, d.output_width);
        entry.fwd_Y = graph->pointwise(entry.fwd_Y, entry.fwd_R,
                                       graph::Pointwise_attributes()
                                       .set_mode(PointwiseMode_t::ADD));
    }

    if (fuse_relu)
        entry.fwd_Y = graph->pointwise(entry.fwd_Y,
                                       graph::Pointwise_attributes()
                                       .set_mode(PointwiseMode_t::RELU_FWD));

    set_nhwc_output(entry.fwd_Y, d.batch, d.kernels, d.output_height, d.output_width);

    entry.fwd.build(graph, "forward");
}

void build_wgrad(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d, Type dtype,
                 bool fp32_output = false)
{
    auto graph = new_graph(dtype);

    entry.wgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);
    entry.wgrad_X  = nhwc_tensor(*graph, "X", d.batch, d.channels, d.height, d.width);

    entry.wgrad_DW = graph->conv_wgrad(entry.wgrad_DY, entry.wgrad_X,
                                       conv_attributes<graph::Conv_wgrad_attributes>(d));
    entry.wgrad_DW->set_output(true)
                   .set_dim({d.kernels, d.channels, d.kernel_height, d.kernel_width})
                   .set_stride(krsc_strides(d));

    if (fp32_output)
        entry.wgrad_DW->set_data_type(DataType_t::FLOAT);

    entry.wgrad.build(graph, "wgrad");
}

void build_bgrad(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d, Type dtype)
{
    auto graph = new_graph(dtype);

    entry.bgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);

    entry.bgrad_DB = graph->reduction(entry.bgrad_DY,
                                      graph::Reduction_attributes()
                                      .set_mode(ReductionMode_t::ADD));

    set_per_channel_output(entry.bgrad_DB, d.kernels);

    entry.bgrad.build(graph, "bgrad");
}

void build_dgrad(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d, Type dtype,
                 bool add_residual = false)
{
    auto graph = new_graph(dtype);

    entry.dgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);
    entry.dgrad_W  = krsc_tensor(*graph, "W", d);
    entry.dgrad_R  = nullptr;

    entry.dgrad_DX = graph->conv_dgrad(entry.dgrad_DY, entry.dgrad_W,
                                       conv_attributes<graph::Conv_dgrad_attributes>(d));

    if (add_residual)
    {
        entry.dgrad_DX->set_dim({d.batch, d.channels, d.height, d.width})
                      .set_stride(nhwc_strides(d.channels, d.height, d.width));
        entry.dgrad_R  = nhwc_tensor(*graph, "R", d.batch, d.channels, d.height, d.width);
        entry.dgrad_DX = graph->pointwise(entry.dgrad_DX, entry.dgrad_R,
                                          graph::Pointwise_attributes()
                                          .set_mode(PointwiseMode_t::ADD));
    }

    set_nhwc_output(entry.dgrad_DX, d.batch, d.channels, d.height, d.width);

    entry.dgrad.build(graph, "dgrad");
}

string conv_timing_label(const ConvolutionOperator& op, const char* kind)
{
    return timing_label("{} {}x{}x{} k{}x{}x{} s{}", kind,
                                          op.input_height, op.input_width, op.kernel_channels,
                                          op.kernel_height, op.kernel_width, op.kernels_number,
                                          op.row_stride);
}

template<typename Build>
bool build_preferred(const ConvolutionOperator& op, const char* kind, int64_t batch,
                     const char* consequence, Build&& build)
{
    try
    {
        build();
        return true;
    }
    catch (const exception& e)
    {
        cerr << "ConvolutionOperator " << kind << " "
             << op.input_height << "x" << op.input_width << "x" << op.kernel_channels
             << " k" << op.kernel_height << "x" << op.kernel_width << "x" << op.kernels_number
             << " batch " << batch << ": no engine (" << e.what() << "); "
             << consequence << ".\n";
        return false;
    }
}

}

}

#endif

ConvolutionOperator::ConvolutionOperator()
    : conv_graph_cache(make_unique<ConvGraphCache>())
{
}

ConvolutionOperator::~ConvolutionOperator() = default;

vector<TensorSpec> ConvolutionOperator::parameter_specs() const
{

    if (!use_bias)
        return {{{kernels_number, kernel_height, kernel_width, kernel_channels}, weights_dtype}};

    return {
        {{kernels_number}, compute_dtype},
        {{kernels_number, kernel_height, kernel_width, kernel_channels}, weights_dtype},
    };
}

vector<Operator::SlotQuantization> ConvolutionOperator::parameter_quantization() const
{
    if (!use_bias)
        return {{kernels_number, 0}};

    return {{}, {kernels_number, 0}};
}

vector<Operator::ParameterSlot> ConvolutionOperator::parameter_slots()
{
    return {
        {&bias,    &bias_gradient,   use_bias},
        {&weights, &weight_gradient},
    };
}

void ConvolutionOperator::link_parameters(span<const TensorView> views)
{
    if (link_slots(views, &ParameterSlot::parameter)) weights_relinked = true;
}

void ConvolutionOperator::link_parameter_scales(span<const TensorView> views)
{
    weight_scale = views.empty() ? TensorView{} : views.back();
}

void ConvolutionOperator::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void ConvolutionOperator::set_parameters_glorot()
{
    if (weights.empty()) return;
    const Index kernel_area = kernel_height * kernel_width;
    const float limit = glorot_limit(kernel_area * kernel_channels, kernel_area * kernels_number);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}

void ConvolutionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode  )
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (input.is_cuda()) apply_gpu(input, output);
    else                  apply_cpu(input, output);
}

void ConvolutionOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& backward_slots = back_propagation.slots[layer];

    const TensorView& input        = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    TensorView empty_input_delta;
    TensorView& input_delta = slot_or(backward_slots, input_delta_slots, 0,
                                      empty_input_delta);
    const TensorView& addend = back_propagation.input_delta_addend(layer, 0);

    if (output_delta.is_cuda())
        apply_delta_gpu(input, output_delta, input_delta, addend);
    else
    {
        apply_delta_cpu(input, output_delta, input_delta);
        if (addend.get_data() && input_delta.get_data())
            input_delta.as_vector() += addend.as_vector();
    }
}

namespace
{

void im2col(const float* image, Index input_height, Index input_width, Index channels,
            Index kernel_height, Index kernel_width,
            Index padding_height, Index padding_width,
            Index row_stride, Index column_stride,
            Index output_height, Index output_width,
            float* col)
{
    const Index patch_size = kernel_height * kernel_width * channels;

    for (Index output_row = 0; output_row < output_height; ++output_row)
        for (Index output_column = 0; output_column < output_width; ++output_column)
        {
            float* patch = col + (output_row * output_width + output_column) * patch_size;
            const Index first_input_column = output_column * column_stride - padding_width;

            for (Index kernel_row = 0; kernel_row < kernel_height; ++kernel_row)
            {
                const Index input_row = output_row * row_stride + kernel_row - padding_height;
                float* patch_row = patch + kernel_row * kernel_width * channels;

                if (input_row < 0 || input_row >= input_height)
                {
                    fill_n(patch_row, kernel_width * channels, 0.0f);
                    continue;
                }

                const float* source = image + (input_row * input_width + first_input_column) * channels;

                if (first_input_column >= 0 && first_input_column + kernel_width <= input_width)
                {
                    copy_n(source, kernel_width * channels, patch_row);
                    continue;
                }

                for (Index kernel_column = 0; kernel_column < kernel_width; ++kernel_column)
                {
                    const Index input_column = first_input_column + kernel_column;
                    if (input_column < 0 || input_column >= input_width)
                        fill_n(patch_row + kernel_column * channels, channels, 0.0f);
                    else
                        copy_n(source + kernel_column * channels, channels,
                               patch_row + kernel_column * channels);
                }
            }
        }
}

void col2im(const float* col, Index input_height, Index input_width, Index channels,
            Index kernel_height, Index kernel_width,
            Index padding_height, Index padding_width,
            Index row_stride, Index column_stride,
            Index output_height, Index output_width,
            float* image)
{
    const Index patch_size = kernel_height * kernel_width * channels;

    for (Index output_row = 0; output_row < output_height; ++output_row)
        for (Index output_column = 0; output_column < output_width; ++output_column)
        {
            const float* patch = col + (output_row * output_width + output_column) * patch_size;
            const Index first_input_column = output_column * column_stride - padding_width;

            for (Index kernel_row = 0; kernel_row < kernel_height; ++kernel_row)
            {
                const Index input_row = output_row * row_stride + kernel_row - padding_height;
                if (input_row < 0 || input_row >= input_height) continue;

                const float* patch_row = patch + kernel_row * kernel_width * channels;
                float* destination = image + (input_row * input_width + first_input_column) * channels;

                if (first_input_column >= 0 && first_input_column + kernel_width <= input_width)
                {
                    Map<VectorR>(destination, kernel_width * channels) +=
                        Map<const VectorR>(patch_row, kernel_width * channels);
                    continue;
                }

                for (Index kernel_column = 0; kernel_column < kernel_width; ++kernel_column)
                {
                    const Index input_column = first_input_column + kernel_column;
                    if (input_column < 0 || input_column >= input_width) continue;

                    Map<VectorR>(destination + kernel_column * channels, channels) +=
                        Map<const VectorR>(patch_row + kernel_column * channels, channels);
                }
            }
        }
}

}

void ConvolutionOperator::apply_cpu(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.get_shape()[0];
    const Index output_height = output.get_shape()[1];
    const Index output_width = output.get_shape()[2];
    const Index output_positions = output_height * output_width;
    const Index patch_size = kernel_height * kernel_width * kernel_channels;
    const Index input_size = input_height * input_width * kernel_channels;

    const Map<const MatrixR> weights_matrix(weights.as<float>(), kernels_number, patch_size);
    const Map<const Matrix<float, 1, Dynamic>> bias_row(use_bias ? bias.as<float>() : nullptr,
                                                        use_bias ? kernels_number : 0);

    // Scratch for the whole team rather than `min(threads, batch)`: a smaller
    // team would make libgomp shrink its pool and rebuild it at the next full
    // region (see Backend::set_threads_number). Threads past the batch draw
    // no iterations.
    const Index col_size = output_positions * patch_size;
    const int workers = max(1, omp_get_max_threads());
    vector<float> scratch(size_t(workers) * size_t(col_size));

    #pragma omp parallel
    {
        float* const col_data =
            scratch.data() + size_t(omp_get_thread_num()) * size_t(col_size);

        #pragma omp for schedule(static)
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            im2col(input.as<float>() + image_index * input_size,
                   input_height, input_width, kernel_channels,
                   kernel_height, kernel_width, padding_height, padding_width,
                   row_stride, column_stride, output_height, output_width,
                   col_data);

            const Map<const MatrixR> col(col_data, output_positions, patch_size);
            Map<MatrixR> output_matrix(output.as<float>() + image_index * output_positions * kernels_number,
                                    output_positions, kernels_number);

            output_matrix.noalias() = col * weights_matrix.transpose();

            if (use_bias)
                output_matrix.rowwise() += bias_row;

            if (fuse_relu)
                output_matrix = output_matrix.cwiseMax(0.0f);
        }
    }
}

void ConvolutionOperator::apply_delta_cpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    const Index batch_size = output_delta.get_shape()[0];
    const Index output_height = output_delta.get_shape()[1];
    const Index output_width = output_delta.get_shape()[2];
    const Index output_positions = output_height * output_width;
    const Index patch_size = kernel_height * kernel_width * kernel_channels;
    const Index input_size = input_height * input_width * kernel_channels;

    const Map<const MatrixR> weights_matrix(weights.as<float>(), kernels_number, patch_size);

    const bool write_input_delta = !input_delta.empty();

    const int workers = max(1, omp_get_max_threads()); // whole team, as in the forward pass
    MatrixR weight_gradient_partials = MatrixR::Zero(workers, kernels_number * patch_size);
    MatrixR bias_gradient_partials = MatrixR::Zero(use_bias ? workers : 0,
                                                   use_bias ? kernels_number : 0);

    const Index col_size = output_positions * patch_size;
    const Index scratch_stride = write_input_delta ? 2 * col_size : col_size;
    vector<float> scratch(size_t(workers) * size_t(scratch_stride));

    #pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        float* const col_data =
            scratch.data() + size_t(thread) * size_t(scratch_stride);
        float* const delta_col_data = write_input_delta ? col_data + col_size : nullptr;

        Map<MatrixR> weight_gradient_partial(weight_gradient_partials.row(thread).data(),
                                          kernels_number, patch_size);

        #pragma omp for schedule(static)
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            im2col(input.as<float>() + image_index * input_size,
                   input_height, input_width, kernel_channels,
                   kernel_height, kernel_width, padding_height, padding_width,
                   row_stride, column_stride, output_height, output_width,
                   col_data);

            const Map<const MatrixR> col(col_data, output_positions, patch_size);
            const Map<const MatrixR> output_deltas(
                output_delta.as<float>() + image_index * output_positions * kernels_number,
                output_positions, kernels_number);

            weight_gradient_partial.noalias() += output_deltas.transpose() * col;

            if (use_bias)
                bias_gradient_partials.row(thread) += output_deltas.colwise().sum();

            if (write_input_delta)
            {
                Map<MatrixR> delta_col(delta_col_data, output_positions, patch_size);
                delta_col.noalias() = output_deltas * weights_matrix;

                float* const image_delta = input_delta.as<float>() + image_index * input_size;
                fill_n(image_delta, input_size, 0.0f);
                col2im(delta_col_data,
                       input_height, input_width, kernel_channels,
                       kernel_height, kernel_width, padding_height, padding_width,
                       row_stride, column_stride, output_height, output_width,
                       image_delta);
            }
        }
    }

    weight_gradient.as_vector() = weight_gradient_partials.colwise().sum().transpose();

    if (use_bias)
        bias_gradient.as_vector() = bias_gradient_partials.colwise().sum().transpose();
}

#ifdef OPENNN_HAS_CUDA

double ConvolutionOperator::minimum_traffic_bytes(const TensorView& input,
                                                  const TensorView& output) const
{
    if (!profiler::is_enabled()) return 0.0;

    return double(type_bytes(input.get_type())) * double(input.size())
         + double(type_bytes(weights.get_type())) * double(weights.size())
         + double(type_bytes(output.get_type())) * double(output.size());
}

void ConvolutionOperator::apply_gpu(const TensorView& input, TensorView& output) const
{
    PROFILE_SCOPE_BYTES("op:conv_fwd", minimum_traffic_bytes(input, output));

    throw_if(!input.is_fp32() && !input.is_bf16(),
             "ConvolutionOperator: GPU convolution requires FP32 or BF16 input.");

    void* weights_data = weights.get_data();
    if (weights.is_int8())
    {
        throw_if(weight_scale.empty() || !input.is_bf16(),
                 "ConvolutionOperator: INT8 kernels require BF16 activations and a per-kernel scale vector.");
        bfloat16* dequantized = ensure_int8_dequant_workspace(weights.size());
        w8_dequant_cuda<bfloat16>(kernels_number, weights.size() / kernels_number, true,
                                  weights.as<int8_t>(), weight_scale.as<float>(), dequantized);
        weights_data = dequantized;
    }

    const bool ran = cudnn_frontend::frontend_enabled()
        && cudnn_frontend::run_frontend(*conv_graph_cache, "ConvolutionOperator", [&](ConvGraphCache& cache)
    {
        auto& entry = detail::bounded_cache_entry(
            cache.entries, input.get_shape()[0],
            cudnn_frontend::graph_cache_capacity);
        if (!entry.fwd.graph)
            cudnn_frontend::build_forward(entry, cudnn_frontend::make_dims(*this, input.get_shape()[0]),
                                    fuse_relu, use_bias, input.get_type());

        cudnn_frontend::VariantPack tensors;
        tensors[entry.fwd_X] = input.get_data();
        tensors[entry.fwd_W] = weights_data;
        if (use_bias) tensors[entry.fwd_B] = bias.get_data();
        tensors[entry.fwd_Y] = output.get_data();

        cudnn_frontend::run_slot(entry.fwd, tensors, "ConvolutionOperator fwd",
                                 cudnn_frontend::conv_timing_label(*this, "conv_fwd"), true);
    });

    if (!ran) cudnn_frontend::throw_frontend_unavailable("ConvolutionOperator: GPU convolution");
}

void ConvolutionOperator::apply_gpu_folded(const TensorView& input,
                                           const TensorView& folded_weights,
                                           const TensorView& folded_bias,
                                           bool relu, TensorView& output,
                                           const TensorView* residual) const
{
    PROFILE_SCOPE_BYTES("op:conv_fwd", minimum_traffic_bytes(input, output));

    // Folding batch norm into the weights is only a win if the convolution it
    // feeds is still cuDNN's. Routing it through a GEMM instead measured 9.9 ms
    // a call against cuDNN's 1.6 ms: a 1x1 convolution becomes an extremely
    // skinny matrix product (M ~ 400k, K = N = 64) and cuBLASLt has nothing
    // good for that shape. The forward graph already ends in ADD and RELU
    // pointwises, so the folded weights and bias drop straight into it and the
    // batch norm pass disappears with no kernel downgrade.
    //
    // The folded graph is a second entry in the same cache, keyed by the
    // negated batch so it cannot collide with the unfolded one, and built with
    // bias on regardless of what the bare convolution was configured for.
    const bool ran = cudnn_frontend::frontend_enabled()
        && cudnn_frontend::run_frontend(*conv_graph_cache, "ConvolutionOperator", [&](ConvGraphCache& cache)
    {
        const Index batch = input.get_shape()[0];
        const bool with_residual = residual && residual->get_data();

        // Two folded variants share the cache: with and without the residual.
        // Keyed off the negated batch so neither collides with the unfolded
        // entry, and separated by a large stride so they do not collide with
        // each other.
        const Index key = with_residual ? -batch - (Index(1) << 20) : -batch;

        auto& entry = detail::bounded_cache_entry(
            cache.entries, key, cudnn_frontend::graph_cache_capacity);
        if (!entry.fwd.graph)
            cudnn_frontend::build_forward(entry, cudnn_frontend::make_dims(*this, batch),
                                          relu, true, input.get_type(), with_residual);

        cudnn_frontend::VariantPack tensors;
        tensors[entry.fwd_X] = input.get_data();
        tensors[entry.fwd_W] = folded_weights.get_data();
        tensors[entry.fwd_B] = folded_bias.get_data();
        if (with_residual) tensors[entry.fwd_R] = residual->get_data();
        tensors[entry.fwd_Y] = output.get_data();

        cudnn_frontend::run_slot(entry.fwd, tensors, "ConvolutionOperator fwd folded",
                                 cudnn_frontend::conv_timing_label(*this, "conv_fwd_folded"), true);
    });

    if (!ran)
        linear_forward(input, folded_weights, folded_bias, output,
                       relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS);
}

void ConvolutionOperator::apply_delta_gpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta,
                                  const TensorView& addend) const
{
    PROFILE_SCOPE("op:conv_bwd");

    assert(output_delta.get_type() == input.get_type());
    assert(weight_gradient.is_fp32());

    throw_if(!input.is_fp32() && !input.is_bf16(),
             "ConvolutionOperator: GPU convolution backward requires FP32 or BF16.");

    const bool ran = cudnn_frontend::frontend_enabled()
        && cudnn_frontend::run_frontend(*conv_graph_cache, "ConvolutionOperator", [&](ConvGraphCache& cache)
    {
        auto& entry = detail::bounded_cache_entry(
            cache.entries, input.get_shape()[0],
            cudnn_frontend::graph_cache_capacity);
        const auto dims = cudnn_frontend::make_dims(*this, input.get_shape()[0]);

        const bool fork_wgrad = device::lanes_available() > 1 && device::active_lane() == 0
            && input_delta.get_data() && input_delta.size() != 0;
        ScopeExit lane_restore([armed = fork_wgrad] { if (armed) device::set_active_lane(0); });
        if (fork_wgrad)
        {
            if (!entry.fork_event) entry.fork_event.create();
            if (!entry.join_event) entry.join_event.create();
            device::record_event(entry.fork_event.get(), device::lane_stream(0));
            device::set_active_lane(1);
            device::stream_wait_event(device::lane_stream(1), entry.fork_event.get());
        }

        if (!entry.wgrad.graph)
        {
            entry.wgrad_fp32_output = input.is_bf16()
                && cudnn_frontend::build_preferred(*this, "wgrad", input.get_shape()[0],
                       "BF16 store + widening cast per step",
                       [&] { cudnn_frontend::build_wgrad(entry, dims, input.get_type(), true); });
            if (!entry.wgrad_fp32_output)
                cudnn_frontend::build_wgrad(entry, dims, input.get_type(), false);
        }

        const bool wgrad_bf16 = input.is_bf16() && !entry.wgrad_fp32_output;
        bfloat16* dw_bf16 = wgrad_bf16 ? ensure_bf16_gradient_workspace(weight_gradient.size()) : nullptr;

        cudnn_frontend::VariantPack tensors;
        tensors[entry.wgrad_DY] = output_delta.get_data();
        tensors[entry.wgrad_X]  = input.get_data();
        tensors[entry.wgrad_DW] = wgrad_bf16 ? static_cast<void*>(dw_bf16) : weight_gradient.get_data();

        cudnn_frontend::run_slot(entry.wgrad, tensors, "ConvolutionOperator wgrad",
                                 cudnn_frontend::conv_timing_label(*this, "conv_wgrad"), false);

        if (wgrad_bf16)
            cast_bf16_to_fp32(weight_gradient.size(), dw_bf16, weight_gradient.as<float>());

        if (use_bias)
        {
            if (!entry.bgrad.graph) cudnn_frontend::build_bgrad(entry, dims, input.get_type());

            cudnn_frontend::VariantPack bgrad_tensors;
            bgrad_tensors[entry.bgrad_DY] = output_delta.get_data();
            bgrad_tensors[entry.bgrad_DB] = bias_gradient.get_data();

            cudnn_frontend::run_slot(entry.bgrad, bgrad_tensors, "ConvolutionOperator bgrad",
                                     cudnn_frontend::conv_timing_label(*this, "conv_bgrad"), false);
        }

        if (fork_wgrad)
        {
            device::record_event(entry.join_event.get(), device::lane_stream(1));
            device::set_active_lane(0);
        }

        if (input_delta.get_data() && input_delta.size() != 0)
        {
            const bool want_add = addend.get_data() && addend.size() == input_delta.size();

            if (!entry.dgrad.graph)
            {
                entry.dgrad_adds = want_add
                    && cudnn_frontend::build_preferred(*this, "dgrad", input.get_shape()[0],
                           "residual delta added separately",
                           [&] { cudnn_frontend::build_dgrad(entry, dims, input.get_type(), true); });
                if (!entry.dgrad_adds)
                    cudnn_frontend::build_dgrad(entry, dims, input.get_type(), false);
            }

            cudnn_frontend::VariantPack dgrad_tensors;
            dgrad_tensors[entry.dgrad_DY] = output_delta.get_data();
            dgrad_tensors[entry.dgrad_W]  = weights.get_data();
            dgrad_tensors[entry.dgrad_DX] = input_delta.get_data();
            if (entry.dgrad_adds) dgrad_tensors[entry.dgrad_R] = addend.get_data();

            cudnn_frontend::run_slot(entry.dgrad, dgrad_tensors, "ConvolutionOperator dgrad",
                                     cudnn_frontend::conv_timing_label(*this, "conv_dgrad"), false);

            if (want_add && !entry.dgrad_adds)
                add(input_delta, addend, input_delta);
        }

        if (fork_wgrad)
            device::stream_wait_event(device::lane_stream(0), entry.join_event.get());
    });

    if (!ran) cudnn_frontend::throw_frontend_unavailable("ConvolutionOperator: GPU convolution backward");
}

#else

void ConvolutionOperator::apply_gpu(const TensorView&, TensorView&) const                          OPENNN_CUDA_STUB_BODY(apply_gpu)
void ConvolutionOperator::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&, const TensorView&) const OPENNN_CUDA_STUB_BODY(apply_delta_gpu)

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
