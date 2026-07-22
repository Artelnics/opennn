//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef OPENNN_HAS_CUDA
#include <cudnn_frontend.h>
#endif

#include "convolution_operator.h"
#include "device_backend.h"
#ifdef OPENNN_HAS_CUDA
#include "kernel.cuh"
#endif
#include "random_utilities.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"
#include "cudnn_frontend_utilities.h"

namespace opennn
{

ConvolutionOperator::ConvolutionOperator() = default;
ConvolutionOperator::~ConvolutionOperator() = default;

#ifdef OPENNN_HAS_CUDA

struct ConvolutionOperator::ConvGraphCache
{
    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> fwd, wgrad, bgrad, dgrad;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_W, fwd_B, fwd_Y;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> wgrad_X, wgrad_DY, wgrad_DW;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bgrad_DY, bgrad_DB;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> dgrad_W, dgrad_DY, dgrad_DX;
        int64_t fwd_workspace_bytes = 0;
        int64_t wgrad_workspace_bytes = 0;
        int64_t bgrad_workspace_bytes = 0;
        int64_t dgrad_workspace_bytes = 0;
        bool fwd_autotune = false;
        bool wgrad_autotune = false;
        bool bgrad_autotune = false;
        bool dgrad_autotune = false;
    };

    unordered_map<Index, Entry> entries;
    bool disabled = false;
};

namespace cudnn_frontend
{
using namespace ::cudnn_frontend;

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
                   bool fuse_relu, bool use_bias, Type dtype)
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

    if (fuse_relu)
        entry.fwd_Y = graph->pointwise(entry.fwd_Y,
                                       graph::Pointwise_attributes()
                                       .set_mode(PointwiseMode_t::RELU_FWD));

    set_nhwc_output(entry.fwd_Y, d.batch, d.kernels, d.output_height, d.output_width);

    entry.fwd_autotune = finalize(*graph, entry.fwd_workspace_bytes, "forward", autotune_enabled());
    entry.fwd = graph;
}

void build_wgrad(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d, Type dtype)
{
    auto graph = new_graph(dtype);

    entry.wgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);
    entry.wgrad_X  = nhwc_tensor(*graph, "X", d.batch, d.channels, d.height, d.width);

    entry.wgrad_DW = graph->conv_wgrad(entry.wgrad_DY, entry.wgrad_X,
                                       conv_attributes<graph::Conv_wgrad_attributes>(d));
    entry.wgrad_DW->set_output(true)
                   .set_dim({d.kernels, d.channels, d.kernel_height, d.kernel_width})
                   .set_stride(krsc_strides(d));

    entry.wgrad_autotune = finalize(*graph, entry.wgrad_workspace_bytes, "wgrad", autotune_enabled());
    entry.wgrad = graph;
}

void build_bgrad(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d, Type dtype)
{
    auto graph = new_graph(dtype);

    entry.bgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);

    entry.bgrad_DB = graph->reduction(entry.bgrad_DY,
                                      graph::Reduction_attributes()
                                      .set_mode(ReductionMode_t::ADD));
    // The bias gradient is the FP32 master gradient regardless of io dtype.
    entry.bgrad_DB->set_output(true)
                   .set_data_type(DataType_t::FLOAT)
                   .set_dim({1, d.kernels, 1, 1})
                   .set_stride({d.kernels, 1, d.kernels, d.kernels});

    entry.bgrad_autotune = finalize(*graph, entry.bgrad_workspace_bytes, "bgrad", autotune_enabled());
    entry.bgrad = graph;
}

void build_dgrad(ConvolutionOperator::ConvGraphCache::Entry& entry, const Dims& d, Type dtype)
{
    auto graph = new_graph(dtype);

    entry.dgrad_DY = nhwc_tensor(*graph, "DY", d.batch, d.kernels, d.output_height, d.output_width);
    entry.dgrad_W  = krsc_tensor(*graph, "W", d);

    entry.dgrad_DX = graph->conv_dgrad(entry.dgrad_DY, entry.dgrad_W,
                                       conv_attributes<graph::Conv_dgrad_attributes>(d));
    set_nhwc_output(entry.dgrad_DX, d.batch, d.channels, d.height, d.width);

    entry.dgrad_autotune = finalize(*graph, entry.dgrad_workspace_bytes, "dgrad", autotune_enabled());
    entry.dgrad = graph;
}

string timing_label(const ConvolutionOperator& op, const char* kind)
{
    if (!graph_timing_enabled()) return {};
    return format("{} {}x{}x{} k{}x{}x{} s{}", kind,
                  op.input_height, op.input_width, op.kernel_channels,
                  op.kernel_height, op.kernel_width, op.kernels_number, op.row_stride);
}

}

#endif


void ConvolutionOperator::set(Index new_input_h, Index new_input_w,
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
    compute_dtype    = new_compute_dtype;

}

vector<TensorSpec> ConvolutionOperator::parameter_specs() const
{
    // The bias is redundant under batch normalization (its beta absorbs it).
    if (!use_bias)
        return {{{kernels_number, kernel_height, kernel_width, kernel_channels}, compute_dtype}};

    return {
        {{kernels_number}, compute_dtype},
        {{kernels_number, kernel_height, kernel_width, kernel_channels}, compute_dtype},
    };
}

void ConvolutionOperator::link_parameters(span<const TensorView> views)
{
    if (views.empty()) return;
    bias    = use_bias ? views[0] : TensorView{};
    weights = views[use_bias ? 1 : 0];
    weights_relinked = true;
}

void ConvolutionOperator::link_gradients(span<const TensorView> views)
{
    if (views.empty()) return;
    bias_gradient   = use_bias ? views[0] : TensorView{};
    weight_gradient = views[use_bias ? 1 : 0];
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


void ConvolutionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool /*is_training*/)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (input.is_cuda()) apply_gpu(input, output);
    else                  apply_cpu(input, output);
}

void ConvolutionOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& backward_slots = back_propagation.backward_slots[layer];

    const TensorView& input        = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    TensorView& input_delta = slot_or(backward_slots, input_delta_slots, 0);

    if (output_delta.is_cuda()) apply_delta_gpu(input, output_delta, input_delta);
    else                         apply_delta_cpu(input, output_delta, input_delta);
}

namespace
{

// One row per output position, one column block per (kernel_row, kernel_column,
// channel) matching the row-major kernel layout, so the whole convolution is a
// single GEMM against the (kernels x kernel_height*kernel_width*channels)
// weight matrix. Out-of-range taps (padding) become zero columns.
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

// Scatter-add inverse of im2col: accumulates each patch column back onto the
// image position it was read from (padding taps are dropped).
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

void ConvolutionOperator::apply_cpu(const TensorView& input, TensorView& output)
{
    const Index batch_size = input.shape[0];
    const Index output_height = output.shape[1];
    const Index output_width = output.shape[2];
    const Index output_positions = output_height * output_width;
    const Index patch_size = kernel_height * kernel_width * kernel_channels;
    const Index input_size = input_height * input_width * kernel_channels;

    const Map<const MatrixR> weights_matrix(weights.as<float>(), kernels_number, patch_size);
    const Map<const Matrix<float, 1, Dynamic>> bias_row(use_bias ? bias.as<float>() : nullptr,
                                                        use_bias ? kernels_number : 0);

    #pragma omp parallel
    {
        thread_local vector<float> col_storage;
        col_storage.resize(size_t(output_positions * patch_size));

        #pragma omp for schedule(static)
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            im2col(input.as<float>() + image_index * input_size,
                   input_height, input_width, kernel_channels,
                   kernel_height, kernel_width, padding_height, padding_width,
                   row_stride, column_stride, output_height, output_width,
                   col_storage.data());

            const Map<const MatrixR> col(col_storage.data(), output_positions, patch_size);
            Map<MatrixR> output_matrix(output.as<float>() + image_index * output_positions * kernels_number,
                                    output_positions, kernels_number);

            output_matrix.noalias() = col * weights_matrix.transpose();

            if (use_bias)
                output_matrix.rowwise() += bias_row;
        }
    }
}

void ConvolutionOperator::apply_delta_cpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    const Index batch_size = output_delta.shape[0];
    const Index output_height = output_delta.shape[1];
    const Index output_width = output_delta.shape[2];
    const Index output_positions = output_height * output_width;
    const Index patch_size = kernel_height * kernel_width * kernel_channels;
    const Index input_size = input_height * input_width * kernel_channels;

    const Map<const MatrixR> weights_matrix(weights.as<float>(), kernels_number, patch_size);

    const bool write_input_delta = !input_delta.empty();

    // Per-thread partials summed in thread order afterwards, so the result does
    // not depend on OpenMP scheduling.
    const int threads_number = omp_get_max_threads();
    MatrixR weight_gradient_partials = MatrixR::Zero(threads_number, kernels_number * patch_size);
    MatrixR bias_gradient_partials = MatrixR::Zero(use_bias ? threads_number : 0,
                                                   use_bias ? kernels_number : 0);

    #pragma omp parallel
    {
        const int thread = omp_get_thread_num();

        thread_local vector<float> col_storage;
        thread_local vector<float> delta_col_storage;
        col_storage.resize(size_t(output_positions * patch_size));
        if (write_input_delta)
            delta_col_storage.resize(size_t(output_positions * patch_size));

        Map<MatrixR> weight_gradient_partial(weight_gradient_partials.row(thread).data(),
                                          kernels_number, patch_size);

        #pragma omp for schedule(static)
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            im2col(input.as<float>() + image_index * input_size,
                   input_height, input_width, kernel_channels,
                   kernel_height, kernel_width, padding_height, padding_width,
                   row_stride, column_stride, output_height, output_width,
                   col_storage.data());

            const Map<const MatrixR> col(col_storage.data(), output_positions, patch_size);
            const Map<const MatrixR> output_deltas(
                output_delta.as<float>() + image_index * output_positions * kernels_number,
                output_positions, kernels_number);

            weight_gradient_partial.noalias() += output_deltas.transpose() * col;

            if (use_bias)
                bias_gradient_partials.row(thread) += output_deltas.colwise().sum();

            if (write_input_delta)
            {
                Map<MatrixR> delta_col(delta_col_storage.data(), output_positions, patch_size);
                delta_col.noalias() = output_deltas * weights_matrix;

                float* const image_delta = input_delta.as<float>() + image_index * input_size;
                fill_n(image_delta, input_size, 0.0f);
                col2im(delta_col_storage.data(),
                       input_height, input_width, kernel_channels,
                       kernel_height, kernel_width, padding_height, padding_width,
                       row_stride, column_stride, output_height, output_width,
                       image_delta);
            }
        }
    }

    Map<VectorR>(weight_gradient.as<float>(), kernels_number * patch_size) =
        weight_gradient_partials.colwise().sum().transpose();

    if (use_bias)
        bias_gradient.as_vector() = bias_gradient_partials.colwise().sum().transpose();
}

#ifdef OPENNN_HAS_CUDA

void ConvolutionOperator::apply_gpu(const TensorView& input, TensorView& output)
{
    PROFILE_SCOPE("op:conv_fwd");

    throw_if(!input.is_fp32() && !input.is_bf16(),
             "ConvolutionOperator: GPU convolution requires FP32 or BF16 input.");

    const bool ran = cudnn_frontend::frontend_enabled()
        && cudnn_frontend::run_frontend(conv_graph_cache, "ConvolutionOperator", [&](ConvGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.fwd)
            cudnn_frontend::build_forward(entry, cudnn_frontend::make_dims(*this, input.shape[0]),
                                    fuse_relu, use_bias, input.type);

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.fwd_X] = input.data;
        tensors[entry.fwd_W] = weights.data;
        if (use_bias) tensors[entry.fwd_B] = bias.data;
        tensors[entry.fwd_Y] = output.data;

        cudnn_frontend::autotune_now(entry.fwd_autotune, *entry.fwd, tensors, entry.fwd_workspace_bytes);

        cudnn_frontend::execute_graph(*entry.fwd, tensors, cudnn_frontend::shared_workspace(entry.fwd_workspace_bytes),
                                "forward execute", cudnn_frontend::timing_label(*this, "conv_fwd"));
    });

    throw_if(!ran, "ConvolutionOperator: GPU convolution requires SM 8.0+ (Ampere).");
}


void ConvolutionOperator::apply_gpu_folded(const TensorView& input,
                                           const TensorView& folded_weights,
                                           const TensorView& folded_bias,
                                           bool relu, TensorView& output)
{
    PROFILE_SCOPE("op:conv_fwd");

    // A stride-1 unpadded 1x1 convolution is a GEMM over the channel
    // dimension; folded_weights comes transposed to [channels, kernels],
    // the layout linear_forward expects, and cuBLASLt fuses the folded
    // bias (and ReLU) into the epilogue.
    linear_forward(input, folded_weights, folded_bias, output,
                   relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS);
}

void ConvolutionOperator::apply_delta_gpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    PROFILE_SCOPE("op:conv_bwd");

    assert(output_delta.type == input.type);
    assert(weight_gradient.is_fp32());

    throw_if(!input.is_fp32() && !input.is_bf16(),
             "ConvolutionOperator: GPU convolution backward requires FP32 or BF16.");

    const bool ran = cudnn_frontend::frontend_enabled()
        && cudnn_frontend::run_frontend(conv_graph_cache, "ConvolutionOperator", [&](ConvGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        const auto dims = cudnn_frontend::make_dims(*this, input.shape[0]);

        if (!entry.wgrad) cudnn_frontend::build_wgrad(entry, dims, input.type);

        // cuDNN emits DW in the io dtype; in BF16 accumulate it into the FP32
        // master gradient through a scratch buffer + cast (as the dense path does).
        const bool wgrad_bf16 = input.is_bf16();
        bfloat16* dw_bf16 = wgrad_bf16 ? ensure_bf16_gradient_workspace(weight_gradient.size()) : nullptr;

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.wgrad_DY] = output_delta.data;
        tensors[entry.wgrad_X]  = input.data;
        tensors[entry.wgrad_DW] = wgrad_bf16 ? static_cast<void*>(dw_bf16) : weight_gradient.data;

        cudnn_frontend::autotune_now(entry.wgrad_autotune, *entry.wgrad, tensors, entry.wgrad_workspace_bytes);

        cudnn_frontend::execute_graph(*entry.wgrad, tensors, cudnn_frontend::shared_workspace(entry.wgrad_workspace_bytes),
                                "wgrad execute", cudnn_frontend::timing_label(*this, "conv_wgrad"));

        if (wgrad_bf16)
            cast_bf16_to_fp32(weight_gradient.size(), dw_bf16, weight_gradient.as<float>());

        if (use_bias)
        {
            if (!entry.bgrad) cudnn_frontend::build_bgrad(entry, dims, input.type);

            unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> bgrad_tensors;
            bgrad_tensors[entry.bgrad_DY] = output_delta.data;
            bgrad_tensors[entry.bgrad_DB] = bias_gradient.data;

            cudnn_frontend::autotune_now(entry.bgrad_autotune, *entry.bgrad, bgrad_tensors, entry.bgrad_workspace_bytes);

            cudnn_frontend::execute_graph(*entry.bgrad, bgrad_tensors, cudnn_frontend::shared_workspace(entry.bgrad_workspace_bytes),
                                    "bgrad execute", cudnn_frontend::timing_label(*this, "conv_bgrad"));
        }

        if (input_delta.data && input_delta.size() != 0)
        {
            if (!entry.dgrad) cudnn_frontend::build_dgrad(entry, dims, input.type);

            unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> dgrad_tensors;
            dgrad_tensors[entry.dgrad_DY] = output_delta.data;
            dgrad_tensors[entry.dgrad_W]  = weights.data;
            dgrad_tensors[entry.dgrad_DX] = input_delta.data;

            cudnn_frontend::autotune_now(entry.dgrad_autotune, *entry.dgrad, dgrad_tensors, entry.dgrad_workspace_bytes);

            cudnn_frontend::execute_graph(*entry.dgrad, dgrad_tensors, cudnn_frontend::shared_workspace(entry.dgrad_workspace_bytes),
                                    "dgrad execute", cudnn_frontend::timing_label(*this, "conv_dgrad"));
        }
    });

    throw_if(!ran, "ConvolutionOperator: GPU convolution backward requires SM 8.0+ (Ampere).");
}

#else

void ConvolutionOperator::apply_gpu(const TensorView&, TensorView&)                                { throw runtime_error("apply_gpu requires CUDA."); }
void ConvolutionOperator::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("apply_delta_gpu requires CUDA."); }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
