//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/registry.h"
#include "opennn/core/string_utilities.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_normalization.cuh"
#endif

namespace opennn
{

namespace
{

void validate_convolution_configuration(const Shape& input_shape,
                                        const Shape& kernel_shape,
                                        const Shape& stride_shape,
                                        const string& convolution_type,
                                        bool batch_normalization,
                                        bool residual,
                                        const string& label)
{
    throw_if(input_shape.get_rank() != 3,
             "Convolutional layer '{}': input shape must have 3 dimensions, read {}.",
             label, input_shape.get_rank());
    throw_if(kernel_shape.get_rank() != 4,
             "Convolutional layer '{}': kernel shape must have 4 dimensions, read {}.",
             label, kernel_shape.get_rank());
    throw_if(stride_shape.get_rank() != 2,
             "Convolutional layer '{}': stride must have 2 dimensions, read {}.",
             label, stride_shape.get_rank());

    throw_if(kernel_shape[0] <= 0 || kernel_shape[1] <= 0
             || kernel_shape[2] <= 0 || kernel_shape[3] <= 0,
             "Convolutional layer '{}': every kernel dimension must be positive, read {}.",
             label, shape_to_string(kernel_shape));
    throw_if(stride_shape[0] <= 0 || stride_shape[1] <= 0,
             "Convolutional layer '{}': stride must be positive, read {}.",
             label, shape_to_string(stride_shape));

    throw_if(kernel_shape[2] != input_shape[2],
             "Convolutional layer '{}': kernel channels ({}) must match input channels ({}).",
             label, kernel_shape[2], input_shape[2]);
    throw_if(stride_shape[0] > input_shape[0] || stride_shape[1] > input_shape[1],
             "Convolutional layer '{}': stride {} cannot be bigger than input shape {}.",
             label, shape_to_string(stride_shape), shape_to_string(input_shape));

    throw_if(!contains({"Valid", "Same"}, convolution_type),
             "Convolutional layer '{}': convolution type must be 'Valid' or 'Same', read '{}'.",
             label, convolution_type);
    throw_if(convolution_type == "Valid"
             && (kernel_shape[0] > input_shape[0] || kernel_shape[1] > input_shape[1]),
             "Convolutional layer '{}': kernel shape {} cannot be bigger than input shape {} "
             "with Valid convolution.",
             label, shape_to_string(kernel_shape), shape_to_string(input_shape));
    throw_if(convolution_type == "Same"
             && (kernel_shape[0] % 2 == 0 || kernel_shape[1] % 2 == 0),
             "Convolutional layer '{}': kernel height and width must be odd with Same convolution, "
             "read {}.",
             label, shape_to_string(kernel_shape));
    throw_if(residual && !batch_normalization,
             "Convolutional layer '{}': a residual input requires batch normalization.",
             label);
}

}

Convolutional::Convolutional(const Shape& new_input_shape,
                             const Shape& new_kernel_shape,
                             const string& new_activation_function,
                             const Shape& new_stride_shape,
                             const string& new_convolution_type,
                             BatchNormalization new_batch_normalization,
                             const string& new_label)
    : Layer(LayerType::Convolutional)
{
    operators = {&convolution, &batch_norm, &activation_operator};

    set(new_input_shape,
        new_kernel_shape,
        new_activation_function,
        new_stride_shape,
        new_convolution_type,
        new_batch_normalization,
        new_label);
}

Shape Convolutional::get_output_shape() const
{
    return { get_output_height(), get_output_width(), convolution.kernels_number };
}

Index Convolutional::get_output_height() const
{
    return use_padding
        ? ceil_div(convolution.input_height, convolution.row_stride)
        : (convolution.input_height - convolution.kernel_height) / convolution.row_stride + 1;
}

Index Convolutional::get_output_width() const
{
    return use_padding
        ? ceil_div(convolution.input_width, convolution.column_stride)
        : (convolution.input_width - convolution.kernel_width) / convolution.column_stride + 1;
}

Index Convolutional::get_padding_height() const
{
    if (!use_padding) return 0;

    const Index total_padding =
        max(Index(0), (get_output_height() - 1) * convolution.row_stride + convolution.kernel_height - convolution.input_height);

    return (total_padding + 1) / 2;
}

Index Convolutional::get_padding_width() const
{
    if (!use_padding) return 0;

    const Index total_padding =
        max(Index(0), (get_output_width() - 1) * convolution.column_stride + convolution.kernel_width - convolution.input_width);

    return (total_padding + 1) / 2;
}

vector<TensorSpec> Convolutional::get_forward_specs(Index batch_size) const
{
    const Shape output_shape = {batch_size, get_output_height(), get_output_width(), convolution.kernels_number};
    const Type act = compute_dtype;

    const Shape convolution_view_shape = batch_norm.active() ? output_shape          : Shape{};
    const Shape bn_stat_shape          = batch_norm.active() ? Shape{convolution.kernels_number} : Shape{};

    const bool relu_mask = batch_norm.active() && batch_norm.fuse_relu && convolution.kernels_number % 8 == 0;
    const Shape relu_mask_shape = relu_mask
        ? Shape{batch_size, get_output_height(), get_output_width(), convolution.kernels_number / 8}
        : Shape{};

    return {
                                     {convolution_view_shape, act},
                                     {bn_stat_shape,          Type::FP32},
                                     {bn_stat_shape,          Type::FP32},
                                     {relu_mask_shape,        Type::INT8},
                                     {output_shape,           act},
    };
}

vector<TensorSpec> Convolutional::get_backward_specs(Index batch_size) const
{
    vector<TensorSpec> specs = {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};

    if (residual)
        specs.push_back({Shape{batch_size}.append(get_output_shape()), compute_dtype});

    return specs;
}

void Convolutional::set_residual(bool new_residual)
{
    throw_if(new_residual && !batch_norm.active(),
             "Convolutional: a residual input requires batch normalization.");

    residual = new_residual;

    update_convolution_operator();
}

void Convolutional::update_convolution_operator()
{
    convolution.use_bias = !batch_norm.active();

    convolution.padding_height = get_padding_height();
    convolution.padding_width  = get_padding_width();
    convolution.compute_dtype  = compute_dtype;

    convolution.output_slots = batch_norm.active()
        ? vector<size_t>{ConvolutionView}
        : vector<size_t>{Output};

    if (batch_norm.active())
    {
        batch_norm.input_slots  = {ConvolutionView};
        batch_norm.output_slots = {Output, BatchNormMean, BatchNormInverseVariance, ReluMask};
    }

    activation_operator.input_slots  = {Output};
    activation_operator.output_slots = {Output};

    const bool relu = (activation_operator.activation_function == ActivationFunction::ReLU);
    const bool fuse_bn_relu = relu && batch_norm.active();
    const bool fuse_bn_add  = residual && batch_norm.active();

    convolution.fuse_relu = relu && !batch_norm.active();
    batch_norm.fuse_relu          = fuse_bn_relu;
    batch_norm.fuse_add           = fuse_bn_add;
    batch_norm.residual_delta_slot = fuse_bn_add ? 2 : 0;
    activation_operator.forward_fused      = relu;
    activation_operator.backward_fused     = fuse_bn_relu;
}

void Convolutional::set(const Shape& new_input_shape,
                        const Shape& new_kernel_shape,
                        const string& new_activation_function,
                        const Shape& new_stride_shape,
                        const string& new_convolution_type,
                        BatchNormalization new_batch_normalization,
                        const string& new_label)
{
    const bool use_batch_normalization = new_batch_normalization == BatchNormalization::Yes;

    validate_convolution_configuration(new_input_shape,
                                       new_kernel_shape,
                                       new_stride_shape,
                                       new_convolution_type,
                                       use_batch_normalization,
                                       residual,
                                       new_label);

    convolution.input_height    = new_input_shape[0];
    convolution.input_width     = new_input_shape[1];
    input_channels  = new_input_shape[2];

    convolution.kernel_height   = new_kernel_shape[0];
    convolution.kernel_width    = new_kernel_shape[1];
    convolution.kernel_channels = new_kernel_shape[2];
    convolution.kernels_number  = new_kernel_shape[3];

    convolution.row_stride      = new_stride_shape[0];
    convolution.column_stride   = new_stride_shape[1];

    use_padding     = (new_convolution_type == "Same");

    set_label(new_label);

    set_activation_function(new_activation_function);

    batch_norm.set_enabled(use_batch_normalization, convolution.kernels_number);

    update_convolution_operator();
}

void Convolutional::apply_input_shape(const Shape& new_input_shape)
{
    throw_if(new_input_shape.get_rank() != 3, "Input shape rank must be 3.");

    convolution.input_height = new_input_shape[0];
    convolution.input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    update_convolution_operator();
}

void Convolutional::set_activation_function(const string& new_activation_function)
{
    const ActivationFunction function = ActivationOperator::from_string(new_activation_function);

    throw_if(function == ActivationFunction::Softmax,
             "Softmax is not a valid activation for a convolutional layer.");
    throw_if(activation_needs_input(function),
             "Convolutional: input-derivative activations (e.g. GELU, SiLU) are not supported; "
             "use a standalone Activation layer after the convolution.");

    activation_operator.set_activation_function(function);
    update_convolution_operator();
}

void Convolutional::set_batch_normalization(bool new_batch_normalization)
{
    batch_norm.set_enabled(new_batch_normalization, convolution.kernels_number);
    update_convolution_operator();
}

void Convolutional::read_JSON_body(const Json* convolutional_layer_element)
{
    const Index new_kernel_height   = read_json_index(convolutional_layer_element, "KernelsHeight");
    const Index new_kernel_width    = read_json_index(convolutional_layer_element, "KernelsWidth");
    const Index new_kernel_channels = read_json_index(convolutional_layer_element, "KernelsChannels");
    const Index new_kernels_number  = read_json_index(convolutional_layer_element, "KernelsNumber");

    const Shape stride_shape = string_to_shape(read_json_string(convolutional_layer_element, "StrideDimensions"));
    const string convolution_type = read_json_string(convolutional_layer_element, "Convolution");
    const bool new_batch_normalization = read_json_bool(convolutional_layer_element, "BatchNormalization");
    const bool new_residual = convolutional_layer_element->has("Residual")
                           && read_json_bool(convolutional_layer_element, "Residual");

    const Shape kernel_shape{
        new_kernel_height,
        new_kernel_width,
        new_kernel_channels,
        new_kernels_number
    };

    validate_convolution_configuration(get_input_shape(),
                                       kernel_shape,
                                       stride_shape,
                                       convolution_type,
                                       new_batch_normalization,
                                       new_residual,
                                       label);

    convolution.kernel_height   = new_kernel_height;
    convolution.kernel_width    = new_kernel_width;
    convolution.kernel_channels = new_kernel_channels;
    convolution.kernels_number  = new_kernels_number;
    convolution.row_stride      = stride_shape[0];
    convolution.column_stride   = stride_shape[1];
    use_padding     = (convolution_type == "Same");
    batch_norm.set_enabled(new_batch_normalization, convolution.kernels_number);
    residual = new_residual;
}

void Convolutional::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"KernelsNumber", get_kernels_number()},
        {"KernelsHeight", get_kernel_height()},
        {"KernelsWidth", get_kernel_width()},
        {"KernelsChannels", get_kernel_channels()},
        {"StrideDimensions", shape_to_string({get_row_stride(), get_column_stride()})},
        {"Convolution", use_padding ? "Same" : "Valid"},
        {"BatchNormalization", batch_norm.active()},
        {"Residual", residual}
    });
}

void Convolutional::load_darknet_weights(FILE* f)
{
    throw_if(!f, "load_darknet_weights: file handle is null.");

    const Index O  = convolution.kernels_number;
    const Index kH = convolution.kernel_height;
    const Index kW = convolution.kernel_width;
    const Index I  = convolution.kernel_channels;
    const Index total_weights = O * kH * kW * I;

    if (batch_norm.active())
    {
        const size_t n = static_cast<size_t>(batch_norm.features);
        const auto read_bn = [&](TensorView& tv)
        {
            throw_if(fread(tv.as<float>(), sizeof(float), n, f) != n,
                     "load_darknet_weights: short read on BN parameters.");
        };
        read_bn(batch_norm.beta);
        read_bn(batch_norm.gamma);
        read_bn(batch_norm.running_mean);
        read_bn(batch_norm.running_variance);
        batch_norm.running_mean.as_vector().setZero();
        batch_norm.running_variance.as_vector().setOnes();
        batch_norm.invalidate_inference_cache();
    }
    else
    {
        const size_t n_out = static_cast<size_t>(O);
        throw_if(fread(convolution.bias.as<float>(), sizeof(float), n_out, f) != n_out,
                 "load_darknet_weights: short read on bias.");
    }

    const size_t n_weights = static_cast<size_t>(total_weights);
    vector<float> tmp(n_weights, 0.0f);
    throw_if(fread(tmp.data(), sizeof(float), n_weights, f) != n_weights,
             "load_darknet_weights: short read on conv weights.");

    float* const dst = convolution.weights.as<float>();
    for (Index o = 0; o < O; ++o)
        for (Index h = 0; h < kH; ++h)
            for (Index w = 0; w < kW; ++w)
                for (Index ic = 0; ic < I; ++ic)
                    dst[o*kH*kW*I + h*kW*I + w*I + ic] =
                        tmp[static_cast<size_t>(o*I*kH*kW + ic*kH*kW + h*kW + w)];

    if (batch_norm.active())
        batch_norm.invalidate_inference_cache();

#ifdef OPENNN_HAS_CUDA
    folded_dirty = true;
#endif
}

void Convolutional::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode pass)
{
#ifdef OPENNN_HAS_CUDA
    if (is_training(pass))
        folded_dirty = true;
    else if (forward_propagate_folded(forward_propagation, layer))
        return;
#endif

    Layer::forward_propagate(forward_propagation, layer, pass);
}

void Convolutional::recompute_forward_slot(ForwardPropagation& forward_propagation,
                                           size_t layer)
{
    throw_if(!batch_norm.active(),
             "Convolutional::recompute_forward_slot requires batch normalization.");
    convolution.forward_propagate(forward_propagation, layer, ForwardPropagationMode::Training);
}

#ifdef OPENNN_HAS_CUDA

bool Convolutional::forward_propagate_folded(ForwardPropagation& forward_propagation, size_t layer)
{
    // No longer restricted to 1x1: the fold rescales weights, and the cuDNN
    // forward graph it now feeds convolves any shape. The restriction existed
    // because the old folded path was a GEMM, which only expresses a pointwise
    // convolution.
    if (!batch_norm.active())
        return false;

    const TensorView& input = convolution.get_input(forward_propagation, layer);

    if (!input.is_cuda() || !(input.is_fp32() || input.is_bf16())
        || !(convolution.weights.is_fp32() || convolution.weights.is_bf16()))
        return false;

    if (convolution.weights_relinked)
    {
        convolution.weights_relinked = false;
        folded_dirty = true;
    }

    const Index weight_count = convolution.weights.size();
    const Index kernel_size  = convolution.kernel_height * convolution.kernel_width * convolution.kernel_channels;

    // The folded parameters carry the network's own precision. Emitting fp32
    // into a bf16 network sent linear_forward down data_for_gemm_dtype(input,
    // weights.get_type()), which upcasts the activations to match the weights.
    const Type folded_type = convolution.weights.get_type();
    const Index folded_element = convolution.weights.is_bf16()
        ? Index(sizeof(bfloat16)) : Index(sizeof(float));

    if (folded_dirty)
    {
        folded_parameters.resize_bytes((weight_count + convolution.kernels_number) * folded_element,
                                       Device::CUDA);
        convolution.weights.dispatch([&]<typename W>()
        {
            conv_bn_fold_cuda(convolution.kernels_number, kernel_size,
                              convolution.weights.as<W>(),
                              batch_norm.gamma.as<float>(), batch_norm.beta.as<float>(),
                              batch_norm.running_mean.as<float>(),
                              batch_norm.running_variance.as<float>(),
                              BN_EPSILON,
                              folded_parameters.as<W>(),
                              folded_parameters.as<W>() + weight_count);
        });
        folded_dirty = false;
    }

    const TensorView folded_weights(folded_parameters.data(),
        convolution.weights.get_shape(), folded_type, Device::CUDA);

    const TensorView folded_bias(
        static_cast<char*>(folded_parameters.data()) + weight_count * folded_element,
        Shape{convolution.kernels_number}, folded_type, Device::CUDA);

    const bool relu = batch_norm.fuse_relu;
    TensorView& output = batch_norm.get_output(forward_propagation, layer);

    if (batch_norm.fuse_add)
    {
        // One graph: convolution, folded bias, residual, activation. The add
        // used to be its own pass over the block output and became the largest
        // kernel in the network once the fold removed batch norm.
        const TensorView& residual_view = forward_propagation.inputs[layer][1];
        convolution.apply_gpu_folded(input, folded_weights, folded_bias, relu, output,
                                     &residual_view);
    }
    else
        convolution.apply_gpu_folded(input, folded_weights, folded_bias, relu, output);

    activation_operator.forward_propagate(forward_propagation, layer, ForwardPropagationMode::Inference);

    return true;
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
