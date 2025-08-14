//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "pooling_layer.h"

namespace opennn
{

Pooling::Pooling(const dimensions& new_input_dimensions,
                 const dimensions& new_pool_dimensions,
                 const dimensions& new_stride_dimensions,
                 const dimensions& new_padding_dimensions,
                 const PoolingMethod& new_pooling_method,
                 const string new_name) : Layer()
{
    name = "Pooling";

    set(new_input_dimensions,
        new_pool_dimensions,
        new_stride_dimensions,
        new_padding_dimensions,
        new_pooling_method,
        new_name);
}


dimensions Pooling::get_output_dimensions() const
{
    const Index rows_number = get_output_height();
    const Index columns_number = get_output_width();
    const Index channels = input_dimensions[2];

    return { rows_number, columns_number, channels };
}


Index Pooling::get_input_height() const
{
    return input_dimensions[0];
}


Index Pooling::get_input_width() const
{
    return input_dimensions[1];
}


Index Pooling::get_channels_number() const
{
    return input_dimensions[2];
}


Index Pooling::get_output_height() const
{
    return (get_input_height() - pool_height + 2 * padding_height) / row_stride + 1;
}


Index Pooling::get_output_width() const
{
    return (get_input_width() - pool_width + 2 * padding_width) / column_stride + 1;
}


Index Pooling::get_padding_height() const
{
    return padding_height;
}


Index Pooling::get_padding_width() const
{
    return padding_width;
}


Index Pooling::get_row_stride() const
{
    return row_stride;
}


Index Pooling::get_column_stride() const
{
    return column_stride;
}


Index Pooling::get_pool_height() const
{
    return pool_height;
}


Index Pooling::get_pool_width() const
{
    return pool_width;
}


Pooling::PoolingMethod Pooling::get_pooling_method() const
{
    return pooling_method;
}


dimensions Pooling::get_input_dimensions() const
{
    return input_dimensions;
}


void Pooling::print() const
{
    cout << "Pooling layer" << endl;
    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);
    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


string Pooling::write_pooling_method() const
{
    switch(pooling_method)
    {
    case PoolingMethod::MaxPooling:
        return "MaxPooling";

    case PoolingMethod::AveragePooling:
        return "AveragePooling";
    }

    return string();
}


void Pooling::set(const dimensions& new_input_dimensions,
                  const dimensions& new_pool_dimensions,
                  const dimensions& new_stride_dimensions,
                  const dimensions& new_padding_dimensions,
                  const PoolingMethod& new_pooling_method,
                  const string new_label)
{
    if(new_pool_dimensions.size() != 2)
        throw runtime_error("Pool dimensions must be 2");

    if (new_stride_dimensions.size() != 2)
        throw runtime_error("Stride dimensions must be 2");

    if (new_padding_dimensions.size() != 2)
        throw runtime_error("Padding dimensions must be 2");

    if (new_pool_dimensions[0] > new_input_dimensions[0] || new_pool_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("Pool dimensions cannot be bigger than input dimensions");

    if (new_stride_dimensions[0] <= 0 || new_stride_dimensions[1] <= 0)
        throw runtime_error("Stride dimensions cannot be 0 or lower");

    if (new_stride_dimensions[0] > new_input_dimensions[0] || new_stride_dimensions[1] > new_input_dimensions[0])
        throw runtime_error("Stride dimensions cannot be bigger than input dimensions");

    if (new_padding_dimensions[0] < 0 || new_padding_dimensions[1] < 0)
        throw runtime_error("Padding dimensions cannot be lower than 0");

    input_dimensions = new_input_dimensions;

    set_pool_size(new_pool_dimensions[0], new_pool_dimensions[1]);

    set_row_stride(new_stride_dimensions[0]);
    set_column_stride(new_stride_dimensions[1]);

    set_padding_height(new_padding_dimensions[0]);
    set_padding_width(new_padding_dimensions[1]);

    set_pooling_method(new_pooling_method);

    set_label(new_label);

    label = "pooling_layer";

#ifdef OPENNN_CUDA

    // Pooling descriptor

    cudnnCreatePoolingDescriptor(&pooling_descriptor);

    cudnnSetPooling2dDescriptor(pooling_descriptor,
                                pooling_mode,
                                CUDNN_PROPAGATE_NAN,
                                pool_height, pool_width,
                                padding_height, padding_width,
                                row_stride, column_stride);

#endif

}


void Pooling::set_input_dimensions(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions must be 3");

    input_dimensions = new_input_dimensions;
}


void Pooling::set_padding_height(const Index& new_padding_height)
{
    padding_height = new_padding_height;
}


void Pooling::set_padding_width(const Index& new_padding_width)
{
    padding_width = new_padding_width;
}


void Pooling::set_row_stride(const Index& new_row_stride)
{
    row_stride = new_row_stride;
}


void Pooling::set_column_stride(const Index& new_column_stride)
{
    column_stride = new_column_stride;
}


void Pooling::set_pool_size(const Index& new_pool_rows_number,
                            const Index& new_pool_columns_number)
{
    pool_height = new_pool_rows_number;
    pool_width = new_pool_columns_number;
}


void Pooling::set_pooling_method(const PoolingMethod& new_pooling_method)
{
    pooling_method = new_pooling_method;
}


void Pooling::set_pooling_method(const string& new_pooling_method)
{
    if(new_pooling_method == "MaxPooling")
        pooling_method = PoolingMethod::MaxPooling;
    else if(new_pooling_method == "AveragePooling")
        pooling_method = PoolingMethod::AveragePooling;
    else
        throw runtime_error("Unknown pooling type: " + new_pooling_method);
}


void Pooling::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map<4>(input_pairs[0]);

    switch(pooling_method)
    {
    case PoolingMethod::MaxPooling:
        forward_propagate_max_pooling(inputs,
                                      layer_forward_propagation,
                                      is_training);
        break;

    case PoolingMethod::AveragePooling:
        forward_propagate_average_pooling(inputs,
                                          layer_forward_propagation,
                                          is_training);
        break;
    }
}


void Pooling::forward_propagate_average_pooling(const Tensor<type, 4>& inputs,
                                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                                const bool& is_training) const
{
    PoolingForwardPropagation* this_forward_propagation =
        static_cast<PoolingForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 5>& image_patches = this_forward_propagation->image_patches;
    Tensor<type, 4>& outputs = this_forward_propagation->outputs;

    image_patches.device(*thread_pool_device) = inputs.extract_image_patches(
        pool_height,
        pool_width,
        row_stride,
        column_stride,
        1,
        1,
        PADDING_VALID,
        type(padding_width)
        );

    outputs.device(*thread_pool_device) = image_patches
                                              .mean(array_2(1, 2))
                                              .reshape(array_4(outputs.dimension(0), outputs.dimension(1), outputs.dimension(2), outputs.dimension(3)));
}


void Pooling::forward_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                            unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                            const bool& is_training) const
{
    PoolingForwardPropagation* pooling_layer_forward_propagation =
        static_cast<PoolingForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 5>& image_patches = pooling_layer_forward_propagation->image_patches;
    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    const Index batch_size = outputs.dimension(0);
    const Index output_width = outputs.dimension(1);
    const Index output_height = outputs.dimension(2);
    const Index channels = outputs.dimension(3);

    image_patches.device(*thread_pool_device) = inputs.extract_image_patches(
        pool_height,
        pool_width,
        row_stride,
        column_stride,
        1, 1,
        PADDING_VALID,
        type(padding_width));

    outputs.device(*thread_pool_device) = image_patches
                                              .maximum(array_2(1, 2))
                                              .reshape(array_4(batch_size, output_width, output_height, channels));

    if (!is_training) return;

    // Maximal indices

    Tensor<Index, 4>& maximal_indices = pooling_layer_forward_propagation->maximal_indices;

    const Index pool_size = pool_height * pool_width;
    const Index output_size = output_height * output_width * channels;

    const array<Index, 3> output_dimensions({ output_height, output_width, channels });
    const array<Index, 2> reshape_dimensions = { pool_size, output_size };

#pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_size; batch_index++)
    {
        const Tensor<type, 2> patches_flat = image_patches.chip(batch_index, 0).reshape(reshape_dimensions);

        maximal_indices.chip(batch_index, 0) = patches_flat.argmax(0).reshape(output_dimensions);
    }
}


void Pooling::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                             const vector<pair<type*, dimensions>>& delta_pairs,
                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map<4>(input_pairs[0]);
    const TensorMap<Tensor<type, 4>> deltas = tensor_map<4>(delta_pairs[0]);

    switch(pooling_method)
    {
    case PoolingMethod::MaxPooling:
        back_propagate_max_pooling(inputs,
                                   deltas,
                                   forward_propagation,
                                   back_propagation);
        break;

    case PoolingMethod::AveragePooling:
        back_propagate_average_pooling(inputs,
                                       deltas,
                                       back_propagation);
        break;
    }
}


void Pooling::back_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                         const Tensor<type, 4>& deltas,
                                         unique_ptr<LayerForwardPropagation>& forward_propagation,
                                         unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = inputs.dimension(0);

    const Index channels = inputs.dimension(3);

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);

    // Forward propagation

    PoolingForwardPropagation* pooling_layer_forward_propagation =
        static_cast<PoolingForwardPropagation*>(forward_propagation.get());

    Tensor<Index, 4>& maximal_indices = pooling_layer_forward_propagation->maximal_indices;

    // Back propagation

    PoolingBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& input_deltas = pooling_layer_back_propagation->input_deltas;

    // Input derivatives

    input_deltas.setZero();

#pragma omp parallel for collapse (2)
    for (Index channel_index = 0; channel_index < channels; channel_index++)
        for (Index batch_index = 0; batch_index < batch_size; batch_index++)
            for (Index output_height_index = 0; output_height_index < output_height; output_height_index++)
                for (Index output_width_index = 0; output_width_index < output_width; output_width_index++)
                {
                    const Index maximal_index = maximal_indices(batch_index, output_height_index, output_width_index, channel_index);

                    const Index input_row = output_height_index * row_stride + maximal_index % pool_height;
                    const Index input_column = output_width_index * column_stride + maximal_index / pool_width;

                    input_deltas(batch_index, input_row, input_column, channel_index)
                        += deltas(batch_index, output_height_index, output_width_index, channel_index);
                }
}


void Pooling::back_propagate_average_pooling(const Tensor<type, 4>& inputs,
                                             const Tensor<type, 4>& deltas,
                                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = inputs.dimension(0);

    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);

    const Index pool_size = pool_height * pool_width;

    const array<Index, 4> grad_extents = { batch_size, 1, 1, 1 };

    // Back propagation

    PoolingBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& input_deltas = pooling_layer_back_propagation->input_deltas;

    Tensor<type, 4>& deltas_by_pool_size = pooling_layer_back_propagation->deltas_by_pool_size;

    deltas_by_pool_size.device(*thread_pool_device) = deltas / type(pool_size);

    // Input derivatives

#pragma omp parallel for
    for (Index channel_index = 0; channel_index < channels; channel_index++)
        for (Index output_height_index = 0; output_height_index < output_height; output_height_index++)
        {
            const Index height_start = output_height_index * row_stride;
            const Index height_end = min(height_start + pool_height, input_height);

            for (Index output_width_index = 0; output_width_index < output_width; output_width_index++)
            {
                const Index width_start = output_width_index * column_stride;
                const Index width_end = min(width_start + pool_width, input_width);

                const array<Index, 4> grad_offsets = {0, output_height_index, output_width_index, channel_index};
                const array<Index, 4> offsets = {0, height_start, width_start, channel_index };
                const array<Index, 4> extents = {batch_size, height_end - height_start, width_end - width_start, 1};

                input_deltas.slice(offsets, extents) += deltas_by_pool_size
                                                            .slice(grad_offsets, grad_extents)
                                                            .broadcast(array_4(1, height_end - height_start, width_end - width_start, 1));
            }
        }
}


void Pooling::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Pooling");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(get_input_dimensions()));
    add_xml_element(printer, "PoolHeight", to_string(get_pool_height()));
    add_xml_element(printer, "PoolWidth", to_string(get_pool_width()));
    add_xml_element(printer, "PoolingMethod", write_pooling_method());
    add_xml_element(printer, "ColumnStride", to_string(get_column_stride()));
    add_xml_element(printer, "RowStride", to_string(get_row_stride()));
    add_xml_element(printer, "PaddingHeight", to_string(get_padding_height()));
    add_xml_element(printer, "PaddingWidth", to_string(get_padding_width()));

    printer.CloseElement();
}


void Pooling::from_XML(const XMLDocument& document)
{
    const XMLElement* pooling_layer_element = document.FirstChildElement("Pooling");

    if(!pooling_layer_element)
        throw runtime_error("Pooling layer element is nullptr.\batch_index");

    set_label(read_xml_string(pooling_layer_element, "Label"));
    set_input_dimensions(string_to_dimensions(read_xml_string(pooling_layer_element, "InputDimensions")));
    set_pool_size(read_xml_index(pooling_layer_element, "PoolHeight"), read_xml_index(pooling_layer_element, "PoolWidth"));
    set_pooling_method(read_xml_string(pooling_layer_element, "PoolingMethod"));
    set_column_stride(read_xml_index(pooling_layer_element, "ColumnStride"));
    set_row_stride(read_xml_index(pooling_layer_element, "RowStride"));
    set_padding_height(read_xml_index(pooling_layer_element, "PaddingHeight"));
    set_padding_width(read_xml_index(pooling_layer_element, "PaddingWidth"));
}


PoolingForwardPropagation::PoolingForwardPropagation(const Index& new_batch_size,
                                                     Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> PoolingForwardPropagation::get_output_pair() const
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();
    const Index channels = pooling_layer->get_channels_number();

    return {(type*)outputs.data(), {batch_size, output_height, output_width, channels}};
}


void PoolingForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index pool_height = pooling_layer->get_pool_height();
    const Index pool_width = pooling_layer->get_pool_width();

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();

    const Index channels = pooling_layer->get_channels_number();

    outputs.resize(batch_size,
                   output_height,
                   output_width,
                   channels);

    image_patches.resize(batch_size,
                         pool_height,
                         pool_width,
                         output_height * output_width,
                         channels);

    maximal_indices.resize(batch_size,
                           output_height,
                           output_width,
                           channels);
}


void PoolingForwardPropagation::print() const
{
    cout << "Pooling layer forward propagation" << endl
         << "Outputs:" << endl
         << outputs(0) << endl;
}


PoolingBackPropagation::PoolingBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void PoolingBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const dimensions& input_dimensions = pooling_layer->get_input_dimensions();
    const dimensions& output_dimensions = pooling_layer->get_output_dimensions();

    deltas_by_pool_size.resize(batch_size, output_dimensions[0], output_dimensions[1], output_dimensions[2]);

    input_deltas.resize(batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]);
}


vector<pair<type*, dimensions>> PoolingBackPropagation::get_input_derivative_pairs() const
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const dimensions& input_dimensions = pooling_layer->get_input_dimensions();

    return {{(type*)(input_deltas.data()),
             {batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]}} };
}


void PoolingBackPropagation::print() const
{
    cout << "Pooling layer back propagation" << endl;
    cout << "Input derivatives:" << endl
         << input_deltas << endl;
}


#ifdef OPENNN_CUDA

void Pooling::forward_propagate_cuda(const vector<float*>& inputs_device,
                                     unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                     const bool& is_training)
{
    // Forward propagation

    PoolingForwardPropagationCuda* pooling_layer_forward_propagation_cuda
        = static_cast<PoolingForwardPropagationCuda*>(forward_propagation_cuda.get());

    const Index batch_size = pooling_layer_forward_propagation_cuda->batch_size;

    type* outputs = pooling_layer_forward_propagation_cuda->outputs;

    cudnnTensorDescriptor_t& input_tensor_descriptor = pooling_layer_forward_propagation_cuda->input_tensor_descriptor;
    cudnnTensorDescriptor_t& output_tensor_descriptor = pooling_layer_forward_propagation_cuda->output_tensor_descriptor;

    // Pooling

    cudnnStatus_t status = cudnnPoolingForward(cudnn_handle,
                                               pooling_descriptor,
                                               &alpha,
                                               input_tensor_descriptor,
                                               inputs_device[0],
                                               &beta,
                                               output_tensor_descriptor,
                                               outputs);

    if (status != CUDNN_STATUS_SUCCESS)
        cout << "cudnnPoolingForward failed: " << cudnnGetErrorString(status) << endl;
}


void Pooling::back_propagate_cuda(const vector<float*>& inputs_device,
                                  const vector<float*>& deltas_device,
                                  unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                  unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Forward propagation

    PoolingForwardPropagationCuda* pooling_layer_forward_propagation_cuda
        = static_cast<PoolingForwardPropagationCuda*>(forward_propagation_cuda.get());

    const Index batch_size = pooling_layer_forward_propagation_cuda->batch_size;

    const type* outputs = pooling_layer_forward_propagation_cuda->outputs;

    cudnnTensorDescriptor_t& input_tensor_descriptor = pooling_layer_forward_propagation_cuda->input_tensor_descriptor;
    cudnnTensorDescriptor_t& output_tensor_descriptor = pooling_layer_forward_propagation_cuda->output_tensor_descriptor;

    // Back propagation

    PoolingBackPropagationCuda* pooling_layer_back_propagation_cuda
        = static_cast<PoolingBackPropagationCuda*>(back_propagation_cuda.get());

    type* input_deltas = pooling_layer_back_propagation_cuda->input_deltas;

    // Pooling

    cudnnStatus_t status = cudnnPoolingBackward(cudnn_handle,
                                                pooling_descriptor,
                                                &alpha,
                                                output_tensor_descriptor,
                                                outputs,
                                                output_tensor_descriptor,
                                                deltas_device[0],
                                                input_tensor_descriptor,
                                                inputs_device[0],
                                                &beta,
                                                input_tensor_descriptor,
                                                input_deltas);

    if (status != CUDNN_STATUS_SUCCESS)
        cout << "cudnnPoolingBackward failed: " << cudnnGetErrorString(status) << endl;
}


PoolingForwardPropagationCuda::PoolingForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void PoolingForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "PoolingForwardPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    const Index pool_height = pooling_layer->get_pool_height();
    const Index pool_width = pooling_layer->get_pool_width();

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();

    const Index padding_height = pooling_layer->get_padding_height();
    const Index padding_width = pooling_layer->get_padding_width();

    const Index row_stride = pooling_layer->get_row_stride();
    const Index column_stride = pooling_layer->get_column_stride();

    pooling_mode = (pooling_layer->get_pooling_method() == Pooling::PoolingMethod::MaxPooling)
                       ? CUDNN_POOLING_MAX
                       : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    // Inputs

    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               channels,
                               input_height,
                               input_width);

    // Outputs

    //CHECK_CUDA(cudaMalloc(&outputs, batch_size * output_height * output_width * channels * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(outputs, batch_size * output_height * output_width * channels * sizeof(float));

    cudnnCreateTensorDescriptor(&output_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               channels,
                               output_height,
                               output_width);

}


void PoolingForwardPropagationCuda::print() const
{
    const Pooling* pooling_layer = static_cast<const Pooling*>(layer);

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();
    const Index channels = pooling_layer->get_channels_number();

    cout << "Pooling layer forward propagation CUDA" << endl
         << "Outputs:" << endl
         << matrix_4d_from_device(outputs, batch_size, output_height, output_width, channels) << endl;
}


void PoolingForwardPropagationCuda::free()
{
    cudaFree(outputs);

    outputs = nullptr;

    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    cudnnDestroyTensorDescriptor(output_tensor_descriptor);
}


PoolingBackPropagationCuda::PoolingBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void PoolingBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "PoolingBackPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    // Input derivatives

    //CHECK_CUDA(cudaMalloc(&input_deltas, batch_size * input_height * input_width * channels * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(input_deltas, batch_size * input_height * input_width * channels * sizeof(float));
}


void PoolingBackPropagationCuda::print() const
{
    const Pooling* pooling_layer = static_cast<const Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    cout << "Pooling layer back propagation CUDA" << endl;
    cout << "Input derivatives:" << endl
         << matrix_4d_from_device(input_deltas, batch_size, input_height, input_width, channels) << endl;
}


void PoolingBackPropagationCuda::free()
{
    cudaFree(input_deltas);

    input_deltas = nullptr;
}


REGISTER(LayerForwardPropagationCuda, PoolingForwardPropagationCuda, "Pooling")
REGISTER(LayerBackPropagationCuda, PoolingBackPropagationCuda, "Pooling")

#endif

REGISTER(Layer, Pooling, "Pooling")
REGISTER(LayerForwardPropagation, PoolingForwardPropagation, "Pooling")
REGISTER(LayerBackPropagation, PoolingBackPropagation, "Pooling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
