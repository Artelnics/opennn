//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"
#include "tensors.h"

namespace opennn
{

Layer::Layer()
{
    const unsigned int threads_number = thread::hardware_concurrency();

    thread_pool = make_unique<ThreadPool>(threads_number);
    device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);
}

Layer::~Layer() = default;

const bool& Layer::get_display() const
{
    return display;
}


const string& Layer::get_label() const
{
    return label;
}


const string& Layer::get_name() const
{
    return name;
}


void Layer::set_label(const string& new_label)
{
    label = new_label;
}


void Layer::set_display(const bool& new_display)
{
    display = new_display;
}


void Layer::set_parameters_random()
{
    const vector<ParameterView> parameter_views = get_parameter_views();

    for (const auto& view : parameter_views)
    {
        TensorMap<Tensor<type, 1>> this_parameters(view.data, view.size);

        set_random(this_parameters);
    }
}


void Layer::set_parameters_glorot()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const type limit = sqrt(6.0 / (inputs_number + outputs_number));

    const vector<ParameterView> parameter_views = get_parameter_views();

    for (const auto& view : parameter_views)
    {
        TensorMap<Tensor<type, 1>> this_parameters(view.data, view.size);

        set_random(this_parameters, -limit, limit);
    }
}


Index Layer::get_parameters_number() const
{
    const vector<ParameterView> parameter_views = get_parameter_views();

    Index parameters_number = 0;

    for(Index i = 0; i < Index(parameter_views.size()); i++)
        parameters_number += parameter_views[i].size;

    return parameters_number;
}


vector<ParameterView > Layer::get_parameter_views() const
{
    return vector<ParameterView>();
}


void Layer::set_threads_number(const int& new_threads_number)
{
    thread_pool.reset();
    device.reset();

    thread_pool = make_unique<ThreadPool>(new_threads_number);
    device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


string Layer::get_expression(const vector<string> &, const vector<string> &) const
{
    return string();
}


vector<string> Layer::get_default_feature_names() const
{
    const Index inputs_number = get_inputs_number();

    vector<string> feature_names(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        feature_names[i] = "input_" + to_string(i);

    return feature_names;
}


vector<string> Layer::get_default_output_names() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> output_names(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names[i] = "output_" + to_string(i);

    return output_names;
}


bool Layer::get_is_trainable() const
{
    return is_trainable;
}


void Layer::add_deltas(const vector<TensorView> &delta_views) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_views[0]);

    for (Index i = 1; i < Index(delta_views.size()); i++)
        deltas.device(*device) += tensor_map<3>(delta_views[i]);
}


Index Layer::get_inputs_number() const
{
    const dimensions input_dimensions = get_input_dimensions();

    return accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
}


Index Layer::get_outputs_number() const
{
    const dimensions output_dimensions = get_output_dimensions();

    return accumulate(output_dimensions.begin(), output_dimensions.end(), 1, multiplies<Index>());
}


void Layer::forward_propagate(const vector<TensorView>&,
                              unique_ptr<LayerForwardPropagation>&, const bool&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}



void Layer::set_input_dimensions(const dimensions&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}


void Layer::set_output_dimensions(const dimensions&)
{
    throw runtime_error("This method is not implemented in the layer type (" + name + ").\n");
}


void Layer::softmax(Tensor<type, 2>& y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);

    y.device(*device) = y - y.maximum(array_1(1))
                                            .eval()
                                            .reshape(array_2(rows_number, 1))
                                            .broadcast(array_2(1, columns_number));

    y.device(*device) = y.exp();

    y.device(*device) = y / y.sum(array<Index, 1>({1}))
                                            .eval()
                                            .reshape(array_2(rows_number, 1))
                                            .broadcast(array_2(1, columns_number));
}


void Layer::softmax(Tensor<type, 3>& y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);

    y.device(*device) = y - y.maximum(array<Index, 1>({2}))
                                            .eval()
                                            .reshape(array<Index, 3>({rows_number, columns_number, 1}))
                                            .broadcast(array<Index, 3>({1, 1, channels}));

    y.device(*device) = y.exp();

    y.device(*device) = y / y.sum(array<Index, 1>({2}))
                                            .eval()
                                            .reshape(array<Index, 3>({rows_number, columns_number, 1}))
                                            .broadcast(array<Index, 3>({1, 1, channels}));
}


void Layer::softmax(Tensor<type, 4>& y) const
{
    const Index rows_number    = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels       = y.dimension(2);
    const Index blocks_number  = y.dimension(3);

    y.device(*device) =
        y - y.maximum(array_1(3)).eval()
                .reshape(array_4(rows_number, columns_number, channels, 1))
                .broadcast(array_4(1, 1, 1, blocks_number));

    y.device(*device) = y.exp();

    y.device(*device) =
        y / y.sum(array_1(3)).eval()
                .reshape(array_4(rows_number, columns_number, channels, 1))
                .broadcast(array_4(1, 1, 1, blocks_number));
}


void Layer::softmax_derivatives_times_tensor(const Tensor<type, 3>& softmax,
                                             TensorMap<Tensor<type, 3>>& result,
                                             Tensor<type, 1>& aux_rows) const
{
    const Index rows = softmax.dimension(0);
    const Index columns = softmax.dimension(1);
    const Index depth = softmax.dimension(2);


    type* softmax_data = (type*)softmax.data();
    type* result_data = result.data();

    type* softmax_vector_data = nullptr;
    type* result_vector_data = nullptr;

    Tensor<type, 0> sum;

    for (Index i = 0; i < depth; i++)
    {
        for (Index j = 0; j < columns; j++)
        {
            softmax_vector_data = softmax_data + rows * (i * columns + j);
            result_vector_data = result_data + rows * (i * columns + j);

            const TensorMap<Tensor<type, 1>> softmax_vector(softmax_vector_data, rows);
            const TensorMap<Tensor<type, 1>> tensor_vector(result_vector_data, rows);

            TensorMap<Tensor<type, 1>> result_vector(result_vector_data, rows);

            aux_rows.device(*device) = softmax_vector * tensor_vector;

            sum.device(*device) = aux_rows.sum();

            result_vector.device(*device) = aux_rows - softmax_vector * sum(0);
        }
    }
}


#ifdef OPENNN_CUDA

void Layer::create_cuda()
{
    cublasCreate(&cublas_handle);
    cudnnCreate(&cudnn_handle);

    // Multiplication

    cudnnCreateOpTensorDescriptor(&operator_multiplication_descriptor);

    cudnnSetOpTensorDescriptor(operator_multiplication_descriptor,
                               CUDNN_OP_TENSOR_MUL,
                               CUDNN_DATA_FLOAT,
                               CUDNN_NOT_PROPAGATE_NAN);

    // Sum

    cudnnCreateOpTensorDescriptor(&operator_sum_descriptor);

    cudnnSetOpTensorDescriptor(operator_sum_descriptor,
                               CUDNN_OP_TENSOR_ADD,
                               CUDNN_DATA_FLOAT,
                               CUDNN_NOT_PROPAGATE_NAN);
}


void Layer::destroy_cuda()
{
    cublasDestroy(cublas_handle);
    cudnnDestroy(cudnn_handle);

    cudnnDestroyOpTensorDescriptor(operator_multiplication_descriptor);
    cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
}


cudnnHandle_t Layer::get_cudnn_handle()
{
    return cudnn_handle;
}


vector<ParameterView> Layer::get_parameter_views_device() const
{
    return vector<ParameterView>();
}

#endif

}

// namespace opennn
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
