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

}


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
    const vector<pair<type*, Index>> parameter_pairs = get_parameter_pairs();

    for(Index i = 0; i < Index(parameter_pairs.size()); i++)
    {
        TensorMap<Tensor<type, 1>> this_parameters(parameter_pairs[i].first, parameter_pairs[i].second);

        set_random(this_parameters);
    }
}


void Layer::set_parameters_glorot()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const type limit = sqrt(6.0 / (inputs_number + outputs_number));

    const vector<pair<type*, Index>> parameter_pairs = get_parameter_pairs();

    for(Index i = 0; i < Index(parameter_pairs.size()); i++)
    {
        TensorMap<Tensor<type, 1>> this_parameters(parameter_pairs[i].first, parameter_pairs[i].second);

        set_random(this_parameters, -limit, limit);
    }
}


Index Layer::get_parameters_number() const
{
    const vector<pair<type*, Index>> parameter_pairs = get_parameter_pairs();

    Index parameters_number = 0;

    for(Index i = 0; i < Index(parameter_pairs.size()); i++)
        parameters_number += parameter_pairs[i].second;

    return parameters_number;
}


vector<pair<type *, Index> > Layer::get_parameter_pairs() const
{
    return vector<pair<type*, Index>>();
}


string Layer::get_expression(const vector<string> &, const vector<string> &) const
{
    return string();
}


vector<string> Layer::get_default_input_names() const
{
    const Index inputs_number = get_inputs_number();

    vector<string> input_names(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        input_names[i] = "input_" + to_string(i);

    return input_names;
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


void Layer::add_deltas(const vector<pair<type *, dimensions> > &delta_pairs) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_pairs[0]);

    type* deltas_data = deltas.data();

    for (size_t i = 1; i < delta_pairs.size(); ++i)
    {
        const TensorMap<Tensor<type, 3>> current_delta = tensor_map<3>(delta_pairs[i]);
        const type* current_delta_data = current_delta.data();

        #pragma omp parallel for
        for (Index j = 0; j < deltas.size(); ++j)
            deltas_data[j] += current_delta_data[j];
    }
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


void Layer::forward_propagate(const vector<pair<type*, dimensions>>&, 
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
    
    #pragma omp parallel for
    for (Index i = 0; i < rows_number; ++i)
    {
        type max_val = y(i, 0);

        for (Index j = 1; j < columns_number; ++j)
            if (y(i, j) > max_val)
                max_val = y(i, j);

        type sum_exp = 0.0;

        for (Index j = 0; j < columns_number; ++j)
        {
            y(i, j) = exp(y(i, j) - max_val);
            sum_exp += y(i, j);
        }

        for (Index j = 0; j < columns_number; ++j)
            y(i, j) /= sum_exp;
    }
}


void Layer::softmax(Tensor<type, 3>& y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);
    
    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < rows_number; ++i)
    {
        for (Index j = 0; j < columns_number; ++j)
        {
            type max_val = y(i, j, 0);
            for (Index k = 1; k < channels; ++k)
                if (y(i, j, k) > max_val)
                    max_val = y(i, j, k);

            type sum_exp = type(0);

            for (Index k = 0; k < channels; ++k)
                sum_exp += std::exp(y(i, j, k) - max_val);

            for (Index k = 0; k < channels; ++k)
                y(i, j, k) = std::exp(y(i, j, k) - max_val) / sum_exp;
        }
    }
}


void Layer::softmax(Tensor<type, 4>& y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);
    const Index blocks_number = y.dimension(3);

    #pragma omp parallel for collapse(3)
    for (Index i = 0; i < rows_number; ++i)
    {
        for (Index j = 0; j < columns_number; ++j)
        {
            for (Index k = 0; k < channels; ++k)
            {
                type max_val = y(i, j, k, 0);

                for (Index l = 1; l < blocks_number; ++l)
                    if (y(i, j, k, l) > max_val)
                        max_val = y(i, j, k, l);

                type sum_exp = type(0);

                for (Index l = 0; l < blocks_number; ++l)
                    sum_exp += std::exp(y(i, j, k, l) - max_val);

                for (Index l = 0; l < blocks_number; ++l)
                    y(i, j, k, l) = std::exp(y(i, j, k, l) - max_val) / sum_exp;
            }
        }
    }
}


//void Layer::softmax_derivatives_times_tensor(const Tensor<type, 3>& softmax,
//                                             const Tensor<type, 3>& tensor,
//                                             TensorMap<Tensor<type, 3>>& result,
//                                             Tensor<type, 1>& aux_rows) const
//{
//    const Index rows_number = softmax.dimension(0);
//    const Index columns_number = softmax.dimension(1);
//    const Index channels = softmax.dimension(2);

//    type* softmax_data = (type*)softmax.data();
//    type* tensor_data = (type*)tensor.data();
//    type* result_data = result.data();

//    type* softmax_vector_data = nullptr;
//    type* tensor_vector_data = nullptr;
//    type* result_vector_data = nullptr;

//    Tensor<type, 0> sum;

//    for(Index i = 0; i < channels; i++)
//    {
//        for(Index j = 0; j < columns_number; j++)
//        {
//            softmax_vector_data = softmax_data + rows_number * (i * columns_number + j);
//            tensor_vector_data = tensor_data + rows_number * (i * columns_number + j);
//            result_vector_data = result_data + rows_number * (i * columns_number + j);

//            const TensorMap<Tensor<type, 1>> softmax_vector(softmax_vector_data, rows_number);
//            const TensorMap<Tensor<type, 1>> tensor_vector(tensor_vector_data, rows_number);

//            TensorMap<Tensor<type, 1>> result_vector(result_vector_data, rows_number);

//            aux_rows.device(*thread_pool_device) = softmax_vector * tensor_vector;
            
//            sum.device(*thread_pool_device) = aux_rows.sum();

//            result_vector.device(*thread_pool_device) = aux_rows - softmax_vector * sum(0);
//        }
//    }
//}


void Layer::softmax_derivatives_times_tensor(const Tensor<type, 3>& softmax,
    TensorMap<Tensor<type, 3>>& result) const
{
    const Index rows = softmax.dimension(0);
    const Index columns = softmax.dimension(1);
    const Index depth = softmax.dimension(2);

    #pragma omp parallel for
    for (Index i = 0; i < depth; i++)
    {
        for (Index j = 0; j < columns; j++)
        {
            auto softmax_vec = softmax.chip(i, 2).chip(j, 1);
            auto tensor_vec = result.chip(i, 2).chip(j, 1);

            const Tensor<type, 0> sum_tensor = (softmax_vec * tensor_vec).sum();
            const type sum = sum_tensor(0);

            tensor_vec = softmax_vec * tensor_vec - softmax_vec * sum;
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


vector<pair<float*, Index>> Layer::get_parameter_pair_device() const
{
    return vector<pair<float*, Index>>();
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
