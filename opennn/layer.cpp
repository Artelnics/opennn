//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"
#include "layer_back_propagation_lm.h"

namespace opennn
{

Layer::Layer()
{
    const unsigned int threads_number = thread::hardware_concurrency();

    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);
}


const bool& Layer::get_display() const
{
    return display;
}


string Layer::layer_type_to_string(const Layer::Type& this_layer_type)
{
    switch(this_layer_type)
    {
    case Type::Perceptron:
        return "Perceptron";

    case Type::Perceptron3D:
        return "Perceptron3D";

    case Type::Bounding:
        return "Bounding";

    case Type::Pooling:
        return "Pooling";

    case Type::Probabilistic:
        return "Probabilistic";

    case Type::Probabilistic3D:
        return "Probabilistic3D";

    case Type::Convolutional:
        return "Convolutional";

    case Type::LongShortTermMemory:
        return "LongShortTermMemory";

    case Type::Recurrent:
        return "Recurrent";

    case Type::Scaling2D:
        return "Scaling2D";

    case Type::Scaling4D:
        return "Scaling4D";

    case Type::Unscaling:
        return "Unscaling";

    case Type::Flatten:
        return "Flatten";

    case Type::NonMaxSuppression:
        return "NonMaxSuppression";

    case Type::Addition3D:
        return "Addition3D";

    case Type::Normalization3D:
        return "Normalization3D";

    case Type::Embedding:
        return "Embedding";

    case Type::MultiheadAttention:
        return "MultiheadAttention";

    default:
        throw runtime_error("Unknown layer type.");
    }

}


Layer::Type Layer::string_to_layer_type(const string& this_layer_type)
{
    if(this_layer_type == "Perceptron")
        return Type::Perceptron;

    if(this_layer_type == "Perceptron3D")
        return Type::Perceptron3D;

    if(this_layer_type == "Bounding")
        return Type::Bounding;

    if(this_layer_type == "Pooling")
        return Type::Pooling;

    if(this_layer_type == "Probabilistic")
        return Type::Probabilistic;

    if(this_layer_type == "Probabilistic3D")
        return Type::Probabilistic3D;

    if(this_layer_type == "Convolutional")
        return Type::Convolutional;

    if(this_layer_type == "LongShortTermMemory")
        return Type::LongShortTermMemory;

    if(this_layer_type == "Recurrent")
        return Type::Recurrent;

    if(this_layer_type == "Scaling2D")
        return Type::Scaling2D;

    if(this_layer_type == "Scaling4D")
        return Type::Scaling4D;

    if(this_layer_type == "Unscaling")
        return Type::Unscaling;

    if(this_layer_type == "Flatten")
        return Type::Flatten;

    if(this_layer_type == "NonMaxSuppression")
        return Type::NonMaxSuppression;

    if(this_layer_type == "Addition3D")
        return Type::Addition3D;

    if(this_layer_type == "Normalization3D")
        return Type::Normalization3D;

    if(this_layer_type == "Embedding")
        return Type::Embedding;

    if(this_layer_type == "MultiheadAttention")
        return Type::MultiheadAttention;

    throw runtime_error("Unknown layer type.");
}


string Layer::get_name() const
{
    return name;
}


Layer::Type Layer::get_type () const
{
    return layer_type;
}


string Layer::get_type_string() const
{
    switch(layer_type)
    {
    case Type::Perceptron:
        return "Perceptron";

    case Type::Perceptron3D:
        return "Perceptron3D";

    case Type::Bounding:
        return "Bounding";

    case Type::Pooling:
        return "Pooling";

    case Type::Probabilistic:
        return "Probabilistic";

    case Type::Probabilistic3D:
        return "Probabilistic3D";

    case Type::Convolutional:
        return "Convolutional";

    case Type::LongShortTermMemory:
        return "LongShortTermMemory";

    case Type::Recurrent:
        return "Recurrent";

    case Type::Scaling2D:
        return "Scaling2D";

    case Type::Scaling4D:
        return "Scaling4D";

    case Type::Unscaling:
        return "Unscaling";

    case Type::Flatten:
        return "Flatten";

    case Type::NonMaxSuppression:
        return "NonMaxSuppression";

    case Type::Addition3D:
        return "Addition3D";

    case Type::Normalization3D:
        return "Normalization3D";

    case Type::Embedding:
        return "Embedding";

    case Type::MultiheadAttention:
        return "MultiheadAttention";

    default:
        return "Unkown type";
    }
}


void Layer::set_name(const string& new_name)
{
    name = new_name;
}


void Layer::set_display(const bool& new_display)
{
    display = new_display;
}


void Layer::set_threads_number(const int& new_threads_number)
{
    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
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


void Layer::set_parameters_constant(const type&)
{
}


void Layer::set_parameters_random()
{
}


void Layer::set_parameters(const Tensor<type, 1>&, const Index&)
{
}


Index Layer::get_parameters_number() const
{
    return 0;
}


Tensor<type, 1> Layer::get_parameters() const
{
    return Tensor<type, 1>();
}


Index Layer::get_inputs_number() const
{
    const dimensions input_dimensions = get_input_dimensions();

    return accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
}


Index Layer::get_outputs_number() const
{
    const dimensions output_dimensions = get_output_dimensions();

    //cout << output_dimensions.begin().base() << " --- " << output_dimensions.end().base() << endl;

    return accumulate(output_dimensions.begin(), output_dimensions.end(), 1, multiplies<Index>());
}


void Layer::forward_propagate(const vector<pair<type*, dimensions>>&, 
                              unique_ptr<LayerForwardPropagation>&, const bool&)
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}



void Layer::set_input_dimensions(const dimensions&)
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}


void Layer::set_output_dimensions(const dimensions&)
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}


void Layer::competitive(Tensor<type, 2>& y) const
{
    const Tensor<Index, 1> maximum_indices = y.argmax(1);

    y.setZero();

    #pragma omp parallel for
    for(Index i = 0; i < y.dimension(0); i++)
        y(i, Index(maximum_indices(i))) = type(1);
}


void Layer::softmax(Tensor<type, 2>& y) const
{
    const Eigen::array<Index, 1> softmax_dimension{{1}};
    
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    
    const Eigen::array<Index, 2> range_2{ { rows_number, 1 }};
    const Eigen::array<Index, 2> expand_softmax_dim{ { 1, columns_number} };

    y.device(*thread_pool_device) = y - y.maximum(softmax_dimension)
                                         .eval()
                                         .reshape(range_2)
                                         .broadcast(expand_softmax_dim);

    y.device(*thread_pool_device) = y.exp();

    y.device(*thread_pool_device) = y / y.sum(softmax_dimension)
                                         .eval()
                                         .reshape(range_2)
                                         .broadcast(expand_softmax_dim);
}


void Layer::softmax(Tensor<type, 3>& y) const
{
    const Eigen::array<Index, 1> softmax_dimension{{2}}; 
    
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);
    
    const Eigen::array<Index, 3> range_3{ { rows_number, columns_number, 1 }};
    const Eigen::array<Index, 3> expand_softmax_dim{ { 1, 1, channels }};

    y.device(*thread_pool_device) = y - y.maximum(softmax_dimension)
                                         .eval()
                                         .reshape(range_3)
                                         .broadcast(expand_softmax_dim);

    y.device(*thread_pool_device) = y.exp();

    y.device(*thread_pool_device) = y / y.sum(softmax_dimension)
                                         .eval()
                                         .reshape(range_3)
                                         .broadcast(expand_softmax_dim);
}


void Layer::softmax(Tensor<type, 4>& y) const
{
    const Index rows_number = y.dimension(0);
    const Index columns_number = y.dimension(1);
    const Index channels = y.dimension(2);
    const Index blocks_number = y.dimension(3);

    const Eigen::array<Index, 1> softmax_dimension{{0}};
    const Eigen::array<Index, 4> range_4{{1, columns_number, channels, blocks_number}};
    const Eigen::array<Index, 4> expand_softmax_dim{{rows_number, 1, 1, 1 }};

    y.device(*thread_pool_device) = y - y.maximum(softmax_dimension)
                                         .eval()
                                         .reshape(range_4)
                                         .broadcast(expand_softmax_dim);

    y.device(*thread_pool_device) = y.exp();

    y.device(*thread_pool_device) = y / y.sum(softmax_dimension)
                                         .eval()
                                         .reshape(range_4)
                                         .broadcast(expand_softmax_dim);
}


void Layer::softmax_derivatives_times_tensor(const Tensor<type, 3>& softmax, 
                                             const Tensor<type, 3>& tensor, 
                                             TensorMap<Tensor<type, 3>>& result, 
                                             Tensor<type, 1>& aux_rows) const
{
    const Index rows_number = softmax.dimension(0);
    const Index columns_number = softmax.dimension(1);
    const Index channels = softmax.dimension(2);

    type* softmax_data = (type*)softmax.data();
    type* tensor_data = (type*)tensor.data();
    type* result_data = result.data();

    type* softmax_vector_data = nullptr;
    type* tensor_vector_data = nullptr;
    type* result_vector_data = nullptr;

    Tensor<type, 0> sum;

    for(Index i = 0; i < channels; i++)
    {        
        for(Index j = 0; j < columns_number; j++)
        {
            softmax_vector_data = softmax_data + rows_number * (i * columns_number + j);
            tensor_vector_data = tensor_data + rows_number * (i * columns_number + j);
            result_vector_data = result_data + rows_number * (i * columns_number + j);

            const TensorMap<Tensor<type, 1>> softmax_vector(softmax_vector_data, rows_number);
            const TensorMap<Tensor<type, 1>> tensor_vector(tensor_vector_data, rows_number);

            TensorMap<Tensor<type, 1>> result_vector(result_vector_data, rows_number);

            aux_rows.device(*thread_pool_device) = softmax_vector * tensor_vector;
            
            sum.device(*thread_pool_device) = aux_rows.sum();

            result_vector.device(*thread_pool_device) = aux_rows - softmax_vector * sum(0);
        }
    }
}

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
