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
        return "PerceptronLayer";

    case Type::PerceptronLayer3D:
        return "PerceptronLayer3D";

    case Type::Bounding:
        return "BoundingLayer";

    case Type::Pooling:
        return "PoolingLayer";

    case Type::Probabilistic:
        return "ProbabilisticLayer2D";

    case Type::Probabilistic3D:
        return "ProbabilisticLayer3D";

    case Type::Convolutional:
        return "ConvolutionalLayer";

    case Type::LongShortTermMemory:
        return "LongShortTermMemoryLayer";

    case Type::Recurrent:
        return "RecurrentLayer";

    case Type::Scaling2D:
        return "ScalingLayer2D";

    case Type::Scaling4D:
        return "Scaling4D";

    case Type::Unscaling:
        return "UnscalingLayer";

    case Type::Flatten:
        return "Flatten";

    case Type::RegionProposal:
        return "RegionProposal";

    case Type::NonMaxSuppression:
        return "NonMaxSuppression";

    case Type::Addition3D:
        return "AdditionLayer3D";

    case Type::Normalization3D:
        return "NormalizationLayer3D";

    case Type::Embedding:
        return "EmbeddingLayer";

    case Type::MultiheadAttention:
        return "MultiheadAttentionLayer";

    default:
        return "Unkown type";
    }
}


void Layer::set_name(const string& new_name)
{
    name = new_name;
}


void Layer::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
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


dimensions Layer::get_output_dimensions() const
{
    return dimensions();
}


void Layer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>&, LayerForwardPropagation*, const bool&)
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}


Index Layer::get_inputs_number() const
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}


Index Layer::get_neurons_number() const
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}


void Layer::set_inputs_number(const Index&)
{
    throw runtime_error("This method is not implemented in the layer type (" + get_type_string() + ").\n");
}


void Layer::set_neurons_number(const Index&)
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
    const Eigen::array<Index, 1> softmax_dimension{ { 2 }}; 
    
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

    const Eigen::array<Index, 1> softmax_dimension{ { 0 }};
    const Eigen::array<Index, 4> range_4{ { 1, columns_number, channels, blocks_number }};
    const Eigen::array<Index, 4> expand_softmax_dim{ { rows_number, 1, 1, 1 }};

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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
