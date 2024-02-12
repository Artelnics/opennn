//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <ctype.h>

#include "layer.h"
#include "tensor_utilities.h"
#include "statistics.h"
#include "scaling.h"
#include <tuple>

namespace opennn
{

Layer::~Layer()
{
    delete thread_pool;
    delete thread_pool_device;
}


/// Default constructor.
/// It creates a layer object with zero parameters.
/// It also initializes the rest of the class members to their default values.

Layer::Type Layer::get_type () const
{
    return layer_type;
}


/// Takes the type of layer used by the model.

string Layer::get_type_string() const
{
    switch(layer_type)
    {
    case Type::Perceptron:
        return "Perceptron";

    case Type::Bounding:
        return "Bounding";

    case Type::Pooling:
        return "Pooling";

    case Type::Probabilistic:
        return "Probabilistic";

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

    case Type::RegionProposal:
        return "RegionProposal";

    case Type::NonMaxSuppression:
        return "NonMaxSuppression";

    default:
        return "Unkown type";
    }
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
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters(const Tensor<type, 1>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw runtime_error(buffer.str());
}


Index Layer::get_parameters_number() const
{
    return 0;
}


Tensor<type, 1> Layer::get_parameters() const
{
    return Tensor<type, 1>();
}


void Layer::forward_propagate(const pair<type*, dimensions>&, LayerForwardPropagation*, const bool&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw runtime_error(buffer.str());
}


/// Returns the number of inputs

Index Layer::get_inputs_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_inputs_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw runtime_error(buffer.str());
}


Index Layer::get_neurons_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_neurons_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw runtime_error(buffer.str());
}


void Layer::set_inputs_number(const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_inputs_number(const Index&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw runtime_error(buffer.str());
}


void Layer::set_neurons_number(const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_neurons_number(const Index&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw runtime_error(buffer.str());
}


void Layer::competitive(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Index rows_number = x.dimension(0);

    Index maximum_index = 0;

    y.setZero();

    for (Index i = 0; i < rows_number; i++)
    {
        maximum_index = maximal_index(x.chip(i, 1));

        y(i, maximum_index) = type(1);
    }
}


void Layer::softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Index rows_number = x.dimension(0);
    const Index raw_variables_number = x.dimension(1);

    const Eigen::array<Index, 1> last_dimension{ {2} };
    const Eigen::array<Index, 2> range_2{ {rows_number, 1} };
    const Eigen::array<Index, 2> expand_last_dim{ {1, raw_variables_number} };

    y.device(*thread_pool_device) = x - x.maximum(last_dimension)
        .reshape(range_2)
        .broadcast(expand_last_dim);

    y.device(*thread_pool_device) = y.exp();

    Tensor<type, 2> y_sum = y.sum(last_dimension)
        .reshape(range_2)
        .broadcast(expand_last_dim);

    y.device(*thread_pool_device) = y / y.sum(last_dimension)
        .reshape(range_2)
        .broadcast(expand_last_dim);
}


void Layer::softmax(const Tensor<type, 3>& x, Tensor<type, 3>& y) const
{
    const Index rows_number = x.dimension(0);
    const Index raw_variables_number = x.dimension(1);
    const Index channels_number = x.dimension(2);

    const Eigen::array<Index, 1> last_dimension{ {2} };
    const Eigen::array<Index, 3> range_3{ {rows_number, raw_variables_number, 1} };
    const Eigen::array<Index, 3> expand_last_dim{ {1, 1, channels_number} };

    y.device(*thread_pool_device) = x - x.maximum(last_dimension)
        .reshape(range_3)
        .broadcast(expand_last_dim);

    y.device(*thread_pool_device) = y.exp();

    Tensor<type, 3> y_sum = y.sum(last_dimension)
        .reshape(range_3)
        .broadcast(expand_last_dim);

    y.device(*thread_pool_device) = y / y_sum;
}


void Layer::softmax(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Index rows_number = x.dimension(0);
    const Index raw_variables_number = x.dimension(1);
    const Index channels_number = x.dimension(2);
    const Index blocks_number = x.dimension(3);

    const Eigen::array<Index, 1> last_dimension{ {2} };
    const Eigen::array<Index, 4> range_4{ {rows_number, raw_variables_number, 1, blocks_number} };
    const Eigen::array<Index, 4> expand_last_dim{ {1, 1, channels_number, 1} };

    y.device(*thread_pool_device) = x - x.maximum(last_dimension)
        .reshape(range_4)
        .broadcast(expand_last_dim);

    y.device(*thread_pool_device) = y.exp();

    Tensor<type, 4> y_sum = y.sum(last_dimension)
        .reshape(range_4)
        .broadcast(expand_last_dim);

    y.device(*thread_pool_device) = y / y_sum;
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
