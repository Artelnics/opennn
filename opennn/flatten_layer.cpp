//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a scaling layer object with no scaling neurons.

FlattenLayer::FlattenLayer() : Layer()
{
    //set();
}


/// Scaling neurons number constructor.
/// This constructor creates a scaling layer with a given size.
/// It initializes the members of this object with the default values.
/// @param new_neurons_number Number of scaling neurons in the layer.

FlattenLayer::FlattenLayer(const Index& new_neurons_number) : Layer()
{
    //set(new_neurons_number);
}


/// Destructor.

FlattenLayer::~FlattenLayer()
{
}


/// Obtain the connection between the convolutional and the conventional part
/// of a neural network. That is a column vector in which each of its elements
/// links to a neuron in the perceptron layer.
/// @param inputs

Tensor<type, 2> FlattenLayer::calculate_outputs_2d(const Tensor<type, 4>& inputs)
{
    const Index batch = inputs.dimension(0);
    const Index channels = inputs.dimension(1);
    const Index width = inputs.dimension(2);
    const Index heights = inputs.dimension(3);

    inputs(batch, channels, width, heights);
    Eigen::array<Index, 2> new_dims{{batch, channels*width*heights}};
    Tensor<type, 2> result = inputs.reshape(new_dims);

    return result;
}

void FlattenLayer::forward_propagate(const Tensor<type, 4> &inputs, LayerForwardPropagation* forward_propagation)
{

    FlattenLayerForwardPropagation* flatten_layer_forward_propagation
            = static_cast<FlattenLayerForwardPropagation*>(forward_propagation);

}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
