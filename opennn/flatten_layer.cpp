//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer.h"

namespace opennn
{


/// Default constructor.
/// It creates a flatten layer.
/// This constructor also initializes the rest of the class members to their default values.

FlattenLayer::FlattenLayer() : Layer()
{
    //set();

    layer_type = Type::Flatten;
}


FlattenLayer::FlattenLayer(const Tensor<Index, 1>& new_input_variables_dimensions) : Layer()
{
    input_variables_dimensions = new_input_variables_dimensions;

    layer_type = Type::Flatten;
}


Tensor<Index, 1> FlattenLayer::get_input_variables_dimensions() const
{
    return input_variables_dimensions;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// @todo change to memcpy approach
/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (number of images (batch), number of channels, width, height).

void FlattenLayer::set(const Tensor<Index, 1>& new_inputs_dimensions)
{
    input_variables_dimensions = new_inputs_dimensions;
}


/// Obtain the connection between the convolutional and the conventional part
/// of a neural network. That is a matrix which links to the perceptron layer.
/// @param inputs 4d tensor(batch, channels, width, height)
/// @return result 2d tensor(batch, number of pixels)

Tensor<type, 2> FlattenLayer::calculate_outputs_2d(const Tensor<type, 4>& inputs)
{
    const Index batch = inputs.dimension(0);
    const Index channels = inputs.dimension(1);
    const Index width = inputs.dimension(2);
    const Index heights = inputs.dimension(3);

    const Eigen::array<Index, 2> new_dims{{batch, channels*width*heights}};

    const Tensor<type, 2> outputs = inputs.reshape(new_dims);

    return outputs;
}


void FlattenLayer::forward_propagate(const Tensor<type, 4>& inputs, LayerForwardPropagation* forward_propagation)
{
    cout << "Hello flatten layer!" << endl;

    FlattenLayerForwardPropagation* flatten_layer_forward_propagation
            = static_cast<FlattenLayerForwardPropagation*>(forward_propagation);

#ifdef OPENNN_DEBUG

    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    if(outputs_dimensions[0] != flatten_layer_forward_propagation->outputs.dimension(0))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer::forward_propagate.\n"
               << "outputs_dimensions[0]" <<outputs_dimensions[0] <<"must be equal to" << flatten_layer_forward_propagation->outputs.dimension(0)<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs_dimensions[1] != flatten_layer_forward_propagation->outputs.dimension(1))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer::forward_propagate.\n"
               << "outputs_dimensions[1]" <<outputs_dimensions[1] <<"must be equal to" << flatten_layer_forward_propagation->outputs.dimension(1)<<".\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Index batch = inputs.dimension(0);
    const Index channels = inputs.dimension(1);
    const Index width = inputs.dimension(2);
    const Index heights = inputs.dimension(3);

    const Eigen::array<Index, 2> new_dims{{batch, channels*width*heights}};

    flatten_layer_forward_propagation->outputs = inputs.reshape(new_dims);
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
