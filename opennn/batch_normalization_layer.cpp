//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M A L I Z A T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch_normalization_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

BatchNormalizationLayer::BatchNormalizationLayer() : Layer()
{
}

BatchNormalizationLayer::BatchNormalizationLayer(const Index& new_inputs_number) : Layer()
{
    set(new_inputs_number);
    layer_type = Type::BatchNormalization;
}


void BatchNormalizationLayer::set(const Index& new_inputs_number)
{
    normalization_weights.resize(2, new_inputs_number);

    set_parameters_random();

    set_default();
}

void BatchNormalizationLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    for(Index i = 0; i < normalization_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        normalization_weights(i) = minimum + (maximum - minimum)*random;
    }
}


void BatchNormalizationLayer::set_default()
{
    layer_name = "batch_normalization_layer";

    display = true;

    layer_type = Type::BatchNormalization;
}


Index BatchNormalizationLayer::get_inputs_number() const
{
    return normalization_weights.dimension(1);
}

void BatchNormalizationLayer::perform_normalization(const Tensor<type, 2>& inputs, BatchNormalizationLayerForwardPropagation* batch_norm_forward_propagation) const
{
    const int rows_number = static_cast<int>(inputs.dimension(0));
    const int cols_number = static_cast<int>(inputs.dimension(1));

    Tensor<float,1>batch_size(cols_number);
    batch_size.setConstant(rows_number);

    const Eigen::array<ptrdiff_t, 1> dims = {0};
    const Eigen::array<Index, 2> dims_2D = {1, cols_number};
    const Eigen::array<Index, 2> bcast({rows_number, 1});

    batch_norm_forward_propagation->mean = inputs.mean(dims);
    batch_norm_forward_propagation->variance = ((batch_norm_forward_propagation->mean.reshape(dims_2D).broadcast(bcast) - inputs).square()).sum(dims)/batch_size;

    cout<<"batch_norm_forward_propagation->mean"<<endl;
    cout<<batch_norm_forward_propagation->mean<<endl;
    cout<<"batch_norm_forward_propagation->variance"<<endl;
    cout<<batch_norm_forward_propagation->variance<<endl;

//    Tensor<type,2>inputs_normalized(rows_number,cols_number);
}


void BatchNormalizationLayer::forward_propagate(type* inputs_data,
                                        const Tensor<Index,1>& inputs_dimensions,
                                        LayerForwardPropagation* forward_propagation)
{
    BatchNormalizationLayerForwardPropagation* batch_norm_layer_forward_propagation
            = static_cast<BatchNormalizationLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    perform_normalization(inputs, batch_norm_layer_forward_propagation);

//    calculate_combinations(inputs,
//                           biases,
//                           synaptic_weights,
//                           perceptron_layer_forward_propagation->combinations.data());

//    const Tensor<Index, 1> combinations_dimensions = get_dimensions(perceptron_layer_forward_propagation->combinations);
//    const Tensor<Index, 1> derivatives_dimensions = get_dimensions(perceptron_layer_forward_propagation->activations_derivatives);

}

void BatchNormalizationLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                                     const Tensor<type, 2>& weights,
                                                     Tensor<type, 2>& outputs)
{


}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
