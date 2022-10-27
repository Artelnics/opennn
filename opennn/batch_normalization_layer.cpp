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
    synaptic_weights.resize(2, new_inputs_number);

    set_parameters_random();

    set_default();
}

void BatchNormalizationLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    for(Index i = 0; i < synaptic_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        synaptic_weights(i) = minimum + (maximum - minimum)*random;
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
    return synaptic_weights.dimension(1);
}

Tensor<type, 2> BatchNormalizationLayer::perform_inputs_normalization(const Tensor<type, 2>& inputs, BatchNormalizationLayerForwardPropagation* batch_norm_forward_propagation) const
{
    const int rows_number = static_cast<int>(inputs.dimension(0));
    const int cols_number = static_cast<int>(inputs.dimension(1));

    Tensor<float,1>batch_size(cols_number);
    Tensor<float,2>epsilon(rows_number, cols_number);

    batch_size.setConstant(rows_number);
    epsilon.setConstant(numeric_limits<type>::epsilon());

    const Eigen::array<ptrdiff_t, 1> dims = {0};
    const Eigen::array<Index, 2> dims_2D = {1, cols_number};
    const Eigen::array<Index, 2> bcast({rows_number, 1});

    batch_norm_forward_propagation->mean = inputs.mean(dims);
    batch_norm_forward_propagation->variance = ((batch_norm_forward_propagation->mean.reshape(dims_2D).broadcast(bcast) - inputs).square()).sum(dims)/batch_size;

    Tensor<type,2> outputs = (inputs - inputs.mean(dims).reshape(dims_2D).broadcast(bcast))/
          (batch_norm_forward_propagation->variance.reshape(dims_2D).broadcast(bcast) + epsilon).sqrt();

    return outputs;
}


void BatchNormalizationLayer::forward_propagate(type* inputs_data,
                                        const Tensor<Index,1>& inputs_dimensions,
                                        LayerForwardPropagation* forward_propagation)
{
    BatchNormalizationLayerForwardPropagation* batch_norm_layer_forward_propagation
            = static_cast<BatchNormalizationLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    Tensor<type, 2> inputs_normalized = perform_inputs_normalization(inputs, batch_norm_layer_forward_propagation);

    calculate_combinations(inputs_normalized,
                           synaptic_weights,
                           batch_norm_layer_forward_propagation->outputs_data);

}

void BatchNormalizationLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                                     const Tensor<type, 2>& weights,
                                                     type* outputs)
{
    const Index batch_number = inputs.dimension(0);
    const Index input_number = inputs.dimension(1);

    float* subtensor_inputs = nullptr;
    float* subtensor_weights = nullptr;

    subtensor_inputs = (float*) malloc(static_cast<size_t>(inputs.size()*sizeof(type)));
    subtensor_weights = (float*) malloc(static_cast<size_t>(weights.size()*sizeof(type)));

    for(int i = 0; i<input_number; i++)
    {
        memcpy(subtensor_inputs, inputs.data() + i* batch_number , static_cast<size_t>(batch_number*sizeof(float)));
        memcpy(subtensor_weights, weights.data() + i* 2 , static_cast<size_t>(2*sizeof(float)));

        for(int j=0; j<batch_number; j++)
        {
            outputs[i*batch_number + j] = subtensor_inputs[j] * subtensor_weights[0] + subtensor_weights[1];
        }
    }
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
