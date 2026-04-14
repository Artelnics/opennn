//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "normalization_layer_3d.h"
#include "neural_network.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Normalization3d::Normalization3d(const Shape& new_input_shape,
                                 const string& new_name) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_name);
}

Shape Normalization3d::get_input_shape() const
{
    return { sequence_length, embedding_dimension };
}

Shape Normalization3d::get_output_shape() const
{
    return { sequence_length, embedding_dimension };
}

vector<Shape> Normalization3d::get_parameter_shapes() const
{
    return {{embedding_dimension},
            {embedding_dimension}};
}

void Normalization3d::set(const Index new_sequence_length,
                          Index new_embedding_dimension,
                          const string& new_label)
{
    sequence_length = new_sequence_length;
    embedding_dimension = new_embedding_dimension;

    label = new_label;
    name = "Normalization3d";
    layer_type = LayerType::Normalization3d;
}

void Normalization3d::set_parameters_random()
{
    VectorMap(parameters[Gammas].data, parameters[Gammas].size()).setOnes();

    VectorMap(parameters[Betas].data, parameters[Betas].size()).setZero();
}

void Normalization3d::set_parameters_glorot()
{
    set_parameters_random();
}


void Normalization3d::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const Index batch_size = forward_propagation.batch_size;
    const Index E = embedding_dimension;

    const TensorView& input_view = forward_propagation.views[layer][Inputs][0];
    TensorView& means_view = forward_propagation.views[layer][Means][0];
    TensorView& stddevs_view = forward_propagation.views[layer][StandardDeviations][0];
    TensorView& output_view = forward_propagation.views[layer][Outputs][0];

    const TensorView& gammas = parameters[Gammas];
    const TensorView& betas = parameters[Betas];

#ifndef CUDA
    TensorView& normalized_view = forward_propagation.views[layer][NormalizedInputs][0];

    const TensorMap3 inputs(input_view.data, batch_size, sequence_length, E);
    TensorMap2 means(means_view.data, batch_size, sequence_length);
    TensorMap2 standard_deviations(stddevs_view.data, batch_size, sequence_length);
    TensorMap3 normalized_inputs(normalized_view.data, batch_size, sequence_length, E);
    TensorMap3 outputs(output_view.data, batch_size, sequence_length, E);

    const array<Index, 3> reshape_dims({batch_size, sequence_length, 1});
    const array<Index, 3> broadcast_dims({1, 1, E});

    means = inputs.mean(array<Index, 1>({2}));
    normalized_inputs = inputs - means.reshape(reshape_dims).broadcast(broadcast_dims);

    auto variance = normalized_inputs.square().mean(array<Index, 1>({2}));
    standard_deviations = (variance + EPSILON).sqrt();

    normalized_inputs = normalized_inputs / standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims);

    TensorMap1 gamma_map(gammas.data, E);
    TensorMap1 beta_map(betas.data, E);

    outputs = normalized_inputs
              * gamma_map.reshape(array<Index, 3>({1, 1, E})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}))
              + beta_map.reshape(array<Index, 3>({1, 1, E})).broadcast(array<Index, 3>({batch_size, sequence_length, 1}));
#else
    const int N = static_cast<int>(batch_size * sequence_length);
    const int D = static_cast<int>(E);

    layernorm_forward_cuda(N, D,
        input_view.data, output_view.data,
        means_view.data, stddevs_view.data,
        gammas.data, betas.data, EPSILON);
#endif
}


void Normalization3d::back_propagate(ForwardPropagation& forward_propagation,
                                     BackPropagation& back_propagation,
                                     size_t layer) const
{
    const Index batch_size = forward_propagation.batch_size;
    const Index E = embedding_dimension;

    const TensorView& input_view = forward_propagation.views[layer][Inputs][0];
    const TensorView& means_view = forward_propagation.views[layer][Means][0];
    const TensorView& stddevs_view = forward_propagation.views[layer][StandardDeviations][0];
    const TensorView& delta_view = back_propagation.backward_views[layer][OutputGradients][0];

    TensorView& gamma_grad_view = back_propagation.gradient_views[layer][Gammas];
    TensorView& beta_grad_view = back_propagation.gradient_views[layer][Betas];

    const TensorView& gammas = parameters[Gammas];

#ifndef CUDA
    const TensorView& normalized_view = forward_propagation.views[layer][NormalizedInputs][0];

    const TensorMap2 standard_deviations(stddevs_view.data, batch_size, sequence_length);
    const TensorMap3 X_hat(normalized_view.data, batch_size, sequence_length, E);
    const TensorMap3 output_gradients(delta_view.data, batch_size, sequence_length, E);

    TensorMap1 dGamma(gamma_grad_view.data, E);
    TensorMap1 dBeta(beta_grad_view.data, E);

    dGamma = (output_gradients * X_hat).sum(array<Index, 2>({0, 1}));
    dBeta = output_gradients.sum(array<Index, 2>({0, 1}));

    if(!is_first_layer)
    {
        TensorView& input_grad_view = back_propagation.backward_views[layer][InputGradients][0];
        TensorMap3 dX(input_grad_view.data, batch_size, sequence_length, E);

        TensorMap1 gamma_map(gammas.data, E);

        auto gamma_bcast = gamma_map.reshape(array<Index, 3>({1, 1, E}))
                               .broadcast(array<Index, 3>({batch_size, sequence_length, 1}));

        Tensor2 sum_dY = output_gradients.sum(array<Index, 1>({2}));
        Tensor2 sum_dY_xhat = (output_gradients * X_hat).sum(array<Index, 1>({2}));

        auto sum_dY_bcast = sum_dY.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                                .broadcast(array<Index, 3>({1, 1, E}));
        auto sum_dY_xhat_bcast = sum_dY_xhat.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                                     .broadcast(array<Index, 3>({1, 1, E}));
        auto std_dev_bcast = standard_deviations.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                                 .broadcast(array<Index, 3>({1, 1, E}));

        const type inv_E = type(1.0) / static_cast<type>(E);
        dX = gamma_bcast * (output_gradients - sum_dY_bcast * inv_E - X_hat * sum_dY_xhat_bcast * inv_E) / std_dev_bcast;
    }
#else
    const int N = static_cast<int>(batch_size * sequence_length);
    const int D = static_cast<int>(E);

    TensorView& input_grad_view = back_propagation.backward_views[layer][InputGradients][0];

    layernorm_backward_cuda(N, D,
        delta_view.data, input_view.data,
        means_view.data, stddevs_view.data,
        gammas.data,
        input_grad_view.data,
        gamma_grad_view.data, beta_grad_view.data);
#endif
}


void Normalization3d::from_XML(const XmlDocument& document)
{
    const XmlElement* element = get_xml_root(document, "Normalization3d");

    const string new_name = read_xml_string(element, "Label");
    const Index new_sequence_length = read_xml_index(element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(element, "EmbeddingDimension");

    set(new_sequence_length, new_embedding_dimension, new_name);
}

void Normalization3d::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Normalization3d");
    write_xml_properties(printer, {
        {"Label", label},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingDimension", to_string(get_embedding_dimension())}
    });
    printer.close_element();
}

#ifdef CUDA
    // @todo CUDA path
#endif

REGISTER(Layer, Normalization3d, "Normalization3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
