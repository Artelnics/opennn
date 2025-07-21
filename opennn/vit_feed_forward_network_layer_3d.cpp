//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I T  F E E D  F O R W A R D  N E T W O R K   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "vit_feed_forward_network_layer_3d.h"
#include "tensors.h"
#include "strings_utilities.h"

namespace opennn
{

    VitFeedForwardNetworkLayer3D::VitFeedForwardNetworkLayer3D(const Index& new_tokens_number,
        const Index& new_inputs_depth,
        const Index& new_neurons_number,
        const VitFeedForwardNetworkLayer3D::ActivationFunction& new_activation_function,
        const string& new_name) : Layer()
    {
        set(new_tokens_number, new_inputs_depth, new_neurons_number, new_activation_function, new_name);
    }


    Index VitFeedForwardNetworkLayer3D::get_tokens_number() const
    {
        return tokens_number;
    }


    Index VitFeedForwardNetworkLayer3D::get_inputs_depth() const
    {
        return synaptic_weights1.dimension(0);
    }


    void VitFeedForwardNetworkLayer3D::set_dropout_rate(const type& new_dropout_rate)
    {
        dropout_rate = new_dropout_rate;
    }


    Index VitFeedForwardNetworkLayer3D::get_neurons_number() const
    {
        return biases1.size();
    }


    dimensions VitFeedForwardNetworkLayer3D::get_output_dimensions() const
    {
        return { tokens_number, biases2.size() };
    }


    Index VitFeedForwardNetworkLayer3D::get_parameters_number() const
    {
        
        return  synaptic_weights1.size() + biases1.size() + synaptic_weights2.size() + biases2.size() ;
    }


    type VitFeedForwardNetworkLayer3D::get_dropout_rate() const
    {
        return dropout_rate;
    }

    
    Tensor<type, 1> VitFeedForwardNetworkLayer3D::get_parameters() const
    {
        Tensor<type, 1> parameters(synaptic_weights1.size() + biases1.size() + synaptic_weights2.size() + biases2.size());

        memcpy(parameters.data(), synaptic_weights1.data(), synaptic_weights1.size() * sizeof(type));

        memcpy(parameters.data() + synaptic_weights1.size(), biases1.data(), biases1.size() * sizeof(type));

        memcpy(parameters.data() + synaptic_weights1.size() + biases1.size(), synaptic_weights2.data(), synaptic_weights2.size() * sizeof(type));

        memcpy(parameters.data() + synaptic_weights1.size() + biases1.size() + synaptic_weights2.size(), biases2.data(), biases2.size() * sizeof(type));

        return parameters;
    }
    

    const VitFeedForwardNetworkLayer3D::ActivationFunction& VitFeedForwardNetworkLayer3D::get_activation_function() const
    {
        return activation_function;
    }


    string VitFeedForwardNetworkLayer3D::get_activation_function_string() const
    {
        switch (activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            return "HyperbolicTangent";

        case ActivationFunction::Linear:
            return "Linear";

        case ActivationFunction::RectifiedLinear:
            return "RectifiedLinear";
        }

        return string();
    }

    
    void VitFeedForwardNetworkLayer3D::set(const Index& new_tokens_number,
        const Index& new_inputs_depth,
        const Index& new_neurons_number,
        const VitFeedForwardNetworkLayer3D::ActivationFunction& new_activation_function,
        const string& new_name)
    {
        tokens_number = new_tokens_number;

        biases1.resize(new_neurons_number);

        biases2.resize(new_inputs_depth);

        synaptic_weights1.resize(new_inputs_depth, new_neurons_number);

        synaptic_weights2.resize(new_neurons_number, new_inputs_depth);

        set_parameters_glorot();

        activation_function = new_activation_function;

        name = new_name;

        layer_type = Type::VitFeedForwardNetwork3D;

        dropout_rate = 0;
    }


    void VitFeedForwardNetworkLayer3D::set_tokens_number(Index new_tokens_number)
    {
        tokens_number = new_tokens_number;
    }


    void VitFeedForwardNetworkLayer3D::set_input_dimensions(const dimensions& new_input_dimensions)
    {
        /*
            inputs_number = new_inputs_number;
        */
    }


    void VitFeedForwardNetworkLayer3D::set_inputs_depth(const Index& new_inputs_depth)
    {
        const Index neurons_number = get_neurons_number();

        biases1.resize(neurons_number);

        biases2.resize(new_inputs_depth);

        synaptic_weights1.resize(new_inputs_depth, neurons_number);

        synaptic_weights2.resize(neurons_number, new_inputs_depth);
    }


    void VitFeedForwardNetworkLayer3D::set_output_dimensions(const dimensions& new_output_dimensions)
    {
        const Index inputs_depth = get_inputs_depth();
        const Index neurons_number = new_output_dimensions[0];

        biases1.resize(neurons_number);

        biases2.resize(inputs_depth);

        synaptic_weights1.resize(inputs_depth, neurons_number);

        synaptic_weights2.resize(neurons_number, inputs_depth);
    }


    void VitFeedForwardNetworkLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
    {
#pragma omp parallel sections
        {
#pragma omp section
            memcpy(synaptic_weights1.data(), new_parameters.data() + index, synaptic_weights1.size() * sizeof(type));

#pragma omp section
            memcpy(biases1.data(), new_parameters.data() + index + synaptic_weights1.size(), biases1.size() * sizeof(type));

#pragma omp section
            memcpy(synaptic_weights2.data(), new_parameters.data() + index + synaptic_weights1.size() + biases1.size(), synaptic_weights2.size() * sizeof(type));

#pragma omp section
            memcpy(biases2.data(), new_parameters.data() + index + synaptic_weights1.size() + biases1.size() + synaptic_weights2.size(), biases2.size() * sizeof(type));
        }
    }


    void VitFeedForwardNetworkLayer3D::set_activation_function(const VitFeedForwardNetworkLayer3D::ActivationFunction& new_activation_function)
    {
        activation_function = new_activation_function;
    }


    void VitFeedForwardNetworkLayer3D::set_activation_function(const string& new_activation_function_name)
    {
        if (new_activation_function_name == "HyperbolicTangent")
            activation_function = ActivationFunction::HyperbolicTangent;
        else if (new_activation_function_name == "Linear")
            activation_function = ActivationFunction::Linear;
        else if (new_activation_function_name == "RectifiedLinear")
            activation_function = ActivationFunction::RectifiedLinear;
        else
            throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
    }


    void VitFeedForwardNetworkLayer3D::set_parameters_constant(const type& value)
    {
        biases1.setZero();

        biases2.setZero();

        synaptic_weights1.setConstant(value);

        synaptic_weights2.setConstant(value);
    }


    void VitFeedForwardNetworkLayer3D::set_parameters_random()
    {
        set_random(biases1);

        set_random(biases2);

        set_random(synaptic_weights1);

        set_random(synaptic_weights2);
    }


    void VitFeedForwardNetworkLayer3D::set_parameters_glorot()
    {
        biases1.setZero();

        biases2.setZero();

        const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

        const type minimum = -limit;
        const type maximum = limit;

#pragma omp parallel for
        for (Index i = 0; i < synaptic_weights1.size(); i++)
            synaptic_weights1(i) = get_random_type(minimum, maximum);

#pragma omp parallel for
        for (Index i = 0; i < synaptic_weights2.size(); i++)
            synaptic_weights2(i) = get_random_type(minimum, maximum);
    }


    void VitFeedForwardNetworkLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
        Tensor<type, 3>& combinations) const
    {
        if (inputs.dimension(2) == synaptic_weights1.dimension(0)) {
            combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights1, contraction_indices);
            sum_matrices(thread_pool_device.get(), biases1, combinations);
        }
        else if (inputs.dimension(2) == synaptic_weights2.dimension(0)) {
            combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights2, contraction_indices);
            sum_matrices(thread_pool_device.get(), biases2, combinations);
        }
        else
            cout << "dimension error feed forward network" << endl;

    }


    void VitFeedForwardNetworkLayer3D::dropout(Tensor<type, 3>& outputs) const
    {
        const type scaling_factor = type(1) / (type(1) - dropout_rate);

#pragma omp parallel for

        for (Index i = 0; i < outputs.size(); i++)
            outputs(i) = (get_random_type(type(0), type(1)) < dropout_rate)
            ? 0
            : outputs(i) * scaling_factor;
    }


    void VitFeedForwardNetworkLayer3D::calculate_activations(Tensor<type, 3>& activations, Tensor<type, 3>& activation_derivatives) const
    {
        switch (activation_function)
        {
        case ActivationFunction::Linear: linear(activations, activation_derivatives); return;

        case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

        case ActivationFunction::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

        default: return;
        }
    }


    void VitFeedForwardNetworkLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
        const bool& is_training)
    {
        const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

        VitFeedForwardNetworkLayer3DForwardPropagation* vit_feed_forward_network_layer_3d_forward_propagation =
            static_cast<VitFeedForwardNetworkLayer3DForwardPropagation*>(layer_forward_propagation.get());

        Tensor<type, 3>& outputs = vit_feed_forward_network_layer_3d_forward_propagation->outputs;
        Tensor<type, 3> activations_inputs(inputs.dimension(0), inputs.dimension(1), get_neurons_number());

        calculate_combinations(inputs,
            activations_inputs);

        if (is_training) {
            if (dropout_rate > type(0)) {
                //cout << "ffn dropout" << endl;
                dropout(outputs);
            }
        }

        Tensor<type, 3>& activation_derivatives = vit_feed_forward_network_layer_3d_forward_propagation->activation_derivatives;

        calculate_activations(activations_inputs, activation_derivatives);

        vit_feed_forward_network_layer_3d_forward_propagation->set_transformed_inputs(activations_inputs);
        calculate_combinations(activations_inputs,
            outputs);

        //cout << "Feed Forward Network layer outputs dimensions: " << outputs.dimensions() << endl;
        //cout << "Feed Forward Network layer outputs:" << endl;
        //cout << outputs << endl;
    }


    void VitFeedForwardNetworkLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
        const vector<pair<type*, dimensions>>& delta_pairs,
        unique_ptr<LayerForwardPropagation>& forward_propagation,
        unique_ptr<LayerBackPropagation>& back_propagation) const
    {
        const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);
        const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

        const VitFeedForwardNetworkLayer3DForwardPropagation* vit_feed_forward_network_layer_3d_forward_propagation =
            static_cast<VitFeedForwardNetworkLayer3DForwardPropagation*>(forward_propagation.get());

        const Tensor<type, 3>& activation_derivatives = vit_feed_forward_network_layer_3d_forward_propagation->activation_derivatives;

        VitFeedForwardNetworkLayer3DBackPropagation* vit_feed_forward_network_layer_3d_back_propagation =
            static_cast<VitFeedForwardNetworkLayer3DBackPropagation*>(back_propagation.get());

        Tensor<type, 3>& combination_derivatives1 = vit_feed_forward_network_layer_3d_back_propagation->combination_derivatives1;
        Tensor<type, 1>& bias_derivatives1 = vit_feed_forward_network_layer_3d_back_propagation->bias_derivatives1;
        Tensor<type, 2>& synaptic_weight_derivatives1 = vit_feed_forward_network_layer_3d_back_propagation->synaptic_weight_derivatives1;

        Tensor<type, 3>& activation_deltas = vit_feed_forward_network_layer_3d_back_propagation->activation_deltas;

        Tensor<type, 3>& combination_derivatives2 = vit_feed_forward_network_layer_3d_back_propagation->combination_derivatives2;
        Tensor<type, 1>& bias_derivatives2 = vit_feed_forward_network_layer_3d_back_propagation->bias_derivatives2;
        Tensor<type, 2>& synaptic_weight_derivatives2 = vit_feed_forward_network_layer_3d_back_propagation->synaptic_weight_derivatives2;

        Tensor<type, 3>& input_derivatives = vit_feed_forward_network_layer_3d_back_propagation->input_derivatives;

        combination_derivatives2.device(*thread_pool_device) = deltas;

        bias_derivatives2.device(*thread_pool_device) =
            combination_derivatives2.sum(sum_dimensions);

        activation_deltas = vit_feed_forward_network_layer_3d_forward_propagation->transformed_inputs; // entrada intermedia antes de la segunda combinación
        synaptic_weight_derivatives2.device(*thread_pool_device) =
            activation_deltas.contract(combination_derivatives2, double_contraction_indices);

        Tensor<type, 3> backprop_through_second_layer;
        backprop_through_second_layer.resize(activation_derivatives.dimensions());
        backprop_through_second_layer.device(*thread_pool_device) =
            combination_derivatives2.contract(synaptic_weights2, single_contraction_indices);

        combination_derivatives1.device(*thread_pool_device) =
            backprop_through_second_layer * activation_derivatives;

        bias_derivatives1.device(*thread_pool_device) =
            combination_derivatives1.sum(sum_dimensions);

        synaptic_weight_derivatives1.device(*thread_pool_device) =
            inputs.contract(combination_derivatives1, double_contraction_indices);

        input_derivatives.device(*thread_pool_device) =
            combination_derivatives1.contract(synaptic_weights1, single_contraction_indices);

        //cout << "FFN layer synaptic_weight_derivatives1 size: " << synaptic_weight_derivatives1.dimensions() << endl;
        //cout << "FFN layer synaptic_weight_derivatives1: " << endl;
        
    }


    void VitFeedForwardNetworkLayer3D::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
    {
        TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

        for (Index i = 1; i < Index(delta_pairs.size()); i++)
            deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
    }

    
    void VitFeedForwardNetworkLayer3D::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
        const Index& index,
        Tensor<type, 1>& gradient) const
    {
        const Index biases1_number = biases1.size();
        const Index weights1_number = synaptic_weights1.size();
        const Index biases2_number = biases2.size();
        const Index weights2_number = synaptic_weights2.size();

        VitFeedForwardNetworkLayer3DBackPropagation* vit_feed_forward_network_layer_3d_back_propagation =
            static_cast<VitFeedForwardNetworkLayer3DBackPropagation*>(back_propagation.get());

        const type* synaptic_weight_derivatives1_data = vit_feed_forward_network_layer_3d_back_propagation->synaptic_weight_derivatives1.data();
        const type* bias_derivatives1_data = vit_feed_forward_network_layer_3d_back_propagation->bias_derivatives1.data();
        const type* synaptic_weight_derivatives2_data = vit_feed_forward_network_layer_3d_back_propagation->synaptic_weight_derivatives2.data();
        const type* bias_derivatives2_data = vit_feed_forward_network_layer_3d_back_propagation->bias_derivatives2.data();

        type* gradient_data = gradient.data();

#pragma omp parallel sections
        {
#pragma omp section
            memcpy(gradient_data + index, synaptic_weight_derivatives1_data, weights1_number * sizeof(type));

#pragma omp section
            memcpy(gradient_data + index + weights1_number, bias_derivatives1_data, biases1_number * sizeof(type));

#pragma omp section
            memcpy(gradient_data + index + weights1_number + biases1_number, synaptic_weight_derivatives2_data, weights2_number * sizeof(type));

#pragma omp section
            memcpy(gradient_data + index + weights1_number + biases1_number + weights2_number,
                bias_derivatives2_data, biases2_number * sizeof(type));
        }
    }
    


    
    void VitFeedForwardNetworkLayer3D::from_XML(const XMLDocument& document)
    {
        const XMLElement* feed_forward_network_layer_element = document.FirstChildElement("VitFeedForwardNetwork3D");

        if (!feed_forward_network_layer_element)
            throw runtime_error("FeedForwardNetwork element is nullptr.\n");

        set_name(read_xml_string(feed_forward_network_layer_element, "Name"));
        set_tokens_number(read_xml_index(feed_forward_network_layer_element, "InputsNumber"));
        set_inputs_depth(read_xml_index(feed_forward_network_layer_element, "InputsDepth"));
        set_output_dimensions({ read_xml_index(feed_forward_network_layer_element, "NeuronsNumber") });
        set_activation_function(read_xml_string(feed_forward_network_layer_element, "ActivationFunction"));
        set_parameters(to_type_vector(read_xml_string(feed_forward_network_layer_element, "Parameters"), " "));

    }


    void VitFeedForwardNetworkLayer3D::to_XML(XMLPrinter& printer) const
    {
        printer.OpenElement("VitFeedForwardNetwork3D");

        add_xml_element(printer, "Name", name);
        add_xml_element(printer, "InputsNumber", to_string(get_tokens_number()));
        add_xml_element(printer, "InputsDepth", to_string(get_inputs_depth()));
        add_xml_element(printer, "NeuronsNumber", to_string(get_neurons_number()));
        add_xml_element(printer, "ActivationFunction", get_activation_function_string());
        add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

        printer.CloseElement();
    }




    void VitFeedForwardNetworkLayer3DForwardPropagation::set_transformed_inputs(const Tensor<type, 3>& new_transformed_inputs)
    {
        transformed_inputs = new_transformed_inputs;
    }


    void VitFeedForwardNetworkLayer3DForwardPropagation::print() const
    {
        cout << "Outputs:" << endl
            << outputs << endl
            << "Activation derivatives:" << endl
            << activation_derivatives << endl;
    }


    VitFeedForwardNetworkLayer3DForwardPropagation::VitFeedForwardNetworkLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    pair<type*, dimensions> VitFeedForwardNetworkLayer3DForwardPropagation::get_outputs_pair() const
    {
        VitFeedForwardNetworkLayer3D* vit_feed_forward_network_layer_3d = static_cast<VitFeedForwardNetworkLayer3D*>(layer);

        const Index inputs_depth = vit_feed_forward_network_layer_3d->get_inputs_depth();

        const Index tokens_number = vit_feed_forward_network_layer_3d->get_tokens_number();

        return { (type*)outputs.data(), { batch_samples_number, tokens_number, inputs_depth } };
    }


    void VitFeedForwardNetworkLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
    {
        layer = new_layer;

        VitFeedForwardNetworkLayer3D* vit_feed_forward_network_layer_3d = static_cast<VitFeedForwardNetworkLayer3D*>(layer);

        batch_samples_number = new_batch_samples_number;

        const Index input_depth = vit_feed_forward_network_layer_3d->get_inputs_depth();

        const Index tokens_number = vit_feed_forward_network_layer_3d->get_tokens_number();

        const Index neurons_number = vit_feed_forward_network_layer_3d->get_neurons_number();

        outputs.resize(batch_samples_number, tokens_number, input_depth);

        activation_derivatives.resize(batch_samples_number, tokens_number, neurons_number);
    }


    void VitFeedForwardNetworkLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
    {
        layer = new_layer;

        VitFeedForwardNetworkLayer3D* vit_feed_forward_network_layer_3d = static_cast<VitFeedForwardNetworkLayer3D*>(layer);

        batch_samples_number = new_batch_samples_number;

        const Index input_depth = vit_feed_forward_network_layer_3d->get_inputs_depth();
        const Index neurons_number = vit_feed_forward_network_layer_3d->get_neurons_number();
        const Index tokens_number = vit_feed_forward_network_layer_3d->get_tokens_number();

        synaptic_weight_derivatives1.resize(input_depth, neurons_number);
        synaptic_weight_derivatives2.resize(neurons_number, input_depth);

        bias_derivatives1.resize(neurons_number);
        bias_derivatives2.resize(input_depth);

        combination_derivatives1.resize(batch_samples_number, tokens_number, neurons_number);
        combination_derivatives2.resize(batch_samples_number, tokens_number, input_depth);

        input_derivatives.resize(batch_samples_number, tokens_number, input_depth);
    }


    void VitFeedForwardNetworkLayer3DBackPropagation::print() const
    {
        cout << "Biases derivatives 1:" << endl
            << bias_derivatives1 << endl
            << "Synaptic weights derivatives 1:" << endl
            << synaptic_weight_derivatives1 << endl
            << "Biases derivatives 2:" << endl
            << bias_derivatives2 << endl
            << "Synaptic weights derivatives 2:" << endl
            << synaptic_weight_derivatives2 << endl;
    }


    VitFeedForwardNetworkLayer3DBackPropagation::VitFeedForwardNetworkLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    vector<pair<type*, dimensions>> VitFeedForwardNetworkLayer3DBackPropagation::get_input_derivative_pairs() const
    {
        VitFeedForwardNetworkLayer3D* vit_feed_forward_network_layer_3d = static_cast<VitFeedForwardNetworkLayer3D*>(layer);

        const Index tokens_number = vit_feed_forward_network_layer_3d->get_tokens_number();
        const Index inputs_depth = vit_feed_forward_network_layer_3d->get_inputs_depth();

        return { {(type*)(input_derivatives.data()),
                {batch_samples_number, tokens_number, inputs_depth}} };
    }


}

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
