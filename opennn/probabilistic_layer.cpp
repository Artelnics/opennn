//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "strings_utilities.h"
#include "probabilistic_layer.h"

namespace opennn
{

ProbabilisticLayer::ProbabilisticLayer(const dimensions& new_input_dimensions,
                                       const dimensions& new_output_dimensions,
                                       const string& new_name) : Layer()
{
    set(new_input_dimensions, 
        new_output_dimensions, 
        new_name);

}


dimensions ProbabilisticLayer::get_input_dimensions() const
{
    return { synaptic_weights.dimension(0) };
}


dimensions ProbabilisticLayer::get_output_dimensions() const
{
    return { biases.size() };
}


const type& ProbabilisticLayer::get_decision_threshold() const
{
    return decision_threshold;
}


const ProbabilisticLayer::ActivationFunction& ProbabilisticLayer::get_activation_function() const
{
    return activation_function;
}


string ProbabilisticLayer::get_activation_function_string() const
{
    if(activation_function == ActivationFunction::Binary)
        return "Binary";
    else if(activation_function == ActivationFunction::Logistic)
        return "Logistic";
    else if(activation_function == ActivationFunction::Competitive)
        return "Competitive";
    else if(activation_function == ActivationFunction::Softmax)
        return "Softmax";
    else
        throw runtime_error("Unknown probabilistic method.\n");
}


Index ProbabilisticLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


Tensor<type, 1> ProbabilisticLayer::get_parameters() const
{
    const Index synaptic_weights_number = synaptic_weights.size();
    const Index biases_number = biases.size();

    Tensor<type, 1> parameters(synaptic_weights_number + biases_number);

    memcpy(parameters.data(), synaptic_weights.data(), synaptic_weights_number*sizeof(type));

    memcpy(parameters.data() + synaptic_weights_number, biases.data(), biases_number*sizeof(type));

    return parameters;
}


void ProbabilisticLayer::set(const dimensions& new_input_dimensions, 
                             const dimensions& new_output_dimensions, 
                             const string& new_name)
{
    if (new_input_dimensions.size() != 1)
        throw runtime_error("Input dimensions rank is not 1");

    if (new_output_dimensions.size() != 1)
        throw runtime_error("Output dimensions rank is not 1");

    biases.resize(new_output_dimensions[0]);

    synaptic_weights.resize(new_input_dimensions[0], new_output_dimensions[0]);

    set_parameters_random();

    layer_type = Layer::Type::Probabilistic;

    new_output_dimensions[0] == 1
        ? activation_function = ActivationFunction::Logistic
        : activation_function = ActivationFunction::Softmax;

    decision_threshold = type(0.5);

    name = new_name;
}



void ProbabilisticLayer::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const dimensions output_dimensions = get_output_dimensions();

    biases.resize(output_dimensions[0]);

    synaptic_weights.resize(new_input_dimensions[0], output_dimensions[0]);
}


void ProbabilisticLayer::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const dimensions input_dimensions = get_input_dimensions();

    biases.resize(new_output_dimensions[0]);

    synaptic_weights.resize(input_dimensions[0], new_output_dimensions[0]);
}


void ProbabilisticLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    memcpy(synaptic_weights.data(), new_parameters.data() + index, synaptic_weights_number*sizeof(type));

    memcpy(biases.data(), new_parameters.data() + index + synaptic_weights_number, biases_number*sizeof(type));
}


void ProbabilisticLayer::set_decision_threshold(const type& new_decision_threshold)
{
    decision_threshold = new_decision_threshold;
}


void ProbabilisticLayer::set_activation_function(const ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void ProbabilisticLayer::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Binary")
        set_activation_function(ActivationFunction::Binary);
    else if(new_activation_function == "Logistic")
        set_activation_function(ActivationFunction::Logistic);
    else if(new_activation_function == "Competitive")
        set_activation_function(ActivationFunction::Competitive);
    else if(new_activation_function == "Softmax")
        set_activation_function(ActivationFunction::Softmax);
    else
        throw runtime_error("Unknown probabilistic method: " + new_activation_function + ".\n");
}


void ProbabilisticLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void ProbabilisticLayer::set_parameters_random()
{
    set_random(biases);
    
    set_random(synaptic_weights);
}


void ProbabilisticLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                                Tensor<type, 2>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, A_B);

    sum_columns(thread_pool_device.get(), biases, combinations);
}


void ProbabilisticLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                           unique_ptr<LayerForwardPropagation>& forward_propagation,
                                           const bool& is_training)
{
    const Index outputs_number = get_outputs_number();

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation =
        static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.get());

    Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    if (outputs_number == 1 && !is_training)
    {
        logistic(outputs, empty);
    }
    else if (outputs_number == 1 && is_training)
    {
        Tensor<type, 2>& activation_derivatives = probabilistic_layer_forward_propagation->activation_derivatives;

        logistic(outputs, activation_derivatives);
    }
    else if (outputs_number > 1)
    {
        softmax(outputs);
    }
    else
    {
		throw runtime_error("Unknown case in forward propagation.\n");
    }
}


void ProbabilisticLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index outputs_number = get_outputs_number();
    
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

    // Forward propagation

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation =
        static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.get());

    // Back propagation

    ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
            static_cast<ProbabilisticLayerBackPropagation*>(back_propagation.get());
    
    Tensor<type, 2>& input_derivatives = probabilistic_layer_back_propagation->input_derivatives;

    Tensor<type, 2>& combination_derivatives = probabilistic_layer_back_propagation->combination_derivatives;

    if(outputs_number == 1)
    {
        const Tensor<type, 2>& activation_derivatives = probabilistic_layer_forward_propagation->activation_derivatives;

        combination_derivatives.device(*thread_pool_device) = deltas * activation_derivatives;
    }
    else
    {
        combination_derivatives.device(*thread_pool_device) = deltas;
    }

    Tensor<type, 1>& bias_derivatives = probabilistic_layer_back_propagation->bias_derivatives;

    Tensor<type, 2>& synaptic_weight_derivatives = probabilistic_layer_back_propagation->synaptic_weight_derivatives;

    synaptic_weight_derivatives.device(*thread_pool_device) = inputs.contract(combination_derivatives, AT_B);

    bias_derivatives.device(*thread_pool_device) = combination_derivatives.sum(sum_dimensions);

    input_derivatives.device(*thread_pool_device) = combination_derivatives.contract(synaptic_weights, A_BT);
}


void ProbabilisticLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                         const Index& index,
                                         Tensor<type, 1>& gradient) const
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    const ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
        static_cast<ProbabilisticLayerBackPropagation*>(back_propagation.get());

    const type* synaptic_weights_derivatives_data = probabilistic_layer_back_propagation->synaptic_weight_derivatives.data();
    const type* biases_derivatives_data = probabilistic_layer_back_propagation->bias_derivatives.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient.data() + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient.data() + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void ProbabilisticLayer::insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                                           const Index& index,
                                                           Tensor<type, 2>& squared_errors_Jacobian) const
{
    ProbabilisticLayerBackPropagationLM* probabilistic_layer_back_propagation_lm =
        static_cast<ProbabilisticLayerBackPropagationLM*>(back_propagation.get());

    const Index batch_samples_number = back_propagation->batch_samples_number;
    const Index parameters_number = get_parameters_number();

    type* this_squared_errors_Jacobian_data = probabilistic_layer_back_propagation_lm->squared_errors_Jacobian.data();

    memcpy(squared_errors_Jacobian.data() + index,
           this_squared_errors_Jacobian_data, 
           parameters_number * batch_samples_number*sizeof(type));
}


void ProbabilisticLayer::print() const
{
    cout << "Probabilistic layer" << endl
         << "Input dimensions: " << get_input_dimensions()[0] << endl
         << "Output dimensions: " << get_output_dimensions()[0] << endl
         << "Biases dimensions: " << biases.dimensions() << endl
         << "Synaptic weights dimensions: " << synaptic_weights.dimensions() << endl;

    cout << "Biases:" << endl;
    cout << biases << endl;
    cout << "Synaptic weights:" << endl;
    cout << synaptic_weights << endl;

    cout << "Activation function:" << endl;
    cout << get_activation_function_string() << endl;
}


void ProbabilisticLayer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Probabilistic");

    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "ActivationFunction", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));
    add_xml_element(printer, "DecisionThreshold", to_string(decision_threshold));

    printer.CloseElement();
}


void ProbabilisticLayer::from_XML(const XMLDocument& document)
{
    const XMLElement* probabilistic_layer_element = document.FirstChildElement("Probabilistic");

    if(!probabilistic_layer_element)
        throw runtime_error("Probabilistic layer element is nullptr.\n");

    const Index new_inputs_number = read_xml_index(probabilistic_layer_element, "InputsNumber");
    const Index new_neurons_number = read_xml_index(probabilistic_layer_element, "NeuronsNumber");

    set({ new_inputs_number }, { new_neurons_number });

    set_activation_function(read_xml_string(probabilistic_layer_element, "ActivationFunction"));    
    set_parameters(to_type_vector(read_xml_string(probabilistic_layer_element, "Parameters"), " "));
    set_decision_threshold(read_xml_type(probabilistic_layer_element, "DecisionThreshold"));
}


string ProbabilisticLayer::write_binary_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = binary(" << input_names[j] << ");\n";

    return buffer.str();
}


string ProbabilisticLayer::write_logistic_expression(const vector<string>& input_names,
                                                     const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = logistic(" << input_names[j] << ");\n";

    return buffer.str();
}


string ProbabilisticLayer::write_competitive_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = competitive(" << input_names[j] << ");\n";

    return buffer.str();
}


string ProbabilisticLayer::write_softmax_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = softmax(" << input_names[j] << ");\n";

    return buffer.str();
}


string ProbabilisticLayer::write_combinations(const vector<string>& input_names) const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer << "probabilistic_layer_combinations_" << to_string(i) << " = " << biases(i);

        for(Index j = 0; j < inputs_number; j++)
            buffer << " +" << synaptic_weights(j, i) << "*" << input_names[j] << "";

        buffer << " " << endl;
    }

    buffer << "\t" << endl;

    return buffer.str();
}


string ProbabilisticLayer::write_activations(const vector<string>& output_names) const
{
    ostringstream buffer;

    const Index outputs_number = get_outputs_number();

    for(Index i = 0; i < outputs_number; i++)
    {
        switch(activation_function)
        {
        case ActivationFunction::Binary:
            buffer << "\tif" << "probabilistic_layer_combinations_" << to_string(i) << " < 0.5, " << output_names[i] << "= 0.0. Else " << output_names[i] << " = 1.0\n";
            break;

        case ActivationFunction::Logistic:
            buffer <<  output_names[i] << " = 1.0/(1.0 + exp(-" <<  "probabilistic_layer_combinations_" << to_string(i) << "));\n";
            break;

        case ActivationFunction::Competitive:
            if(i == 0)
                buffer << "\tfor each probabilistic_layer_combinations_i:" << endl
                       << "\t\tif probabilistic_layer_combinations_i is equal to max(probabilistic_layer_combinations_i):"<<endl
                       << "\t\t\tactivations[i] = 1"<<endl
                       << "\t\telse:"<<endl
                       << "\t\t\tactivations[i] = 0"<<endl;
            break;

        case ActivationFunction::Softmax:

            if (i == 0)
            {
                buffer << "sum = ";

                for (Index i = 0; i < outputs_number; i++)
                {
                    buffer << "exp(probabilistic_layer_combinations_" << to_string(i) << ")";

                    if (i != outputs_number - 1)
                        buffer << " + ";
                }

                buffer << ";\n" << endl;

                for (Index i = 0; i < outputs_number; i++)
                    buffer << output_names[i] << " = exp(probabilistic_layer_combinations_" << to_string(i) << ")/sum;\n";
            }
            break;
        default:
            break;
        }
    }

    return buffer.str();
}


string ProbabilisticLayer::get_expression(const vector<string>& input_names,
                                            const vector<string>& output_names) const
{
    ostringstream buffer;

    buffer << write_combinations(input_names)
           << write_activations(output_names);

    return buffer.str();
}


ProbabilisticLayerForwardPropagation::ProbabilisticLayerForwardPropagation(
    const Index& new_batch_samples_number, Layer *new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type *, dimensions> ProbabilisticLayerForwardPropagation::get_outputs_pair() const
{
    const Index outputs_number = layer->get_outputs_number();

    return pair<type *, dimensions>((type*)outputs.data(), {{batch_samples_number, outputs_number}});
}


void ProbabilisticLayerForwardPropagation::set(const Index &new_batch_samples_number, Layer *new_layer) 
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_samples_number, outputs_number);

    activation_derivatives.resize(0, 0);

    if(outputs_number == 1)
        activation_derivatives.resize(batch_samples_number, outputs_number);
}


void ProbabilisticLayerForwardPropagation::print() const 
{
    cout << "Probabilistic layer forward-propagation" << endl
         << "Outputs dimensions:" << endl
         << outputs.dimensions() << endl;

    const Index outputs_number = layer->get_outputs_number();

    if(outputs_number == 1)
       cout << "Activation derivatives:" << endl
            << activation_derivatives << endl;
}


ProbabilisticLayerBackPropagation::ProbabilisticLayerBackPropagation(const Index &new_batch_samples_number, Layer *new_layer)
    : LayerBackPropagation() 
{
    set(new_batch_samples_number, new_layer);
}


void ProbabilisticLayerBackPropagation::set(const Index &new_batch_samples_number, Layer *new_layer) 
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    //if(neurons_number > 1)
    //    targets.resize(batch_samples_number, neurons_number);

    bias_derivatives.resize(outputs_number);

    synaptic_weight_derivatives.resize(inputs_number, outputs_number);

    //deltas_row.resize(neurons_number);
    //activations_derivatives_matrix.resize(neurons_number, neurons_number);

    combination_derivatives.resize(batch_samples_number, outputs_number);

    input_derivatives.resize(batch_samples_number, inputs_number);
}


vector<pair<type*, dimensions>> ProbabilisticLayerBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_derivatives.data()), {batch_samples_number, inputs_number}} };
}


void ProbabilisticLayerBackPropagation::print() const 
{
    cout << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << synaptic_weight_derivatives << endl;
}


ProbabilisticLayerBackPropagationLM::ProbabilisticLayerBackPropagationLM(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagationLM()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> ProbabilisticLayerBackPropagationLM::get_input_derivative_pairs() const
{
    return vector<pair<type*, dimensions>>();
}


void ProbabilisticLayerBackPropagationLM::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index outputs_number = layer->get_outputs_number();
    const Index parameters_number = layer->get_parameters_number();

    //deltas.resize(batch_samples_number, neurons_number);
    //deltas_row.resize(neurons_number);

    squared_errors_Jacobian.resize(batch_samples_number, parameters_number);

    combination_derivatives.resize(batch_samples_number, outputs_number);
}


void ProbabilisticLayerBackPropagationLM::print() const
{
    cout << "Squared errors Jacobian: " << endl
        << squared_errors_Jacobian << endl;
}
} // namespace opennn

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
