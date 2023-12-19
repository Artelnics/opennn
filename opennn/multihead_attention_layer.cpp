//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "multihead_attention_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.


MultiheadAttentionLayer::MultiheadAttentionLayer() : Layer()
{
    set();

    layer_type = Type::MultiheadAttention;
}


/// Layer architecture constructor.
/// It creates a layer object with given input size, embedding depth and number of attention heads.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.

MultiheadAttentionLayer::MultiheadAttentionLayer(const Index& new_input_size,
                                                 const Index& new_context_size,
                                                 const Index& new_depth,
                                                 const Index& new_number_of_heads,
                                                 const bool& apply_causal_mask) : Layer()
{
    set(new_input_size, new_context_size, new_depth, new_number_of_heads);

    set_causal_mask(apply_causal_mask);

    layer_type = Type::MultiheadAttention;

    layer_name = "multihead_attention_layer";
}


/// Returns the size of the input to the layer.

Index MultiheadAttentionLayer::get_input_size() const
{
    return input_size;
}


/// Returns the size of the context to the layer.

Index MultiheadAttentionLayer::get_context_size() const
{
    return context_size;
}


/// Returns the embedding depth used in the layer.

Index MultiheadAttentionLayer::get_depth() const
{
    return depth;
}


/// Returns the number of attention heads of the layer.

Index MultiheadAttentionLayer::get_number_of_heads() const
{
    return number_of_heads;
}


/// Returns linear transformation kernels

Tensor<type, 3> MultiheadAttentionLayer::get_query_kernel() const
{
    return query_kernel;
}

Tensor<type, 3> MultiheadAttentionLayer::get_key_kernel() const
{
    return key_kernel;
}

Tensor<type, 3> MultiheadAttentionLayer::get_value_kernel() const
{
    return value_kernel;
}


/// Returns the linear projection kernel

Tensor<type, 3> MultiheadAttentionLayer::get_projection_kernel() const
{
    return projection_kernel;
}


/// Returns the number of parameters of the layer.

Index MultiheadAttentionLayer::get_parameters_number() const
{
    return query_kernel.size() + key_kernel.size() + value_kernel.size() + projection_kernel.size();
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& MultiheadAttentionLayer::get_display() const
{
    return display;
}


/// Sets an empty layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set()
{
    input_size = 0;

    depth = 0;

    number_of_heads = 0;

    query_kernel.resize(0, 0, 0);
    key_kernel.resize(0, 0, 0);
    value_kernel.resize(0, 0, 0);

    projection_kernel.resize(0, 0, 0);

    set_default();
}


/// Sets new input size, embedding depth, number of attention heads and activation function of the layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set(const Index& new_input_size,
                                  const Index& new_context_size,
                                  const Index& new_depth,
                                  const Index& new_number_of_heads)
{
    input_size = new_input_size;

    context_size = new_context_size;

    depth = new_depth;

    number_of_heads = new_number_of_heads;

    set_kernels();

    set_default();
}


/// Sets those members not related to the perceptrons to their default value.

void MultiheadAttentionLayer::set_default()
{
    layer_name = "multihead_attention_layer";

    display = true;

    layer_type = Type::MultiheadAttention;
}


void MultiheadAttentionLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new input size in the layer.

void MultiheadAttentionLayer::set_input_size(const Index& new_input_size)
{
    input_size = new_input_size;
}


/// Sets a new input size in the layer.

void MultiheadAttentionLayer::set_context_size(const Index& new_context_size)
{
    context_size = new_context_size;
}


/// Sets a new embedding depth in the layer.

void MultiheadAttentionLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_kernels();
}


/// Sets a new number of attention heads in the layer.

void MultiheadAttentionLayer::set_number_of_heads(const Index& new_number_of_heads)
{
    number_of_heads = new_number_of_heads;

    set_kernels();
}


/// Sets the layer's kernels according to the parameters.

void MultiheadAttentionLayer::set_kernels()
{
    query_kernel.resize(depth, depth, number_of_heads);
    key_kernel.resize(depth, depth, number_of_heads);
    value_kernel.resize(depth, depth, number_of_heads);

    projection_kernel.resize(depth, depth, number_of_heads);

    set_parameters_random();
}


void MultiheadAttentionLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

#pragma omp parallel for
    for(Index i = 0; i < query_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        query_kernel(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < key_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        key_kernel(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < value_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        value_kernel(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < projection_kernel.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        projection_kernel(i) = minimum + (maximum - minimum)*random;
    }
}


void MultiheadAttentionLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void MultiheadAttentionLayer::set_causal_mask(const bool& apply_causal_mask)
{
    if(apply_causal_mask && input_size != context_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void set_causal_mask(const bool&) method.\n"
               << "Causal mask can only be applied to self-attention. In this case, input size (" << input_size << ") should be equal to context size (" << context_size << ").";

        throw invalid_argument(buffer.str());
    }

    causal_mask = apply_causal_mask;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void MultiheadAttentionLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// @todo explain

void MultiheadAttentionLayer::calculate_transformation(const Tensor<type, 3>& data, DynamicTensor<type>& transformed_data, const Tensor<type, 3>& kernel)
{
    const Index batch_size = get_dimensions(data)(0);

    TensorMap<Tensor<type, 4>> transformed_data_map = transformed_data.to_tensor_map<4>();

#pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; batch_index++)
    {
        transformed_data_map.chip(batch_index, 0) = data.chip(batch_index, 0).contract(kernel, A_B);
    }
}


void MultiheadAttentionLayer::calculate_output_projection(DynamicTensor<type>& attention_outputs, DynamicTensor<type>& outputs)
{
    const Index batch_size = attention_outputs.get_dimension(0);

    TensorMap<Tensor<type, 4>> attention_outputs_map = attention_outputs.to_tensor_map<4>();
    TensorMap<Tensor<type, 3>> outputs_map = outputs.to_tensor_map<3>();

    const Eigen::array<IndexPair<Index>, 2> contraction_indeces = {IndexPair<Index>(1, 0), IndexPair<Index>(2, 2)};

#pragma omp parallel for
    for(Index batch_index = 0; batch_index < batch_size; batch_index++)
    {
        outputs_map.chip(batch_index, 0) = attention_outputs_map.chip(batch_index, 0).contract(projection_kernel, contraction_indeces);
    }
}


/// Computes the attention scores by comparing (via dot product) query and key.
/// Attention scores must be computed separately for each batch element and each attention head (batch matrix multiplication).

void MultiheadAttentionLayer::compute_attention_scores(DynamicTensor<type>& transformed_query,
                                                       DynamicTensor<type>& transformed_key,
                                                       DynamicTensor<type>& attention_scores)
{
    if(transformed_query.get_dimension(0) != transformed_key.get_dimension(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_scores(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Input batch size (" << transformed_query.get_dimension(0) << ") and context batch size (" << transformed_key.get_dimension(0) << ") not equal.\n";

        throw invalid_argument(buffer.str());
    }

    const Index batch_size = transformed_query.get_dimension(0);

    const TensorMap<Tensor<type, 4>>  transformed_query_map = transformed_query.to_tensor_map<4>();
    const TensorMap<Tensor<type, 4>>  transformed_key_map = transformed_key.to_tensor_map<4>();

    const Tensor<type, 4> scaled_query = transformed_query_map / transformed_query_map.constant(sqrt(depth));

    TensorMap<Tensor<type, 4>> attention_scores_map = attention_scores.to_tensor_map<4>();

    Tensor<type, 2> raw_attention_scores(input_size, context_size);
    Tensor<type, 2> attention_scores_matrix(input_size, context_size);

    Tensor<Index, 1> attention_scores_dimensions(2);
    attention_scores_dimensions.setValues({input_size, context_size});

//#pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; batch_index++)
    {
        for(Index head_index = 0; head_index < number_of_heads ; head_index++)
        {
            attention_scores_matrix.setZero();

            raw_attention_scores/*.device(thread_pool_device)*/ =
                scaled_query.chip(batch_index, 0).chip(head_index, 2).contract(
                transformed_key_map.chip(batch_index, 0).chip(head_index, 2), A_BT);

            if(causal_mask)
            {
                type m_inf = -1 * numeric_limits<type>::infinity();

#pragma omp parallel for collapse(2)
                for(Index input_index = 0; input_index < input_size; input_index++)
                {
                    for(Index context_index = input_index + 1; context_index < context_size; context_index++)
                    {
                        raw_attention_scores(input_index, context_index) = m_inf;
                    }
                }

            }

            softmax(raw_attention_scores.data(),
                    attention_scores_dimensions,
                    attention_scores_matrix.data(),
                    attention_scores_dimensions);
            attention_scores_map.chip(batch_index, 0).chip(head_index, 2) = attention_scores_matrix;
        }
    }
    /// @todo add dropout
}


void MultiheadAttentionLayer::compute_attention_output(DynamicTensor<type>& transformed_value,
                                                       DynamicTensor<type>& attention_scores,
                                                       DynamicTensor<type>& attention_outputs)
{    
    const Index batch_size = transformed_value.get_dimension(0);

    TensorMap<Tensor<type, 4>> transformed_value_map = transformed_value.to_tensor_map<4>();
    TensorMap<Tensor<type, 4>> attention_scores_map = attention_scores.to_tensor_map<4>();

    TensorMap<Tensor<type, 4>> attention_output_map = attention_outputs.to_tensor_map<4>();

#pragma omp parallel for collapse(2)
        for(Index batch_index = 0; batch_index < batch_size; batch_index++)
        {
            for(Index head_index = 0; head_index < number_of_heads ; head_index++)
            {
                attention_output_map.chip(batch_index, 0).chip(head_index, 2) =
                    attention_scores_map.chip(batch_index, 0).chip(head_index, 2).contract(
                    transformed_value_map.chip(batch_index, 0).chip(head_index, 2), A_B);
            }
        };
}


void MultiheadAttentionLayer::forward_propagate(const Tensor<DynamicTensor<type>, 1>& inputs,
                                        LayerForwardPropagation* forward_propagation,
                                        const bool& is_training)
{
    if(inputs.size() != 2)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "Number of input tensors (" << inputs.size() << ") must be 2 (input and context).\n";
        throw invalid_argument(buffer.str());
    }

    if(inputs(0).get_dimension(0) != inputs(1).get_dimension(0))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "Batch sizes of input and context must be equal.\n";
        throw invalid_argument(buffer.str());
    }

    if(inputs(0).get_dimension(1) != input_size)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "2nd dimension of input must be equal to layer input_size.\n";
        throw invalid_argument(buffer.str());
    }

    if(inputs(1).get_dimension(1) != context_size)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "2nd dimension of context must be equal to layer context_size.\n";
        throw invalid_argument(buffer.str());
    }

    if(inputs(0).get_dimension(2) != depth || inputs(1).get_dimension(2) != depth)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index,1>, 1>&, LayerForwardPropagation*, const bool&)\n"
               << "3rd dimension of input and context must be equal to layer depth.\n";
        throw invalid_argument(buffer.str());
    }

    MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation
        = static_cast<MultiheadAttentionLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 3>>& query_map = inputs(0).to_tensor_map<3>();
    const TensorMap<Tensor<type, 3>>& key_map = inputs(1).to_tensor_map<3>();
    const TensorMap<Tensor<type, 3>>& value_map = inputs(1).to_tensor_map<3>();

    DynamicTensor<type> transformed_query = multihead_attention_layer_forward_propagation->get_transformed_query();
/*
    DynamicTensor<type> transformed_key = multihead_attention_layer_forward_propagation->get_transformed_key();
    DynamicTensor<type> transformed_value = multihead_attention_layer_forward_propagation->get_transformed_value();

    calculate_transformation(query_map, transformed_query, query_kernel);
    calculate_transformation(key_map, transformed_key, key_kernel);
    calculate_transformation(value_map, transformed_value, value_kernel);

    DynamicTensor<type> attention_scores = multihead_attention_layer_forward_propagation->get_attention_scores();

    compute_attention_scores(transformed_query,
                             transformed_key,
                             attention_scores);

    DynamicTensor<type> attention_outputs = multihead_attention_layer_forward_propagation->get_attention_outputs();

    compute_attention_output(transformed_value,
                             attention_scores,
                             attention_outputs);

    calculate_output_projection(attention_outputs,
                                multihead_attention_layer_forward_propagation->outputs(0));
    */
}


/*
void PerceptronLayer::forward_propagate(type* inputs_data,
                                        const Tensor<Index, 1>& inputs_dimensions,
                                        Tensor<type, 1>& potential_parameters,
                                        LayerForwardPropagation* forward_propagation)
{
#ifdef OPENNN_DEBUG
    if(inputs_dimensions(1) != get_inputs_number())
    {
        ostringstream buffer;
        buffer << "OpenNN Exception:" << LOG << endl
               << "void forward_propagate(type*, const Tensor<Index, 1>&, Tensor<type, 1>&, LayerForwardPropagation*) final method.\n"
               << "Inputs columns number must be equal to " << get_inputs_number() << ", (inputs number).\n";

        throw invalid_argument(buffer.str());
    }

    check_size(potential_parameters, get_parameters_number(), LOG);
#endif

    const TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(), 1, neurons_number);

    const TensorMap<Tensor<type, 2>> potential_synaptic_weights(potential_parameters.data()+neurons_number, inputs_number, neurons_number);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
        = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation);

    const Tensor<Index, 1> activations_dimensions = perceptron_layer_forward_propagation->outputs_dimensions;

    const Tensor<Index, 1> combinations_dimensions = get_dimensions(perceptron_layer_forward_propagation->combinations);

    const Tensor<Index, 1> derivatives_dimensions = get_dimensions(perceptron_layer_forward_propagation->activations_derivatives);


    calculate_combinations(inputs,
                           potential_biases,
                           potential_synaptic_weights,
                           perceptron_layer_forward_propagation->get_combinations_data());


    calculate_activations_derivatives(perceptron_layer_forward_propagation->combinations.data(),
                                      combinations_dimensions,
                                      perceptron_layer_forward_propagation->outputs_data,
                                      activations_dimensions,
                                      perceptron_layer_forward_propagation->activations_derivatives.data(),
                                      derivatives_dimensions);
}
*/

/// @todo
///// Returns a string with the expression of the inputs-outputs relationship of the layer.
///// @param inputs_names vector of strings with the name of the layer inputs.
///// @param outputs_names vector of strings with the name of the layer outputs.

//string PerceptronLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
//{
//#ifdef OPENNN_DEBUG
//    //    check_size(inputs_names, get_inputs_number(), LOG);
//    //    check_size(outputs_names, get_neurons_number(), LOG);
//#endif

//    ostringstream buffer;

//    for(Index j = 0; j < outputs_names.size(); j++)
//    {
//        const Tensor<type, 1> synaptic_weights_column =  synaptic_weights.chip(j,1);

//        buffer << outputs_names[j] << " = " << write_activation_function_expression() << "( " << biases(0,j) << " +";

//        for(Index i = 0; i < inputs_names.size() - 1; i++)
//        {
//            buffer << " (" << inputs_names[i] << "*" << synaptic_weights_column(i) << ") +";
//        }

//        buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << synaptic_weights_column[inputs_names.size() - 1] << ") );\n";
//    }

//    return buffer.str();
//}


//void PerceptronLayer::from_XML(const tinyxml2::XMLDocument& document)
//{
//    ostringstream buffer;

//    // Perceptron layer

//    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer");

//    if(!perceptron_layer_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "PerceptronLayer element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    // Layer name

//    const tinyxml2::XMLElement* layer_name_element = perceptron_layer_element->FirstChildElement("LayerName");

//    if(!layer_name_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "LayerName element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(layer_name_element->GetText())
//    {
//        set_name(layer_name_element->GetText());
//    }

//    // Inputs number

//    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

//    if(!inputs_number_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "InputsNumber element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(inputs_number_element->GetText())
//    {
//        set_inputs_number(static_cast<Index>(stoi(inputs_number_element->GetText())));
//    }

//    // Neurons number

//    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

//    if(!neurons_number_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "NeuronsNumber element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(neurons_number_element->GetText())
//    {
//        set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
//    }

//    // Activation function

//    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

//    if(!activation_function_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "ActivationFunction element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(activation_function_element->GetText())
//    {
//        set_activation_function(activation_function_element->GetText());
//    }

//    // Parameters

//    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

//    if(!parameters_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "Parameters element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(parameters_element->GetText())
//    {
//        const string parameters_string = parameters_element->GetText();

//        set_parameters(to_type_vector(parameters_string, ' '));
//    }
//}


//void PerceptronLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
//{
//    ostringstream buffer;

//    // Perceptron layer

//    file_stream.OpenElement("PerceptronLayer");

//    // Layer name
//    file_stream.OpenElement("LayerName");
//    buffer.str("");
//    buffer << layer_name;
//    file_stream.PushText(buffer.str().c_str());
//    file_stream.CloseElement();

//    // Inputs number
//    file_stream.OpenElement("InputsNumber");

//    buffer.str("");
//    buffer << get_inputs_number();

//    file_stream.PushText(buffer.str().c_str());

//    file_stream.CloseElement();

//    // Outputs number

//    file_stream.OpenElement("NeuronsNumber");

//    buffer.str("");
//    buffer << get_neurons_number();

//    file_stream.PushText(buffer.str().c_str());

//    file_stream.CloseElement();

//    // Activation function

//    file_stream.OpenElement("ActivationFunction");

//    file_stream.PushText(write_activation_function().c_str());

//    file_stream.CloseElement();

//    // Parameters

//    file_stream.OpenElement("Parameters");

//    buffer.str("");

//    const Tensor<type, 1> parameters = get_parameters();
//    const Index parameters_size = parameters.size();

//    for(Index i = 0; i < parameters_size; i++)
//    {
//        buffer << parameters(i);

//        if(i != (parameters_size-1)) buffer << " ";
//    }

//    file_stream.PushText(buffer.str().c_str());

//    file_stream.CloseElement();

//    // Peceptron layer (end tag)

//    file_stream.CloseElement();
//}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
