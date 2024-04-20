//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "multihead_attention_layer.h"
#include "perceptron_layer_3d.h"

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
                                                 const Index& new_heads_number,
                                                 const bool& apply_causal_mask) : Layer()
{
    set(new_input_size, new_context_size, new_depth, new_heads_number);

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

Index MultiheadAttentionLayer::get_heads_number() const
{
    return heads_number;
}


Index MultiheadAttentionLayer::get_weights_depth() const
{
    return weights_depth;
}


dimensions MultiheadAttentionLayer::get_output_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { input_size, depth };
}


/// Return linear transformation weights and biases

Tensor<type, 3> MultiheadAttentionLayer::get_query_weights() const
{
    return query_weights;
}


Tensor<type, 2> MultiheadAttentionLayer::get_query_biases() const
{
    return query_biases;
}


Tensor<type, 3> MultiheadAttentionLayer::get_key_weights() const
{
    return key_weights;
}


Tensor<type, 2> MultiheadAttentionLayer::get_key_biases() const
{
    return key_biases;
}


Tensor<type, 3> MultiheadAttentionLayer::get_value_weights() const
{
    return value_weights;
}


Tensor<type, 2> MultiheadAttentionLayer::get_value_biases() const
{
    return value_biases;
}


/// Returns the linear projection weights

Tensor<type, 3> MultiheadAttentionLayer::get_projection_weights() const
{
    return projection_weights;
}

Tensor<type, 1> MultiheadAttentionLayer::get_projection_biases() const
{
    return projection_biases;
}


/// Returns the number of parameters of the layer.

Index MultiheadAttentionLayer::get_parameters_number() const
{
    return
        query_weights.size() + query_biases.size()
        + key_weights.size() + key_biases.size()
        + value_weights.size() + value_biases.size()
        + projection_weights.size() + projection_biases.size();
}


Tensor<type, 1> MultiheadAttentionLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    Index parameters_index = 0;

    copy(/*execution::par,*/
        query_weights.data(),
        query_weights.data() + query_weights.size(),
        parameters.data());

    parameters_index += query_weights.size();

    copy(/*execution::par,*/
        query_biases.data(),
        query_biases.data() + query_biases.size(),
        parameters.data() + parameters_index);

    parameters_index += query_biases.size();


    copy(/*execution::par,*/
        key_weights.data(),
        key_weights.data() + key_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += key_weights.size();

    copy(/*execution::par,*/
        key_biases.data(),
        key_biases.data() + key_biases.size(),
        parameters.data() + parameters_index);

    parameters_index += key_biases.size();


    copy(/*execution::par,*/
        value_weights.data(),
        value_weights.data() + value_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += value_weights.size();

    copy(/*execution::par,*/
        value_biases.data(),
        value_biases.data() + value_biases.size(),
        parameters.data() + parameters_index);

    parameters_index += value_biases.size();

    copy(/*execution::par,*/
        projection_weights.data(),
        projection_weights.data() + projection_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += projection_weights.size();

    copy(/*execution::par,*/
        projection_biases.data(),
        projection_biases.data() + projection_biases.size(),
        parameters.data() + parameters_index);

    return parameters;
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

    heads_number = 0;

    query_weights.resize(0, 0, 0);
    query_biases.resize(0, 0);

    key_weights.resize(0, 0, 0);
    key_biases.resize(0, 0);

    value_weights.resize(0, 0, 0);
    value_biases.resize(0, 0);

    projection_weights.resize(0, 0, 0);
    projection_biases.resize(0);

    set_default();
}


/// Sets new input size, embedding depth, number of attention heads and activation function of the layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set(const Index& new_input_size,
                                  const Index& new_context_size,
                                  const Index& new_depth,
                                  const Index& new_heads_number)
{
    input_size = new_input_size;

    context_size = new_context_size;

    depth = new_depth;

    heads_number = new_heads_number;

    weights_depth = depth; //Index(depth / heads_number);

    scaling_factor = type(1) / type(sqrt(weights_depth));

    set_weights();

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

void MultiheadAttentionLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    type* new_parameters_data = (type*) new_parameters.data();

    type* query_weights_data = (type*)query_weights.data();
    type* query_biases_data = (type*)query_biases.data();

    type* key_weights_data = (type*)key_weights.data();
    type* key_biases_data = (type*)key_biases.data();

    type* value_weights_data = (type*)value_weights.data();
    type* value_biases_data = (type*)value_biases.data();

    type* projection_weights_data = (type*)projection_weights.data();
    type* projection_biases_data = (type*)projection_biases.data();

    Index parameters_index = index;

    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + query_weights.size(),
        query_weights_data);

    parameters_index += query_weights.size();

    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + query_biases.size(),
        query_biases_data);

    parameters_index += query_biases.size();

    
    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + key_weights.size(),
        key_weights_data);

    parameters_index += key_weights.size();

    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + key_biases.size(),
        key_biases_data);

    parameters_index += key_biases.size();

    
    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + value_weights.size(),
        value_weights_data);

    parameters_index += value_weights.size();

    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + value_biases.size(),
        value_biases_data);

    parameters_index += value_biases.size();

    
    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + projection_weights.size(),
        projection_weights_data);

    parameters_index += projection_weights.size();
    
    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + projection_biases.size(),
        projection_biases_data);
}


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

    set_weights();
}


/// Sets a new number of attention heads in the layer.

void MultiheadAttentionLayer::set_heads_number(const Index& new_heads_number)
{
    heads_number = new_heads_number;

    set_weights();
}


/// Sets the layer's weights according to the parameters.

void MultiheadAttentionLayer::set_weights()
{
    query_weights.resize(depth, weights_depth, heads_number);
    query_biases.resize(weights_depth, heads_number);

    key_weights.resize(depth, weights_depth, heads_number);
    key_biases.resize(weights_depth, heads_number);

    value_weights.resize(depth, weights_depth, heads_number);
    value_biases.resize(weights_depth, heads_number);

    projection_weights.resize(weights_depth, depth, heads_number);
    projection_biases.resize(depth);

    set_parameters_random();
}


void MultiheadAttentionLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);


    /// @todo in Tensor form

#pragma omp parallel for
    for(Index i = 0; i < query_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        query_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for (Index i = 0; i < query_biases.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        query_biases(i) = minimum + (maximum - minimum) * random;
    }



#pragma omp parallel for
    for(Index i = 0; i < key_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        key_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for (Index i = 0; i < key_biases.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        key_biases(i) = minimum + (maximum - minimum) * random;
    }


#pragma omp parallel for
    for(Index i = 0; i < value_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        value_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for (Index i = 0; i < value_biases.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        value_biases(i) = minimum + (maximum - minimum) * random;
    }


#pragma omp parallel for
    for(Index i = 0; i < projection_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        projection_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for (Index i = 0; i < projection_biases.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        projection_biases(i) = minimum + (maximum - minimum) * random;
    }
}

void MultiheadAttentionLayer::set_parameters_constant(const type& value)
{
    query_weights.setConstant(value);
    query_biases.setConstant(value);

    key_weights.setConstant(value);
    key_biases.setConstant(value);

    value_weights.setConstant(value);
    value_biases.setConstant(value);

    projection_weights.setConstant(value);
    projection_biases.setConstant(value);
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

        throw runtime_error(buffer.str());
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


/// @todo check 

void MultiheadAttentionLayer::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    #pragma omp parallel for

    for(Index input_index = 0; input_index < input_size ; input_index++)
    {
        for(Index context_index = input_index + 1; context_index < context_size; context_index++)
        {
            attention_scores.chip(input_index, 1).chip(context_index, 0).setConstant(m_inf);
        }
    }
}


/// @todo Check if we can do this with transposed matrices in contract. 

void MultiheadAttentionLayer::calculate_transformation(const Tensor<type, 3>& input,
                                                       Tensor<type, 4>& transformed_input,
                                                       const Tensor<type, 3>& weights,
                                                       const Tensor<type, 2>& biases) const
{
    const Index batch_size = input.dimension(0);
    const Index variables_number = input.dimension(1);

    type* weights_data = (type*)weights.data();
    type* biases_data = (type*)biases.data();
    type* transformed_input_data = transformed_input.data();

    /// @todo move to struct or do tensor map? 
    Tensor<type, 2> sample_input;

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        type* head_weights_data = weights_data + head_index * depth * weights_depth;
        type* head_biases_data = biases_data + head_index * weights_depth;
        type* head_transformed_input_data = transformed_input_data + head_index * batch_size * variables_number * weights_depth;

        const TensorMap<Tensor<type, 2>> head_weights(head_weights_data, depth, weights_depth);
        const TensorMap<Tensor<type, 1>> head_biases(head_biases_data, weights_depth);

        for (Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            sample_input = input.chip(sample_index, 0);

            type* sample_transformed_input_data = head_transformed_input_data + sample_index * variables_number * weights_depth;

            TensorMap<Tensor<type, 2>> sample_transformed_input(sample_transformed_input_data, variables_number, weights_depth);

            sample_transformed_input.device(*thread_pool_device)
                = sample_input.contract(head_weights, A_B);

            sum_columns(thread_pool_device, head_biases, sample_transformed_input);
        }
    }
}


void MultiheadAttentionLayer::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
                                                          Tensor<type, 4>& projection_outputs,
                                                          Tensor<type, 3>& outputs) const 
{
    const Index batch_size = outputs.dimension(0);

    type* attention_outputs_data = (type*)attention_outputs.data();
    type* projection_outputs_data = projection_outputs.data();
    type* projection_weights_data = (type*)projection_weights.data();

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        type* head_projection_output_data = projection_outputs_data + head_index * batch_size * input_size * depth;
        type* head_projection_weights_data = projection_weights_data + head_index * weights_depth * depth;
        type* head_attention_output_data = attention_outputs_data + head_index * input_size * weights_depth * batch_size;

        TensorMap<Tensor<type, 3>> head_projection_output(head_projection_output_data, batch_size, input_size, depth);

        const TensorMap<Tensor<type, 2>> head_projection_weights(head_projection_weights_data, weights_depth, depth);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            type* sample_attention_output_data = head_attention_output_data + sample_index * input_size * weights_depth;

            const TensorMap<Tensor<type, 2>> sample_attention_output(sample_attention_output_data, input_size, weights_depth);

            head_projection_output.chip(sample_index, 0).device(*thread_pool_device)
                = sample_attention_output.contract(head_projection_weights, A_B);
        }
    }

    outputs.device(*thread_pool_device) = projection_outputs.sum(projection_sum_index);

    sum_matrices(thread_pool_device, projection_biases, outputs);
}


/// Computes the attention scores by comparing (via dot product) query and key.
/// Attention scores must be computed separately for each batch element and each attention head 
/// (batch matrix multiplication).

void MultiheadAttentionLayer::compute_attention_scores(const Tensor<type, 4>& query,
                                                       const Tensor<type, 4>& key,
                                                       Tensor<type, 4>& attention_scores,
                                                       Tensor<type, 4>& softmax_attention_scores) const
{
    batch_matrix_multiplication(thread_pool_device, key, query, attention_scores, A_BT);

    attention_scores.device(*thread_pool_device) = attention_scores * scaling_factor;

    if (causal_mask)
    {
        apply_causal_mask(attention_scores);
    }

    softmax(attention_scores, softmax_attention_scores);
}


void MultiheadAttentionLayer::compute_attention_outputs(const Tensor<type, 4>& transformed_value,
                                                       const Tensor<type, 4>& softmax_attention_scores,
                                                       Tensor<type, 4>& attention_outputs) const 
{    
    batch_matrix_multiplication(thread_pool_device, softmax_attention_scores, transformed_value, attention_outputs, AT_B);
}


// @todo
void MultiheadAttentionLayer::dropout(Tensor<type, 4>& attention_scores) const
{
   /*
    const Index batch_samples_number = attention_scores.dimension(0);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        for(Index )
        {
            TensorMap<Tensor<type, 2>> matrix(attention_scores.data() + neuron_index * batch_samples_number * inputs_number,
                batch_samples_number, inputs_number);

            random = calculate_random_uniform((type)0, (type)1);

            random < dropout_rate ? matrix.setZero()
                : matrix = matrix * scaling_factor;
        }
    }
*/
}


void MultiheadAttentionLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                                LayerForwardPropagation* layer_forward_propagation,
                                                const bool& is_training)
{
    MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation
        = static_cast<MultiheadAttentionLayerForwardPropagation*>(layer_forward_propagation);

    const TensorMap<Tensor<type, 3>> input(inputs_pair(0).first,
                                           inputs_pair(0).second[0],
                                           inputs_pair(0).second[1],
                                           inputs_pair(0).second[2]);

    const TensorMap<Tensor<type, 3>> context(inputs_pair(1).first,
                                             inputs_pair(1).second[0],
                                             inputs_pair(1).second[1],
                                             inputs_pair(1).second[2]);

    Tensor<type, 4>& query = multihead_attention_layer_forward_propagation->query;
    Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;

    Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;
    Tensor<type, 4>& softmax_attention_scores = multihead_attention_layer_forward_propagation->softmax_attention_scores;

    Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    Tensor<type, 4>& projection_outputs = multihead_attention_layer_forward_propagation->projection_outputs;
    Tensor<type, 3>& outputs = multihead_attention_layer_forward_propagation->outputs;

    calculate_transformation(input, query, query_weights, query_biases);

    calculate_transformation(context, key, key_weights, key_biases);

    calculate_transformation(context, value, value_weights, value_biases);
    
    compute_attention_scores(query,
                             key,
                             attention_scores,
                             softmax_attention_scores);

    if (is_training && dropout_rate > type(0)) 
        dropout(softmax_attention_scores);
    
    compute_attention_outputs(value,
                             softmax_attention_scores,
                             attention_outputs);

    calculate_output_projection(attention_outputs,
                                projection_outputs,
                                outputs);
}


void MultiheadAttentionLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                                       const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                                       LayerForwardPropagation* forward_propagation,
                                                       LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input(inputs_pair(0).first,
                                           inputs_pair(0).second[0],
                                           inputs_pair(0).second[1],
                                           inputs_pair(0).second[2]);

    const TensorMap<Tensor<type, 3>> context(inputs_pair(1).first,
                                             inputs_pair(1).second[0],
                                             inputs_pair(1).second[1],
                                             inputs_pair(1).second[2]);

    const TensorMap<Tensor<type, 3>> deltas(deltas_pair(0).first,
                                            deltas_pair(0).second[0],
                                            deltas_pair(0).second[1],
                                            deltas_pair(0).second[2]);

    Index batch_samples_number = inputs_pair(0).second[0];

    type* query_weights_data = (type*)query_weights.data();
    type* key_weights_data = (type*)key_weights.data();
    type* value_weights_data = (type*)value_weights.data();
    type* projection_weights_data = (type*)projection_weights.data();

    // Forward propagation

    const MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation =
        static_cast<MultiheadAttentionLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 4>& softmax_attention_scores = multihead_attention_layer_forward_propagation->softmax_attention_scores;
    const Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    const Tensor<type, 4>& query = multihead_attention_layer_forward_propagation-> query;
    const Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    const Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;
    
    type* softmax_attention_scores_data = (type*)softmax_attention_scores.data();
    type* attention_outputs_data = (type*)attention_outputs.data();

    type* query_data = (type*)query.data();
    type* key_data = (type*)key.data();
    type* value_data = (type*)value.data();

    // Back propagation

    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    Tensor<type, 3>& projection_weights_derivatives = multihead_attention_layer_back_propagation->projection_weights_derivatives;

    Tensor<type, 4>& error_attention_scores_derivatives = multihead_attention_layer_back_propagation->error_attention_scores_derivatives;
    Tensor<type, 4>& error_softmax_attention_scores_derivatives = multihead_attention_layer_back_propagation->error_softmax_attention_scores_derivatives;
    Tensor<type, 4>& error_attention_output_derivatives = multihead_attention_layer_back_propagation->error_attention_output_derivatives;

    Tensor<type, 4>& error_query_derivatives = multihead_attention_layer_back_propagation->error_query_derivatives;
    Tensor<type, 4>& error_key_derivatives = multihead_attention_layer_back_propagation->error_key_derivatives;
    Tensor<type, 4>& error_value_derivatives = multihead_attention_layer_back_propagation->error_value_derivatives;

    Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;
    Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;
    Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;

    Tensor<type, 3>& input_derivatives = multihead_attention_layer_back_propagation->input_derivatives;
    input_derivatives.setZero();
    Tensor<type, 3>& context_derivatives = multihead_attention_layer_back_propagation->context_derivatives;
    context_derivatives.setZero();

    Tensor<type, 1>& aux_rows = multihead_attention_layer_back_propagation->aux_rows;

    Tensor<type, 2>& query_biases_derivatives = multihead_attention_layer_back_propagation->query_biases_derivatives;
    Tensor<type, 2>& key_biases_derivatives = multihead_attention_layer_back_propagation->key_biases_derivatives;
    Tensor<type, 2>& value_biases_derivatives = multihead_attention_layer_back_propagation->value_biases_derivatives;
    Tensor<type, 1>& projection_biases_derivatives = multihead_attention_layer_back_propagation->projection_biases_derivatives;

    type* projection_weights_derivatives_data = projection_weights_derivatives.data();

    type* error_attention_scores_derivatives_data = error_attention_scores_derivatives.data();
    type* error_softmax_attention_scores_derivatives_data = error_softmax_attention_scores_derivatives.data();
    type* error_attention_output_derivatives_data = error_attention_output_derivatives.data();

    type* error_query_derivatives_data = error_query_derivatives.data();
    type* error_key_derivatives_data = error_key_derivatives.data();
    type* error_value_derivatives_data = error_value_derivatives.data();

    type* query_weights_derivatives_data = query_weights_derivatives.data();
    type* key_weights_derivatives_data = key_weights_derivatives.data();
    type* value_weights_derivatives_data = value_weights_derivatives.data();

    type* query_biases_derivatives_data = query_biases_derivatives.data();
    type* key_biases_derivatives_data = key_biases_derivatives.data();
    type* value_biases_derivatives_data = value_biases_derivatives.data();
    
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        type* head_query_weights_data = query_weights_data + head_index * depth * weights_depth;
        type* head_key_weights_data = key_weights_data + head_index * depth * weights_depth;
        type* head_value_weights_data = value_weights_data + head_index * depth * weights_depth;

        type* head_query_data = query_data + head_index * input_size * weights_depth * batch_samples_number;
        type* head_key_data = key_data + head_index * context_size * weights_depth * batch_samples_number;
        type* head_value_data = value_data + head_index * context_size * weights_depth * batch_samples_number;

        type* head_projection_weights_data = projection_weights_data + head_index * weights_depth * depth;
        type* head_softmax_attention_scores_data = softmax_attention_scores_data + head_index * context_size * input_size * batch_samples_number;
        type* head_attention_outputs_data = attention_outputs_data + head_index * input_size * weights_depth * batch_samples_number;

        type* head_projection_weights_derivatives_data = projection_weights_derivatives_data + head_index * weights_depth * depth;

        type* head_attention_scores_derivatives_data = error_attention_scores_derivatives_data + head_index * context_size * input_size * batch_samples_number;
        type* head_softmax_attention_scores_derivatives_data = error_softmax_attention_scores_derivatives_data + head_index * context_size * input_size * batch_samples_number;
        type* head_attention_output_derivatives_data = error_attention_output_derivatives_data + head_index * input_size * weights_depth * batch_samples_number;

        type* head_query_derivatives_data = error_query_derivatives_data + head_index * input_size * weights_depth * batch_samples_number;
        type* head_key_derivatives_data = error_key_derivatives_data + head_index * context_size * weights_depth * batch_samples_number;
        type* head_value_derivatives_data = error_value_derivatives_data + head_index * context_size * weights_depth * batch_samples_number;

        type* head_query_weights_derivatives_data = query_weights_derivatives_data + head_index * depth * weights_depth;
        type* head_key_weights_derivatives_data = key_weights_derivatives_data + head_index * depth * weights_depth;
        type* head_value_weights_derivatives_data = value_weights_derivatives_data + head_index * depth * weights_depth;
        
        type* head_query_biases_derivatives_data = query_biases_derivatives_data + head_index * weights_depth;
        type* head_key_biases_derivatives_data = key_biases_derivatives_data + head_index * weights_depth;
        type* head_value_biases_derivatives_data = value_biases_derivatives_data + head_index * weights_depth;

        const TensorMap<Tensor<type, 2>> head_query_weights(head_query_weights_data, depth, weights_depth);
        const TensorMap<Tensor<type, 2>> head_key_weights(head_key_weights_data, depth, weights_depth);
        const TensorMap<Tensor<type, 2>> head_value_weights(head_value_weights_data, depth, weights_depth);

        const TensorMap<Tensor<type, 3>> head_query(head_query_data, input_size, weights_depth, batch_samples_number);
        const TensorMap<Tensor<type, 3>> head_key(head_key_data, context_size, weights_depth, batch_samples_number);
        const TensorMap<Tensor<type, 3>> head_value(head_value_data, context_size, weights_depth, batch_samples_number);

        const TensorMap<Tensor<type, 2>> head_projection_weights(head_projection_weights_data, weights_depth, depth);

        const TensorMap<Tensor<type, 3>> head_softmax_attention_scores(head_softmax_attention_scores_data, context_size, input_size, batch_samples_number);
        const TensorMap<Tensor<type, 3>> head_attention_outputs(head_attention_outputs_data, input_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_projection_weights_derivatives(head_projection_weights_derivatives_data, weights_depth, depth);

        TensorMap<Tensor<type, 3>> head_attention_scores_derivatives(head_attention_scores_derivatives_data, context_size, input_size, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_softmax_attention_scores_derivatives(head_softmax_attention_scores_derivatives_data, context_size, input_size, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_attention_output_derivatives(head_attention_output_derivatives_data, input_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 3>> head_query_derivatives(head_query_derivatives_data, input_size, weights_depth, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_key_derivatives(head_key_derivatives_data, context_size, weights_depth, batch_samples_number);
        TensorMap<Tensor<type, 3>> head_value_derivatives(head_value_derivatives_data, context_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_query_weights_derivatives(head_query_weights_derivatives_data, depth, weights_depth);
        TensorMap<Tensor<type, 2>> head_key_weights_derivatives(head_key_weights_derivatives_data, depth, weights_depth);
        TensorMap<Tensor<type, 2>> head_value_weights_derivatives(head_value_weights_derivatives_data, depth, weights_depth);

        TensorMap<Tensor<type, 1>> head_query_biases_derivatives(head_query_biases_derivatives_data, weights_depth);
        TensorMap<Tensor<type, 1>> head_key_biases_derivatives(head_key_biases_derivatives_data, weights_depth);
        TensorMap<Tensor<type, 1>> head_value_biases_derivatives(head_value_biases_derivatives_data, weights_depth);
        
        // PROJECTION WEIGHTS DERIVATIVES

        head_projection_weights_derivatives.device(*thread_pool_device)
            = head_attention_outputs.contract(deltas, projection_weights_derivatives_contraction_indices);
        
        // ATTENTION OUTPUT DERIVATIVES
        
        for (Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            type* sample_attention_output_derivatives_data = head_attention_output_derivatives_data + sample_index * input_size * weights_depth;

            TensorMap<Tensor<type, 2>> sample_attention_output_derivatives(sample_attention_output_derivatives_data, input_size, weights_depth);

            const Tensor<type, 2> sample_deltas = deltas.chip(sample_index, 0);

            sample_attention_output_derivatives.device(*thread_pool_device)
                = sample_deltas.contract(head_projection_weights, A_BT);
        }

        // VALUE DERIVATIVES

        batch_matrix_multiplication(thread_pool_device, head_softmax_attention_scores, head_attention_output_derivatives, head_value_derivatives, A_B);

        // VALUE WEIGHTS DERIVATIVES

        head_value_weights_derivatives.device(*thread_pool_device)
            = context.contract(head_value_derivatives, transformation_weights_derivatives_contraction_indices);
        
        // SOFTMAX ATTENTION SCORES DERIVATIVES

        batch_matrix_multiplication(thread_pool_device, head_value, head_attention_output_derivatives, head_softmax_attention_scores_derivatives, A_BT);

        // ATTENTION SCORES DERIVATIVES
        
        softmax_derivatives_times_tensor(head_softmax_attention_scores, head_softmax_attention_scores_derivatives, head_attention_scores_derivatives, aux_rows);

        // QUERY DERIVATIVES

        batch_matrix_multiplication(thread_pool_device, head_attention_scores_derivatives, head_key, head_query_derivatives, AT_B);

        head_query_derivatives.device(*thread_pool_device) = head_query_derivatives * scaling_factor;
    
        // KEY DERIVATIVES

        batch_matrix_multiplication(thread_pool_device, head_attention_scores_derivatives, head_query, head_key_derivatives, A_B);
    
        head_key_derivatives.device(*thread_pool_device) = head_key_derivatives * scaling_factor;

        // QUERY WEIGHTS DERIVATIVES

        head_query_weights_derivatives.device(*thread_pool_device)
            = input.contract(head_query_derivatives, transformation_weights_derivatives_contraction_indices);

        // KEY WEIGHTS DERIVATIVES

        head_key_weights_derivatives.device(*thread_pool_device)
            = context.contract(head_key_derivatives, transformation_weights_derivatives_contraction_indices);

        /// @todo try to simplify

        for (Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            type* sample_query_derivatives_data = head_query_derivatives_data + sample_index * input_size * weights_depth;
            type* sample_key_derivatives_data = head_key_derivatives_data + sample_index * context_size * weights_depth;
            type* sample_value_derivatives_data = head_value_derivatives_data + sample_index * context_size * weights_depth;

            const TensorMap<Tensor<type, 2>> sample_query_derivatives(sample_query_derivatives_data, input_size, weights_depth);
            const TensorMap<Tensor<type, 2>> sample_key_derivatives(sample_key_derivatives_data, context_size, weights_depth);
            const TensorMap<Tensor<type, 2>> sample_value_derivatives(sample_value_derivatives_data, context_size, weights_depth);

            // INPUT DERIVATIVES

            input_derivatives.chip(sample_index, 0).device(*thread_pool_device)
                += sample_query_derivatives.contract(head_query_weights, A_BT);

            // CONTEXT DERIVATIVES

            context_derivatives.chip(sample_index, 0).device(*thread_pool_device) 
                += sample_key_derivatives.contract(head_key_weights, A_BT)
                + sample_value_derivatives.contract(head_value_weights, A_BT);
        }

        // BIASES DERIVATIVES

        head_query_biases_derivatives.device(*thread_pool_device) = head_query_derivatives.sum(biases_derivatives_sum_indices);
        head_key_biases_derivatives.device(*thread_pool_device) = head_key_derivatives.sum(biases_derivatives_sum_indices);
        head_value_biases_derivatives.device(*thread_pool_device) = head_value_derivatives.sum(biases_derivatives_sum_indices);
    }

    projection_biases_derivatives.device(*thread_pool_device) = deltas.sum(projection_biases_derivatives_sum_indices);
}


void MultiheadAttentionLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                              const Index& index,
                                              Tensor<type, 1>& gradient) const
{
    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    const Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;
    const Tensor<type, 2>& query_biases_derivatives = multihead_attention_layer_back_propagation->query_biases_derivatives;

    const Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;
    const Tensor<type, 2>& key_biases_derivatives = multihead_attention_layer_back_propagation->key_biases_derivatives;

    const Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;
    const Tensor<type, 2>& value_biases_derivatives = multihead_attention_layer_back_propagation->value_biases_derivatives;

    const Tensor<type, 3>& projection_weights_derivatives = multihead_attention_layer_back_propagation->projection_weights_derivatives;
    const Tensor<type, 1>& projection_biases_derivatives = multihead_attention_layer_back_propagation->projection_biases_derivatives;

    type* gradient_data = gradient.data();

    Index gradient_index = index;

    copy(/*execution::par,*/
        query_weights_derivatives.data(),
        query_weights_derivatives.data() + query_weights_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += query_weights_derivatives.size();

    copy(/*execution::par,*/
        query_biases_derivatives.data(),
        query_biases_derivatives.data() + query_biases_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += query_biases_derivatives.size();


    copy(/*execution::par,*/
        key_weights_derivatives.data(),
        key_weights_derivatives.data() + key_weights_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += key_weights_derivatives.size();

    copy(/*execution::par,*/
        key_biases_derivatives.data(),
        key_biases_derivatives.data() + key_biases_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += key_biases_derivatives.size();


    copy(/*execution::par,*/
        value_weights_derivatives.data(),
        value_weights_derivatives.data() + value_weights_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += value_weights_derivatives.size();

    copy(/*execution::par,*/
        value_biases_derivatives.data(),
        value_biases_derivatives.data() + value_biases_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += value_biases_derivatives.size();


    copy(/*execution::par,*/
        projection_weights_derivatives.data(),
        projection_weights_derivatives.data() + projection_weights_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += projection_weights_derivatives.size();

    copy(/*execution::par,*/
        projection_biases_derivatives.data(),
        projection_biases_derivatives.data() + projection_biases_derivatives.size(),
        gradient_data + gradient_index);
}


pair<type*, dimensions> MultiheadAttentionLayerForwardPropagation::get_outputs_pair() const
{
    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    const Index input_size = multihead_attention_layer->get_input_size();

    const Index depth = multihead_attention_layer->get_depth();

    return pair<type*, dimensions>(outputs_data, { { batch_samples_number, input_size, depth } });
}


void MultiheadAttentionLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index input_size = multihead_attention_layer->get_input_size();

    const Index context_size = multihead_attention_layer->get_context_size();

    const Index depth = multihead_attention_layer->get_depth();

    const Index heads_number = multihead_attention_layer->get_heads_number();

    const Index weights_depth = multihead_attention_layer->get_weights_depth();

    // Outputs

    outputs.resize(batch_samples_number, input_size, depth);

    outputs_data = outputs.data();

    // Rest of quantities

    query.resize(input_size, weights_depth, batch_samples_number, heads_number);
    key.resize(context_size, weights_depth, batch_samples_number, heads_number);
    value.resize(context_size, weights_depth, batch_samples_number, heads_number);

    attention_scores.resize(context_size, input_size, batch_samples_number, heads_number);
    softmax_attention_scores.resize(context_size, input_size, batch_samples_number, heads_number);
    attention_outputs.resize(input_size, weights_depth, batch_samples_number, heads_number);

    projection_outputs.resize(batch_samples_number, input_size, depth, heads_number);
}


void MultiheadAttentionLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    MultiheadAttentionLayer* multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index input_size = multihead_attention_layer->get_input_size();
    const Index context_size = multihead_attention_layer->get_context_size();
    const Index depth = multihead_attention_layer->get_depth();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index weights_depth = multihead_attention_layer->get_weights_depth();

    error_attention_scores_derivatives.resize(context_size, input_size, batch_samples_number, heads_number);
    error_softmax_attention_scores_derivatives.resize(context_size, input_size, batch_samples_number, heads_number);
    error_attention_output_derivatives.resize(input_size, weights_depth, batch_samples_number, heads_number);

    error_query_derivatives.resize(input_size, weights_depth, batch_samples_number, heads_number);
    error_key_derivatives.resize(context_size, weights_depth, batch_samples_number, heads_number);
    error_value_derivatives.resize(context_size, weights_depth, batch_samples_number, heads_number);

    query_weights_derivatives.resize(depth, weights_depth, heads_number);
    key_weights_derivatives.resize(depth, weights_depth, heads_number);
    value_weights_derivatives.resize(depth, weights_depth, heads_number);

    projection_weights_derivatives.resize(weights_depth, depth, heads_number);

    query_biases_derivatives.resize(weights_depth, heads_number);
    key_biases_derivatives.resize(weights_depth, heads_number);
    value_biases_derivatives.resize(weights_depth, heads_number);
    projection_biases_derivatives.resize(depth);

    aux_rows.resize(context_size);

    input_derivatives.resize(batch_samples_number, input_size, depth);
    context_derivatives.resize(batch_samples_number, context_size, depth);

    inputs_derivatives.resize(2);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, input_size, depth };
    inputs_derivatives(1).first = context_derivatives.data();
    inputs_derivatives(1).second = { batch_samples_number, context_size, depth };
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
