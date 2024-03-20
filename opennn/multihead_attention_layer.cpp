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


/// Returns linear transformation weights

Tensor<type, 3> MultiheadAttentionLayer::get_query_weights() const
{
    return query_weights;
}

Tensor<type, 3> MultiheadAttentionLayer::get_key_weights() const
{
    return key_weights;
}

Tensor<type, 3> MultiheadAttentionLayer::get_value_weights() const
{
    return value_weights;
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
    return query_weights.size() + key_weights.size() + value_weights.size() + projection_weights.size() + projection_biases.size();
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
        key_weights.data(),
        key_weights.data() + key_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += key_weights.size();

    copy(/*execution::par,*/
        value_weights.data(),
        value_weights.data() + value_weights.size(),
        parameters.data() + parameters_index);

    parameters_index += value_weights.size();

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
    key_weights.resize(0, 0, 0);
    value_weights.resize(0, 0, 0);

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

    weights_depth = Index(depth / heads_number);

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
    type* key_weights_data = (type*)key_weights.data();
    type* value_weights_data = (type*)value_weights.data();
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
        new_parameters_data + parameters_index + key_weights.size(),
        key_weights_data);

    parameters_index += key_weights.size();
    
    copy(/*execution::par,*/
        new_parameters_data + parameters_index,
        new_parameters_data + parameters_index + value_weights.size(),
        value_weights_data);

    parameters_index += value_weights.size();
    
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
    key_weights.resize(depth, weights_depth, heads_number);
    value_weights.resize(depth, weights_depth, heads_number);

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
    for(Index i = 0; i < key_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        key_weights(i) = minimum + (maximum - minimum)*random;
    }

#pragma omp parallel for
    for(Index i = 0; i < value_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        value_weights(i) = minimum + (maximum - minimum)*random;
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
    key_weights.setConstant(value);
    value_weights.setConstant(value);

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


void MultiheadAttentionLayer::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    constexpr type m_inf = -numeric_limits<type>::infinity(); // superior triangular = m_inf

    for(Index input_index = 0; input_index < input_size ; input_index++)
    {
        for(Index context_index = input_index + 1; context_index < context_size; context_index++)
        {
            attention_scores.chip(context_index, 1).chip(input_index, 0).setConstant(m_inf);
        }
    }
}


/// @todo explain

void MultiheadAttentionLayer::calculate_transformation(const Tensor<type, 3>& data,
                                                       Tensor<type, 4>& transformed_data,
                                                       const Tensor<type, 3>& weights) const
{
    const Index batch_size = data.dimension(0);

    // This is NOT batch matrix multiplication

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        transformed_data.chip(sample_index, 2).device(*thread_pool_device)
            = data.chip(sample_index, 0).contract(weights, A_B);
    }
}


void MultiheadAttentionLayer::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
                                                          Tensor<type, 3>& outputs) const 
{
    const Index batch_size = outputs.dimension(0);

    Tensor<type, 4> projection_outputs(batch_size, input_size, depth, heads_number);

    // This is NOT batch matrix multiplication

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        TensorMap<Tensor<type, 3>> head_projection_output(projection_outputs.data() + head_index * batch_size*input_size*depth,
            batch_size, input_size, depth);

        const TensorMap<Tensor<type, 2>> head_projection_weights((type*) projection_weights.data() + head_index * weights_depth*depth,
            weights_depth, depth);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            const TensorMap<Tensor<type, 2>> sample_attention_output((type*) attention_outputs.data() + sample_index * input_size*weights_depth + head_index * input_size*weights_depth*batch_size,
                input_size, weights_depth);

            head_projection_output.chip(sample_index, 0).device(*thread_pool_device) =
                sample_attention_output.contract(head_projection_weights, A_B);
        }
    }

    outputs = projection_outputs.sum(Eigen::array<Index, 1>({ 3 }));

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
    const Index batch_size = query.dimension(0);

    batch_matrix_multiplication(thread_pool_device, key, query, attention_scores, A_BT);

    attention_scores = attention_scores * scaling_factor;

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
    const Index batch_size = transformed_value.dimension(0);

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
    
    calculate_transformation(input, query, query_weights);

    calculate_transformation(context, key, key_weights);

    calculate_transformation(context, value, value_weights);

    Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;
    Tensor<type, 4>& softmax_attention_scores = multihead_attention_layer_forward_propagation->softmax_attention_scores;
    
    compute_attention_scores(query,
                             key,
                             attention_scores,
                             softmax_attention_scores);

    if (dropout_rate > type(0))
    {
        dropout(softmax_attention_scores);
    }

    Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;
    
    compute_attention_outputs(value,
                             softmax_attention_scores,
                             attention_outputs);


    Tensor<type, 3>& outputs = multihead_attention_layer_forward_propagation->outputs;
    
    calculate_output_projection(attention_outputs,
                                outputs);
}


void MultiheadAttentionLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                                     LayerBackPropagation* next_back_propagation,
                                                     LayerForwardPropagation*,
                                                     LayerBackPropagation* back_propagation) const
{
    MultiheadAttentionLayerBackPropagation* multihead_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    switch (next_back_propagation->layer->get_type())
    {

    case Type::Perceptron3D:
    {
        PerceptronLayer3DForwardPropagation* next_perceptron_layer_forward_propagation =
            reinterpret_cast<PerceptronLayer3DForwardPropagation*>(next_forward_propagation);

        PerceptronLayer3DBackPropagation* next_perceptron_layer_back_propagation =
            reinterpret_cast<PerceptronLayer3DBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_perceptron_layer_forward_propagation,
                               next_perceptron_layer_back_propagation,
                               multihead_layer_back_propagation);
    }
    return;

    case Type::MultiheadAttention:
    {
        MultiheadAttentionLayerForwardPropagation* next_multihead_attention_layer_forward_propagation =
            reinterpret_cast<MultiheadAttentionLayerForwardPropagation*>(next_forward_propagation);

        MultiheadAttentionLayerBackPropagation* next_multihead_attention_layer_back_propagation =
            reinterpret_cast<MultiheadAttentionLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta(next_multihead_attention_layer_forward_propagation,
                               next_multihead_attention_layer_back_propagation,
                               multihead_layer_back_propagation);
    }
    return;

    default:

        return;
    }
}


void MultiheadAttentionLayer::calculate_hidden_delta(PerceptronLayer3DForwardPropagation* next_forward_propagation,
                                                     PerceptronLayer3DBackPropagation* next_back_propagation,
                                                     MultiheadAttentionLayerBackPropagation* back_propagation) const
{
    // Next layer

    const PerceptronLayer3D* next_perceptron_layer = static_cast<PerceptronLayer3D*>(next_back_propagation->layer);

    const Tensor<type, 2>& next_synaptic_weights = next_perceptron_layer->get_synaptic_weights();

    // Next back propagation

    const Tensor<type, 3>& next_error_combinations_derivatives = next_back_propagation->error_combinations_derivatives;

    // This back propagation

    Tensor<type, 3>& deltas = back_propagation->deltas;

    const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 1) };

    deltas.device(*thread_pool_device) = next_error_combinations_derivatives.contract(next_synaptic_weights, contraction_indices);
}


void MultiheadAttentionLayer::calculate_hidden_delta(MultiheadAttentionLayerForwardPropagation* next_forward_propagation,
                                                     MultiheadAttentionLayerBackPropagation* next_back_propagation,
                                                     MultiheadAttentionLayerBackPropagation* back_propagation) const
{
    // Next layer

    const MultiheadAttentionLayer* next_multihead_attention_layer = static_cast<MultiheadAttentionLayer*>(next_back_propagation->layer);

    // This back propagation

    Tensor<type, 3>& deltas = back_propagation->deltas;

    // Next back propagation

    /* Transformer's cross-attention layer takes MHA as input and Perceptron3D as context
    bool is_context; // @todo

    if(is_context)
        deltas.device(*thread_pool_device) = next_back_propagation->error_context_derivatives;
    else
    */

    deltas.device(*thread_pool_device) = next_back_propagation->error_input_derivatives;
}


void MultiheadAttentionLayer::calculate_error_gradient(const Tensor<pair<type*, dimensions>, 1>& inputs,
                                                       LayerForwardPropagation* forward_propagation,
                                                       LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input(inputs(0).first,
                                           inputs(0).second[0],
                                           inputs(0).second[1],
                                           inputs(0).second[2]);

    const TensorMap<Tensor<type, 3>> context(inputs(1).first,
                                             inputs(1).second[0],
                                             inputs(1).second[1],
                                             inputs(1).second[2]);

    Index batch_samples_number = inputs(0).second[0];

    // Forward propagation

    const MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation =
        static_cast<MultiheadAttentionLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 4>& attention_scores = multihead_attention_layer_forward_propagation->attention_scores;
    const Tensor<type, 4>& softmax_attention_scores = multihead_attention_layer_forward_propagation->softmax_attention_scores;
    const Tensor<type, 4>& attention_outputs = multihead_attention_layer_forward_propagation->attention_outputs;

    const Tensor<type, 4>& query = multihead_attention_layer_forward_propagation-> query;
    const Tensor<type, 4>& key = multihead_attention_layer_forward_propagation->key;
    const Tensor<type, 4>& value = multihead_attention_layer_forward_propagation->value;

    // Back propagation

    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    const Tensor<type, 3>& deltas = multihead_attention_layer_back_propagation->deltas;

    Tensor<type, 4>& error_attention_scores_derivatives = multihead_attention_layer_back_propagation->error_attention_scores_derivatives;
    Tensor<type, 4>& error_softmax_attention_scores_derivatives = multihead_attention_layer_back_propagation->error_softmax_attention_scores_derivatives;
    Tensor<type, 4>& error_attention_output_derivatives = multihead_attention_layer_back_propagation->error_attention_output_derivatives;

    Tensor<type, 4>& error_query_derivatives = multihead_attention_layer_back_propagation->error_query_derivatives;
    Tensor<type, 4>& error_key_derivatives = multihead_attention_layer_back_propagation->error_key_derivatives;
    Tensor<type, 4>& error_value_derivatives = multihead_attention_layer_back_propagation->error_value_derivatives;

    Tensor<type, 3>& error_input_derivatives = multihead_attention_layer_back_propagation->error_input_derivatives;
    Tensor<type, 3>& error_context_derivatives = multihead_attention_layer_back_propagation->error_context_derivatives;

    Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;
    Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;
    Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;

    Tensor<type, 3>& projection_weights_derivatives = multihead_attention_layer_back_propagation->projection_weights_derivatives;
    Tensor<type, 1>& projection_biases_derivatives = multihead_attention_layer_back_propagation->projection_biases_derivatives;

    
    // PROJECTION DERIVATIVES

    //calculate_error_projection_weights_derivatives() // using attention_outputs and deltas

    const Eigen::array<IndexPair<Index>, 2> projection_weights_derivatives_contraction_indices = { IndexPair<Index>(2, 0), IndexPair<Index>(0, 1) };
    
    // This is NOT batch matrix multiplication. 
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_attention_outputs((type*)attention_outputs.data() + head_index * input_size*weights_depth*batch_samples_number,
            input_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_projection_weights_derivatives((type*)projection_weights_derivatives.data() + head_index * weights_depth*depth,
            weights_depth, depth);

        head_projection_weights_derivatives.device(*thread_pool_device) =
            head_attention_outputs.contract(deltas, projection_weights_derivatives_contraction_indices);
    }

    
    //calculate_error_projection_biases_derivatives() // using deltas

    projection_biases_derivatives.device(*thread_pool_device) = deltas.sum(Eigen::array<Index, 2>({ 0, 1 }));

    
    // VALUE DERIVATIVES

    //calculate_error_value_derivatives() // using softmax_attention_scores, deltas, projection_weights and value_weights

    const Eigen::array<IndexPair<Index>, 1> attention_output_derivatives_contraction_indices = { IndexPair<Index>(2, 1) };

    // This is NOT batch matrix multiplication. 
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 2>> head_projection_weights((type*)projection_weights.data() + head_index * weights_depth*depth,
            weights_depth, depth);

        TensorMap<Tensor<type, 3>> head_attention_output_derivatives((type*)error_attention_output_derivatives.data() + head_index * input_size*weights_depth*batch_samples_number,
            input_size, weights_depth, batch_samples_number);

        head_attention_output_derivatives.device(*thread_pool_device) =
            deltas.contract(head_projection_weights, attention_output_derivatives_contraction_indices);
    }

    const Eigen::array<IndexPair<Index>, 1> value_derivatives_contraction_indices = { IndexPair<Index>(2, 1) };

    batch_matrix_multiplication(thread_pool_device, softmax_attention_scores, error_attention_output_derivatives, error_value_derivatives, A_B);
    

    //calculate_value_weights_derivatives() // using context and error_value_derivatives

    const Eigen::array<IndexPair<Index>, 2> transformation_weights_derivatives_contraction_indices = { IndexPair<Index>(1, 0), IndexPair<Index>(0, 2) };
    
    // This is NOT batch matrix multiplication. 
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_value_derivatives((type*)error_value_derivatives.data() + head_index * context_size*weights_depth*batch_samples_number,
            context_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_value_weights_derivatives((type*)value_weights_derivatives.data() + head_index * depth*weights_depth,
            depth, weights_depth);
        
        head_value_weights_derivatives.device(*thread_pool_device) =
            context.contract(head_value_derivatives, transformation_weights_derivatives_contraction_indices);
    }

    
    // QUERY AND KEY DERIVATIVES
    
    //calculate_error_attention_scores_derivatives(); // using deltas, projection_weights, value and softmax_derivatives(attention_scores)
    
    batch_matrix_multiplication(thread_pool_device, value, error_attention_output_derivatives, error_softmax_attention_scores_derivatives, A_BT);

    Tensor<type, 4> softmax_activations_derivatives(context_size, context_size, input_size, batch_samples_number);

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_softmax_attention_scores((type*) softmax_attention_scores.data() + head_index * context_size*input_size*batch_samples_number,
            context_size, input_size, batch_samples_number);

        softmax_derivatives(head_softmax_attention_scores, softmax_activations_derivatives);

        const TensorMap<Tensor<type, 3>> head_softmax_attention_scores_derivatives((type*)error_softmax_attention_scores_derivatives.data() + head_index * context_size*input_size*batch_samples_number,
            context_size, input_size, batch_samples_number);

        TensorMap<Tensor<type, 3>> head_attention_scores_derivatives((type*) error_attention_scores_derivatives.data() + head_index * context_size*input_size*batch_samples_number,
            context_size, input_size, batch_samples_number);

        batch_matrix_multiplication(thread_pool_device, softmax_activations_derivatives, head_softmax_attention_scores_derivatives, head_attention_scores_derivatives, AT_B);
    }
    
    
    //calculate_error_query_derivatives() // using error_attention_scores_derivatives, key and query_weights

    batch_matrix_multiplication(thread_pool_device, error_attention_scores_derivatives, key, error_query_derivatives, AT_B);
    error_query_derivatives.device(*thread_pool_device) = error_query_derivatives * scaling_factor;
    

    //calculate_error_key_derivatives() // using error_attention_scores_derivatives, query and key_weights
    
    batch_matrix_multiplication(thread_pool_device, error_attention_scores_derivatives, query, error_key_derivatives, A_B);
    error_key_derivatives.device(*thread_pool_device) = error_key_derivatives * scaling_factor;
    
    
    //calculate_error_input_derivatives() // using error_query_derivatives and query_weights (sum)
    
    error_input_derivatives.setZero();
    
    // This is NOT batch matrix multiplication
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_query_derivatives(error_query_derivatives.data() + head_index * input_size*weights_depth*batch_samples_number,
            input_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_query_weights((type*) query_weights.data() + head_index * depth*weights_depth,
            depth, weights_depth);

        for (Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            error_input_derivatives.chip(sample_index, 0).device(*thread_pool_device) += head_query_derivatives.chip(sample_index, 2)
                                                                                         .contract(head_query_weights, A_BT);
        }
    }
    

    //calculate_error_context_derivatives() // using error_key_derivatives, key_weights, error_value_derivatives and value_weights (sum)

    error_context_derivatives.setZero();

    // This is NOT batch matrix multiplication
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_key_derivatives(error_key_derivatives.data() + head_index * context_size*weights_depth*batch_samples_number,
            context_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_key_weights((type*)key_weights.data() + head_index * depth*weights_depth,
            depth, weights_depth);

        const TensorMap<Tensor<type, 3>> head_value_derivatives(error_value_derivatives.data() + head_index * context_size*weights_depth*batch_samples_number,
            context_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_value_weights((type*)value_weights.data() + head_index * depth * weights_depth,
            depth, weights_depth);

        for (Index sample_index = 0; sample_index < batch_samples_number; sample_index++)
        {
            error_context_derivatives.chip(sample_index, 0).device(*thread_pool_device) +=
                head_key_derivatives.chip(sample_index, 2).contract(head_key_weights, A_BT) +
                head_value_derivatives.chip(sample_index, 2).contract(head_value_weights, A_BT);
        }
    }
    

    //calculate_query_weights_derivatives() // using input and error_query_derivatives
    
    // This is NOT batch matrix multiplication. 
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_query_derivatives((type*)error_query_derivatives.data() + head_index * input_size*weights_depth*batch_samples_number,
            input_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_query_weights_derivatives((type*)query_weights_derivatives.data() + head_index * depth*weights_depth,
            depth, weights_depth);

        head_query_weights_derivatives.device(*thread_pool_device) =
            input.contract(head_query_derivatives, transformation_weights_derivatives_contraction_indices);
    }
    

    //calculate_key_weights_derivatives() // using context and error_key_derivatives

    // This is NOT batch matrix multiplication. 
    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_key_derivatives((type*)error_key_derivatives.data() + head_index * batch_samples_number*context_size*weights_depth,
            context_size, weights_depth, batch_samples_number);

        TensorMap<Tensor<type, 2>> head_key_weights_derivatives((type*)key_weights_derivatives.data() + head_index * depth*weights_depth,
            depth, weights_depth);
        
        head_key_weights_derivatives.device(*thread_pool_device) =
            context.contract(head_key_derivatives, transformation_weights_derivatives_contraction_indices);
    }
}


void MultiheadAttentionLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                              const Index& index,
                                              Tensor<type, 1>& gradient) const
{
    MultiheadAttentionLayerBackPropagation* multihead_attention_layer_back_propagation =
        static_cast<MultiheadAttentionLayerBackPropagation*>(back_propagation);

    const Tensor<type, 3>& query_weights_derivatives = multihead_attention_layer_back_propagation->query_weights_derivatives;

    const Tensor<type, 3>& key_weights_derivatives = multihead_attention_layer_back_propagation->key_weights_derivatives;

    const Tensor<type, 3>& value_weights_derivatives = multihead_attention_layer_back_propagation->value_weights_derivatives;

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
        key_weights_derivatives.data(),
        key_weights_derivatives.data() + key_weights_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += key_weights_derivatives.size();

    copy(/*execution::par,*/
        value_weights_derivatives.data(),
        value_weights_derivatives.data() + value_weights_derivatives.size(),
        gradient_data + gradient_index);

    gradient_index += value_weights_derivatives.size();

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

    query.resize(input_size, weights_depth, new_batch_samples_number, heads_number);
    key.resize(context_size, weights_depth, new_batch_samples_number, heads_number);
    value.resize(context_size, weights_depth, new_batch_samples_number, heads_number);

    attention_scores.resize(context_size, input_size, new_batch_samples_number, heads_number);
    softmax_attention_scores.resize(context_size, input_size, new_batch_samples_number, heads_number);
    attention_outputs.resize(input_size, weights_depth, new_batch_samples_number, heads_number);
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

    deltas.resize(batch_samples_number, input_size, depth);

    deltas_data = deltas.data();

    error_attention_scores_derivatives.resize(context_size, input_size, batch_samples_number, heads_number);
    error_softmax_attention_scores_derivatives.resize(context_size, input_size, batch_samples_number, heads_number);
    error_attention_output_derivatives.resize(input_size, weights_depth, batch_samples_number, heads_number);

    error_query_derivatives.resize(input_size, weights_depth, batch_samples_number, heads_number);
    error_key_derivatives.resize(context_size, weights_depth, batch_samples_number, heads_number);
    error_value_derivatives.resize(context_size, weights_depth, batch_samples_number, heads_number);

    error_input_derivatives.resize(batch_samples_number, input_size, depth);
    error_context_derivatives.resize(batch_samples_number, context_size, depth);

    query_weights_derivatives.resize(depth, weights_depth, heads_number);
    key_weights_derivatives.resize(depth, weights_depth, heads_number);
    value_weights_derivatives.resize(depth, weights_depth, heads_number);

    projection_weights_derivatives.resize(weights_depth, depth, heads_number);
    projection_biases_derivatives.resize(depth);
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
