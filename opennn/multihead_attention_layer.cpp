//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "strings_utilities.h"
#include "tensors.h"
#include "multihead_attention_layer.h"

namespace opennn
{

MultiHeadAttention::MultiHeadAttention(const Index& new_query_sequence_length,
                                       const Index& new_source_sequence_length,
                                       const Index& new_embedding_dimension,
                                       const Index& new_heads_number,
                                       const bool& new_use_causal_mask,
                                       const string& new_name) : Layer()
{
    set(new_query_sequence_length,
        new_source_sequence_length,
        new_embedding_dimension,
        new_heads_number,
        new_use_causal_mask,
        new_name);

    layer_type = Type::MultiheadAttention;
}


Index MultiHeadAttention::get_query_sequence_length() const
{
    return query_sequence_length;
}


Index MultiHeadAttention::get_source_sequence_length() const
{
    return source_sequence_length;
}


Index MultiHeadAttention::get_embedding_dimension() const
{
    return embedding_dimension;
}


Index MultiHeadAttention::get_heads_number() const
{
    return heads_number;
}


type MultiHeadAttention::get_scaling_factor() const
{
    const Index hidden_depth = get_hidden_depth();

    return (hidden_depth == 0)
               ? 0.25
               : type(1) / type(sqrt(hidden_depth));
}


Index MultiHeadAttention::get_hidden_depth() const
{
    return (heads_number == 0)
               ? 0
               : Index(embedding_dimension / heads_number);
}


dimensions MultiHeadAttention::get_input_dimensions() const
{// @todo
    return { query_sequence_length, embedding_dimension};
}


dimensions MultiHeadAttention::get_output_dimensions() const
{
    return { query_sequence_length, embedding_dimension};
}


Index MultiHeadAttention::get_parameters_number() const
{
    return query_weights.size() + query_biases.size()
           + key_weights.size() + key_biases.size()
           + value_weights.size() + value_biases.size()
           + projection_weights.size() + projection_biases.size();
}


Tensor<type, 1> MultiHeadAttention::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    Index index = 0;

    copy_to_vector(parameters, query_weights, index);
    copy_to_vector(parameters, query_biases, index);
    copy_to_vector(parameters, key_weights, index);
    copy_to_vector(parameters, key_biases, index);
    copy_to_vector(parameters, value_weights, index);
    copy_to_vector(parameters, value_biases, index);
    copy_to_vector(parameters, projection_weights, index);
    copy_to_vector(parameters, projection_biases, index);

    return parameters;
}


void MultiHeadAttention::set(const Index& new_query_sequence_length,
                             const Index& new_source_sequence_length,
                             const Index& new_embedding_dimension,
                             const Index& new_heads_number,
                             const bool& new_use_causal_mask,
                             const string& new_name)
{
    query_sequence_length = new_query_sequence_length;

    source_sequence_length = new_source_sequence_length;

    heads_number = new_heads_number;

    embedding_dimension = new_embedding_dimension;

    name = new_name;

    dropout_rate = 0;

    const Index hidden_depth = get_hidden_depth();

    query_weights.resize(embedding_dimension, hidden_depth, heads_number);
    query_biases.resize(hidden_depth, heads_number);

    key_weights.resize(embedding_dimension, hidden_depth, heads_number);
    key_biases.resize(hidden_depth, heads_number);

    value_weights.resize(embedding_dimension, hidden_depth, heads_number);
    value_biases.resize(hidden_depth, heads_number);

    projection_weights.resize(embedding_dimension, embedding_dimension);
    projection_biases.resize(embedding_dimension);

    set_parameters_glorot();

    use_causal_mask = new_use_causal_mask;

    if (!use_causal_mask) return;

    causal_mask.resize(query_sequence_length, source_sequence_length);
    causal_mask.setZero();

#pragma omp parallel for

    for (Index i = 0; i < query_sequence_length; i++)
        for (Index j = i + 1; j < source_sequence_length; j++)
            causal_mask(i, j) = minus_inf;
}


void MultiHeadAttention::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(query_weights, new_parameters, index);
    copy_from_vector(query_biases, new_parameters, index);
    copy_from_vector(key_weights, new_parameters, index);
    copy_from_vector(key_biases, new_parameters, index);
    copy_from_vector(value_weights, new_parameters, index);
    copy_from_vector(value_biases, new_parameters, index);
    copy_from_vector(projection_weights, new_parameters, index);
    copy_from_vector(projection_biases, new_parameters, index);
}


void MultiHeadAttention::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    set_random(query_weights, minimum, maximum);
    set_random(query_biases, minimum, maximum);
    set_random(key_weights, minimum, maximum);
    set_random(key_biases, minimum, maximum);
    set_random(value_weights, minimum, maximum);
    set_random(value_biases, minimum, maximum);
    set_random(projection_weights, minimum, maximum);
    set_random(projection_biases, minimum, maximum);
}


void MultiHeadAttention::set_parameters_glorot()
{
    query_biases.setZero();
    key_biases.setZero();
    value_biases.setZero();
    projection_biases.setZero();

    const Index hidden_depth = get_hidden_depth();
    const Index embedding_dimension = get_embedding_dimension();
    const Index heads_number = get_heads_number();
    const type limit = sqrt(6 / type(embedding_dimension + hidden_depth * heads_number));
    const type minimum = -limit;
    const type maximum = limit;

    set_random(query_weights, minimum, maximum);
    set_random(key_weights, minimum, maximum);
    set_random(value_weights, minimum, maximum);
    set_random(projection_weights, minimum, maximum);
}


void MultiHeadAttention::set_parameters_constant(const type& value)
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


void MultiHeadAttention::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void MultiHeadAttention::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    const Index batch_size = attention_scores.dimension(2);

    const Index context_input_size = source_sequence_length * query_sequence_length;

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            type* sample_attention_scores_data = attention_scores.data()
                                                 // + (sample_index + head_index) * context_input_size * batch_size;
                                                 + (sample_index + head_index * batch_size) * context_input_size;
            // + (sample_index * heads_number + head_index) * context_input_size * batch_size;

            TensorMap<Tensor<type, 2>> sample_attention_scores(sample_attention_scores_data,
                                                               source_sequence_length,
                                                               query_sequence_length);

            sample_attention_scores.device(*thread_pool_device) += causal_mask;
        }
    }
}


void MultiHeadAttention::calculate_query(const Tensor<type, 3>& query_input, Tensor<type, 4>& query) const
{
    // Verify embedding dimension matches
    assert(embedding_dimension == query_weights.dimension(0));

    // Tensor contraction dimensions:
    // query_input: (batch_size, query_sequence_length, embedding_dimension)
    // query_weights: (embedding_dimension, hidden_depth, heads_number)

    const array<Index, 4> shuffle_dims = { 1, 2, 0, 3 };

    // Perform contraction to get (batch_size, query_sequence_length, hidden_depth, heads_number)

    query.device(*thread_pool_device)
        = query_input.contract(query_weights, axes(2,0)).shuffle(shuffle_dims);

    // @todo try this

    const Index hidden_depth = get_hidden_depth();

    const array<ptrdiff_t, 4> reshape_dims = { 1, hidden_depth, 1, heads_number };

    // Broadcast it along the dimensions (query_sequence_length and batch_size)
    // Here, query.dimension(0) is query_sequence_length and query.dimension(2) is batch_size.
    const array<ptrdiff_t, 4> broadcast_dims = { query.dimension(0), 1, query.dimension(2), 1 };

    // Using Eigen's tensor expressions, the bias is now added to every appropriate slice.
    query.device(*thread_pool_device) = query + query_biases.reshape(reshape_dims).broadcast(broadcast_dims);

/*
    const Index batch_size = query_input.dimension(0);

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 1>> head_query_biases = tensor_map(query_biases, head_index);

        for (Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            TensorMap<Tensor<type, 2>> head_sample_query = tensor_map(query, head_index, sample_index);

            sum_columns(thread_pool_device.get(), head_query_biases, head_sample_query);
        }
    }
*/
}


void MultiHeadAttention::calculate_key(const Tensor<type, 3>& source_input, Tensor<type, 4>& key) const
{
    // Verify embedding dimension matches
    assert(embedding_dimension == key_weights.dimension(0));

    // Tensor contraction dimensions:
    // query_input: (batch_size, query_sequence_length, embedding_dimension)
    // query_weights: (embedding_dimension, hidden_depth, heads_number)

    const array<Index, 4> shuffle_dims = { 1, 2, 0, 3 };
    // Perform contraction to get (batch_size, query_sequence_length, hidden_depth, heads_number)

    key.device(*thread_pool_device) = source_input.contract(key_weights, axes(2,0))
                .shuffle(shuffle_dims);

    const Index batch_size = source_input.dimension(0);

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 1>> head_key_biases = tensor_map(key_biases, head_index);

        for (Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            TensorMap<Tensor<type, 2>> head_sample_key = tensor_map(key, head_index, sample_index);

            sum_columns(thread_pool_device.get(), head_key_biases, head_sample_key);
        }
    }
}


void MultiHeadAttention::calculate_value(const Tensor<type, 3>& source_input, Tensor<type, 4>& value) const
{
    const Index batch_size = source_input.dimension(0);

    assert(embedding_dimension == query_weights.dimension(0));

    // Tensor contraction dimensions:
    // query_input: (batch_size, query_sequence_length, embedding_dimension)
    // query_weights: (embedding_dimension, hidden_depth, heads_number)

    const array<Index, 4> shuffle_dims = { 1, 2, 0, 3 };
    // Perform contraction to get (batch_size, query_sequence_length, hidden_depth, heads_number)

    value.device(*thread_pool_device) = source_input.contract(value_weights, axes(2,0))
                .shuffle(shuffle_dims);

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 1>> head_value_biases = tensor_map(value_biases, head_index);

        for (Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            TensorMap<Tensor<type, 2>> head_sample_value = tensor_map(value, head_index, sample_index);

            sum_columns(thread_pool_device.get(), head_value_biases, head_sample_value);
        }
    }
}


void MultiHeadAttention::calculate_attention_weights(const Tensor<type, 4>& query,
                                                     const Tensor<type, 4>& key,
                                                     Tensor<type, 4>& attention_weights) const
{

    batch_matrix_multiplication(thread_pool_device.get(), key, query, attention_weights, axes<Index>(1,1));

    const type scaling_factor = get_scaling_factor();

    attention_weights.device(*thread_pool_device) = attention_weights * scaling_factor;

    if(use_causal_mask)
        apply_causal_mask(attention_weights);

    softmax(attention_weights);
}


void MultiHeadAttention::calculate_attention_outputs(const Tensor<type, 4>& value,
                                                     const Tensor<type, 4>& attention_weights,
                                                     Tensor<type, 4>& attention_outputs) const
{
    batch_matrix_multiplication(thread_pool_device.get(),
                                attention_weights,
                                value,
                                attention_outputs,
                                axes<Index>(0,0));
}


void MultiHeadAttention::concatenate_heads(const Tensor<type, 4>& attention_outputs,
                                           Tensor<type, 3>& concatenated_attention_outputs) const
{
    const Index batch_size = attention_outputs.dimension(2);
    const Index hidden_depth = attention_outputs.dimension(1);

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        TensorMap<Tensor<type, 2>> sample_output = tensor_map(concatenated_attention_outputs, sample_index);

        for(Index head_index = 0; head_index < heads_number; head_index++)
        {
            const TensorMap<Tensor<type, 2>> head_output = tensor_map(attention_outputs, head_index, sample_index);

            const Index start_col = head_index * hidden_depth;

            sample_output.slice(array<Index,2>{0, start_col},
                                array<Index,2>{query_sequence_length, hidden_depth}) = head_output;
        }
    }
}
/*
void MultiHeadAttention::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
                                                     Tensor<type, 4>& projection_outputs,
                                                     Tensor<type, 3>& outputs) const
{

    const Index batch_size = outputs.dimension(0);
    const Index heads_number = get_heads_number();

    for (Index head_index = 0; head_index < heads_number; head_index++)
    {
        // const Tensor<type, 2> head_projection_weights = projection_weights.slice(array<Index,2>{head_index*get_hidden_depth(),0}, array<Index,2>{get_hidden_depth(),embedding_dimension});
        const TensorMap<Tensor<type, 2>> head_projection_weights = tensor_map(projection_weights, head_index);
        TensorMap<Tensor<type, 3>> head_projection_outputs = tensor_map(projection_outputs, head_index);

        for (Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            const TensorMap<Tensor<type, 2>> sample_attention_output = tensor_map(attention_outputs, head_index, sample_index);

            head_projection_outputs.chip(sample_index, 0).device(*thread_pool_device)
                = sample_attention_output.contract(head_projection_weights, A_B);
            // if(sample_index < 2 && iteration == 3){
            //     cout << "sample " << sample_index << " attention_output of head " << head_index << ":\n" << sample_attention_output << endl;
            //     cout << "head_projection_weights of head " << head_index << ":\n" << head_projection_weights << endl;
            // }
        }
    }

    outputs.device(*thread_pool_device) = projection_outputs.sum(projection_sum_index);

    // cout << "Outputs before:\n" << outputs << endl<<endl<<endl<<endl;

    sum_matrices(thread_pool_device.get(), projection_biases, outputs);

}
*/
void MultiHeadAttention::calculate_output_projection(const Tensor<type, 3>& concatenated_attention_outputs,
                                                     Tensor<type, 3>& outputs) const
{
    const Index batch_size = outputs.dimension(0);

    for (Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        const TensorMap<Tensor<type, 2>> sample_attention_output = tensor_map(concatenated_attention_outputs, sample_index);

        outputs.chip(sample_index, 0).device(*thread_pool_device)
            = sample_attention_output.contract(projection_weights, axes(1,0));
    }

    sum_matrices(thread_pool_device.get(), projection_biases, outputs);
}

void MultiHeadAttention::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                           const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> query_input = tensor_map_3(input_pairs[0]);
    const TensorMap<Tensor<type, 3>> source_input = tensor_map_3(input_pairs[1]);

    MultiheadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiheadAttentionForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& query = this_forward_propagation->query;
    Tensor<type, 4>& key = this_forward_propagation->key;
    Tensor<type, 4>& value = this_forward_propagation->value;

    Tensor<type, 4>& attention_weights = this_forward_propagation->attention_weights;

    Tensor<type, 4>& attention_outputs = this_forward_propagation->attention_outputs;

    Tensor<type, 3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    // Tensor<type, 4>& projection_outputs = this_forward_propagation->projection_outputs;
    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    calculate_query(query_input, query);
    calculate_key(source_input, key);
    calculate_value(source_input, value);

    calculate_attention_weights(query,
                                key,
                                attention_weights);

    if(is_training && dropout_rate > type(0))
        dropout(attention_weights, dropout_rate);

    calculate_attention_outputs(value,
                                attention_weights,
                                attention_outputs);

    concatenate_heads(attention_outputs,
                      concatenated_attention_outputs);

    calculate_output_projection(concatenated_attention_outputs,
                                outputs);
}


void MultiHeadAttention::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input = tensor_map_3(input_pairs[0]);
    const TensorMap<Tensor<type, 3>> context = tensor_map_3(input_pairs[1]);
    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    const Index batch_size = input_pairs[0].second[0];
    const Index hidden_depth = get_hidden_depth();

    const type scaling_factor = get_scaling_factor();

    // Forward propagation

    const MultiheadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiheadAttentionForwardPropagation*>(forward_propagation.get());

    // Back propagation

    MultiheadAttentionBackPropagation* this_back_propagation =
        static_cast<MultiheadAttentionBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_query_derivatives = this_back_propagation->input_query_derivatives;
    input_query_derivatives.setZero();

    Tensor<type, 3>& input_source_derivatives = this_back_propagation->input_source_derivatives;
    input_source_derivatives.setZero();

    Tensor<type, 1>& aux_rows = this_back_propagation->aux_rows;

    Tensor<type, 1>& projection_bias_derivatives = this_back_propagation->projection_bias_derivatives;

    const Tensor<type,3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;
    Tensor<type, 2>& projection_weights_derivatives = this_back_propagation->projection_weight_derivatives;
    Tensor<type,3>& concatenated_attention_output_deltas = this_back_propagation->concatenated_attention_output_deltas;
    Tensor<type,4>& attention_output_deltas = this_back_propagation->attention_output_deltas;

    // Calculation

    projection_bias_derivatives.device(*thread_pool_device) = deltas.sum(projection_bias_derivatives_sum_indices);
    projection_weights_derivatives.device(*thread_pool_device) = concatenated_attention_outputs.contract(deltas, projection_weight_derivatives_contraction_indices);

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        TensorMap<Tensor<type, 2>> sample_concatenated_attention_output_deltas
            = tensor_map(concatenated_attention_output_deltas, sample_index);

        sample_concatenated_attention_output_deltas.device(*thread_pool_device)
            = deltas.chip(sample_index, 0).contract(projection_weights, axes(1,1));

        // Attention output deltas

        for(Index head_index = 0; head_index < heads_number; head_index++)
        {
            TensorMap<Tensor<type, 2>> head_attention_output_deltas
                = tensor_map(attention_output_deltas, head_index, sample_index);

            head_attention_output_deltas.device(*thread_pool_device)
                = sample_concatenated_attention_output_deltas.slice(array<Index,2>{0, head_index * hidden_depth},
                                                                    array<Index,2>{query_sequence_length, hidden_depth});
        }
    }

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        // Forward propagation

        const TensorMap<Tensor<type, 3>> head_query = tensor_map(this_forward_propagation->query, head_index);
        const TensorMap<Tensor<type, 3>> head_key = tensor_map(this_forward_propagation->key, head_index);
        const TensorMap<Tensor<type, 3>> head_value = tensor_map(this_forward_propagation->value, head_index);

        const TensorMap<Tensor<type, 3>> head_attention_weights = tensor_map(this_forward_propagation->attention_weights, head_index);

        // Back propagation

        TensorMap<Tensor<type, 3>> head_attention_weight_derivatives_xxx = tensor_map(this_back_propagation->attention_weight_deltas_xxx, head_index);
        TensorMap<Tensor<type, 3>> head_attention_output_deltas = tensor_map(this_back_propagation->attention_output_deltas, head_index);

        TensorMap<Tensor<type, 3>> head_query_derivatives = tensor_map(this_back_propagation->query_deltas, head_index);
        TensorMap<Tensor<type, 3>> head_key_derivatives = tensor_map(this_back_propagation->key_deltas, head_index);
        TensorMap<Tensor<type, 3>> head_value_derivatives = tensor_map(this_back_propagation->value_deltas, head_index);

        //

        TensorMap<Tensor<type, 1>> head_query_bias_derivatives = tensor_map(this_back_propagation->query_bias_derivatives, head_index);
        TensorMap<Tensor<type, 2>> head_query_weight_derivatives = tensor_map(this_back_propagation->query_weight_derivatives, head_index);

        TensorMap<Tensor<type, 1>> head_key_bias_derivatives = tensor_map(this_back_propagation->key_bias_derivatives, head_index);
        TensorMap<Tensor<type, 2>> head_key_weight_derivatives = tensor_map(this_back_propagation->key_weight_derivatives, head_index);

        TensorMap<Tensor<type, 1>> head_value_bias_derivatives = tensor_map(this_back_propagation->value_bias_derivatives, head_index);
        TensorMap<Tensor<type, 2>> head_value_weight_derivatives = tensor_map(this_back_propagation->value_weight_derivatives, head_index);

        // VALUE DERIVATIVES

        batch_matrix_multiplication(thread_pool_device.get(),
                                    head_attention_weights,
                                    head_attention_output_deltas,
                                    head_value_derivatives,
                                    axes<Index>(1,0));

        // VALUE WEIGHT DERIVATIVES

        head_value_weight_derivatives.device(*thread_pool_device)
            = context.contract(head_value_derivatives, transformation_weight_derivatives_contraction_indices);

        head_value_bias_derivatives.device(*thread_pool_device) = head_value_derivatives.sum(bias_derivatives_sum_indices);

        // ATTENTION WEIGHT DERIVATIVES

        batch_matrix_multiplication(thread_pool_device.get(),
                                    head_value,
                                    head_attention_output_deltas,
                                    head_attention_weight_derivatives_xxx,
                                    axes<Index>(1,1));

        softmax_derivatives_times_tensor(head_attention_weights,
                                         head_attention_weight_derivatives_xxx,
                                         aux_rows);

        head_attention_weight_derivatives_xxx.device(*thread_pool_device) = head_attention_weight_derivatives_xxx * scaling_factor;

        // QUERY DERIVATIVES

        batch_matrix_multiplication(thread_pool_device.get(),
                                    head_attention_weight_derivatives_xxx,
                                    head_key,
                                    head_query_derivatives,
                                    axes<Index>(0,0));

        // KEY DERIVATIVES

        batch_matrix_multiplication(thread_pool_device.get(),
                                    head_attention_weight_derivatives_xxx,
                                    head_query,
                                    head_key_derivatives,
                                    axes<Index>(1,0));

        // QUERY WEIGHTS DERIVATIVES

        head_query_weight_derivatives.device(*thread_pool_device)
            = input.contract(head_query_derivatives, transformation_weight_derivatives_contraction_indices);

        head_query_bias_derivatives.device(*thread_pool_device) = head_query_derivatives.sum(bias_derivatives_sum_indices);

        // KEY WEIGHTS DERIVATIVES

        head_key_weight_derivatives.device(*thread_pool_device)
            = context.contract(head_key_derivatives, transformation_weight_derivatives_contraction_indices);

        head_key_bias_derivatives.device(*thread_pool_device) = head_key_derivatives.sum(bias_derivatives_sum_indices);
    }


    // INPUT QUERY DERIVATIVES

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 2>> head_query_weights = tensor_map(query_weights, head_index);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            const TensorMap<Tensor<type, 2>> sample_query_derivatives = tensor_map(this_back_propagation->query_deltas, head_index, sample_index);

            input_query_derivatives.chip(sample_index, 0).device(*thread_pool_device)
                += sample_query_derivatives.contract(head_query_weights, axes(1,1));
        }
    }

/*
    // @todo try this

    auto heads_result = query_deltas.contract(query_weights, axes(3,1));

    const array<int, 1> reduction_dims = { 0 };

    auto result = heads_result.sum(reduction_dims);

    input_query_derivatives.device(*thread_pool_device) += result;
*/
    // INPUT SOURCE DERIVATIVES

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 2>> head_key_weights = tensor_map(key_weights, head_index);
        const TensorMap<Tensor<type, 2>> head_value_weights = tensor_map(value_weights, head_index);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            const TensorMap<Tensor<type, 2>> sample_key_derivatives = tensor_map(this_back_propagation->key_deltas, head_index, sample_index);
            const TensorMap<Tensor<type, 2>> sample_value_derivatives = tensor_map(this_back_propagation->value_deltas, head_index, sample_index);

            input_source_derivatives.chip(sample_index, 0).device(*thread_pool_device)
                += sample_key_derivatives.contract(head_key_weights, axes(1,1))
                   + sample_value_derivatives.contract(head_value_weights, axes(1,1));
        }
    }
}


void MultiHeadAttention::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                         Index& index,
                                         Tensor<type, 1>& gradient) const
{
    MultiheadAttentionBackPropagation* multihead_attention_back_propagation =
        static_cast<MultiheadAttentionBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, multihead_attention_back_propagation->query_weight_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->query_bias_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->key_weight_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->key_bias_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->value_weight_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->value_bias_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->projection_weight_derivatives, index);
    copy_to_vector(gradient, multihead_attention_back_propagation->projection_bias_derivatives, index);
}


void MultiHeadAttention::from_XML(const XMLDocument& document)
{
    // @todo update notation

    const XMLElement* multihead_attention_layer_element = document.FirstChildElement("MultiheadAttention");

    if(!multihead_attention_layer_element)
        throw runtime_error("MultiheadAttention element is nullptr.\n");

    const string new_name = read_xml_string(multihead_attention_layer_element, "Name");
    const Index new_input_size = read_xml_index(multihead_attention_layer_element, "InputSize");
    const Index new_context_size = read_xml_index(multihead_attention_layer_element, "ContextSize");
    const Index new_depth = read_xml_index(multihead_attention_layer_element, "Depth");
    const Index new_heads_number = read_xml_index(multihead_attention_layer_element, "HeadsNumber");
    const Index new_use_causal_mask = read_xml_bool(multihead_attention_layer_element, "CausalMask");

    set(new_input_size, new_context_size, new_depth, new_heads_number, new_use_causal_mask, new_name);

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(multihead_attention_layer_element, "Parameters"), " "), index);
}


void MultiHeadAttention::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("MultiheadAttention");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputSize", to_string(get_query_sequence_length()));
    add_xml_element(printer, "ContextSize", to_string(get_source_sequence_length()));
    add_xml_element(printer, "Depth", to_string(get_embedding_dimension()));
    add_xml_element(printer, "HiddenDepth", to_string(get_hidden_depth()));
    add_xml_element(printer, "HeadsNumber", to_string(get_heads_number()));
    add_xml_element(printer, "CausalMask", to_string(use_causal_mask ? 1 : 0));
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


MultiheadAttentionForwardPropagation::MultiheadAttentionForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> MultiheadAttentionForwardPropagation::get_outputs_pair() const
{
    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();

    return { (type*)outputs.data(), {{ batch_size, query_sequence_length, embedding_dimension }} };
}


void MultiheadAttentionForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    batch_size = new_batch_size;

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index hidden_depth = multihead_attention_layer->get_hidden_depth();

    query.resize(query_sequence_length, hidden_depth, batch_size, heads_number);
    key.resize(source_sequence_length, hidden_depth, batch_size, heads_number);
    value.resize(source_sequence_length, hidden_depth, batch_size, heads_number);

    attention_weights.resize(source_sequence_length, query_sequence_length, batch_size, heads_number);
    attention_outputs.resize(query_sequence_length, hidden_depth, batch_size, heads_number);

    concatenated_attention_outputs.resize(query_sequence_length, embedding_dimension, batch_size);

    projection_outputs.resize(batch_size, query_sequence_length, embedding_dimension, heads_number);

    outputs.resize(batch_size, query_sequence_length, embedding_dimension);

    // Auxiliar

    sample_matrix.resize(query_sequence_length, hidden_depth);

}


void MultiheadAttentionForwardPropagation::print() const
{
    cout << "Outputs dimensions:" << endl;
    //cout << output_dimensions << endl;
    cout << "Outputs:" << endl;
    //cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
}


void MultiheadAttentionBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    batch_size = new_batch_size;

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index hidden_depth = multihead_attention_layer->get_hidden_depth();

    query_weight_derivatives.resize(embedding_dimension, hidden_depth, heads_number);
    key_weight_derivatives.resize(embedding_dimension, hidden_depth, heads_number);
    value_weight_derivatives.resize(embedding_dimension, hidden_depth, heads_number);
    projection_weight_derivatives.resize(embedding_dimension, embedding_dimension);

    query_bias_derivatives.resize(hidden_depth, heads_number);
    key_bias_derivatives.resize(hidden_depth, heads_number);
    value_bias_derivatives.resize(hidden_depth, heads_number);
    projection_bias_derivatives.resize(embedding_dimension);

    input_query_derivatives.resize(batch_size, query_sequence_length, embedding_dimension);
    input_source_derivatives.resize(batch_size, source_sequence_length, embedding_dimension);

    // Auxiliar

    attention_weight_deltas_xxx.resize(source_sequence_length, query_sequence_length, batch_size, heads_number);

    attention_output_deltas.resize(query_sequence_length, hidden_depth, batch_size, heads_number);

    concatenated_attention_output_deltas.resize(query_sequence_length, embedding_dimension, batch_size);

    query_deltas.resize(query_sequence_length, hidden_depth, batch_size, heads_number);

    key_deltas.resize(source_sequence_length, hidden_depth, batch_size, heads_number);
    value_deltas.resize(source_sequence_length, hidden_depth, batch_size, heads_number);

    sample_deltas.resize(query_sequence_length, embedding_dimension);
    aux_rows.resize(source_sequence_length);
}


void MultiheadAttentionBackPropagation::print() const
{
}


MultiheadAttentionBackPropagation::MultiheadAttentionBackPropagation(const Index& new_batch_size,
                                                                     Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> MultiheadAttentionBackPropagation::get_input_derivative_pairs() const
{
    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();

    return
        {{(type*)(input_query_derivatives.data()), {batch_size, query_sequence_length, embedding_dimension}},
         {(type*)(input_source_derivatives.data()), {batch_size, source_sequence_length, embedding_dimension}} };
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
