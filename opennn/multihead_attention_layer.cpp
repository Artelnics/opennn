//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "multihead_attention_layer.h"

namespace opennn
{

MultiHeadAttention::MultiHeadAttention(const dimensions& new_input_dimensions,
                                       const Index& new_heads_number,
                                       const string& new_name) : Layer()
{
    // self-attention
    set(new_input_dimensions[0],    // query_sequence_length
        new_input_dimensions[0],    // source_sequence_length
        new_input_dimensions[1],    // embedding_dimension
        new_heads_number,
        false,
        new_name);
}

MultiHeadAttention::MultiHeadAttention(const dimensions& new_query_dimensions,
                                       const dimensions& new_source_dimensions,
                                       const Index& new_heads_number,
                                       const string& new_name) : Layer()
{
    if (new_query_dimensions[1] != new_source_dimensions[1])
        throw runtime_error("MultiHeadAttention Error: embedding dimension must be the same for query and source.");

    // cross-attention
    set(new_query_dimensions[0],    // query_sequence_length
        new_source_dimensions[0],   // source_sequence_length
        new_query_dimensions[1],    // embedding_dimension
        new_heads_number,
        false,
        new_name);
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
    const Index head_dimension = get_head_dimension();

    return (head_dimension == 0)
               ? 0.25
               : type(1) / type(sqrt(head_dimension));
}


Index MultiHeadAttention::get_head_dimension() const
{
    return (heads_number == 0)
               ? 0
               : Index(embedding_dimension / heads_number);
}


dimensions MultiHeadAttention::get_input_dimensions() const
{
    return { query_sequence_length, embedding_dimension };
}


dimensions MultiHeadAttention::get_output_dimensions() const
{
    return { query_sequence_length, embedding_dimension };
}


vector<pair<type *, Index> > MultiHeadAttention::get_parameter_pairs() const
{

    return {
        {(type*)query_weights.data(), query_weights.size()},
        {(type*)query_biases.data(), query_biases.size()},
        {(type*)key_weights.data(), key_weights.size()},
        {(type*)key_biases.data(), key_biases.size()},
        {(type*)value_weights.data(), value_weights.size()},
        {(type*)value_biases.data(), value_biases.size()},
        {(type*)projection_weights.data(), projection_weights.size()},
        {(type*)projection_biases.data(), projection_biases.size()}
    };
}


void MultiHeadAttention::set(const Index& new_query_sequence_length,
                             const Index& new_source_sequence_length,
                             const Index& new_embedding_dimension,
                             const Index& new_heads_number,
                             const bool& new_use_causal_mask,
                             const string& new_label)
{
    name = "MultiHeadAttention";
    query_sequence_length = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    heads_number = new_heads_number;
    embedding_dimension = new_embedding_dimension;
    label = new_label;

    if (embedding_dimension > 0 && heads_number > 0 && embedding_dimension % heads_number != 0)
        throw runtime_error("MultiHeadAttention Error: The embedding dimension must be divisible by the number of heads.");

    const Index head_dimension = get_head_dimension();

    query_weights.resize(embedding_dimension, embedding_dimension);
    query_biases.resize(embedding_dimension);

    key_weights.resize(embedding_dimension, embedding_dimension);
    key_biases.resize(embedding_dimension);

    value_weights.resize(embedding_dimension, embedding_dimension);
    value_biases.resize(embedding_dimension);

    projection_weights.resize(embedding_dimension, embedding_dimension);
    projection_biases.resize(embedding_dimension);

    set_parameters_glorot();

    use_causal_mask = new_use_causal_mask;

    if (use_causal_mask)
        causal_mask = Tensor<type, 2>(query_sequence_length, source_sequence_length)
                          .generate([=](const Eigen::array<Index, 2>& idx) -> type {
                              Index row = idx[0];
                              Index col = idx[1];
                              return (col > row) ? minus_inf : 0;
                          });
}


void MultiHeadAttention::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void MultiHeadAttention::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    // const Index batch_size = attention_scores.dimension(2);

    // const Index context_input_size = source_sequence_length * query_sequence_length;

    // for(Index head_index = 0; head_index < heads_number; head_index++)
    // {
    //     for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    //     {
    //         type* sample_attention_scores_data = attention_scores.data()
    //         + (sample_index + head_index * batch_size) * context_input_size;

    //         // + (sample_index + head_index) * context_input_size * batch_size;
    //         // + (sample_index * heads_number + head_index) * context_input_size * batch_size;

    //         TensorMap<Tensor<type, 2>> sample_attention_scores(sample_attention_scores_data,
    //                                                            source_sequence_length,
    //                                                            query_sequence_length);

    //         sample_attention_scores.device(*thread_pool_device) += causal_mask;
    //     }
    // }
}


void MultiHeadAttention::apply_key_padding_mask(const Tensor<bool, 2>& key_padding_mask,
                                                Tensor<type, 4>& attention_weights) const
{
    // @Todo (I don't know if it is building the mask correctly)
    const Index batch_size  = attention_weights.dimension(2);

    Tensor<type, 2> key_padding_mask_type(key_padding_mask.dimension(0),key_padding_mask.dimension(1));

    for(Index h = 0; h < heads_number; ++h)
    {
        for(Index b = 0; b < batch_size; ++b)
        {
            TensorMap<Tensor<type, 2>> head_sample_attention_weights = tensor_map(attention_weights,h,b);

            head_sample_attention_weights.device(*thread_pool_device)
                += key_padding_mask.chip(b, 0)
                       .cast<type>()
                       .reshape(array<Index,2>{source_sequence_length, 1})
                       .broadcast(array<Index,2>{1, query_sequence_length})
                   * type(-10e9);
        }
    }
}


void MultiHeadAttention::calculate_attention_outputs(const Tensor<type, 4>& value,
                                                     const Tensor<type, 4>& attention_weights,
                                                     Tensor<type, 4>& attention_outputs) const
{
    batch_matrix_multiplication<type, 4>(thread_pool_device.get(),
                                         attention_weights,
                                         value,
                                         attention_outputs,
                                         axes<Index>(0,0));
}


void MultiHeadAttention::concatenate_heads(const Tensor<type, 4>& attention_outputs,
                                           Tensor<type, 3>& concatenated_attention_outputs) const
{
/*
    const Index batch_size = attention_outputs.dimension(2);
    const Index head_dimension = attention_outputs.dimension(1);

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        TensorMap<Tensor<type, 2>> sample_output = tensor_map(concatenated_attention_outputs, sample_index);

        for(Index head_index = 0; head_index < heads_number; head_index++)
        {
            const TensorMap<Tensor<type, 2>> head_output = tensor_map(attention_outputs, head_index, sample_index);

            const Index start_col = head_index * head_dimension;

            sample_output.slice(array<Index,2>{0, start_col},
                                array<Index,2>{query_sequence_length, head_dimension}) = head_output;
        }
    }
*/
    // @todo check This gives the AI

    // Original dimensions of attention_outputs: (query_sequence_length, head_dimension, batch_size, heads_number)
    // We want to merge dimensions 1 (head_dimension) and 3 (heads_number) to form the new embedding_dimension.
    // The desired output shape is (batch_size, query_sequence_length, embedding_dimension)

    // Define the shuffling pattern: map old dimensions to new ones.
    // Old: 0(seq), 1(depth), 2(batch), 3(heads)
    // New: 2(batch), 0(seq), 3(heads), 1(depth)

    // const array<int, 4> shuffling_pattern = {2, 0, 3, 1};

    // concatenated_attention_outputs.device(*thread_pool_device) =
    //     attention_outputs.shuffle(shuffling_pattern)
    //         .reshape(array<Index, 3>{
    //             attention_outputs.dimension(2), // batch_size
    //             query_sequence_length,
    //             heads_number * get_hidden_depth() // embedding_dimension
    //         });

    const array<int, 4> shuffling_pattern = {0, 3, 1, 2};

    const array<Index, 3> target_shape = {
        query_sequence_length,
        heads_number * get_head_dimension(),
        attention_outputs.dimension(2) // batch_size
    };

    concatenated_attention_outputs.device(*thread_pool_device) =
        attention_outputs.shuffle(shuffling_pattern)
            .reshape(target_shape);
}


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
    /*
    outputs.device(*thread_pool_device) = outputs
        + projection_biases.reshape(array<Index, 3>{1, 1, projection_biases.dimension(0)})
                           .broadcast(array<Index, 3>{outputs.dimension(0), outputs.dimension(1), 1});
    */
    sum_matrices(thread_pool_device.get(), projection_biases, outputs);
}


void MultiHeadAttention::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                           const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> query_input = tensor_map<3>(input_pairs[0]);

    const TensorMap<Tensor<type, 3>> source_input = (input_pairs.size() == 1)
                                                    ? query_input
                                                    : tensor_map<3>(input_pairs[1]);

    MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& query = this_forward_propagation->query;
    Tensor<type, 4>& key = this_forward_propagation->key;
    Tensor<type, 4>& value = this_forward_propagation->value;

    Tensor<type, 4>& attention_weights = this_forward_propagation->attention_weights;
    Tensor<type, 4>& attention_outputs = this_forward_propagation->attention_outputs;
    Tensor<type, 3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;
    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    const Index batch_size = this_forward_propagation->batch_size;

    const Index head_dimension = get_head_dimension();

    //const array<IndexPair<Index>, 1> contract_dims = { IndexPair<Index>(2, 0) };
    //const array<Index, 3> bias_reshape_dims = {1, 1, embedding_dimension};
    //const array<Index, 3> bias_broadcast_dims_query = {batch_size, query_sequence_length, 1}; // @todo for key and value is not the same
    //const array<Index, 3> bias_broadcast_dims_source = {batch_size, source_sequence_length, 1}; // @todo for key and value is not the same
    //const array<Index, 4> final_reshape_dims_query = {batch_size, query_sequence_length, heads_number, head_dimension};
    //const array<Index, 4> final_reshape_dims_source = {batch_size, source_sequence_length, heads_number, head_dimension};
    //const array<Index, 4> shuffle_order = {0, 2, 1, 3};
    //const array<int, 4> key_transpose_order = {0, 1, 3, 2};
    //array<IndexPair<Index>, 1> contract_dims_attention_scores = { IndexPair<Index>(3, 2) };

    query.device(*thread_pool_device) = (query_input.contract(query_weights, axes(2,0))
        + query_biases.reshape(array_3(1, 1, embedding_dimension))
                      .broadcast(array_3(batch_size, query_sequence_length, 1)))
        .reshape(array_4(batch_size, query_sequence_length, heads_number, head_dimension))
        .shuffle(array_4(0, 2, 1, 3));

    key.device(*thread_pool_device) = (source_input.contract(key_weights, axes(2,0))
        + key_biases.reshape(array_3(1, 1, embedding_dimension))
                    .broadcast(array_3(batch_size, source_sequence_length, 1)))
        .reshape(array_4(batch_size, source_sequence_length, heads_number, head_dimension))
        .shuffle(array_4(0, 2, 1, 3));

    value.device(*thread_pool_device) = (source_input.contract(value_weights, axes(2,0))
        + value_biases.reshape(array_3(1, 1, embedding_dimension))
                      .broadcast(array_3(batch_size, source_sequence_length, 1)))
        .reshape(array_4(batch_size, source_sequence_length, heads_number, head_dimension))
        .shuffle(array_4(0, 2, 1, 3));

    attention_weights.device(*thread_pool_device) =
        query.contract(key.shuffle(array_4(0, 1, 3, 2)), axes(3,2)) / sqrt(type(head_dimension));

    softmax(attention_weights);

    attention_outputs.device(*thread_pool_device) = attention_weights.contract(value, axes(3, 2));

/*
    Tensor<float, 3> final_mha_output =
        (attention_outputs.shuffle(array_4(0, 2, 1, 3))
             .reshape(array_3(batch_size, sequence_length, embedding_dimension)))
            .contract(output_weights, axes(2, 0)) +
        output_bias.reshape(make_shape(1, 1, embedding_dimension))
            .broadcast(make_shape(batch_size, sequence_length, 1));

    // batch_matrix_multiplication<type, 4>(thread_pool_device.get(), key, query, attention_weights, axes<Index>(1,1));

    // const type scaling_factor = get_scaling_factor();

    // attention_weights.device(*thread_pool_device) = attention_weights * scaling_factor;

    // if(use_causal_mask)
    //     apply_causal_mask(attention_weights);

    // // @Todo (The key mask is hardcoded to function on a self-attention)
    // // Tensor<bool,2> key_mask(key.dimension(2), source_sequence_length);

    // // #pragma omp parallel for
    // // for(Index i = 0; i < key_mask.dimension(0); i++)
    // //     for(Index j = 0; j < key_mask.dimension(1);j++)
    // //         key_mask(i,j)=(attention_weights(j,j,i,0)==0);

    // // apply_key_padding_mask(key_mask, attention_weights);

     //softmax(attention_weights);

    ////////////////////

    // @todo fix the dropout implementation
    // if(is_training && dropout_rate > type(0)) {
    //     dropout(attention_weights, dropout_rate);
    //     cout << "Dropout aplicado" << endl;
    // }
/*
    calculate_attention_outputs(value, attention_weights, attention_outputs);

    concatenate_heads(attention_outputs, concatenated_attention_outputs);

    calculate_output_projection(concatenated_attention_outputs, outputs);
*/
}


void MultiHeadAttention::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
/*
    const TensorMap<Tensor<type, 3>> input = tensor_map<3>(input_pairs[0]);
    const TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_pairs[0]);

    const TensorMap<Tensor<type, 3>> context = (input_pairs.size() > 1) ?
                                               tensor_map<3>(input_pairs[1]) :
                                               input;

    const Index batch_size = input_pairs[0].second[0];
    const Index head_dimension = get_head_dimension();

    const type scaling_factor = get_scaling_factor();

    const MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(forward_propagation.get());

    MultiHeadAttentionBackPropagation* this_back_propagation =
        static_cast<MultiHeadAttentionBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_query_deltas = this_back_propagation->input_query_deltas;
    input_query_deltas.setZero();

    Tensor<type, 3>& input_source_deltas = this_back_propagation->input_source_deltas;
    input_source_deltas.setZero();

    Tensor<type, 1>& aux_rows = this_back_propagation->aux_rows;

    Tensor<type, 1>& projection_bias_deltas = this_back_propagation->projection_bias_deltas;

    const Tensor<type,3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;
    Tensor<type, 2>& projection_weight_deltas = this_back_propagation->projection_weight_deltas;
    Tensor<type,3>& concatenated_attention_output_deltas = this_back_propagation->concatenated_attention_output_deltas;
    Tensor<type,4>& attention_output_deltas = this_back_propagation->attention_output_deltas;

    // Calculation

    projection_bias_deltas.device(*thread_pool_device) = deltas.sum(array<Index, 2>({0,1}));

    projection_weight_deltas.device(*thread_pool_device) = concatenated_attention_outputs.contract(deltas, axes(2,0,0,1));

    const Tensor<type, 2> projection_weights_transposed = projection_weights.shuffle(array<Index, 2>{1, 0});

    // concatenated_attention_output_deltas.device(*thread_pool_device) =
    //     deltas.contract(projection_weights_transposed, array<IndexPair<Index>, 1>{{IndexPair<Index>(2, 0)}});

    concatenated_attention_output_deltas.device(*thread_pool_device) =
        deltas.contract(projection_weights_transposed, array<IndexPair<Index>, 1>{{IndexPair<Index>(2, 0)}})
            .shuffle(array<int, 3>{1, 2, 0});

    // for(Index head_index = 0; head_index < heads_number; ++head_index)
    // {
    //     attention_output_deltas.chip(head_index, 0).device(*thread_pool_device) =
    //         concatenated_attention_output_deltas.slice(
    //             array<Index, 3>{0, 0, head_index * head_dimension},
    //             array<Index, 3>{batch_size, query_sequence_length, head_dimension});
    // }

    for(Index head_index = 0; head_index < heads_number; ++head_index)
        attention_output_deltas.chip(head_index, 3).device(*thread_pool_device) =
            concatenated_attention_output_deltas.slice(
            array<Index, 3>{0, head_index * head_dimension, 0},
            array<Index, 3>{query_sequence_length, head_dimension, batch_size}
            );

    // Value deltas

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_attention_weights = tensor_map(this_forward_propagation->attention_weights, head_index);

        TensorMap<Tensor<type, 3>> head_attention_output_deltas = tensor_map(this_back_propagation->attention_output_deltas, head_index);

        TensorMap<Tensor<type, 3>> head_value_deltas = tensor_map(this_back_propagation->value_deltas, head_index);

        batch_matrix_multiplication<type, 3>(thread_pool_device.get(),
                                             head_attention_weights,
                                             head_attention_output_deltas,
                                             head_value_deltas,
                                             axes<Index>(1,0));
    }

    // Value weight deltas

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        TensorMap<Tensor<type, 3>> head_value_deltas = tensor_map(this_back_propagation->value_deltas, head_index);

        TensorMap<Tensor<type, 1>> head_value_bias_deltas = tensor_map(this_back_propagation->value_bias_deltas, head_index);
        TensorMap<Tensor<type, 2>> head_value_weight_deltas = tensor_map(this_back_propagation->value_weight_deltas, head_index);

        head_value_weight_deltas.device(*thread_pool_device)
            = context.contract(head_value_deltas, axes(1,0,0,2));

        head_value_bias_deltas.device(*thread_pool_device) = head_value_deltas.sum(array<Index, 2>({0,2}));
    }


    // Attention weight deltas

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_value = tensor_map(this_forward_propagation->value, head_index);

        TensorMap<Tensor<type, 3>> head_attention_output_deltas = tensor_map(this_back_propagation->attention_output_deltas, head_index);
        TensorMap<Tensor<type, 3>> head_attention_weight_deltas_xxx = tensor_map(this_back_propagation->attention_weight_deltas_xxx, head_index);
        const TensorMap<Tensor<type, 3>> head_attention_weights = tensor_map(this_forward_propagation->attention_weights, head_index);

        batch_matrix_multiplication<type, 3>(thread_pool_device.get(),
                                             head_value,
                                             head_attention_output_deltas,
                                             head_attention_weight_deltas_xxx,
                                             axes<Index>(1,1));

        softmax_derivatives_times_tensor(head_attention_weights,
                                         head_attention_weight_deltas_xxx,
                                         aux_rows);

        head_attention_weight_deltas_xxx.device(*thread_pool_device) = head_attention_weight_deltas_xxx * scaling_factor;
    }

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_query = tensor_map(this_forward_propagation->query, head_index);
        const TensorMap<Tensor<type, 3>> head_key = tensor_map(this_forward_propagation->key, head_index);

        TensorMap<Tensor<type, 3>> head_query_derivatives = tensor_map(this_back_propagation->query_deltas, head_index);
        TensorMap<Tensor<type, 3>> head_key_deltas = tensor_map(this_back_propagation->key_deltas, head_index);

        TensorMap<Tensor<type, 3>> head_attention_weight_deltas_xxx = tensor_map(this_back_propagation->attention_weight_deltas_xxx, head_index);

        // Query derivatives

        batch_matrix_multiplication<type, 3>(thread_pool_device.get(),
                                             head_attention_weight_deltas_xxx,
                                             head_key,
                                             head_query_derivatives,
                                             axes<Index>(0,0));

        // Key derivatives

        batch_matrix_multiplication<type, 3>(thread_pool_device.get(),
                                             head_attention_weight_deltas_xxx,
                                             head_query,
                                             head_key_deltas,
                                             axes<Index>(1,0));
    }

    // Query weight deltas

    // @todo clean?
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        TensorMap<Tensor<type, 3>> head_query_derivatives = tensor_map(this_back_propagation->query_deltas, head_index);

        TensorMap<Tensor<type, 1>> head_query_bias_deltas = tensor_map(this_back_propagation->query_bias_deltas, head_index);
        TensorMap<Tensor<type, 2>> head_query_weight_deltas = tensor_map(this_back_propagation->query_weight_deltas, head_index);

        head_query_weight_deltas.device(*thread_pool_device)
            = input.contract(head_query_derivatives, axes(1,0,0,2));

        head_query_bias_deltas.device(*thread_pool_device) = head_query_derivatives.sum(array<Index, 2>({0,2}));
    }

    // Key weight deltas

    // @todo clean?
    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        TensorMap<Tensor<type, 3>> head_key_deltas = tensor_map(this_back_propagation->key_deltas, head_index);
        TensorMap<Tensor<type, 1>> head_key_bias_deltas = tensor_map(this_back_propagation->key_bias_deltas, head_index);
        TensorMap<Tensor<type, 2>> head_key_weight_deltas = tensor_map(this_back_propagation->key_weight_deltas, head_index);

        head_key_weight_deltas.device(*thread_pool_device)
            = context.contract(head_key_deltas, axes(1,0,0,2));

        head_key_bias_deltas.device(*thread_pool_device) = head_key_deltas.sum(array<Index, 2>({0,2}));
    }

    // Input query deltas

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 2>> head_query_weights = tensor_map(query_weights, head_index);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            const TensorMap<Tensor<type, 2>> sample_query_derivatives = tensor_map(this_back_propagation->query_deltas, head_index, sample_index);

            input_query_deltas.chip(sample_index, 0).device(*thread_pool_device)
                += sample_query_derivatives.contract(head_query_weights, axes(1,1));
        }
    }

    // Input source deltas

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 2>> head_key_weights = tensor_map(key_weights, head_index);
        const TensorMap<Tensor<type, 2>> head_value_weights = tensor_map(value_weights, head_index);

        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            const TensorMap<Tensor<type, 2>> sample_key_derivatives = tensor_map(this_back_propagation->key_deltas, head_index, sample_index);
            const TensorMap<Tensor<type, 2>> sample_value_derivatives = tensor_map(this_back_propagation->value_deltas, head_index, sample_index);

            input_source_deltas.chip(sample_index, 0).device(*thread_pool_device)
                += sample_key_derivatives.contract(head_key_weights, axes(1,1))
                   + sample_value_derivatives.contract(head_value_weights, axes(1,1));
        }
    }
*/
}


void MultiHeadAttention::from_XML(const XMLDocument& document)
{
    // @todo update notation

    const XMLElement* multihead_attention_layer_element = document.FirstChildElement("MultiHeadAttention");

    if(!multihead_attention_layer_element)
        throw runtime_error("MultiHeadAttention element is nullptr.\n");

    const string new_name = read_xml_string(multihead_attention_layer_element, "Name");
    const Index new_input_size = read_xml_index(multihead_attention_layer_element, "InputSize");
    const Index new_context_size = read_xml_index(multihead_attention_layer_element, "ContextSize");
    const Index new_depth = read_xml_index(multihead_attention_layer_element, "Depth");
    const Index new_heads_number = read_xml_index(multihead_attention_layer_element, "HeadsNumber");
    const Index new_use_causal_mask = read_xml_bool(multihead_attention_layer_element, "CausalMask");

    set(new_input_size, new_context_size, new_depth, new_heads_number, new_use_causal_mask, new_name);

    string_to_tensor<type, 1>(read_xml_string(multihead_attention_layer_element, "QueryBiases"), query_biases);
    string_to_tensor<type, 2>(read_xml_string(multihead_attention_layer_element, "QueryWeights"), query_weights);
    string_to_tensor<type, 1>(read_xml_string(multihead_attention_layer_element, "KeyBiases"), key_biases);
    string_to_tensor<type, 2>(read_xml_string(multihead_attention_layer_element, "KeyWeights"), key_weights);
    string_to_tensor<type, 1>(read_xml_string(multihead_attention_layer_element, "ValueBiases"), value_biases);
    string_to_tensor<type, 2>(read_xml_string(multihead_attention_layer_element, "ValueWeights"), value_weights);
    string_to_tensor<type, 1>(read_xml_string(multihead_attention_layer_element, "ProjectionBiases"), projection_biases);
    string_to_tensor<type, 2>(read_xml_string(multihead_attention_layer_element, "ProjectionWeights"), projection_weights);
}


void MultiHeadAttention::print() const
{
    cout << "Multi-head attention Layer" << endl
         << "Label: " << label << endl
         << "Type: Embedding" << endl
         << "Input dimensions: ";
    print_vector(get_input_dimensions());

    cout << "Output dimensions: ";
    print_vector(get_output_dimensions());
}


void MultiHeadAttention::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("MultiHeadAttention");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputSize", to_string(get_query_sequence_length()));
    add_xml_element(printer, "ContextSize", to_string(get_source_sequence_length()));
    add_xml_element(printer, "Depth", to_string(get_embedding_dimension()));
    add_xml_element(printer, "HeadDimension", to_string(get_head_dimension()));
    add_xml_element(printer, "HeadsNumber", to_string(get_heads_number()));
    add_xml_element(printer, "CausalMask", to_string(use_causal_mask ? 1 : 0));
    add_xml_element(printer, "QueryBiases", tensor_to_string<type, 1>(query_biases));
    add_xml_element(printer, "QueryWeights", tensor_to_string<type, 2>(query_weights));
    add_xml_element(printer, "KeyBiases", tensor_to_string<type, 1>(key_biases));
    add_xml_element(printer, "KeyWeights", tensor_to_string<type, 2>(key_weights));
    add_xml_element(printer, "ValueBiases", tensor_to_string<type, 1>(value_biases));
    add_xml_element(printer, "ValueWeights", tensor_to_string<type, 2>(value_weights));
    add_xml_element(printer, "ProjectionBiases", tensor_to_string<type, 1>(projection_biases));
    add_xml_element(printer, "ProjectionWeights", tensor_to_string<type, 2>(projection_weights));

    printer.CloseElement();
}


MultiHeadAttentionForwardPropagation::MultiHeadAttentionForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> MultiHeadAttentionForwardPropagation::get_output_pair() const
{
    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();

    return { (type*)outputs.data(), {{ batch_size, query_sequence_length, embedding_dimension }} };
}


void MultiHeadAttentionForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    batch_size = new_batch_size;

    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index head_dimension = multihead_attention_layer->get_head_dimension();

    query.resize(batch_size, heads_number, query_sequence_length, head_dimension);
    key.resize(batch_size, heads_number, source_sequence_length, head_dimension);
    value.resize(batch_size, heads_number, source_sequence_length, head_dimension);

    attention_weights.resize(batch_size, heads_number, source_sequence_length, query_sequence_length);
    attention_outputs.resize(query_sequence_length, head_dimension, batch_size, heads_number);

    concatenated_attention_outputs.resize(query_sequence_length, embedding_dimension, batch_size);

    projection_outputs.resize(batch_size, query_sequence_length, embedding_dimension, heads_number);

    outputs.resize(batch_size, query_sequence_length, embedding_dimension);

    // Auxiliar

    sample_matrix.resize(query_sequence_length, head_dimension);
}


void MultiHeadAttentionForwardPropagation::print() const
{
    cout << "Outputs dimensions:" << endl;
    //cout << output_dimensions << endl;
    cout << "Outputs:" << endl;
    //cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
}


void MultiHeadAttentionBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    batch_size = new_batch_size;

    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index head_dimension = multihead_attention_layer->get_head_dimension();

    query_weight_deltas.resize(embedding_dimension, head_dimension, heads_number);
    key_weight_deltas.resize(embedding_dimension, head_dimension, heads_number);
    value_weight_deltas.resize(embedding_dimension, head_dimension, heads_number);
    projection_weight_deltas.resize(embedding_dimension, embedding_dimension);

    query_bias_deltas.resize(head_dimension, heads_number);
    key_bias_deltas.resize(head_dimension, heads_number);
    value_bias_deltas.resize(head_dimension, heads_number);
    projection_bias_deltas.resize(embedding_dimension);

    input_query_deltas.resize(batch_size, query_sequence_length, embedding_dimension);
    input_source_deltas.resize(batch_size, source_sequence_length, embedding_dimension);

    // Auxiliar

    attention_weight_deltas_xxx.resize(source_sequence_length, query_sequence_length, batch_size, heads_number);

    attention_output_deltas.resize(query_sequence_length, head_dimension, batch_size, heads_number);

    concatenated_attention_output_deltas.resize(query_sequence_length, embedding_dimension, batch_size);

    query_deltas.resize(query_sequence_length, head_dimension, batch_size, heads_number);

    key_deltas.resize(source_sequence_length, head_dimension, batch_size, heads_number);
    value_deltas.resize(source_sequence_length, head_dimension, batch_size, heads_number);

    aux_rows.resize(source_sequence_length);
}


void MultiHeadAttentionBackPropagation::print() const
{
}


MultiHeadAttentionBackPropagation::MultiHeadAttentionBackPropagation(const Index& new_batch_size,
                                                                     Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> MultiHeadAttentionBackPropagation::get_input_derivative_pairs() const
{
    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();

    return
        {{(type*)(input_query_deltas.data()), {batch_size, query_sequence_length, embedding_dimension}},
         {(type*)(input_source_deltas.data()), {batch_size, source_sequence_length, embedding_dimension}} };
}


vector<pair<type*, Index>> MultiHeadAttentionBackPropagation::get_parameter_delta_pairs() const
{
    return {
        {(type*)query_weight_deltas.data(), query_weight_deltas.size()},
        {(type*)query_bias_deltas.data(), query_bias_deltas.size()},
        {(type*)key_weight_deltas.data(), key_weight_deltas.size()},
        {(type*)key_bias_deltas.data(), key_bias_deltas.size()},
        {(type*)value_weight_deltas.data(), value_weight_deltas.size()},
        {(type*)value_bias_deltas.data(), value_bias_deltas.size()},
        {(type*)projection_weight_deltas.data(), projection_weight_deltas.size()},
        {(type*)projection_bias_deltas.data(), projection_bias_deltas.size()}
    };
}

REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")
REGISTER(LayerForwardPropagation, MultiHeadAttentionForwardPropagation, "MultiHeadAttention")
REGISTER(LayerBackPropagation, MultiHeadAttentionBackPropagation, "MultiHeadAttention")

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
// License along with this library; if not, write to the Free Software Foundation.er the terms of the GNU Lesser General Public
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
