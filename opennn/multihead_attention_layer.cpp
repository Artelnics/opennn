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

MultiHeadAttention::MultiHeadAttention(const dimensions& new_input_dimensions,
                                       const Index& new_heads_number,
                                       const string& new_name) : Layer()
{
    set(new_input_dimensions[0],
        new_input_dimensions[0],
        new_input_dimensions[1],
        new_heads_number,
        false,
        new_name);
}


MultiHeadAttention::MultiHeadAttention(const dimensions& new_query_dimensions,
                                       const dimensions& new_source_dimensions,
                                       const Index& new_heads_number,
                                       const string& new_name) : Layer()
{
/*
    set(new_query_sequence_length,
        new_source_sequence_length,
        new_embedding_dimension,
        new_heads_number,
        new_use_causal_mask,
        new_name);
*/
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
{
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


void MultiHeadAttention::get_parameters(Tensor<type, 1>& parameters) const
{
    parameters.resize(get_parameters_number());

    Index index = 0;

    copy_to_vector(parameters, query_weights, index);
    copy_to_vector(parameters, query_biases, index);
    copy_to_vector(parameters, key_weights, index);
    copy_to_vector(parameters, key_biases, index);
    copy_to_vector(parameters, value_weights, index);
    copy_to_vector(parameters, value_biases, index);
    copy_to_vector(parameters, projection_weights, index);
    copy_to_vector(parameters, projection_biases, index);

}


void MultiHeadAttention::set(const Index& new_query_sequence_length,
                             const Index& new_source_sequence_length,
                             const Index& new_embedding_dimension,
                             const Index& new_heads_number,
                             const bool& new_use_causal_mask,
                             const string& new_name)
{
    layer_type = Type::MultiheadAttention;
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
                                                 + (sample_index + head_index * batch_size) * context_input_size;

            // + (sample_index + head_index) * context_input_size * batch_size;
            // + (sample_index * heads_number + head_index) * context_input_size * batch_size;

            TensorMap<Tensor<type, 2>> sample_attention_scores(sample_attention_scores_data,
                                                               source_sequence_length,
                                                               query_sequence_length);

            sample_attention_scores.device(*thread_pool_device) += causal_mask;
        }
    }
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


void MultiHeadAttention::calculate_attention_weights(const Tensor<type, 4>& query,
                                                     const Tensor<type, 4>& key,
                                                     Tensor<type, 4>& attention_weights) const
{

    batch_matrix_multiplication(thread_pool_device.get(), key, query, attention_weights, axes<Index>(1,1));

    const type scaling_factor = get_scaling_factor();

    attention_weights.device(*thread_pool_device) = attention_weights * scaling_factor;

    if(use_causal_mask)
        apply_causal_mask(attention_weights);

    // @Todo (The key mask is hardcoded to function on a self-attention)
    // Tensor<bool,2> key_mask(key.dimension(2), source_sequence_length);

    // #pragma omp parallel for
    // for(Index i = 0; i < key_mask.dimension(0); i++)
    //     for(Index j = 0; j < key_mask.dimension(1);j++)
    //         key_mask(i,j)=(attention_weights(j,j,i,0)==0);

    // apply_key_padding_mask(key_mask, attention_weights);

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
    const TensorMap<Tensor<type, 3>> source_input = tensor_map<3>(input_pairs[1]);

    MultiheadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiheadAttentionForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& query = this_forward_propagation->query;
    Tensor<type, 4>& key = this_forward_propagation->key;
    Tensor<type, 4>& value = this_forward_propagation->value;

    Tensor<type, 4>& attention_weights = this_forward_propagation->attention_weights;

    Tensor<type, 4>& attention_outputs = this_forward_propagation->attention_outputs;

    Tensor<type, 3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    const Index batch_size = query_input.dimension(0);
    const Index hidden_depth = get_hidden_depth();

    query.device(*thread_pool_device)
        = query_input.contract(query_weights, axes(2, 0))
                .shuffle(array<Index, 4>({1, 2, 0, 3}))
          + query_biases.reshape(array<Index, 4>({1, hidden_depth, 1, heads_number}))
                .broadcast(array<Index, 4>({query_sequence_length, 1, batch_size, 1 }));

    key.device(*thread_pool_device)
        = source_input.contract(key_weights, axes(2,0))
              .shuffle(array<Index, 4>({1, 2, 0, 3}))
          + key_biases.reshape(array<Index, 4>({1, hidden_depth, 1, heads_number}))
                .broadcast(array<Index, 4>({source_sequence_length, 1, batch_size, 1}));

    value.device(*thread_pool_device)
        = source_input.contract(value_weights, axes(2,0))
              .shuffle(array<Index, 4>({1, 2, 0, 3}))
          + value_biases.reshape(array<Index, 4>({1, hidden_depth, 1, heads_number}))
                .broadcast(array<Index, 4>({source_sequence_length, 1, batch_size, 1}));

    calculate_attention_weights(query, key, attention_weights);

    if(is_training && dropout_rate > type(0))
        dropout(attention_weights, dropout_rate);

    calculate_attention_outputs(value, attention_weights, attention_outputs);

    concatenate_heads(attention_outputs, concatenated_attention_outputs);

    calculate_output_projection(concatenated_attention_outputs, outputs);

}


void MultiHeadAttention::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input = tensor_map<3>(input_pairs[0]);
    const TensorMap<Tensor<type, 3>> context = tensor_map<3>(input_pairs[1]);
    const TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_pairs[0]);

    const Index batch_size = input_pairs[0].second[0];
    const Index hidden_depth = get_hidden_depth();

    const type scaling_factor = get_scaling_factor();

    const MultiheadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiheadAttentionForwardPropagation*>(forward_propagation.get());

    MultiheadAttentionBackPropagation* this_back_propagation =
        static_cast<MultiheadAttentionBackPropagation*>(back_propagation.get());

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

    //

    const Tensor<type, 2> projection_weights_transposed = projection_weights.shuffle(array<Index, 2>{1, 0});

    concatenated_attention_output_deltas.device(*thread_pool_device) =
        deltas.contract(projection_weights_transposed, array<IndexPair<Index>, 1>{{IndexPair<Index>(2, 0)}});

    for(Index head_index = 0; head_index < heads_number; ++head_index)
    {
        attention_output_deltas.chip(head_index, 0).device(*thread_pool_device) =
            concatenated_attention_output_deltas.slice(
                array<Index, 3>{0, 0, head_index * hidden_depth},
                array<Index, 3>{batch_size, query_sequence_length, hidden_depth});
    }

    // Value deltas

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        const TensorMap<Tensor<type, 3>> head_attention_weights = tensor_map(this_forward_propagation->attention_weights, head_index);

        TensorMap<Tensor<type, 3>> head_attention_output_deltas = tensor_map(this_back_propagation->attention_output_deltas, head_index);

        TensorMap<Tensor<type, 3>> head_value_deltas = tensor_map(this_back_propagation->value_deltas, head_index);

        batch_matrix_multiplication(thread_pool_device.get(),
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

        batch_matrix_multiplication(thread_pool_device.get(),
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

        batch_matrix_multiplication(thread_pool_device.get(),
                                    head_attention_weight_deltas_xxx,
                                    head_key,
                                    head_query_derivatives,
                                    axes<Index>(0,0));

        // Key derivatives

        batch_matrix_multiplication(thread_pool_device.get(),
                                    head_attention_weight_deltas_xxx,
                                    head_query,
                                    head_key_deltas,
                                    axes<Index>(1,0));
    }

    // Query weight deltas

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
}


    void MultiHeadAttention::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                             Index& index,
                                             Tensor<type, 1>& gradient) const
    {
        MultiheadAttentionBackPropagation* this_back_propagation =
            static_cast<MultiheadAttentionBackPropagation*>(back_propagation.get());

        copy_to_vector(gradient, this_back_propagation->query_weight_deltas, index);
        copy_to_vector(gradient, this_back_propagation->query_bias_deltas, index);
        copy_to_vector(gradient, this_back_propagation->key_weight_deltas, index);
        copy_to_vector(gradient, this_back_propagation->key_bias_deltas, index);
        copy_to_vector(gradient, this_back_propagation->value_weight_deltas, index);
        copy_to_vector(gradient, this_back_propagation->value_bias_deltas, index);
        copy_to_vector(gradient, this_back_propagation->projection_weight_deltas, index);
        copy_to_vector(gradient, this_back_propagation->projection_bias_deltas, index);
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


    void MultiHeadAttention::print() const
    {
        cout << "Multi-head attention Layer" << endl
             << "Name: " << name << endl
             << "Type: Embedding" << endl
             << "Input dimensions: ";
        print_vector(get_input_dimensions());

        cout << "Output dimensions: ";
        print_vector(get_output_dimensions());
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

        Tensor<type, 1> parameters;
        get_parameters(parameters);

        add_xml_element(printer, "Parameters", tensor_to_string(parameters));

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

        query_weight_deltas.resize(embedding_dimension, hidden_depth, heads_number);
        key_weight_deltas.resize(embedding_dimension, hidden_depth, heads_number);
        value_weight_deltas.resize(embedding_dimension, hidden_depth, heads_number);
        projection_weight_deltas.resize(embedding_dimension, embedding_dimension);

        query_bias_deltas.resize(hidden_depth, heads_number);
        key_bias_deltas.resize(hidden_depth, heads_number);
        value_bias_deltas.resize(hidden_depth, heads_number);
        projection_bias_deltas.resize(embedding_dimension);

        input_query_deltas.resize(batch_size, query_sequence_length, embedding_dimension);
        input_source_deltas.resize(batch_size, source_sequence_length, embedding_dimension);

        // Auxiliar

        attention_weight_deltas_xxx.resize(source_sequence_length, query_sequence_length, batch_size, heads_number);

        attention_output_deltas.resize(query_sequence_length, hidden_depth, batch_size, heads_number);

        concatenated_attention_output_deltas.resize(query_sequence_length, embedding_dimension, batch_size);

        query_deltas.resize(query_sequence_length, hidden_depth, batch_size, heads_number);

        key_deltas.resize(source_sequence_length, hidden_depth, batch_size, heads_number);
        value_deltas.resize(source_sequence_length, hidden_depth, batch_size, heads_number);

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
            {{(type*)(input_query_deltas.data()), {batch_size, query_sequence_length, embedding_dimension}},
             {(type*)(input_source_deltas.data()), {batch_size, source_sequence_length, embedding_dimension}} };
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
