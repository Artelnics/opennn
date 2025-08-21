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


vector<ParameterView > MultiHeadAttention::get_parameter_views() const
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

    head_dimension = get_head_dimension();
    scaling_factor = get_scaling_factor();

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


void MultiHeadAttention::forward_propagate(const vector<TensorView>& input_views,
                                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                           const bool&)
{
    const TensorMap<Tensor<type, 3>> query_input = tensor_map<3>(input_views[0]);
    const TensorMap<Tensor<type, 3>> source_input = (input_views.size() == 1)
                                                        ? query_input
                                                        : tensor_map<3>(input_views[1]);

    MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(layer_forward_propagation.get());

    const Index batch_size = this_forward_propagation->batch_size;

    Tensor<type, 4>& query = this_forward_propagation->query;
    Tensor<type, 4>& key = this_forward_propagation->key;
    Tensor<type, 4>& value = this_forward_propagation->value;

    Tensor<type, 4>& attention_weights = this_forward_propagation->attention_weights;
    Tensor<type, 4>& attention_outputs = this_forward_propagation->attention_outputs;

    Tensor<type, 3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

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
        (query.reshape(array_5(batch_size, heads_number, query_sequence_length, 1, head_dimension)).broadcast(array_5(1, 1, 1, source_sequence_length, 1))
         * key.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension)).broadcast(array_5(1, 1, query_sequence_length, 1, 1)))
         .sum(array_1(4)) * scaling_factor;

    softmax(attention_weights);

    attention_outputs.device(*thread_pool_device) =
        (attention_weights.reshape(array_5(batch_size, heads_number, query_sequence_length, source_sequence_length, 1)).broadcast(array_5(1, 1, 1, 1, head_dimension))
         * value.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension)).broadcast(array_5(1, 1, query_sequence_length, 1, 1)))
         .sum(array_1(3));

#pragma omp parallel for
    for(int head_number = 0; head_number < heads_number; ++head_number)
        concatenated_attention_outputs.slice(array_3(0, 0, head_number * head_dimension), array_3(batch_size, query_sequence_length, head_dimension)) =
            attention_outputs.chip(head_number, 1);

    outputs.device(*thread_pool_device) =
        concatenated_attention_outputs.contract(projection_weights, axes(2, 0))
        + projection_biases.reshape(array_3(1, 1, embedding_dimension))
        .broadcast(array_3(batch_size, query_sequence_length, 1));
}


void MultiHeadAttention::back_propagate(const vector<TensorView>& input_views,
                                        const vector<TensorView>& delta_views,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    // Get forward propagation data
    MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(forward_propagation.get());

    // Get backward propagation storage
    MultiHeadAttentionBackPropagation* this_back_propagation =
        static_cast<MultiHeadAttentionBackPropagation*>(back_propagation.get());

    // Get tensors from forward propagation
    const TensorMap<Tensor<type, 3>> query_input = tensor_map<3>(input_views[0]);
    const TensorMap<Tensor<type, 3>> source_input = (input_views.size() == 1)
                                                        ? query_input
                                                        : tensor_map<3>(input_views[1]);

    const Tensor<type, 4>& query = this_forward_propagation->query;
    const Tensor<type, 4>& key = this_forward_propagation->key;
    const Tensor<type, 4>& value = this_forward_propagation->value;
    const Tensor<type, 4>& attention_weights = this_forward_propagation->attention_weights;
    const Tensor<type, 3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    // Get incoming gradient (ΔY)
    const TensorMap<Tensor<type, 3>> delta_Y = tensor_map<3>(delta_views[0]);

    // Get gradient storage
    Tensor<type, 2>& projection_weight_deltas = this_back_propagation->projection_weight_deltas;
    Tensor<type, 1>& projection_bias_deltas = this_back_propagation->projection_bias_deltas;
    Tensor<type, 3>& concatenated_attention_output_deltas = this_back_propagation->concatenated_attention_output_deltas;
    Tensor<type, 4>& attention_output_deltas = this_back_propagation->attention_output_deltas;
    Tensor<type, 4>& attention_weight_deltas_xxx = this_back_propagation->attention_weight_deltas_xxx;
    Tensor<type, 4>& query_deltas = this_back_propagation->query_deltas;
    Tensor<type, 4>& key_deltas = this_back_propagation->key_deltas;
    Tensor<type, 4>& value_deltas = this_back_propagation->value_deltas;
    Tensor<type, 2>& query_weight_deltas = this_back_propagation->query_weight_deltas;
    Tensor<type, 1>& query_bias_deltas = this_back_propagation->query_bias_deltas;
    Tensor<type, 2>& key_weight_deltas = this_back_propagation->key_weight_deltas;
    Tensor<type, 1>& key_bias_deltas = this_back_propagation->key_bias_deltas;
    Tensor<type, 2>& value_weight_deltas = this_back_propagation->value_weight_deltas;
    Tensor<type, 1>& value_bias_deltas = this_back_propagation->value_bias_deltas;
    Tensor<type, 3>& input_query_deltas = this_back_propagation->input_query_deltas;
    Tensor<type, 3>& input_source_deltas = this_back_propagation->input_source_deltas;
    Tensor<type, 4>& softmax_deltas = this_back_propagation->softmax_deltas;
    Tensor<type, 3>& query_deltas_reshaped = this_back_propagation->query_deltas_reshaped;
    Tensor<type, 3>& key_deltas_reshaped = this_back_propagation->key_deltas_reshaped;
    Tensor<type, 3>& value_deltas_reshaped = this_back_propagation->value_deltas_reshaped;

    const Index batch_size = this_forward_propagation->batch_size;

    // Step 1: Derivative of the final projection
    // ∇W_O L = H_concat^T * ΔY
    projection_weight_deltas.device(*thread_pool_device) =
        concatenated_attention_outputs.reshape(array_2(batch_size * query_sequence_length, embedding_dimension))
        .contract(delta_Y.reshape(array_2(batch_size * query_sequence_length, embedding_dimension)), axes(0, 0));

    // ∇b_O L = sum(ΔY, axis=(0,1))
    projection_bias_deltas.device(*thread_pool_device) =
        delta_Y.sum(array_2(0, 1));

    // ΔH_concat = ΔY * W_O^T
    concatenated_attention_output_deltas.device(*thread_pool_device) =
        delta_Y.contract(projection_weights, axes(2, 1));

// Step 2: Split ΔH_concat by heads

#pragma omp parallel for
    for(int head_index = 0; head_index < heads_number; ++head_index)
    {
        attention_output_deltas.slice(array_4(0, head_index, 0, 0), array_4(batch_size, 1, query_sequence_length, head_dimension))
            = concatenated_attention_output_deltas.slice(array_3(0, 0, head_index * head_dimension),
                                                         array_3(batch_size, query_sequence_length, head_dimension))
                  .reshape(array_4(batch_size, 1, query_sequence_length, head_dimension));
    }

    // Step 3: Derivative of H_i = A_i * V_i
    // ∇V_i L = A_i^T * ΔH_i
    value_deltas.device(*thread_pool_device) =
        (attention_weights.shuffle(array_4(0, 1, 3, 2))
             .reshape(array_5(batch_size, heads_number, source_sequence_length, query_sequence_length, 1))
             .broadcast(array_5(1, 1, 1, 1, head_dimension))
         * attention_output_deltas
               .reshape(array_5(batch_size, heads_number, 1, query_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, source_sequence_length, 1, 1))
         ).sum(array_1(3));

    // ∇A_i L = ΔH_i * V_i^T
    attention_weight_deltas_xxx.device(*thread_pool_device) =
        (attention_output_deltas.reshape(array_5(batch_size, heads_number, query_sequence_length, 1, head_dimension))
             .broadcast(array_5(1, 1, 1, source_sequence_length, 1))
         * value.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, query_sequence_length, 1, 1))
         ).sum(array_1(4));

    // Step 4: Derivative of softmax
    // For each position in each head in each batch
    // ∇S_i L = diag(A_i) * ΔA_i - A_i * (A_i^T * ΔA_i)
#pragma omp parallel for collapse(3)
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index h = 0; h < heads_number; ++h)
        {
            for(Index q = 0; q < query_sequence_length; ++q)
            {
                type dot_product = 0;
                for(Index s = 0; s < source_sequence_length; ++s)
                {
                    dot_product += attention_weights(b, h, q, s) * attention_weight_deltas_xxx(b, h, q, s);
                }

                for(Index s = 0; s < source_sequence_length; ++s)
                {
                    softmax_deltas(b, h, q, s) = attention_weights(b, h, q, s) *
                                               (attention_weight_deltas_xxx(b, h, q, s) - dot_product);
                }
            }
        }
    }

    // Step 5: Derivative of scores S_i = Q_i * K_i^T / sqrt(d_k)
    // ∇Q_i L = (1/sqrt(d_k)) * ΔS_i * K_i
    query_deltas.device(*thread_pool_device) =
        (softmax_deltas.reshape(array_5(batch_size, heads_number, query_sequence_length, source_sequence_length, 1))
             .broadcast(array_5(1, 1, 1, 1, head_dimension))
         * key.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, query_sequence_length, 1, 1))
         ).sum(array_1(3)) * scaling_factor;

    // ∇K_i L = (1/sqrt(d_k)) * ΔS_i^T * Q_i
    key_deltas.device(*thread_pool_device) =
        (softmax_deltas.shuffle(array_4(0, 1, 3, 2))
             .reshape(array_5(batch_size, heads_number, source_sequence_length, query_sequence_length, 1))
             .broadcast(array_5(1, 1, 1, 1, head_dimension))
         * query.reshape(array_5(batch_size, heads_number, 1, query_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, source_sequence_length, 1, 1))
         ).sum(array_1(3)) * scaling_factor;


    // Step 6: Derivative of projections
    // First reshape the deltas back to (batch_size, sequence_length, embedding_dimension)
    query_deltas_reshaped.device(*thread_pool_device) =
        query_deltas.shuffle(array_4(0, 2, 1, 3))
        .reshape(array_3(batch_size, query_sequence_length, embedding_dimension));

    key_deltas_reshaped.device(*thread_pool_device) =
        key_deltas.shuffle(array_4(0, 2, 1, 3))
        .reshape(array_3(batch_size, source_sequence_length, embedding_dimension));

    value_deltas_reshaped.device(*thread_pool_device) =
        value_deltas.shuffle(array_4(0, 2, 1, 3))
        .reshape(array_3(batch_size, source_sequence_length, embedding_dimension));

    // ∇W_Q L = X^T * ∇Q L
    query_weight_deltas.device(*thread_pool_device) =
        query_input.reshape(array_2(batch_size * query_sequence_length, embedding_dimension))
        .contract(query_deltas_reshaped.reshape(array_2(batch_size * query_sequence_length, embedding_dimension)), axes(0, 0));

    // ∇b_Q L = sum(∇Q L, axis=(0,1))
    query_bias_deltas.device(*thread_pool_device) =
        query_deltas_reshaped.sum(array_2(0, 1));

    // ∇W_K L = X^T * ∇K L
    key_weight_deltas.device(*thread_pool_device) =
        source_input.reshape(array_2(batch_size * source_sequence_length, embedding_dimension))
        .contract(key_deltas_reshaped.reshape(array_2(batch_size * source_sequence_length, embedding_dimension)), axes(0, 0));

    // ∇b_K L = sum(∇K L, axis=(0,1))
    key_bias_deltas.device(*thread_pool_device) =
        key_deltas_reshaped.sum(array_2(0, 1));

    // ∇W_V L = X^T * ∇V L
    value_weight_deltas.device(*thread_pool_device) =
        source_input.reshape(array_2(batch_size * source_sequence_length, embedding_dimension))
        .contract(value_deltas_reshaped.reshape(array_2(batch_size * source_sequence_length, embedding_dimension)), axes(0, 0));

    // ∇b_V L = sum(∇V L, axis=(0,1))
    value_bias_deltas.device(*thread_pool_device) =
        value_deltas_reshaped.sum(array_2(0, 1));

    // Step 7: Accumulate input gradient
    // ∇X L (from Q) = ∇Q L * W_Q^T
    input_query_deltas.device(*thread_pool_device) =
        query_deltas_reshaped.contract(query_weights, axes(2, 1));

    // ∇X L (from K) = ∇K L * W_K^T
    input_source_deltas.device(*thread_pool_device) =
        key_deltas_reshaped.contract(key_weights, axes(2, 1));

    // ∇X L (from V) = ∇V L * W_V^T (add to source deltas)
    input_source_deltas.device(*thread_pool_device) +=
        value_deltas_reshaped.contract(value_weights, axes(2, 1));

    // For self-attention, accumulate all gradients into input_query_deltas
    if(input_views.size() == 1)
    {
        input_query_deltas.device(*thread_pool_device) += input_source_deltas;
    }
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


TensorView MultiHeadAttentionForwardPropagation::get_output_pair() const
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

    attention_weights.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

    attention_outputs.resize(batch_size, heads_number, query_sequence_length, head_dimension);
    concatenated_attention_outputs.resize(batch_size, query_sequence_length, embedding_dimension);
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

    query_weight_deltas.resize(embedding_dimension, embedding_dimension);
    key_weight_deltas.resize(embedding_dimension, embedding_dimension);
    value_weight_deltas.resize(embedding_dimension, embedding_dimension);
    projection_weight_deltas.resize(embedding_dimension, embedding_dimension);

    query_bias_deltas.resize(embedding_dimension);
    key_bias_deltas.resize(embedding_dimension);
    value_bias_deltas.resize(embedding_dimension);
    projection_bias_deltas.resize(embedding_dimension);

    input_query_deltas.resize(batch_size, query_sequence_length, embedding_dimension);
    input_source_deltas.resize(batch_size, source_sequence_length, embedding_dimension);

    // Auxiliar

    softmax_deltas.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

    query_deltas_reshaped.resize(batch_size, query_sequence_length, embedding_dimension);
    key_deltas_reshaped.resize(batch_size, source_sequence_length, embedding_dimension);
    value_deltas_reshaped.resize(batch_size, source_sequence_length, embedding_dimension);

    attention_weight_deltas_xxx.resize(batch_size, heads_number, source_sequence_length, query_sequence_length);

    attention_output_deltas.resize(batch_size, heads_number, query_sequence_length, head_dimension);

    concatenated_attention_output_deltas.resize(batch_size, query_sequence_length, embedding_dimension);

    query_deltas.resize(batch_size, heads_number, query_sequence_length, head_dimension);

    key_deltas.resize(batch_size, heads_number, source_sequence_length, head_dimension);
    value_deltas.resize(batch_size, heads_number, source_sequence_length, head_dimension);

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


vector<TensorView> MultiHeadAttentionBackPropagation::get_input_derivative_views() const
{
    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();

    return
        {{(type*)(input_query_deltas.data()), {batch_size, query_sequence_length, embedding_dimension}},
         {(type*)(input_source_deltas.data()), {batch_size, source_sequence_length, embedding_dimension}} };
}


vector<ParameterView> MultiHeadAttentionBackPropagation::get_parameter_delta_views() const
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
