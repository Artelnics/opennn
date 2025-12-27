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
    // Self-attention

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
    return query_biases.dimension(0);
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
        : Index(get_embedding_dimension() / heads_number);
}


dimensions MultiHeadAttention::get_input_dimensions() const
{
    return { query_sequence_length, get_embedding_dimension() };
}


dimensions MultiHeadAttention::get_output_dimensions() const
{
    return { query_sequence_length, get_embedding_dimension() };
}


vector<ParameterView > MultiHeadAttention::get_parameter_views() const
{
    return {{(type*)query_weights.data(), query_weights.size()},
            {(type*)query_biases.data(), query_biases.size()},
            {(type*)key_weights.data(), key_weights.size()},
            {(type*)key_biases.data(), key_biases.size()},
            {(type*)value_weights.data(), value_weights.size()},
            {(type*)value_biases.data(), value_biases.size()},
            {(type*)projection_weights.data(), projection_weights.size()},
            {(type*)projection_biases.data(), projection_biases.size()}};
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
    label = new_label;

    if (heads_number <= 0 || new_embedding_dimension % heads_number != 0)
        throw runtime_error("MultiHeadAttention Error: The embedding dimension must be divisible by the number of heads.");

    query_weights.resize(new_embedding_dimension, new_embedding_dimension);
    query_biases.resize(new_embedding_dimension);

    key_weights.resize(new_embedding_dimension, new_embedding_dimension);
    key_biases.resize(new_embedding_dimension);

    value_weights.resize(new_embedding_dimension, new_embedding_dimension);
    value_biases.resize(new_embedding_dimension);

    projection_weights.resize(new_embedding_dimension, new_embedding_dimension);
    projection_biases.resize(new_embedding_dimension);

    set_parameters_glorot();

    use_causal_mask = new_use_causal_mask;

    if (use_causal_mask)
        causal_mask = Tensor<type, 2>(query_sequence_length, source_sequence_length)
                          .generate([=](const Eigen::array<Index, 2>& idx) -> type {
                              const Index row = idx[0];
                              const Index column = idx[1];
                              return (column > row) ? minus_inf : 0;
                          });
}


void MultiHeadAttention::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
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

    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();
    const type scaling_factor = get_scaling_factor();

    auto project = [&](const TensorMap<Tensor<type, 3>>& inputs,
                       const Tensor<type, 2>& weights, const Tensor<type, 1>& biases,
                       const Index& sequence_length)
    {
        return (inputs.reshape(array_2(batch_size * sequence_length, embedding_dimension)).contract(weights, axes(1,0))
                + biases.reshape(array_2(1, embedding_dimension)).broadcast(array_2(batch_size*sequence_length, 1)))
               .reshape(array_4(batch_size, sequence_length, heads_number, head_dimension))
               .shuffle(array_4(0, 2, 1, 3));
    };

    query.device(*device) = project(query_input, query_weights, query_biases, query_sequence_length);
    key.device(*device) = project(source_input, key_weights, key_biases, source_sequence_length);
    value.device(*device) = project(source_input, value_weights, value_biases, source_sequence_length);

    attention_weights.device(*device) =
        (query.reshape(array_5(batch_size, heads_number, query_sequence_length, 1, head_dimension)).broadcast(array_5(1, 1, 1, source_sequence_length, 1))
         * key.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension)).broadcast(array_5(1, 1, query_sequence_length, 1, 1)))
         .sum(array_1(4)) * scaling_factor;

    softmax(attention_weights);

    attention_outputs.device(*device) =
        (attention_weights.reshape(array_5(batch_size, heads_number, query_sequence_length, source_sequence_length, 1)).broadcast(array_5(1, 1, 1, 1, head_dimension))
         * value.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension)).broadcast(array_5(1, 1, query_sequence_length, 1, 1)))
         .sum(array_1(3));

    concatenated_attention_outputs.device(*device) = attention_outputs.shuffle(array_4(0, 2, 1, 3))
                                                                      .reshape(concatenated_attention_outputs.dimensions());

    outputs.device(*device) =
        concatenated_attention_outputs.contract(projection_weights, axes(2, 0))
        + projection_biases.reshape(array_3(1, 1, embedding_dimension))
        .broadcast(array_3(batch_size, query_sequence_length, 1));
}


void MultiHeadAttention::back_propagate(const vector<TensorView>& input_views,
                                        const vector<TensorView>& delta_views,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> query_input = tensor_map<3>(input_views[0]);
    const TensorMap<Tensor<type, 3>> source_input = (input_views.size() == 1)
                                                        ? query_input
                                                        : tensor_map<3>(input_views[1]);

    const TensorMap<Tensor<type, 3>> delta_Y = tensor_map<3>(delta_views[0]);

    // Forward propagation

    MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 4>& query = this_forward_propagation->query;
    const Tensor<type, 4>& key = this_forward_propagation->key;
    const Tensor<type, 4>& value = this_forward_propagation->value;
    const Tensor<type, 4>& attention_weights = this_forward_propagation->attention_weights;
    const Tensor<type, 3>& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    // Back propagation

    MultiHeadAttentionBackPropagation* this_back_propagation =
        static_cast<MultiHeadAttentionBackPropagation*>(back_propagation.get());

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
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();
    const type scaling_factor = get_scaling_factor();

    projection_weight_deltas.device(*device) =
        concatenated_attention_outputs.reshape(array_2(batch_size * query_sequence_length, embedding_dimension))
        .contract(delta_Y.reshape(array_2(batch_size * query_sequence_length, embedding_dimension)), axes(0, 0));

    projection_bias_deltas.device(*device) = delta_Y.sum(array_2(0, 1));

    concatenated_attention_output_deltas.device(*device) =
        delta_Y.contract(projection_weights, axes(2, 1));

    attention_output_deltas.device(*device) =
        concatenated_attention_output_deltas.reshape(array_4(batch_size, query_sequence_length, heads_number, head_dimension))
                                            .shuffle(array_4(0, 2, 1, 3));

    value_deltas.device(*device) =
        (attention_weights.shuffle(array_4(0, 1, 3, 2))
             .reshape(array_5(batch_size, heads_number, source_sequence_length, query_sequence_length, 1))
             .broadcast(array_5(1, 1, 1, 1, head_dimension))
         * attention_output_deltas
               .reshape(array_5(batch_size, heads_number, 1, query_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, source_sequence_length, 1, 1))
         ).sum(array_1(3));

    attention_weight_deltas_xxx.device(*device) =
        (attention_output_deltas.reshape(array_5(batch_size, heads_number, query_sequence_length, 1, head_dimension))
             .broadcast(array_5(1, 1, 1, source_sequence_length, 1))
         * value.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, query_sequence_length, 1, 1))
         ).sum(array_1(4));

    auto dot_product = (attention_weights * attention_weight_deltas_xxx).sum(array_1(3));

    softmax_deltas.device(*device) = attention_weights * (attention_weight_deltas_xxx -
        dot_product.reshape(array_4(batch_size, heads_number, query_sequence_length, 1))
        .broadcast(array_4(1, 1, 1, source_sequence_length)));

    query_deltas.device(*device) =
        (softmax_deltas.reshape(array_5(batch_size, heads_number, query_sequence_length, source_sequence_length, 1))
             .broadcast(array_5(1, 1, 1, 1, head_dimension))
         * key.reshape(array_5(batch_size, heads_number, 1, source_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, query_sequence_length, 1, 1))
         ).sum(array_1(3)) * scaling_factor;

    key_deltas.device(*device) =
        (softmax_deltas.shuffle(array_4(0, 1, 3, 2))
             .reshape(array_5(batch_size, heads_number, source_sequence_length, query_sequence_length, 1))
             .broadcast(array_5(1, 1, 1, 1, head_dimension))
         * query.reshape(array_5(batch_size, heads_number, 1, query_sequence_length, head_dimension))
               .broadcast(array_5(1, 1, source_sequence_length, 1, 1))
         ).sum(array_1(3)) * scaling_factor;

    const array<Index, 2> flat_q_dims = {batch_size * query_sequence_length, embedding_dimension};
    const array<Index, 2> flat_s_dims = {batch_size * source_sequence_length, embedding_dimension};

    auto project_back = [&](const Tensor<type, 4>& d_head, const Tensor<type, 3>& input, const Tensor<type, 2>& weights,
                            Tensor<type, 3>& d_reshaped, Tensor<type, 2>& d_weights, Tensor<type, 1>& d_biases,
                            Tensor<type, 3>& d_input, const array<Index, 2>& flat_dims, bool accumulate)
    {
        d_reshaped.device(*device) = d_head.shuffle(array_4(0, 2, 1, 3)).reshape(input.dimensions());
        d_weights.device(*device)  = input.reshape(flat_dims).contract(d_reshaped.reshape(flat_dims), axes(0, 0));
        d_biases.device(*device)   = d_reshaped.sum(array_2(0, 1));

        accumulate ? (d_input.device(*device) += d_reshaped.contract(weights, axes(2, 1)))
                   : (d_input.device(*device) =  d_reshaped.contract(weights, axes(2, 1)));
    };

    project_back(query_deltas, query_input, query_weights, query_deltas_reshaped,
                 query_weight_deltas, query_bias_deltas, input_query_deltas, flat_q_dims, false);

    project_back(key_deltas, source_input, key_weights, key_deltas_reshaped,
                 key_weight_deltas, key_bias_deltas, input_source_deltas, flat_s_dims, false);

    project_back(value_deltas, source_input, value_weights, value_deltas_reshaped,
                 value_weight_deltas, value_bias_deltas, input_source_deltas, flat_s_dims, true);

    // For self-attention, accumulate all gradients into input_query_deltas

    if(input_views.size() == 1)
        input_query_deltas.device(*device) += input_source_deltas;
}


void MultiHeadAttention::apply_causal_mask(Tensor<type, 4>& attention_scores) const
{
    //throw runtime_error("MultiHeadAttention::apply_causal_mask is not yet implemented. Please check back in a future version.");
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

    //         sample_attention_scores.device(*device) += causal_mask;
    //     }
    // }
}


void MultiHeadAttention::apply_key_padding_mask(const Tensor<bool, 2>& key_padding_mask,
                                                Tensor<type, 4>& attention_weights) const
{
    // @todo (I don't know if it is building the mask correctly)
    const Index batch_size  = attention_weights.dimension(2);

    Tensor<type, 2> key_padding_mask_type(key_padding_mask.dimension(0),key_padding_mask.dimension(1));

    for(Index h = 0; h < heads_number; ++h)
    {
        for(Index b = 0; b < batch_size; ++b)
        {
            TensorMap<Tensor<type, 2>> head_sample_attention_weights = tensor_map(attention_weights,h,b);

            head_sample_attention_weights.device(*device)
                += key_padding_mask.chip(b, 0)
                       .cast<type>()
                       .reshape(array<Index,2>{source_sequence_length, 1})
                       .broadcast(array<Index,2>{1, query_sequence_length})
                   * type(-10e9);
        }
    }
}


void MultiHeadAttention::print() const
{
    cout << "Multi-head attention Layer" << endl
         << "Label: " << label << endl
         << "Type: Embedding" << endl
         << "Input dimensions: " << get_input_dimensions() << endl
         << "Output dimensions: " << get_output_dimensions();
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


MultiHeadAttentionForwardPropagation::MultiHeadAttentionForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView MultiHeadAttentionForwardPropagation::get_output_view() const
{
    MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();

    return { (type*)outputs.data(), {{ batch_size, query_sequence_length, embedding_dimension }} };
}


void MultiHeadAttentionForwardPropagation::initialize()
{
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

    //sample_matrix.resize(query_sequence_length, head_dimension);
}


void MultiHeadAttentionForwardPropagation::print() const
{
//    cout << "Outputs dimensions:" << output_dimensions << endl
//    cout << "Outputs:" << endl;
    //cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
}


void MultiHeadAttentionBackPropagation::initialize()
{
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

    return {{(type*)(input_query_deltas.data()), {batch_size, query_sequence_length, embedding_dimension}},
            {(type*)(input_source_deltas.data()), {batch_size, source_sequence_length, embedding_dimension}}};
}


vector<ParameterView> MultiHeadAttentionBackPropagation::get_parameter_delta_views() const
{
    return {{(type*)query_weight_deltas.data(), query_weight_deltas.size()},
            {(type*)query_bias_deltas.data(), query_bias_deltas.size()},
            {(type*)key_weight_deltas.data(), key_weight_deltas.size()},
            {(type*)key_bias_deltas.data(), key_bias_deltas.size()},
            {(type*)value_weight_deltas.data(), value_weight_deltas.size()},
            {(type*)value_bias_deltas.data(), value_bias_deltas.size()},
            {(type*)projection_weight_deltas.data(), projection_weight_deltas.size()},
            {(type*)projection_bias_deltas.data(), projection_bias_deltas.size()}};
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
