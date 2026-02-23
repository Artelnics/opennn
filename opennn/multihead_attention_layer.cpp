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

MultiHeadAttention::MultiHeadAttention(const Shape& new_input_shape,
                                       Index new_heads_number,
                                       const string& new_name) : Layer()
{
    // Self-attention

    set(new_input_shape[0],    // query_sequence_length
        new_input_shape[0],    // source_sequence_length
        new_input_shape[1],    // embedding_dimension
        new_heads_number,
        false,
        new_name);
}

MultiHeadAttention::MultiHeadAttention(const Shape& new_query_dimensions,
                                       const Shape& new_source_dimensions,
                                       Index new_heads_number,
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
    return query_biases.shape[0];
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


Shape MultiHeadAttention::get_input_shape() const
{
    return { query_sequence_length, get_embedding_dimension() };
}


Shape MultiHeadAttention::get_output_shape() const
{
    return { query_sequence_length, get_embedding_dimension() };
}


vector<TensorView*> MultiHeadAttention::get_parameter_views()
{
    return {&query_weights, &query_biases,
            &key_weights, &key_biases,
            &value_weights, &value_biases,
            &projection_weights, &projection_biases};
}


void MultiHeadAttention::set(const Index new_query_sequence_length,
                             Index new_source_sequence_length,
                             Index new_embedding_dimension,
                             Index new_heads_number,
                             bool new_use_causal_mask,
                             const string& new_label)
{
    name = "MultiHeadAttention";
    query_sequence_length = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    heads_number = new_heads_number;
    label = new_label;

    if (heads_number <= 0 || new_embedding_dimension % heads_number != 0)
        throw runtime_error("MultiHeadAttention Error: The embedding dimension must be divisible by the number of heads.");

    query_weights.shape = {new_embedding_dimension, new_embedding_dimension};
    query_biases.shape = {new_embedding_dimension};

    key_weights.shape = {new_embedding_dimension, new_embedding_dimension};
    key_biases.shape = {new_embedding_dimension};

    value_weights.shape = {new_embedding_dimension, new_embedding_dimension};
    value_biases.shape = {new_embedding_dimension};

    projection_weights.shape = {new_embedding_dimension, new_embedding_dimension};
    projection_biases.shape = {new_embedding_dimension};

    use_causal_mask = new_use_causal_mask;

    if (use_causal_mask)
        causal_mask = Tensor2(query_sequence_length, source_sequence_length)
                          .generate([=](const Eigen::array<Index, 2>& idx) -> type {
                              const Index row = idx[0];
                              const Index column = idx[1];
                              return (column > row) ? minus_inf : 0;});
}


void MultiHeadAttention::set_dropout_rate(const type new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void MultiHeadAttention::forward_propagate(const vector<TensorView>& input_views,
                                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                           bool)
{
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();
    const type scaling_factor = get_scaling_factor();

    const TensorMap3 query_input = tensor_map<3>(input_views[0]);

    const TensorMap3 source_input = (input_views.size() == 1)
                                    ? query_input
                                    : tensor_map<3>(input_views[1]);

    MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(layer_forward_propagation.get());

    const Index batch_size = this_forward_propagation->batch_size;

    Tensor4& query = this_forward_propagation->query;
    Tensor4& key = this_forward_propagation->key;
    Tensor4& value = this_forward_propagation->value;

    Tensor4& attention_weights = this_forward_propagation->attention_weights;
    Tensor4& attention_outputs = this_forward_propagation->attention_outputs;

    Tensor3& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    TensorMap3 outputs = tensor_map<3>(layer_forward_propagation->outputs);

    calculate_projection(query_input, query_weights, query_biases, query_sequence_length, batch_size, query);
    calculate_projection(source_input, key_weights, key_biases, source_sequence_length, batch_size, key);
    calculate_projection(source_input, value_weights, value_biases, source_sequence_length, batch_size, value);

    const Index total_heads = batch_size * heads_number;

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const auto q_mat = query.reshape(array_2(total_heads, query_sequence_length * head_dimension))
                                .chip(i, 0)
                                .reshape(array_2(query_sequence_length, head_dimension));

        const auto k_mat = key.reshape(array_2(total_heads, source_sequence_length * head_dimension))
                              .chip(i, 0)
                              .reshape(array_2(source_sequence_length, head_dimension));

        auto w_mat = attention_weights.reshape(array_2(total_heads, query_sequence_length * source_sequence_length))
                                      .chip(i, 0)
                                      .reshape(array_2(query_sequence_length, source_sequence_length));

        w_mat = q_mat.contract(k_mat, axes(1, 1)) * scaling_factor;
    }

    if (use_causal_mask)
        attention_weights.device(get_device()) += causal_mask.reshape(array_4(1, 1, query_sequence_length, source_sequence_length))
                                                        .broadcast(array_4(batch_size, heads_number, 1, 1));

    // @todo Optimization: Call the padding mask here if your LanguageDataset provides it
    // apply_key_padding_mask(padding_mask, attention_weights);
/*
    softmax(attention_weights);
*/
    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const auto w_mat = attention_weights.reshape(array_2(total_heads, query_sequence_length * source_sequence_length)).chip(i, 0).reshape(array_2(query_sequence_length, source_sequence_length));
        const auto v_mat = value.reshape(array_2(total_heads, source_sequence_length * head_dimension)).chip(i, 0).reshape(array_2(source_sequence_length, head_dimension));
        auto o_mat = attention_outputs.reshape(array_2(total_heads, query_sequence_length * head_dimension)).chip(i, 0).reshape(array_2(query_sequence_length, head_dimension));

        o_mat.device(get_device()) = w_mat.contract(v_mat, axes(1, 0));
    }

    concatenated_attention_outputs.device(get_device()) = attention_outputs.shuffle(array_4(0, 2, 1, 3))
                                                                      .reshape(concatenated_attention_outputs.dimensions());

    const MatrixMap projection_weights_map = matrix_map(projection_weights);
    const VectorMap projection_biases_map = vector_map(projection_biases);

    outputs.device(get_device()) =
        concatenated_attention_outputs.contract(projection_weights_map, axes(2, 0))
        + projection_biases_map.reshape(array_3(1, 1, embedding_dimension))
        .broadcast(array_3(batch_size, query_sequence_length, 1));
}


void MultiHeadAttention::back_propagate(const vector<TensorView>& input_views,
                                        const vector<TensorView>& output_gradient_views,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{

    const TensorMap3 query_input = tensor_map<3>(input_views[0]);

    const TensorMap3 source_input = (input_views.size() == 1)
                                                        ? query_input
                                                        : tensor_map<3>(input_views[1]);

    const TensorMap3 delta_Y = tensor_map<3>(output_gradient_views[0]);

    // Forward propagation

    MultiHeadAttentionForwardPropagation* this_forward_propagation =
        static_cast<MultiHeadAttentionForwardPropagation*>(forward_propagation.get());

    const Tensor4& query = this_forward_propagation->query;
    const Tensor4& key = this_forward_propagation->key;
    const Tensor4& value = this_forward_propagation->value;
    const Tensor4& attention_weights = this_forward_propagation->attention_weights;
    const Tensor3& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    // Back propagation

    MultiHeadAttentionBackPropagation* this_back_propagation =
        static_cast<MultiHeadAttentionBackPropagation*>(back_propagation.get());

    MatrixMap projection_weight_gradients = matrix_map(this_back_propagation->projection_weight_gradients);
    VectorMap projection_bias_gradients = vector_map(this_back_propagation->projection_bias_gradients);
    Tensor3& concatenated_attention_output_gradients = this_back_propagation->concatenated_attention_output_gradients;
    Tensor4& attention_output_gradients = this_back_propagation->attention_output_gradients;
    Tensor4& attention_weight_gradients = this_back_propagation->attention_weight_gradients;
    Tensor4& query_gradients = this_back_propagation->query_gradients;
    Tensor4& key_gradients = this_back_propagation->key_gradients;
    Tensor4& value_gradients = this_back_propagation->value_gradients;
    MatrixMap query_weight_gradients = matrix_map(this_back_propagation->query_weight_gradients);
    VectorMap query_bias_gradients = vector_map(this_back_propagation->query_bias_gradients);
    MatrixMap key_weight_gradients = matrix_map(this_back_propagation->key_weight_gradients);
    VectorMap key_bias_gradients = vector_map(this_back_propagation->key_bias_gradients);
    MatrixMap value_weight_gradients = matrix_map(this_back_propagation->value_weight_gradients);
    VectorMap value_bias_gradients = vector_map(this_back_propagation->value_bias_gradients);

    TensorMap3 input_query_gradients = tensor_map<3>(back_propagation->input_gradients[0]);
    TensorMap3 input_source_gradients = tensor_map<3>(back_propagation->input_gradients[1]);

    Tensor4& softmax_gradients = this_back_propagation->softmax_gradients;

    const Index batch_size = this_forward_propagation->batch_size;
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();
    const type scaling_factor = get_scaling_factor();
    const Index total_heads = batch_size * heads_number;

    projection_weight_gradients.device(get_device()) =
        concatenated_attention_outputs.reshape(array_2(batch_size * query_sequence_length, embedding_dimension))
        .contract(delta_Y.reshape(array_2(batch_size * query_sequence_length, embedding_dimension)), axes(0, 0));

    projection_bias_gradients.device(get_device()) = delta_Y.sum(array_2(0, 1));
/*
    concatenated_attention_output_gradients.device(get_device()) =
        delta_Y.contract(projection_weights, axes(2, 1));
*/
    attention_output_gradients.device(get_device()) =
        concatenated_attention_output_gradients.reshape(array_4(batch_size, query_sequence_length, heads_number, head_dimension))
                                            .shuffle(array_4(0, 2, 1, 3));

    // @todo improve the following loops as before

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index h = 0; h < heads_number; ++h)
        {
            const auto w_slice = attention_weights.chip(b, 0).chip(h, 0); // [Lq, Ls]
            const auto do_slice = attention_output_gradients.chip(b, 0).chip(h, 0); // [Lq, Dh]
            const auto v_slice = value.chip(b, 0).chip(h, 0); // [Ls, Dh]

            value_gradients.chip(b, 0).chip(h, 0).device(get_device()) =
                w_slice.contract(do_slice, axes(0, 0));

            attention_weight_gradients.chip(b, 0).chip(h, 0).device(get_device()) =
                do_slice.contract(v_slice, axes(1, 1));
        }
    }

    auto dot_product = (attention_weights * attention_weight_gradients).sum(array_1(3));

    softmax_gradients.device(get_device()) = attention_weights * (attention_weight_gradients -
        dot_product.reshape(array_4(batch_size, heads_number, query_sequence_length, 1))
        .broadcast(array_4(1, 1, 1, source_sequence_length)));

    #pragma omp parallel for collapse(2)
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index h = 0; h < heads_number; ++h)
        {
            const auto sd_slice = softmax_gradients.chip(b, 0).chip(h, 0); // [Lq, Ls]
            const auto q_slice = query.chip(b, 0).chip(h, 0); // [Lq, Dh]
            const auto k_slice = key.chip(b, 0).chip(h, 0); // [Ls, Dh]

            query_gradients.chip(b, 0).chip(h, 0).device(get_device()) =
                sd_slice.contract(k_slice, axes(1, 0)) * scaling_factor;

            key_gradients.chip(b, 0).chip(h, 0).device(get_device()) =
                sd_slice.contract(q_slice, axes(0, 0)) * scaling_factor;
        }
    }

    // Query Projection
    calculate_projection_gradient(query_gradients, query_input, query_weights,
                                  query_bias_gradients, query_weight_gradients,
                                  input_query_gradients, batch_size, false);

    // Key Projection
    calculate_projection_gradient(key_gradients, source_input, key_weights,
                                  key_bias_gradients, key_weight_gradients,
                                  input_source_gradients, batch_size, false);

    // Value Projection (accumulate=true because it shares input with Key)
    calculate_projection_gradient(value_gradients, source_input, value_weights,
                                  value_bias_gradients, value_weight_gradients,
                                  input_source_gradients, batch_size, true);

    if(input_views.size() == 1)
        input_query_gradients.device(get_device()) += input_source_gradients;

}


void MultiHeadAttention::apply_causal_mask(Tensor4& attention_scores) const
{
    // @todo

    const Index batch_size = attention_scores.dimension(2);

    const Index context_input_size = source_sequence_length * query_sequence_length;

    for(Index head_index = 0; head_index < heads_number; head_index++)
    {
        for(Index sample_index = 0; sample_index < batch_size; sample_index++)
        {
            type* sample_attention_scores_data = attention_scores.data()
             + (sample_index + head_index * batch_size) * context_input_size;

             MatrixMap sample_attention_scores(sample_attention_scores_data,
                                                source_sequence_length,
                                                query_sequence_length);

             sample_attention_scores.device(get_device()) += causal_mask;
         }
    }
}


void MultiHeadAttention::apply_key_padding_mask(const MatrixB& key_padding_mask,
                                                Tensor4& attention_weights) const
{
/*
    // @todo (I don't know if it is building the mask correctly)
    const Index batch_size  = attention_weights.dimension(2);

    Tensor2 key_padding_mask_type(key_padding_mask.rows(),key_padding_mask.cols());

    for(Index h = 0; h < heads_number; ++h)
    {
        for(Index b = 0; b < batch_size; ++b)
        {
            MatrixMap head_sample_attention_weights = tensor_map(attention_weights,h,b);

            head_sample_attention_weights.device(get_device())
                += key_padding_mask.row(b)
                       .cast<type>()
                       .reshape(array<Index,2>{source_sequence_length, 1})
                       .broadcast(array<Index,2>{1, query_sequence_length})
                   * type(-10e9);
        }
    }
*/
}


void MultiHeadAttention::print() const
{
    cout << "Multi-head attention Layer" << endl
         << "Label: " << label << endl
         << "Type: Embedding" << endl
         << "Input shape: " << get_input_shape() << endl
         << "Output shape: " << get_output_shape();
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
}


MultiHeadAttentionForwardPropagation::MultiHeadAttentionForwardPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void MultiHeadAttentionForwardPropagation::initialize()
{
    const MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

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

    // @todo can we remove concatenated_attention_outputs and assign to outputs?

    concatenated_attention_outputs.resize(batch_size, query_sequence_length, embedding_dimension);

    outputs.shape = {batch_size, query_sequence_length, embedding_dimension};
}


void MultiHeadAttentionForwardPropagation::print() const
{
//    cout << "Output shape:" << output_shape << endl
//    cout << "Outputs:" << endl;
    //cout << TensorMap<Tensor<type,3>>(outputs_data, output_shape(0), output_shape(1), output_shape(2)) << endl;
}


void MultiHeadAttentionBackPropagation::initialize()
{
    const MultiHeadAttention* multihead_attention_layer = static_cast<MultiHeadAttention*>(layer);

    const Index query_sequence_length = multihead_attention_layer->get_query_sequence_length();
    const Index source_sequence_length = multihead_attention_layer->get_source_sequence_length();
    const Index embedding_dimension = multihead_attention_layer->get_embedding_dimension();
    const Index heads_number = multihead_attention_layer->get_heads_number();
    const Index head_dimension = multihead_attention_layer->get_head_dimension();

    query_weight_gradients.shape = {embedding_dimension, embedding_dimension};
    key_weight_gradients.shape = {embedding_dimension, embedding_dimension};
    value_weight_gradients.shape = {embedding_dimension, embedding_dimension};
    projection_weight_gradients.shape = {embedding_dimension, embedding_dimension};

    query_bias_gradients.shape = {embedding_dimension};
    key_bias_gradients.shape = {embedding_dimension};
    value_bias_gradients.shape = {embedding_dimension};
    projection_bias_gradients.shape = {embedding_dimension};

    input_gradients.resize(2);
    input_gradients[0].shape = {batch_size, query_sequence_length, embedding_dimension};
    input_gradients[1].shape = {batch_size, source_sequence_length, embedding_dimension};

    // Auxiliar

    softmax_gradients.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

    attention_weight_gradients.resize(batch_size, heads_number, source_sequence_length, query_sequence_length);

    attention_output_gradients.resize(batch_size, heads_number, query_sequence_length, head_dimension);

    concatenated_attention_output_gradients.resize(batch_size, query_sequence_length, embedding_dimension);

    query_gradients.resize(batch_size, heads_number, query_sequence_length, head_dimension);
    key_gradients.resize(batch_size, heads_number, source_sequence_length, head_dimension);
    value_gradients.resize(batch_size, heads_number, source_sequence_length, head_dimension);

    aux_rows.resize(source_sequence_length);
}


void MultiHeadAttentionBackPropagation::print() const
{
}


MultiHeadAttentionBackPropagation::MultiHeadAttentionBackPropagation(const Index new_batch_size,
                                                                     Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<TensorView*> MultiHeadAttentionBackPropagation::get_gradient_views()
{
    return {&query_weight_gradients, &query_bias_gradients,
            &key_weight_gradients, &key_bias_gradients,
            &value_weight_gradients, &value_bias_gradients,
            &projection_weight_gradients, &projection_bias_gradients};
}

REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")
REGISTER(LayerForwardPropagation, MultiHeadAttentionForwardPropagation, "MultiHeadAttention")
REGISTER(LayerBackPropagation, MultiHeadAttentionBackPropagation, "MultiHeadAttention")
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
