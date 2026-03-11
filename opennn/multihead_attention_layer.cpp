//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
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

    if(new_heads_number == 0 && new_embedding_dimension == 0)
    {
        heads_number = 0;
        return;
    }

    if(new_heads_number <= 0)
        throw runtime_error("MultiHeadAttention Error: Heads number must be greater than 0.");

    if(new_embedding_dimension % new_heads_number != 0)
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
    {
        causal_mask.resize(query_sequence_length, source_sequence_length);

        for(Index row = 0; row < query_sequence_length; ++row)
            for(Index column = 0; column < source_sequence_length; ++column)
                causal_mask(row, column) = (column > row) ? minus_inf : type(0);
    }
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

    type* query_data = query.data();
    type* key_data = key.data();
    type* att_weights_data = attention_weights.data();

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_q = i * query_sequence_length * head_dimension;
        const Index offset_k = i * source_sequence_length * head_dimension;
        const Index offset_w = i * query_sequence_length * source_sequence_length;

        const MatrixMap q_mat(query_data + offset_q, query_sequence_length, head_dimension);
        const MatrixMap k_mat(key_data + offset_k, source_sequence_length, head_dimension);
        MatrixMap w_mat(att_weights_data + offset_w, query_sequence_length, source_sequence_length);

        w_mat.noalias() = (q_mat * k_mat.transpose()) * scaling_factor;
    }

    if (use_causal_mask)
        apply_causal_mask(attention_weights);

    const Index total_rows_softmax = batch_size * heads_number * query_sequence_length;
    MatrixMap attention_weights_map(attention_weights.data(), total_rows_softmax, source_sequence_length);

    softmax(attention_weights_map);

    type* value_data = value.data();
    type* att_outputs_data = attention_outputs.data();

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const Index offset_v = i * source_sequence_length * head_dimension;
        const Index offset_o = i * query_sequence_length * head_dimension;

        const MatrixMap w_mat(att_weights_data + offset_w, query_sequence_length, source_sequence_length);
        const MatrixMap v_mat(value_data + offset_v, source_sequence_length, head_dimension);
        MatrixMap o_mat(att_outputs_data + offset_o, query_sequence_length, head_dimension);

        o_mat.noalias() = w_mat * v_mat;
    }

    concatenated_attention_outputs.device(get_device()) = attention_outputs.shuffle(array_4(0, 2, 1, 3))
                                                                      .reshape(concatenated_attention_outputs.dimensions());

    const MatrixMap projection_weights_map = matrix_map(projection_weights);
    const VectorMap projection_biases_map = vector_map(projection_biases);

    const Index total_rows = batch_size * query_sequence_length;

    const MatrixMap concatenated_map(const_cast<type*>(concatenated_attention_outputs.data()), total_rows, embedding_dimension);
    MatrixMap outputs_map(outputs.data(), total_rows, embedding_dimension);

    outputs_map.noalias() = (concatenated_map * projection_weights_map).rowwise() + projection_biases_map.transpose();
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

    const Index total_rows = batch_size * query_sequence_length;

    const MatrixMap concatenated_map(const_cast<type*>(concatenated_attention_outputs.data()), total_rows, embedding_dimension);
    const MatrixMap delta_Y_map(const_cast<type*>(delta_Y.data()), total_rows, embedding_dimension);

    projection_weight_gradients.noalias() = concatenated_map.transpose() * delta_Y_map;
    projection_bias_gradients.noalias() = delta_Y_map.colwise().sum();

    MatrixMap concat_grad_map(concatenated_attention_output_gradients.data(), total_rows, embedding_dimension);
    const MatrixMap proj_weights_map = matrix_map(projection_weights);

    concat_grad_map.noalias() = delta_Y_map * proj_weights_map.transpose();

    attention_output_gradients.device(get_device()) =
        concatenated_attention_output_gradients.reshape(array_4(batch_size, query_sequence_length, heads_number, head_dimension))
                                            .shuffle(array_4(0, 2, 1, 3));

    const Index total_heads = batch_size * heads_number;
    type* att_weights_data = const_cast<type*>(attention_weights.data());
    type* att_out_grad_data = attention_output_gradients.data();
    type* v_data = const_cast<type*>(value.data());
    type* v_grad_data = value_gradients.data();
    type* att_weight_grad_data = attention_weight_gradients.data();

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const Index offset_do = i * query_sequence_length * head_dimension;
        const Index offset_v = i * source_sequence_length * head_dimension;

        const MatrixMap W(att_weights_data + offset_w, query_sequence_length, source_sequence_length);
        const MatrixMap dO(att_out_grad_data + offset_do, query_sequence_length, head_dimension);
        const MatrixMap V(v_data + offset_v, source_sequence_length, head_dimension);

        MatrixMap dV(v_grad_data + offset_v, source_sequence_length, head_dimension);
        MatrixMap dW(att_weight_grad_data + offset_w, query_sequence_length, source_sequence_length);

        dV.noalias() = W.transpose() * dO;
        dW.noalias() = dO * V.transpose();
    }

    type* sm_grad_data = softmax_gradients.data();
    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const MatrixMap W(att_weights_data + offset_w, query_sequence_length, source_sequence_length);
        const MatrixMap dW(att_weight_grad_data + offset_w, query_sequence_length, source_sequence_length);
        MatrixMap dS(sm_grad_data + offset_w, query_sequence_length, source_sequence_length);

        VectorR dot_product = (W.array() * dW.array()).rowwise().sum();

        for(Index r = 0; r < query_sequence_length; ++r)
            dS.row(r).array() = W.row(r).array() * (dW.row(r).array() - dot_product(r));
    }

    type* q_data = const_cast<type*>(query.data());
    type* k_data = const_cast<type*>(key.data());
    type* q_grad_data = query_gradients.data();
    type* k_grad_data = key_gradients.data();

    #pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const Index offset_q = i * query_sequence_length * head_dimension;
        const Index offset_k = i * source_sequence_length * head_dimension;

        const MatrixMap dS(sm_grad_data + offset_w, query_sequence_length, source_sequence_length);
        const MatrixMap Q(q_data + offset_q, query_sequence_length, head_dimension);
        const MatrixMap K(k_data + offset_k, source_sequence_length, head_dimension);

        MatrixMap dQ(q_grad_data + offset_q, query_sequence_length, head_dimension);
        MatrixMap dK(k_grad_data + offset_k, source_sequence_length, head_dimension);

        dQ.noalias() = (dS * K) * scaling_factor;
        dK.noalias() = (dS.transpose() * Q) * scaling_factor;
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
    {
        input_query_gradients.device(get_device()) += input_source_gradients;
        //back_propagation->input_gradients.resize(1);
    }
}


void MultiHeadAttention::apply_causal_mask(Tensor4& attention_scores) const
{
    const Index batch_size = attention_scores.dimension(0);
    const Index heads = attention_scores.dimension(1);
    const Index query_sequence_length = attention_scores.dimension(2);
    const Index source_sequence_length = attention_scores.dimension(3);

    const Index matrix_size = query_sequence_length * source_sequence_length;

    const Index total_matrices = batch_size * heads;

    MatrixMap scores(attention_scores.data(), total_matrices, matrix_size);

    const VectorMap causal_mask_map(const_cast<type*>(causal_mask.data()), matrix_size);

    scores.rowwise() += causal_mask_map.transpose();
}


void MultiHeadAttention::apply_key_padding_mask(const MatrixB& key_padding_mask,
                                                Tensor4& attention_weights) const
{
    if (key_padding_mask.size() == 0) return;

    const Index batch_size = attention_weights.dimension(0);
    const Index heads = attention_weights.dimension(1);
    const Index query_sequence_length = attention_weights.dimension(2);
    const Index source_sequence_length = attention_weights.dimension(3);

    const Index rows_per_batch = heads * query_sequence_length;
    const type mask_penalty = type(-1e9);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        type* batch_data = attention_weights.data() + (b * rows_per_batch * source_sequence_length);
        MatrixMap batch_map(batch_data, rows_per_batch, source_sequence_length);

        batch_map.rowwise() += key_padding_mask.row(b).template cast<type>() * mask_penalty;
    }
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

    input_gradients_memory.resize(2);
    input_gradients_memory[0].resize(batch_size * query_sequence_length * embedding_dimension);
    input_gradients_memory[1].resize(batch_size * source_sequence_length * embedding_dimension);

    input_gradients.resize(2);
    input_gradients[0].data = input_gradients_memory[0].data();
    input_gradients[0].shape = {batch_size, query_sequence_length, embedding_dimension};

    input_gradients[1].data = input_gradients_memory[1].data();
    input_gradients[1].shape = {batch_size, source_sequence_length, embedding_dimension};

    // Auxiliar

    softmax_gradients.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

    attention_weight_gradients.resize(batch_size, heads_number, query_sequence_length, source_sequence_length);

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
