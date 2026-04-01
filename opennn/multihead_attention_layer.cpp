//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "multihead_attention_layer.h"
#include "neural_network.h"
#include "loss.h"

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

    // Cross-attention

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
    return input_shape.back();
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


vector<Shape> MultiHeadAttention::get_parameter_shapes() const
{
    return {{embedding_dimension, embedding_dimension},
            {embedding_dimension},
            {embedding_dimension, embedding_dimension},
            {embedding_dimension},
            {embedding_dimension, embedding_dimension},
            {embedding_dimension},
            {embedding_dimension, embedding_dimension},
            {embedding_dimension}};
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


void MultiHeadAttention::forward_propagate(ForwardPropagation& forward_propagation,
                                           size_t layer,
                                           bool)
{
    const TensorView& query_input = forward_propagation.views[layer][Inputs][0];

    const TensorView& source_input = (forward_propagation.views[layer][Inputs].size() == 1)
                                        ? query_input
                                        : forward_propagation.views[layer][Inputs][1];

    const TensorView& query_weights = parameters[QueryWeights];
    const TensorView& query_biases = parameters[QueryBiases];
    TensorView& query = forward_propagation.views[layer][Query][0];

    projection(query_input, query_weights, query_biases, query);


    const TensorView& key_weights = parameters[KeyWeights];
    const TensorView& key_biases = parameters[KeyBiases];
    TensorView& key = forward_propagation.views[layer][Key][0];

    projection(source_input, key_weights, key_biases, key);

    const TensorView& value_weights = parameters[ValueWeights];
    const TensorView& value_biases = parameters[KeyBiases];
    TensorView& value = forward_propagation.views[layer][Value][0];

    projection(source_input, value_weights, value_biases, value);

    TensorView& attention_weights = forward_propagation.views[layer][AttentionWeights][0];

    softmax(attention_weights);

#ifndef CUDA
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();
    const type scaling_factor = get_scaling_factor();
    const Index batch_size = forward_propagation.batch_size;
/*
    Tensor3& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    TensorMap3 outputs = tensor_map<3>(forward_propagation->outputs);

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

    apply_key_padding_mask(source_input, attention_weights);

    if (use_causal_mask)
        apply_causal_mask(attention_weights);

    const Index total_rows_softmax = batch_size * heads_number * query_sequence_length;
    MatrixMap attention_weights_map(attention_weights.data(), total_rows_softmax, source_sequence_length);

    softmax(attention_weights_map);

    type* value_data = value.data();
    type* concat_data = concatenated_attention_outputs.data();

    const int original_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
    {
        for (Index h = 0; h < heads_number; ++h)
        {
            const Index offset_v = b * (heads_number * source_sequence_length * head_dimension) + h * (source_sequence_length * head_dimension);
            const Index offset_w = b * (heads_number * query_sequence_length * source_sequence_length) + h * (query_sequence_length * source_sequence_length);

            const MatrixMap w_mat(att_weights_data + offset_w, query_sequence_length, source_sequence_length);
            const MatrixMap v_mat(value_data + offset_v, source_sequence_length, head_dimension);

            type* out_ptr = concat_data + b * (query_sequence_length * embedding_dimension) + h * head_dimension;

            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<MatrixR, 0, StrideType> o_mat(out_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));

            o_mat.noalias() = w_mat * v_mat;
        }
    }

    Eigen::setNbThreads(original_eigen_threads);

    const MatrixMap projection_weights_map = matrix_map(projection_weights);
    const VectorMap projection_biases_map = vector_map(projection_biases);

    const Index total_rows = batch_size * query_sequence_length;

    const MatrixMap concatenated_map(const_cast<type*>(concatenated_attention_outputs.data()), total_rows, embedding_dimension);
    MatrixMap outputs_map(outputs.data(), total_rows, embedding_dimension);

    outputs_map.noalias() = (concatenated_map * projection_weights_map).rowwise() + projection_biases_map.transpose();
*/
#else
    const Index batch_size = forward->batch_size;
    const Index query_sequence_length = this->query_sequence_length;
    const Index source_sequence_length = this->source_sequence_length;
    const Index embedding_dimension = get_embedding_dimension();
    const Index heads_number = this->heads_number;
    const Index head_dimension = get_head_dimension();

    const Index total_weights = batch_size * heads_number * query_sequence_length * source_sequence_length;

    const float scaling_factor = static_cast<float>(get_scaling_factor());

    const float* query_input = forward_propagation->inputs[0].data;
    const float* source_input = (forward_propagation->inputs.size() == 1) ? query_input : forward_propagation->inputs[1].data;

    // Query projection
    linear_projection_cuda(query_input, query_weights_device.data, query_biases_device.data,
                           query_biases_device.get_descriptor(), forward->query.data,
                           forward->query.get_descriptor(),
                           (int)(batch_size * query_sequence_length),
                           (int)embedding_dimension, (int)embedding_dimension);

    // Key projection
    linear_projection_cuda(source_input, key_weights_device.data, key_biases_device.data,
                           key_biases_device.get_descriptor(), forward->key.data,
                           forward->key.get_descriptor(),
                           (int)(batch_size * source_sequence_length),
                           (int)embedding_dimension, (int)embedding_dimension);

    // Value projection
    linear_projection_cuda(source_input, value_weights_device.data, value_biases_device.data,
                           value_biases_device.get_descriptor(), forward->value.data,
                           forward->value.get_descriptor(),
                           (int)(batch_size * source_sequence_length),
                           (int)embedding_dimension, (int)embedding_dimension);

    // Transpositions

    mha_transpose_qkv_cuda(batch_size * query_sequence_length * embedding_dimension,
                           forward->query.data, forward->query_transposed.data,
                           (int)query_sequence_length, (int)heads_number, (int)head_dimension);

    mha_transpose_qkv_cuda(batch_size * source_sequence_length * embedding_dimension,
                           forward->key.data, forward->key_transposed.data,
                           (int)source_sequence_length, (int)heads_number, (int)head_dimension);

    mha_transpose_qkv_cuda(batch_size * source_sequence_length * embedding_dimension,
                           forward->value.data, forward->value_transposed.data,
                           (int)source_sequence_length, (int)heads_number, (int)head_dimension);

    // Attention scores (Q * K^T)

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              (int)source_sequence_length, (int)query_sequence_length, (int)head_dimension,
                              &scaling_factor,
                              forward->key_transposed.data, (int)head_dimension, (int)(source_sequence_length * head_dimension),
                              forward->query_transposed.data, (int)head_dimension, (int)(query_sequence_length * head_dimension),
                              &beta_zero,
                              forward->attention_weights.data, (int)source_sequence_length, (int)(query_sequence_length * source_sequence_length),
                              (int)(batch_size * heads_number));

    // Key padding mask

    mha_key_padding_mask_cuda(
        total_weights,
        source_input,
        forward->attention_weights.data,
        (int)heads_number,
        (int)query_sequence_length,
        (int)source_sequence_length,
        (int)embedding_dimension
        );

    if(use_causal_mask)
    {
        mha_causal_mask_cuda(batch_size * heads_number * query_sequence_length * source_sequence_length,
                             forward->attention_weights.data,
                             (int)query_sequence_length, (int)source_sequence_length);
    }

    // Softmax

    cudnnSoftmaxForward(get_cudnn_handle(),
                        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha_one,
                        forward->attention_weights.get_descriptor(), forward->attention_weights.data,
                        &beta_zero,
                        forward->attention_probabilities.get_descriptor(), forward->attention_probabilities.data);

    // Attention context (Probs * V)

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              (int)head_dimension, (int)query_sequence_length, (int)source_sequence_length,
                              &alpha_one,
                              forward->value_transposed.data, (int)head_dimension, (int)(source_sequence_length * head_dimension),
                              forward->attention_probabilities.data, (int)source_sequence_length, (int)(query_sequence_length * source_sequence_length),
                              &beta_zero,
                              forward->attention_outputs_transposed.data, (int)head_dimension, (int)(query_sequence_length * head_dimension),
                              (int)(batch_size * heads_number));

    // Concatenation and output projection

    mha_transpose_o_cuda(batch_size * query_sequence_length * embedding_dimension,
                         forward->attention_outputs_transposed.data, forward->concatenated_attention_outputs.data,
                         (int)query_sequence_length, (int)heads_number, (int)head_dimension);

    linear_projection_cuda(forward->concatenated_attention_outputs.data,
                           projection_weights_device.data, projection_biases_device.data,
                           projection_biases_device.get_descriptor(), forward->outputs.data,
                           forward->outputs.get_descriptor(),
                           (int)(batch_size * query_sequence_length),
                           (int)embedding_dimension, (int)embedding_dimension);
#endif
}


void MultiHeadAttention::back_propagate(ForwardPropagation& forward_propagation,
                                        BackPropagation& back_propagation,
                                        size_t layer) const
{
/*
    if(back_propagation->output_gradients.size() > 1)
        add_gradients(back_propagation->output_gradients);
*/
    const TensorView& query_input = forward_propagation.views[layer][Inputs][0];

    const TensorView& source_input = (forward_propagation.views[layer][Inputs].size() == 1)
                                         ? query_input
                                         : forward_propagation.views[layer][Inputs][1];

    const TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradient][0];

    const TensorView& query = forward_propagation.views[layer][Query][0];
    const TensorView& key = forward_propagation.views[layer][Query][0];
    const TensorView& value = forward_propagation.views[layer][Query][0];

#ifndef CUDA
/*
    const Tensor4& attention_weights = this_forward_propagation->attention_weights;
    const Tensor3& concatenated_attention_outputs = this_forward_propagation->concatenated_attention_outputs;

    // Back propagation

    MatrixMap projection_weight_gradients = matrix_map(this_back_propagation->projection_weight_gradients);
    VectorMap projection_bias_gradients = vector_map(this_back_propagation->projection_bias_gradients);
    Tensor3& concatenated_attention_output_gradients = this_back_propagation->concatenated_attention_output_gradients;
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

    const bool self_attention = (forward_propagation->inputs.size() == 1);

    const Index batch_size = this_forward_propagation->batch_size;
    const Index embedding_dimension = get_embedding_dimension();
    const Index head_dimension = get_head_dimension();
    const type scaling_factor = get_scaling_factor();

    const Index total_rows = batch_size * query_sequence_length;
    const Index total_heads = batch_size * heads_number;

    const MatrixMap concatenated_map(const_cast<type*>(concatenated_attention_outputs.data()), total_rows, embedding_dimension);
    const MatrixMap delta_Y_map(const_cast<type*>(delta_Y.data()), total_rows, embedding_dimension);

    projection_weight_gradients.noalias() = concatenated_map.transpose() * delta_Y_map;

    projection_bias_gradients.noalias() = delta_Y_map.colwise().sum();

    MatrixMap concat_grad_map(concatenated_attention_output_gradients.data(), total_rows, embedding_dimension);
    const MatrixMap proj_weights_map = matrix_map(projection_weights);

    concat_grad_map.noalias() = delta_Y_map * proj_weights_map.transpose();

    type* att_weights_data = const_cast<type*>(attention_weights.data());
    type* v_data = const_cast<type*>(value.data());
    type* v_grad_data = value_gradients.data();
    type* att_weight_grad_data = attention_weight_gradients.data();
    type* concat_grad_data = concatenated_attention_output_gradients.data();

    const int original_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
    {
        for (Index h = 0; h < heads_number; ++h)
        {
            const Index offset_w = b * (heads_number * query_sequence_length * source_sequence_length) + h * (query_sequence_length * source_sequence_length);
            const Index offset_v = b * (heads_number * source_sequence_length * head_dimension) + h * (source_sequence_length * head_dimension);

            const MatrixMap W(att_weights_data + offset_w, query_sequence_length, source_sequence_length);
            const MatrixMap V(v_data + offset_v, source_sequence_length, head_dimension);

            MatrixMap dV(v_grad_data + offset_v, source_sequence_length, head_dimension);
            MatrixMap dW(att_weight_grad_data + offset_w, query_sequence_length, source_sequence_length);

            type* dO_ptr = concat_grad_data + b * (query_sequence_length * embedding_dimension) + h * head_dimension;

            using StrideType = Eigen::OuterStride<Eigen::Dynamic>;
            Eigen::Map<const MatrixR, 0, StrideType> dO(dO_ptr, query_sequence_length, head_dimension, StrideType(embedding_dimension));

            dV.noalias() = W.transpose() * dO;
            dW.noalias() = dO * V.transpose();
        }
    }
    Eigen::setNbThreads(original_eigen_threads);

#pragma omp parallel for
    for(Index i = 0; i < total_heads; ++i)
    {
        const Index offset_w = i * query_sequence_length * source_sequence_length;
        const MatrixMap W(att_weights_data + offset_w, query_sequence_length, source_sequence_length);

        // Remove 'const' so we can overwrite it
        MatrixMap dW(att_weight_grad_data + offset_w, query_sequence_length, source_sequence_length);

        VectorR dot_product = (W.array() * dW.array()).rowwise().sum();

        // Overwrite dW in-place
        dW.array() = W.array() * (dW.colwise() - dot_product).array();
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

        // Use att_weight_grad_data (which now holds the softmax derivative)
        const MatrixMap dW(att_weight_grad_data + offset_w, query_sequence_length, source_sequence_length);
        const MatrixMap Q(q_data + offset_q, query_sequence_length, head_dimension);
        const MatrixMap K(k_data + offset_k, source_sequence_length, head_dimension);

        MatrixMap dQ(q_grad_data + offset_q, query_sequence_length, head_dimension);
        MatrixMap dK(k_grad_data + offset_k, source_sequence_length, head_dimension);

        dQ.noalias() = (dW * K) * scaling_factor;
        dK.noalias() = (dW.transpose() * Q) * scaling_factor;
    }

    // Query Projection
    calculate_projection_gradient(query_gradients, query_input, query_weights,
                                  query_bias_gradients, query_weight_gradients,
                                  input_query_gradients, batch_size, false);

    if(self_attention)
    {
        // Key Projection
        calculate_projection_gradient(key_gradients, source_input, key_weights,
                                      key_bias_gradients, key_weight_gradients,
                                      input_query_gradients, batch_size, true);

        // Value Projection
        calculate_projection_gradient(value_gradients, source_input, value_weights,
                                      value_bias_gradients, value_weight_gradients,
                                      input_query_gradients, batch_size, true);
    }
    else
    {
        TensorMap3 input_source_gradients = tensor_map<3>(back_propagation->input_gradients[1]);

        // Key Projection
        calculate_projection_gradient(key_gradients, source_input, key_weights,
                                      key_bias_gradients, key_weight_gradients,
                                      input_source_gradients, batch_size, false);

        // Value Projection
        calculate_projection_gradient(value_gradients, source_input, value_weights,
                                      value_bias_gradients, value_weight_gradients,
                                      input_source_gradients, batch_size, true);
    }
*/
#else
    back->ones.fill(1.0f);

    const Index batch_size = forward->batch_size;
    const Index query_sequence_length = this->query_sequence_length;
    const Index source_sequence_length = this->source_sequence_length;
    const Index embedding_dimension = get_embedding_dimension();
    const Index heads_number = this->heads_number;
    const Index head_dimension = get_head_dimension();

    const float scaling_factor = static_cast<float>(get_scaling_factor());

    const float* query_input = forward_propagation->inputs[0].data;
    const float* source_input = (forward_propagation->inputs.size() == 1) ? query_input : forward_propagation->inputs[1].data;
    const float* output_gradients_data = back_propagation->output_gradients[0].data;

    // Projection weight gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int)embedding_dimension, (int)embedding_dimension, (int)(batch_size * query_sequence_length),
                &alpha_one,
                output_gradients_data, (int)embedding_dimension,
                forward->concatenated_attention_outputs.data, (int)embedding_dimension,
                &beta_zero,
                back->projection_weight_gradients.data, (int)embedding_dimension);

    // Projection bias gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)embedding_dimension, 1, (int)(batch_size * query_sequence_length),
                &alpha_one,
                output_gradients_data, (int)embedding_dimension,
                back->ones.data, (int)(batch_size * query_sequence_length),
                &beta_zero,
                back->projection_bias_gradients.data, (int)embedding_dimension);

    // Concatenated attention output gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)embedding_dimension, (int)(batch_size * query_sequence_length), (int)embedding_dimension,
                &alpha_one,
                projection_weights_device.data, (int)embedding_dimension,
                output_gradients_data, (int)embedding_dimension,
                &beta_zero,
                back->concatenated_attention_output_gradients.data, (int)embedding_dimension);

    // Attention output gradients transposed

    mha_transpose_qkv_cuda(batch_size * query_sequence_length * embedding_dimension,
                           back->concatenated_attention_output_gradients.data, back->attention_output_gradients_transposed.data,
                           (int)query_sequence_length, (int)heads_number, (int)head_dimension);

    // Value gradients transposed

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              (int)head_dimension, (int)source_sequence_length, (int)query_sequence_length,
                              &alpha_one,
                              back->attention_output_gradients_transposed.data, (int)head_dimension, (int)(query_sequence_length * head_dimension),
                              forward->attention_probabilities.data, (int)source_sequence_length, (int)(query_sequence_length * source_sequence_length),
                              &beta_zero,
                              back->value_gradients_transposed.data, (int)head_dimension, (int)(source_sequence_length * head_dimension),
                              (int)(batch_size * heads_number));

    // Attention weight gradients

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              (int)source_sequence_length, (int)query_sequence_length, (int)head_dimension,
                              &alpha_one,
                              forward->value_transposed.data, (int)head_dimension, (int)(source_sequence_length * head_dimension),
                              back->attention_output_gradients_transposed.data, (int)head_dimension, (int)(query_sequence_length * head_dimension),
                              &beta_zero,
                              back->attention_weight_gradients.data, (int)source_sequence_length, (int)(query_sequence_length * source_sequence_length),
                              (int)(batch_size * heads_number));

    // Softmax gradients

    cudnnSoftmaxBackward(get_cudnn_handle(),
                         CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                         &alpha_one,
                         forward->attention_probabilities.get_descriptor(), forward->attention_probabilities.data,
                         back->attention_weight_gradients.get_descriptor(), back->attention_weight_gradients.data,
                         &beta_zero,
                         back->softmax_gradients.get_descriptor(), back->softmax_gradients.data);

    // Query and key gradients transposed

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              (int)head_dimension, (int)query_sequence_length, (int)source_sequence_length,
                              &scaling_factor,
                              forward->key_transposed.data, (int)head_dimension, (int)(source_sequence_length * head_dimension),
                              back->softmax_gradients.data, (int)source_sequence_length, (int)(query_sequence_length * source_sequence_length),
                              &beta_zero,
                              back->query_gradients_transposed.data, (int)head_dimension, (int)(query_sequence_length * head_dimension),
                              (int)(batch_size * heads_number));

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              (int)head_dimension, (int)source_sequence_length, (int)query_sequence_length,
                              &scaling_factor,
                              forward->query_transposed.data, (int)head_dimension, (int)(query_sequence_length * head_dimension),
                              back->softmax_gradients.data, (int)source_sequence_length, (int)(query_sequence_length * source_sequence_length),
                              &beta_zero,
                              back->key_gradients_transposed.data, (int)head_dimension, (int)(source_sequence_length * head_dimension),
                              (int)(batch_size * heads_number));

    // Des-transposition to flat

    mha_transpose_o_cuda(batch_size * query_sequence_length * embedding_dimension,
                         back->query_gradients_transposed.data, back->query_gradients.data,
                         (int)query_sequence_length, (int)heads_number, (int)head_dimension);

    mha_transpose_o_cuda(batch_size * source_sequence_length * embedding_dimension,
                         back->key_gradients_transposed.data, back->key_gradients.data,
                         (int)source_sequence_length, (int)heads_number, (int)head_dimension);

    mha_transpose_o_cuda(batch_size * source_sequence_length * embedding_dimension,
                         back->value_gradients_transposed.data, back->value_gradients.data,
                         (int)source_sequence_length, (int)heads_number, (int)head_dimension);

    // Query weight, bias and input gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int)embedding_dimension, (int)embedding_dimension, (int)(batch_size * query_sequence_length),
                &alpha_one,
                back->query_gradients.data, (int)embedding_dimension,
                query_input, (int)embedding_dimension,
                &beta_zero,
                back->query_weight_gradients.data, (int)embedding_dimension);

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)embedding_dimension, 1, (int)(batch_size * query_sequence_length),
                &alpha_one,
                back->query_gradients.data, (int)embedding_dimension,
                back->ones.data, (int)(batch_size * query_sequence_length),
                &beta_zero,
                back->query_bias_gradients.data, (int)embedding_dimension);

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)embedding_dimension, (int)(batch_size * query_sequence_length), (int)embedding_dimension,
                &alpha_one,
                query_weights_device.data, (int)embedding_dimension,
                back->query_gradients.data, (int)embedding_dimension,
                &beta_zero,
                back->query_input_gradients.data, (int)embedding_dimension);

    // Key weight, bias and source projection gradients (Temp)

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int)embedding_dimension, (int)embedding_dimension, (int)(batch_size * source_sequence_length),
                &alpha_one,
                back->key_gradients.data, (int)embedding_dimension,
                source_input, (int)embedding_dimension,
                &beta_zero,
                back->key_weight_gradients.data, (int)embedding_dimension);

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)embedding_dimension, 1, (int)(batch_size * source_sequence_length),
                &alpha_one,
                back->key_gradients.data, (int)embedding_dimension,
                back->ones.data, (int)(batch_size * source_sequence_length),
                &beta_zero,
                back->key_bias_gradients.data, (int)embedding_dimension);

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)embedding_dimension, (int)(batch_size * source_sequence_length), (int)embedding_dimension,
                &alpha_one,
                key_weights_device.data, (int)embedding_dimension,
                back->key_gradients.data, (int)embedding_dimension,
                &beta_zero,
                back->source_input_gradients.data, (int)embedding_dimension);

    // Value weight, bias and source input gradients accumulation

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int)embedding_dimension, (int)embedding_dimension, (int)(batch_size * source_sequence_length),
                &alpha_one,
                back->value_gradients.data, (int)embedding_dimension,
                source_input, (int)embedding_dimension,
                &beta_zero,
                back->value_weight_gradients.data, (int)embedding_dimension);

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)embedding_dimension, 1, (int)(batch_size * source_sequence_length),
                &alpha_one,
                back->value_gradients.data, (int)embedding_dimension,
                back->ones.data, (int)(batch_size * source_sequence_length),
                &beta_zero,
                back->value_bias_gradients.data, (int)embedding_dimension);

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)embedding_dimension, (int)(batch_size * source_sequence_length), (int)embedding_dimension,
                &alpha_one,
                value_weights_device.data, (int)embedding_dimension,
                back->value_gradients.data, (int)embedding_dimension,
                &alpha_one,
                back->source_input_gradients.data, (int)embedding_dimension);

    // Final input gradients

    if(forward_propagation->inputs.size() == 1)
    {
        addition_cuda(batch_size * query_sequence_length * embedding_dimension,
                      back->query_input_gradients.data, back->source_input_gradients.data,
                      back->input_gradients[0].data);
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(back->input_gradients[0].data,
                              back->query_input_gradients.data,
                              batch_size * query_sequence_length * embedding_dimension * sizeof(type),
                              cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaMemcpy(back->input_gradients[1].data,
                              back->source_input_gradients.data,
                              batch_size * source_sequence_length * embedding_dimension * sizeof(type),
                              cudaMemcpyDeviceToDevice));
    }

#endif
}


void MultiHeadAttention::apply_causal_mask(Tensor4& attention_scores) const
{
    const Index batch_size = attention_scores.dimension(0);
    const Index query_sequence_length = attention_scores.dimension(2);
    const Index source_sequence_length = attention_scores.dimension(3);

    const Index matrix_size = query_sequence_length * source_sequence_length;

    const Index total_matrices = batch_size * heads_number;

    MatrixMap scores(attention_scores.data(), total_matrices, matrix_size);

    const VectorMap causal_mask_map(const_cast<type*>(causal_mask.data()), matrix_size);

    scores.rowwise() += causal_mask_map.transpose();
}


void MultiHeadAttention::apply_key_padding_mask(const TensorMap3& source_input,
                                                Tensor4& attention_weights) const
{
    const Index batch_size = attention_weights.dimension(0);
    const Index query_sequence_length = attention_weights.dimension(2);
    const Index source_sequence_length = attention_weights.dimension(3);
    const Index embedding_dimension = source_input.dimension(2);

    #pragma omp parallel for
    for(Index b = 0; b < batch_size; ++b)
    {
        for(Index s = 0; s < source_sequence_length; ++s)
        {
            bool is_pad = true;
            for(Index d = 0; d < embedding_dimension; ++d)
            {
                if(abs(source_input(b, s, d)) > 1e-7f)
                {
                    is_pad = false;
                    break;
                }
            }

            if(is_pad)
                for(Index h = 0; h < heads_number; ++h)
                    for(Index q = 0; q < query_sequence_length; ++q)
                        attention_weights(b, h, q, s) = -1e9f;
        }
    }
}


void MultiHeadAttention::print() const
{
    cout << "Multi-head attention layer" << endl
         << "Label: " << label << endl
         << "Type: MultiHeadAttention" << endl
         << "Input shape: " << get_input_shape() << endl
         << "Output shape: " << get_output_shape() << endl
         << "Query sequence length: " << get_query_sequence_length() << endl
         << "Source sequence length: " << get_source_sequence_length() << endl
         << "Embedding dimension: " << get_embedding_dimension() << endl
         << "Heads number: " << get_heads_number() << endl
         << "Head dimension: " << get_head_dimension() << endl
         << "Use causal mask: " << (use_causal_mask ? "True" : "False") << endl;
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

    const string new_label = read_xml_string(multihead_attention_layer_element, "Label");
    const Index new_input_size = read_xml_index(multihead_attention_layer_element, "InputSize");
    const Index new_context_size = read_xml_index(multihead_attention_layer_element, "ContextSize");
    const Index new_depth = read_xml_index(multihead_attention_layer_element, "Depth");
    const Index new_heads_number = read_xml_index(multihead_attention_layer_element, "HeadsNumber");
    const Index new_use_causal_mask = read_xml_bool(multihead_attention_layer_element, "CausalMask");

    set(new_input_size, new_context_size, new_depth, new_heads_number, new_use_causal_mask, new_label);
}


REGISTER(Layer, MultiHeadAttention, "MultiHeadAttention")
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
