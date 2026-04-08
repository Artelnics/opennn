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
    embedding_dimension = new_embedding_dimension;
    input_shape = {new_query_sequence_length, new_embedding_dimension};
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



void MultiHeadAttention::forward_propagate(ForwardPropagation& forward_propagation,
                                           size_t layer,
                                           bool)
{
    const TensorView& query_input = forward_propagation.views[layer][Inputs][0];

    const TensorView& source_input = (forward_propagation.views[layer][Inputs].size() == 1)
                                        ? query_input
                                        : forward_propagation.views[layer][Inputs][1];

    TensorView& query = forward_propagation.views[layer][Query][0];
    TensorView& key = forward_propagation.views[layer][Key][0];
    TensorView& value = forward_propagation.views[layer][Value][0];
    TensorView& attention_weights_view = forward_propagation.views[layer][AttentionWeights][0];
    TensorView& concatenated = forward_propagation.views[layer][ConcatenatedAttentionOutputs][0];
    TensorView& output = forward_propagation.views[layer].back()[0];

    projection(query_input, parameters[QueryWeights], parameters[QueryBiases], query);
    projection(source_input, parameters[KeyWeights], parameters[KeyBiases], key);
    projection(source_input, parameters[ValueWeights], parameters[ValueBiases], value);

    multihead_attention_forward(
        query, key, value,
        attention_weights_view, concatenated, output,
        parameters[ProjectionWeights], parameters[ProjectionBiases],
        source_input,
        forward_propagation.batch_size, heads_number,
        query_sequence_length, source_sequence_length,
        get_embedding_dimension(), get_head_dimension(),
        get_scaling_factor(), use_causal_mask, causal_mask);


#ifdef CUDA
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
                           static_cast<int>(batch_size * query_sequence_length),
                           static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension));

    // Key projection
    linear_projection_cuda(source_input, key_weights_device.data, key_biases_device.data,
                           key_biases_device.get_descriptor(), forward->key.data,
                           forward->key.get_descriptor(),
                           static_cast<int>(batch_size * source_sequence_length),
                           static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension));

    // Value projection
    linear_projection_cuda(source_input, value_weights_device.data, value_biases_device.data,
                           value_biases_device.get_descriptor(), forward->value.data,
                           forward->value.get_descriptor(),
                           static_cast<int>(batch_size * source_sequence_length),
                           static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension));

    // Transpositions

    mha_transpose_qkv_cuda(batch_size * query_sequence_length * embedding_dimension,
                           forward->query.data, forward->query_transposed.data,
                           static_cast<int>(query_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    mha_transpose_qkv_cuda(batch_size * source_sequence_length * embedding_dimension,
                           forward->key.data, forward->key_transposed.data,
                           static_cast<int>(source_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    mha_transpose_qkv_cuda(batch_size * source_sequence_length * embedding_dimension,
                           forward->value.data, forward->value_transposed.data,
                           static_cast<int>(source_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    // Attention scores (Q * K^T)

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length), static_cast<int>(head_dimension),
                              &scaling_factor,
                              forward->key_transposed.data, static_cast<int>(head_dimension), static_cast<int>(source_sequence_length * head_dimension),
                              forward->query_transposed.data, static_cast<int>(head_dimension), static_cast<int>(query_sequence_length * head_dimension),
                              &zero,
                              forward->attention_weights.data, static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length * source_sequence_length),
                              static_cast<int>(batch_size * heads_number));

    // Key padding mask

    mha_key_padding_mask_cuda(
        total_weights,
        source_input,
        forward->attention_weights.data,
        static_cast<int>(heads_number),
        static_cast<int>(query_sequence_length),
        static_cast<int>(source_sequence_length),
        static_cast<int>(embedding_dimension)
        );

    if(use_causal_mask)
    {
        mha_causal_mask_cuda(batch_size * heads_number * query_sequence_length * source_sequence_length,
                             forward->attention_weights.data,
                             static_cast<int>(query_sequence_length), static_cast<int>(source_sequence_length));
    }

    // Softmax

    cudnnSoftmaxForward(get_cudnn_handle(),
                        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &one,
                        forward->attention_weights.get_descriptor(), forward->attention_weights.data,
                        &zero,
                        forward->attention_probabilities.get_descriptor(), forward->attention_probabilities.data);

    // Attention context (Probs * V)

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(head_dimension), static_cast<int>(query_sequence_length), static_cast<int>(source_sequence_length),
                              &one,
                              forward->value_transposed.data, static_cast<int>(head_dimension), static_cast<int>(source_sequence_length * head_dimension),
                              forward->attention_probabilities.data, static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length * source_sequence_length),
                              &zero,
                              forward->attention_outputs_transposed.data, static_cast<int>(head_dimension), static_cast<int>(query_sequence_length * head_dimension),
                              static_cast<int>(batch_size * heads_number));

    // Concatenation and output projection

    mha_transpose_o_cuda(batch_size * query_sequence_length * embedding_dimension,
                         forward->attention_outputs_transposed.data, forward->concatenated_attention_outputs.data,
                         static_cast<int>(query_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    linear_projection_cuda(forward->concatenated_attention_outputs.data,
                           projection_weights_device.data, projection_biases_device.data,
                           projection_biases_device.get_descriptor(), forward->outputs.data,
                           forward->outputs.get_descriptor(),
                           static_cast<int>(batch_size * query_sequence_length),
                           static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension));
#endif
}


void MultiHeadAttention::back_propagate(ForwardPropagation& forward_propagation,
                                        BackPropagation& back_propagation,
                                        size_t layer) const
{
    const TensorView& query_input = forward_propagation.views[layer][Inputs][0];
    const TensorView& source_input = (forward_propagation.views[layer][Inputs].size() == 1)
                                         ? query_input
                                         : forward_propagation.views[layer][Inputs][1];
    const bool self_attention = (forward_propagation.views[layer][Inputs].size() == 1);

    TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradient][0];

    multihead_attention_backward(
        query_input, source_input, output_gradient,
        forward_propagation.views[layer][Query][0],
        forward_propagation.views[layer][Key][0],
        forward_propagation.views[layer][Value][0],
        forward_propagation.views[layer][AttentionWeights][0],
        forward_propagation.views[layer][ConcatenatedAttentionOutputs][0],
        parameters[ProjectionWeights],
        back_propagation.gradient_views[layer][ProjectionWeights],
        back_propagation.gradient_views[layer][ProjectionBiases],
        back_propagation.backward_views[layer][ConcatenatedOutputGradient][0],
        back_propagation.backward_views[layer][AttentionWeightGradient][0],
        back_propagation.backward_views[layer][QueryGradient][0],
        back_propagation.backward_views[layer][KeyGradient][0],
        back_propagation.backward_views[layer][ValueGradient][0],
        back_propagation.gradient_views[layer][QueryWeights],
        back_propagation.gradient_views[layer][QueryBiases],
        back_propagation.gradient_views[layer][KeyWeights],
        back_propagation.gradient_views[layer][KeyBiases],
        back_propagation.gradient_views[layer][ValueWeights],
        back_propagation.gradient_views[layer][ValueBiases],
        back_propagation.backward_views[layer][InputQueryGradient][0],
        parameters[QueryWeights], parameters[KeyWeights], parameters[ValueWeights],
        forward_propagation.batch_size, heads_number,
        query_sequence_length, source_sequence_length,
        get_embedding_dimension(), get_head_dimension(),
        get_scaling_factor(), self_attention);

#ifdef CUDA
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
                static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension), static_cast<int>(batch_size * query_sequence_length),
                &one,
                output_gradients_data, static_cast<int>(embedding_dimension),
                forward->concatenated_attention_outputs.data, static_cast<int>(embedding_dimension),
                &zero,
                back->projection_weight_gradients.data, static_cast<int>(embedding_dimension));

    // Projection bias gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), 1, static_cast<int>(batch_size * query_sequence_length),
                &one,
                output_gradients_data, static_cast<int>(embedding_dimension),
                back->ones.data, static_cast<int>(batch_size * query_sequence_length),
                &zero,
                back->projection_bias_gradients.data, static_cast<int>(embedding_dimension));

    // Concatenated attention output gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), static_cast<int>(batch_size * query_sequence_length), static_cast<int>(embedding_dimension),
                &one,
                projection_weights_device.data, static_cast<int>(embedding_dimension),
                output_gradients_data, static_cast<int>(embedding_dimension),
                &zero,
                back->concatenated_attention_output_gradients.data, static_cast<int>(embedding_dimension));

    // Attention output gradients transposed

    mha_transpose_qkv_cuda(batch_size * query_sequence_length * embedding_dimension,
                           back->concatenated_attention_output_gradients.data, back->attention_output_gradients_transposed.data,
                           static_cast<int>(query_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    // Value gradients transposed

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              static_cast<int>(head_dimension), static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length),
                              &one,
                              back->attention_output_gradients_transposed.data, static_cast<int>(head_dimension), static_cast<int>(query_sequence_length * head_dimension),
                              forward->attention_probabilities.data, static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length * source_sequence_length),
                              &zero,
                              back->value_gradients_transposed.data, static_cast<int>(head_dimension), static_cast<int>(source_sequence_length * head_dimension),
                              static_cast<int>(batch_size * heads_number));

    // Attention weight gradients

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length), static_cast<int>(head_dimension),
                              &one,
                              forward->value_transposed.data, static_cast<int>(head_dimension), static_cast<int>(source_sequence_length * head_dimension),
                              back->attention_output_gradients_transposed.data, static_cast<int>(head_dimension), static_cast<int>(query_sequence_length * head_dimension),
                              &zero,
                              back->attention_weight_gradients.data, static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length * source_sequence_length),
                              static_cast<int>(batch_size * heads_number));

    // Softmax gradients

    cudnnSoftmaxBackward(get_cudnn_handle(),
                         CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                         &one,
                         forward->attention_probabilities.get_descriptor(), forward->attention_probabilities.data,
                         back->attention_weight_gradients.get_descriptor(), back->attention_weight_gradients.data,
                         &zero,
                         back->softmax_gradients.get_descriptor(), back->softmax_gradients.data);

    // Query and key gradients transposed

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(head_dimension), static_cast<int>(query_sequence_length), static_cast<int>(source_sequence_length),
                              &scaling_factor,
                              forward->key_transposed.data, static_cast<int>(head_dimension), static_cast<int>(source_sequence_length * head_dimension),
                              back->softmax_gradients.data, static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length * source_sequence_length),
                              &zero,
                              back->query_gradients_transposed.data, static_cast<int>(head_dimension), static_cast<int>(query_sequence_length * head_dimension),
                              static_cast<int>(batch_size * heads_number));

    cublasSgemmStridedBatched(get_cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              static_cast<int>(head_dimension), static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length),
                              &scaling_factor,
                              forward->query_transposed.data, static_cast<int>(head_dimension), static_cast<int>(query_sequence_length * head_dimension),
                              back->softmax_gradients.data, static_cast<int>(source_sequence_length), static_cast<int>(query_sequence_length * source_sequence_length),
                              &zero,
                              back->key_gradients_transposed.data, static_cast<int>(head_dimension), static_cast<int>(source_sequence_length * head_dimension),
                              static_cast<int>(batch_size * heads_number));

    // Des-transposition to flat

    mha_transpose_o_cuda(batch_size * query_sequence_length * embedding_dimension,
                         back->query_gradients_transposed.data, back->query_gradients.data,
                         static_cast<int>(query_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    mha_transpose_o_cuda(batch_size * source_sequence_length * embedding_dimension,
                         back->key_gradients_transposed.data, back->key_gradients.data,
                         static_cast<int>(source_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    mha_transpose_o_cuda(batch_size * source_sequence_length * embedding_dimension,
                         back->value_gradients_transposed.data, back->value_gradients.data,
                         static_cast<int>(source_sequence_length), static_cast<int>(heads_number), static_cast<int>(head_dimension));

    // Query weight, bias and input gradients

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension), static_cast<int>(batch_size * query_sequence_length),
                &one,
                back->query_gradients.data, static_cast<int>(embedding_dimension),
                query_input, static_cast<int>(embedding_dimension),
                &zero,
                back->query_weight_gradients.data, static_cast<int>(embedding_dimension));

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), 1, static_cast<int>(batch_size * query_sequence_length),
                &one,
                back->query_gradients.data, static_cast<int>(embedding_dimension),
                back->ones.data, static_cast<int>(batch_size * query_sequence_length),
                &zero,
                back->query_bias_gradients.data, static_cast<int>(embedding_dimension));

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), static_cast<int>(batch_size * query_sequence_length), static_cast<int>(embedding_dimension),
                &one,
                query_weights_device.data, static_cast<int>(embedding_dimension),
                back->query_gradients.data, static_cast<int>(embedding_dimension),
                &zero,
                back->query_input_gradients.data, static_cast<int>(embedding_dimension));

    // Key weight, bias and source projection gradients (Temp)

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension), static_cast<int>(batch_size * source_sequence_length),
                &one,
                back->key_gradients.data, static_cast<int>(embedding_dimension),
                source_input, static_cast<int>(embedding_dimension),
                &zero,
                back->key_weight_gradients.data, static_cast<int>(embedding_dimension));

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), 1, static_cast<int>(batch_size * source_sequence_length),
                &one,
                back->key_gradients.data, static_cast<int>(embedding_dimension),
                back->ones.data, static_cast<int>(batch_size * source_sequence_length),
                &zero,
                back->key_bias_gradients.data, static_cast<int>(embedding_dimension));

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), static_cast<int>(batch_size * source_sequence_length), static_cast<int>(embedding_dimension),
                &one,
                key_weights_device.data, static_cast<int>(embedding_dimension),
                back->key_gradients.data, static_cast<int>(embedding_dimension),
                &zero,
                back->source_input_gradients.data, static_cast<int>(embedding_dimension));

    // Value weight, bias and source input gradients accumulation

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(embedding_dimension), static_cast<int>(embedding_dimension), static_cast<int>(batch_size * source_sequence_length),
                &one,
                back->value_gradients.data, static_cast<int>(embedding_dimension),
                source_input, static_cast<int>(embedding_dimension),
                &zero,
                back->value_weight_gradients.data, static_cast<int>(embedding_dimension));

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), 1, static_cast<int>(batch_size * source_sequence_length),
                &one,
                back->value_gradients.data, static_cast<int>(embedding_dimension),
                back->ones.data, static_cast<int>(batch_size * source_sequence_length),
                &zero,
                back->value_bias_gradients.data, static_cast<int>(embedding_dimension));

    cublasSgemm(get_cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(embedding_dimension), static_cast<int>(batch_size * source_sequence_length), static_cast<int>(embedding_dimension),
                &one,
                value_weights_device.data, static_cast<int>(embedding_dimension),
                back->value_gradients.data, static_cast<int>(embedding_dimension),
                &one,
                back->source_input_gradients.data, static_cast<int>(embedding_dimension));

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

    const XMLElement* multihead_attention_layer_element = get_xml_root(document, "MultiHeadAttention");

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
// Licensed under the GNU Lesser General Public License v2.1 or later.
