//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I T  M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "strings_utilities.h"
#include "tensors.h"
#include "vit_multihead_attention_layer.h"

namespace opennn
{

    VitMultiheadAttentionLayer::VitMultiheadAttentionLayer(const Index& new_input_size,
        const Index& new_depth,
        const Index& new_heads_number,
        const string& new_name) : Layer()
    {
        set(new_input_size, new_depth, new_heads_number, new_name);

        layer_type = Type::VitMultiheadAttention;

        name = new_name;
    }


    Index VitMultiheadAttentionLayer::get_input_size() const
    {
        return input_size;
    }


    Index VitMultiheadAttentionLayer::get_depth() const
    {
        return depth;
    }


    Index VitMultiheadAttentionLayer::get_heads_number() const
    {
        return heads_number;
    }


    Index VitMultiheadAttentionLayer::get_weights_depth() const
    {
        return hidden_depth;
    }


    dimensions VitMultiheadAttentionLayer::get_input_dimensions() const
    {// @todo
        return { input_size, depth };
    }


    dimensions VitMultiheadAttentionLayer::get_output_dimensions() const
    {
        return { input_size, depth };
    }


    Index VitMultiheadAttentionLayer::get_parameters_number() const
    {
        return query_weights.size() + query_biases.size()
            + key_weights.size() + key_biases.size()
            + value_weights.size() + value_biases.size()
            + projection_weights.size() + projection_biases.size();
    }


    Tensor<type, 1> VitMultiheadAttentionLayer::get_parameters() const
    {
        Tensor<type, 1> parameters(get_parameters_number());

        Index parameters_index = 0;

        memcpy(parameters.data(), query_weights.data(), query_weights.size() * sizeof(type));

        parameters_index += query_weights.size();

        memcpy(parameters.data() + parameters_index, query_biases.data(), query_biases.size() * sizeof(type));

        parameters_index += query_biases.size();

        memcpy(parameters.data() + parameters_index, key_weights.data(), key_weights.size() * sizeof(type));

        parameters_index += key_weights.size();

        memcpy(parameters.data() + parameters_index, key_biases.data(), key_biases.size() * sizeof(type));

        parameters_index += key_biases.size();

        memcpy(parameters.data() + parameters_index, value_weights.data(), value_weights.size() * sizeof(type));

        parameters_index += value_weights.size();

        memcpy(parameters.data() + parameters_index, value_biases.data(), value_biases.size() * sizeof(type));

        parameters_index += value_biases.size();

        memcpy(parameters.data() + parameters_index, projection_weights.data(), projection_weights.size() * sizeof(type));

        parameters_index += projection_weights.size();

        memcpy(parameters.data() + parameters_index, projection_biases.data(), projection_biases.size() * sizeof(type));

        return parameters;
    }


    void VitMultiheadAttentionLayer::set(const Index& new_input_size,
        const Index& new_depth,
        const Index& new_heads_number,
        const string& new_name)
    {
        input_size = new_input_size;

        depth = new_depth;

        heads_number = new_heads_number;
        
        heads_number == 0
            ? hidden_depth = 0
            : hidden_depth = Index(depth / heads_number);
        
        scaling_factor = (hidden_depth == 0)
            ? 0.25
            : type(1) / type(sqrt(hidden_depth));

        name = new_name;

        layer_type = Type::VitMultiheadAttention;

        dropout_rate = 0;

        query_weights.resize(depth, hidden_depth, heads_number);
        query_biases.resize(hidden_depth, heads_number);

        key_weights.resize(depth, hidden_depth, heads_number);
        key_biases.resize(hidden_depth, heads_number);

        value_weights.resize(depth, hidden_depth, heads_number);
        value_biases.resize(hidden_depth, heads_number);

        projection_weights.resize(depth, depth);
        projection_biases.resize(depth);

        set_parameters_glorot();

    }


    void VitMultiheadAttentionLayer::set_input_size(const Index& new_input_size) {
        input_size = new_input_size;
    }


    void VitMultiheadAttentionLayer::set_depth(const Index& new_depth) {
        depth = new_depth;
    }


    void VitMultiheadAttentionLayer::set_heads_number(const Index& new_heads_number) {
        heads_number = new_heads_number;
    }


    void VitMultiheadAttentionLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
    {
        const type* new_parameters_data = new_parameters.data();

        type* query_weights_data = query_weights.data();
        type* query_biases_data = query_biases.data();

        type* key_weights_data = key_weights.data();
        type* key_biases_data = key_biases.data();

        type* value_weights_data = value_weights.data();
        type* value_biases_data = value_biases.data();

        type* projection_weights_data = projection_weights.data();
        type* projection_biases_data = projection_biases.data();

        Index parameters_index = index;

        memcpy(query_weights_data, new_parameters_data + parameters_index, query_weights.size() * sizeof(type));

        parameters_index += query_weights.size();

        memcpy(query_biases_data, new_parameters_data + parameters_index, query_biases.size() * sizeof(type));

        parameters_index += query_biases.size();

        memcpy(key_weights_data, new_parameters_data + parameters_index, key_weights.size() * sizeof(type));

        parameters_index += key_weights.size();

        memcpy(key_biases_data, new_parameters_data + parameters_index, key_biases.size() * sizeof(type));

        parameters_index += key_biases.size();

        memcpy(value_weights_data, new_parameters_data + parameters_index, value_weights.size() * sizeof(type));

        parameters_index += value_weights.size();

        memcpy(value_biases_data, new_parameters_data + parameters_index, value_biases.size() * sizeof(type));

        parameters_index += value_biases.size();

        memcpy(projection_weights_data, new_parameters_data + parameters_index, projection_weights.size() * sizeof(type));

        parameters_index += projection_weights.size();

        memcpy(projection_biases_data, new_parameters_data + parameters_index, projection_biases.size() * sizeof(type));
    }


    void VitMultiheadAttentionLayer::set_parameters_random()
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


    void VitMultiheadAttentionLayer::set_parameters_glorot()
    {
        query_biases.setZero();
        key_biases.setZero();
        value_biases.setZero();
        projection_biases.setZero();

        const type limit = sqrt(6 / type(depth + hidden_depth));

        const type minimum = -limit;
        const type maximum = limit;

        set_random(query_weights, minimum, maximum);
        set_random(key_weights, minimum, maximum);
        set_random(value_weights, minimum, maximum);

        
        const type limit_projection = sqrt(6 / type(depth + depth));

        const type minimum_projection = -limit_projection;
        const type maximum_projection = limit_projection;

        set_random(projection_weights, minimum_projection, maximum_projection);
        
    }


    void VitMultiheadAttentionLayer::set_parameters_constant(const type& value)
    {
        query_weights.setConstant(value);
        query_biases.setZero();

        key_weights.setConstant(value);
        key_biases.setZero();

        value_weights.setConstant(value);
        value_biases.setZero();

        projection_weights.setConstant(value);
        projection_biases.setZero();
    }


    void VitMultiheadAttentionLayer::set_dropout_rate(const type& new_dropout_rate)
    {
        dropout_rate = new_dropout_rate;
    }


    void VitMultiheadAttentionLayer::calculate_transformation(const Tensor<type, 3>& input,
        Tensor<type, 4>& transformed_input,
        const Tensor<type, 3>& weights,
        const Tensor<type, 2>& biases,
        Tensor<type, 2>& sample_matrix) const
    {
        const Index batch_size = input.dimension(0);
        const Index variables_number = input.dimension(1);

        type* transformed_input_data = transformed_input.data();

        for (Index head_index = 0; head_index < heads_number; head_index++)
        {
            const TensorMap<Tensor<type, 2>> head_weights((type*)weights.data() + head_index * depth * hidden_depth,
                depth,
                hidden_depth);

            const TensorMap<Tensor<type, 1>> head_biases((type*)biases.data() + head_index * hidden_depth,
                hidden_depth);


            type* head_transformed_input_data = transformed_input_data + head_index * batch_size * variables_number * hidden_depth;

            for (Index sample_index = 0; sample_index < batch_size; sample_index++)
            {
                sample_matrix = input.chip(sample_index, 0);

                TensorMap<Tensor<type, 2>> sample_transformed_input(head_transformed_input_data + sample_index * variables_number * hidden_depth,
                    variables_number,
                    hidden_depth);

                sample_transformed_input.device(*thread_pool_device)
                    = sample_matrix.contract(head_weights, A_B);

                Eigen::Tensor<type, 2> reshaped_vector = head_biases.reshape(Eigen::array<Index, 2>{1, head_biases.dimension(0)});
                sample_transformed_input.device(*thread_pool_device) = sample_transformed_input + reshaped_vector.broadcast(Eigen::array<Index, 2>{sample_transformed_input.dimension(0), 1});
            }
        }
    }


    void VitMultiheadAttentionLayer::calculate_output_projection(const Tensor<type, 4>& attention_outputs,
        Tensor<type, 3>& concatenated_outputs,
        Tensor<type, 3>& outputs) const
    {
        concatenated_outputs = concatenate_heads(attention_outputs);
        outputs.device(*thread_pool_device) = concatenated_outputs.contract(projection_weights, A2_B);
        sum_matrices(thread_pool_device.get(), projection_biases, outputs);
    }


    Tensor<type, 3> VitMultiheadAttentionLayer::concatenate_heads(const Tensor<type, 4>& heads_tensor) const
    {
        const Index batch = heads_tensor.dimension(2);

        Eigen::array<Index, 4> shuffle_order = { 2, 0, 3, 1 };
        Tensor<type, 4> shuffled = heads_tensor.shuffle(shuffle_order);

        Eigen::array<Index, 3> new_shape = { batch, input_size, depth };
        Tensor<type, 3> concatenated = shuffled.reshape(new_shape);

        return concatenated;
    }



    void VitMultiheadAttentionLayer::compute_attention_scores(const Tensor<type, 4>& query,
        const Tensor<type, 4>& key,
        Tensor<type, 4>& attention_scores,
        Tensor<type, 4>& attention_weights) const
    {
        batch_matrix_multiplication(thread_pool_device.get(), query, key, attention_scores, A_BT);

        attention_scores.device(*thread_pool_device) = attention_scores * scaling_factor;

        softmax(attention_scores);

        attention_weights = attention_scores;
    }


    void VitMultiheadAttentionLayer::compute_attention_outputs(const Tensor<type, 4>& value,
        const Tensor<type, 4>& attention_weights,
        Tensor<type, 4>& attention_outputs) const
    {
        batch_matrix_multiplication(thread_pool_device.get(), attention_weights, value, attention_outputs, A_B);
    }


    void VitMultiheadAttentionLayer::dropout(Tensor<type, 4>& attention_weights, Tensor<type, 4>& dropout_mask)
    {
        scaling_factor = type(1) / (type(1) - dropout_rate);

#pragma omp parallel for
        for (Index i = 0; i < attention_weights.size(); i++)
        {
            if (get_random_type(type(0), type(1)) < dropout_rate)
            {
                attention_weights(i) = 0;
                dropout_mask(i) = 0;
            }
            else
            {
                attention_weights(i) *= scaling_factor;
                dropout_mask(i) = scaling_factor;
            }
        }
    }


    void VitMultiheadAttentionLayer::softmax_derivatives_times_tensor(Tensor<type, 4>& A,
        Tensor<type, 4>& dL_dA,
        Tensor<type, 4>& dL_dZ) const
    {
        const Index T = A.dimension(0);
        const Index C = A.dimension(1);
        const Index B = A.dimension(2);
        const Index H = A.dimension(3);

        const Eigen::array<Index, 1> sum_axis = { 0 };

        Tensor<type, 4> A_mul_dL_dA = A * dL_dA;

        Tensor<type, 4> sum_tensor = A_mul_dL_dA.sum(sum_axis)
            .reshape(Eigen::array<Index, 4>{1, C, B, H})
            .broadcast(Eigen::array<Index, 4>{T, 1, 1, 1});

        dL_dZ.device(*thread_pool_device) = (dL_dA - sum_tensor) * A;
    }



    void VitMultiheadAttentionLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
        const bool& is_training)
    {
        VitMultiheadAttentionLayerForwardPropagation* vit_multihead_attention_layer_forward_propagation =
            static_cast<VitMultiheadAttentionLayerForwardPropagation*>(layer_forward_propagation.get());

        const TensorMap<Tensor<type, 3>> input = tensor_map_3(input_pairs[0]);
        
        Tensor<type, 4>& query = vit_multihead_attention_layer_forward_propagation->query;
        Tensor<type, 4>& key = vit_multihead_attention_layer_forward_propagation->key;
        Tensor<type, 4>& value = vit_multihead_attention_layer_forward_propagation->value;

        Tensor<type, 2>& sample_matrix = vit_multihead_attention_layer_forward_propagation->sample_matrix;

        Tensor<type, 4>& attention_scores = vit_multihead_attention_layer_forward_propagation->attention_scores;

        Tensor<type, 4>& attention_weights = vit_multihead_attention_layer_forward_propagation->attention_weights;

        Tensor<type, 4>& attention_outputs = vit_multihead_attention_layer_forward_propagation->attention_outputs;

        Tensor<type, 3>& concatenated_outputs = vit_multihead_attention_layer_forward_propagation->concatenated_outputs;

        Tensor<type, 4>& dropout_mask = vit_multihead_attention_layer_forward_propagation->dropout_mask;
        dropout_mask.setConstant(type(1));

        Tensor<type, 3>& outputs = vit_multihead_attention_layer_forward_propagation->outputs;

        calculate_transformation(input, query, query_weights, query_biases, sample_matrix);

        calculate_transformation(input, key, key_weights, key_biases, sample_matrix);

        calculate_transformation(input, value, value_weights, value_biases, sample_matrix);
       
        compute_attention_scores(query,
            key,
            attention_scores,
            attention_weights);
     
        if (is_training && dropout_rate > type(0)) {
            //cout << "attention dropout" << endl;
            dropout(attention_weights,
                dropout_mask);
        }

        compute_attention_outputs(value,
            attention_weights,
            attention_outputs);

        calculate_output_projection(attention_outputs,
            concatenated_outputs,
            outputs);
        
        //cout << "Multihead Self Attention layer outputs dimensions: " << outputs.dimensions() << endl;
        //cout << "Multihead Self Attention layer outputs:" << endl;
        //cout << outputs << endl;
    }


    
    void VitMultiheadAttentionLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
        const vector<pair<type*, dimensions>>& delta_pairs,
        unique_ptr<LayerForwardPropagation>& forward_propagation,
        unique_ptr<LayerBackPropagation>& back_propagation) const
    {
        TensorMap<Tensor<type, 3>> input = tensor_map_3(input_pairs[0]);

        const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);
        
        const Index batch_size = input_pairs[0].second[0];

        VitMultiheadAttentionLayerForwardPropagation* vit_multihead_attention_layer_forward_propagation =
            static_cast<VitMultiheadAttentionLayerForwardPropagation*>(forward_propagation.get());

        Tensor<type, 3>& concatenated_outputs = vit_multihead_attention_layer_forward_propagation->concatenated_outputs;

        Tensor<type, 4>& value = vit_multihead_attention_layer_forward_propagation->value;

        Tensor<type, 4>& attention_weights = vit_multihead_attention_layer_forward_propagation->attention_weights;
        Tensor<type, 4>& dropout_mask = vit_multihead_attention_layer_forward_propagation->dropout_mask;
        Tensor<type, 4>& attention_scores = vit_multihead_attention_layer_forward_propagation->attention_scores;

        Tensor<type, 4>& key = vit_multihead_attention_layer_forward_propagation->key;
        Tensor<type, 4>& query = vit_multihead_attention_layer_forward_propagation->query;
        
        VitMultiheadAttentionLayerBackPropagation* vit_multihead_attention_layer_back_propagation =
            static_cast<VitMultiheadAttentionLayerBackPropagation*>(back_propagation.get());

        Tensor<type, 2>& projection_weights_derivatives = vit_multihead_attention_layer_back_propagation->projection_weights_derivatives;
        Tensor<type, 1>& projection_biases_derivatives = vit_multihead_attention_layer_back_propagation->projection_biases_derivatives;
        Tensor<type, 3>& concatenate_derivatives = vit_multihead_attention_layer_back_propagation->concatenate_derivatives;
        Tensor<type, 4>& attention_output_derivatives = vit_multihead_attention_layer_back_propagation->attention_output_derivatives;
        Tensor<type, 4>& dropout_derivatives = vit_multihead_attention_layer_back_propagation->dropout_derivatives;

        Tensor<type, 4>& value_derivatives = vit_multihead_attention_layer_back_propagation->value_derivatives;

        Tensor<type, 4>& attention_weights_derivatives = vit_multihead_attention_layer_back_propagation->attention_weights_derivatives;
        Tensor<type, 4>& attention_scores_derivatives = vit_multihead_attention_layer_back_propagation->attention_scores_derivatives;

        Tensor<type, 4>& key_derivatives = vit_multihead_attention_layer_back_propagation->key_derivatives;
        Tensor<type, 4>& query_derivatives = vit_multihead_attention_layer_back_propagation->query_derivatives;

        Tensor<type, 3>& query_weights_derivatives = vit_multihead_attention_layer_back_propagation->query_weights_derivatives;
        query_weights_derivatives.setZero();
        Tensor<type, 2>& query_biases_derivatives = vit_multihead_attention_layer_back_propagation->query_biases_derivatives;
        query_biases_derivatives.setZero();

        Tensor<type, 3>& key_weights_derivatives = vit_multihead_attention_layer_back_propagation->key_weights_derivatives;
        key_weights_derivatives.setZero();
        Tensor<type, 2>& key_biases_derivatives = vit_multihead_attention_layer_back_propagation->key_biases_derivatives;
        key_biases_derivatives.setZero();

        Tensor<type, 3>& value_weights_derivatives = vit_multihead_attention_layer_back_propagation->value_weights_derivatives;
        value_weights_derivatives.setZero();
        Tensor<type, 2>& value_biases_derivatives = vit_multihead_attention_layer_back_propagation->value_biases_derivatives;
        value_biases_derivatives.setZero();

        Tensor<type, 3>& input_derivatives = vit_multihead_attention_layer_back_propagation->input_derivatives;
        input_derivatives.setZero();
        

        projection_weights_derivatives.device(*thread_pool_device) = concatenated_outputs.contract(deltas, projection_weights_derivatives_contraction_indices);

        projection_biases_derivatives.device(*thread_pool_device) = deltas.sum(projection_biases_derivatives_sum_indices);

        concatenate_derivatives.device(*thread_pool_device) = deltas.contract(projection_weights, A2_B1);

        Eigen::array<Index, 4> reverse_shape = { batch_size, input_size, heads_number, hidden_depth };
        Eigen::array<Index, 4> shuffle_order = { 1, 3, 0, 2 };
        attention_output_derivatives = concatenate_derivatives.reshape(reverse_shape).shuffle(shuffle_order); // [input_size, hidden_depth, batch_size, heads]

        batch_matrix_multiplication(thread_pool_device.get(), attention_output_derivatives, value, dropout_derivatives, A_BT); // contract last axis of Z and first of V

        batch_matrix_multiplication(thread_pool_device.get(), attention_weights, attention_output_derivatives, value_derivatives, AT_B);
        
        attention_weights_derivatives.device(*thread_pool_device) = dropout_derivatives * dropout_mask;
        
        softmax_derivatives_times_tensor(attention_scores, attention_weights_derivatives, attention_scores_derivatives);

        batch_matrix_multiplication(thread_pool_device.get(),
            attention_scores_derivatives,
            key,
            query_derivatives,
            A_B);

        batch_matrix_multiplication(thread_pool_device.get(),
            attention_scores_derivatives,
            query,
            key_derivatives,
            AT_B);

        query_derivatives.device(*thread_pool_device) = query_derivatives * scaling_factor;
        key_derivatives.device(*thread_pool_device) = key_derivatives * scaling_factor;


        for (Index head = 0; head < heads_number; ++head)
        {
            for (Index sample = 0; sample < batch_size; ++sample)
            {
                TensorMap<Tensor< type, 2>> X_sample(input.data() + sample * input_size * depth,
                    input_size, depth);
                
                TensorMap<Tensor< type, 2>> dQ_sample(query_derivatives.data() + sample * input_size * hidden_depth * heads_number
                    + head * input_size * hidden_depth,
                    input_size, hidden_depth);

                TensorMap<Tensor< type, 2>> dK_sample(key_derivatives.data() + sample * input_size * hidden_depth * heads_number
                    + head * input_size * hidden_depth,
                    input_size, hidden_depth);

                TensorMap<Tensor< type, 2>> dV_sample(value_derivatives.data() + sample * input_size * hidden_depth * heads_number
                    + head * input_size * hidden_depth,
                    input_size, hidden_depth);

                query_weights_derivatives.chip(head, 2).device(*thread_pool_device) += X_sample.contract(dQ_sample, AT_B);

                key_weights_derivatives.chip(head, 2).device(*thread_pool_device) += X_sample.contract(dK_sample, AT_B);

                value_weights_derivatives.chip(head, 2).device(*thread_pool_device) += X_sample.contract(dV_sample, AT_B);

                const TensorMap<Tensor<const type, 2>> Wq_head(query_weights.data() + head * depth * hidden_depth,
                    depth, hidden_depth);

                const TensorMap<Tensor<const type, 2>> Wk_head(key_weights.data() + head * depth * hidden_depth,
                    depth, hidden_depth);

                const TensorMap<Tensor<const type, 2>> Wv_head(value_weights.data() + head * depth * hidden_depth,
                    depth, hidden_depth);

                TensorMap<Tensor<type, 2>> dX_sample(input_derivatives.data() + sample * input_size * depth,
                    input_size, depth);

                dX_sample.device(*thread_pool_device) += dQ_sample.contract(Wq_head, A_BT)
                    + dK_sample.contract(Wk_head, A_BT)
                    + dV_sample.contract(Wv_head, A_BT);
                
            }
        }


        query_biases_derivatives.device(*thread_pool_device) =
            query_derivatives.sum(biases_derivatives_sum_indices);

        key_biases_derivatives.device(*thread_pool_device) =
            key_derivatives.sum(biases_derivatives_sum_indices);

        value_biases_derivatives.device(*thread_pool_device) =
            value_derivatives.sum(biases_derivatives_sum_indices);

    }


    void VitMultiheadAttentionLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
        const Index& index,
        Tensor<type, 1>& gradient) const
    {
        VitMultiheadAttentionLayerBackPropagation* vit_multihead_attention_layer_back_propagation =
            static_cast<VitMultiheadAttentionLayerBackPropagation*>(back_propagation.get());

        const Tensor<type, 3>& query_weights_derivatives = vit_multihead_attention_layer_back_propagation->query_weights_derivatives;
        const Tensor<type, 2>& query_biases_derivatives = vit_multihead_attention_layer_back_propagation->query_biases_derivatives;

        const Tensor<type, 3>& key_weights_derivatives = vit_multihead_attention_layer_back_propagation->key_weights_derivatives;
        const Tensor<type, 2>& key_biases_derivatives = vit_multihead_attention_layer_back_propagation->key_biases_derivatives;

        const Tensor<type, 3>& value_weights_derivatives = vit_multihead_attention_layer_back_propagation->value_weights_derivatives;
        const Tensor<type, 2>& value_biases_derivatives = vit_multihead_attention_layer_back_propagation->value_biases_derivatives;

        const Tensor<type, 2>& projection_weights_derivatives = vit_multihead_attention_layer_back_propagation->projection_weights_derivatives;
        const Tensor<type, 1>& projection_biases_derivatives = vit_multihead_attention_layer_back_propagation->projection_biases_derivatives;

        type* gradient_data = gradient.data();

        Index gradient_index = index;

        memcpy(gradient_data + gradient_index, query_weights_derivatives.data(), query_weights_derivatives.size() * sizeof(type));

        gradient_index += query_weights_derivatives.size();

        memcpy(gradient_data + gradient_index, query_biases_derivatives.data(), query_biases_derivatives.size() * sizeof(type));

        gradient_index += query_biases_derivatives.size();

        memcpy(gradient_data + gradient_index, key_weights_derivatives.data(), key_weights_derivatives.size() * sizeof(type));

        gradient_index += key_weights_derivatives.size();

        memcpy(gradient_data + gradient_index, key_biases_derivatives.data(), key_biases_derivatives.size() * sizeof(type));

        gradient_index += key_biases_derivatives.size();

        memcpy(gradient_data + gradient_index, value_weights_derivatives.data(), value_weights_derivatives.size() * sizeof(type));

        gradient_index += value_weights_derivatives.size();

        memcpy(gradient_data + gradient_index, value_biases_derivatives.data(), value_biases_derivatives.size() * sizeof(type));

        gradient_index += value_biases_derivatives.size();

        memcpy(gradient_data + gradient_index, projection_weights_derivatives.data(), projection_weights_derivatives.size() * sizeof(type));

        gradient_index += projection_weights_derivatives.size();

        memcpy(gradient_data + gradient_index, projection_biases_derivatives.data(), projection_biases_derivatives.size() * sizeof(type));
    }

    
    void VitMultiheadAttentionLayer::from_XML(const XMLDocument& document)
    {
        
        const XMLElement* vit_multihead_attention_layer_element = document.FirstChildElement("VitMultiheadAttention");

        if (!vit_multihead_attention_layer_element)
            throw runtime_error("VitMultiheadAttention element is nullptr.\n");

        set_name(read_xml_string(vit_multihead_attention_layer_element, "Name"));
        set_input_size(read_xml_index(vit_multihead_attention_layer_element, "InputSize"));
        set_depth(read_xml_index(vit_multihead_attention_layer_element, "Depth"));
        set_heads_number(read_xml_index(vit_multihead_attention_layer_element, "HeadsNumber"));
        set_parameters(to_type_vector(read_xml_string(vit_multihead_attention_layer_element, "Parameters"), " "));
        
    }


    void VitMultiheadAttentionLayer::to_XML(XMLPrinter& printer) const
    {
        
        printer.OpenElement("VitMultiheadAttention");

        add_xml_element(printer, "Name", name);
        add_xml_element(printer, "InputSize", to_string(get_input_size()));
        add_xml_element(printer, "Depth", to_string(get_depth()));
        add_xml_element(printer, "HeadsNumber", to_string(get_heads_number()));
        add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

        printer.CloseElement();
        
    }
    

    VitMultiheadAttentionLayerForwardPropagation::VitMultiheadAttentionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    pair<type*, dimensions> VitMultiheadAttentionLayerForwardPropagation::get_outputs_pair() const
    {
        VitMultiheadAttentionLayer* vit_multihead_attention_layer = static_cast<VitMultiheadAttentionLayer*>(layer);

        const Index input_size = vit_multihead_attention_layer->get_input_size();

        const Index depth = vit_multihead_attention_layer->get_depth();

        return { (type*)outputs.data(), {{ batch_samples_number, input_size, depth }} };
    }


    void VitMultiheadAttentionLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
    {
        layer = new_layer;

        VitMultiheadAttentionLayer* vit_multihead_attention_layer = static_cast<VitMultiheadAttentionLayer*>(layer);

        batch_samples_number = new_batch_samples_number;

        const Index input_size = vit_multihead_attention_layer->get_input_size();

        const Index depth = vit_multihead_attention_layer->get_depth();

        const Index heads_number = vit_multihead_attention_layer->get_heads_number();

        const Index hidden_depth = vit_multihead_attention_layer->get_weights_depth();

        outputs.resize(batch_samples_number, input_size, depth);

        query.resize(input_size, hidden_depth, batch_samples_number, heads_number);
        key.resize(input_size, hidden_depth, batch_samples_number, heads_number);
        value.resize(input_size, hidden_depth, batch_samples_number, heads_number);

        sample_matrix.resize(input_size, hidden_depth);

        attention_scores.resize(input_size, input_size, batch_samples_number, heads_number);
        attention_weights.resize(input_size, input_size, batch_samples_number, heads_number);
        attention_outputs.resize(input_size, hidden_depth, batch_samples_number, heads_number);

        dropout_mask.resize(input_size, input_size, batch_samples_number, heads_number);

    }


    void VitMultiheadAttentionLayerForwardPropagation::print() const
    {
        cout << "Attention scores:" << endl
            << attention_scores.dimensions() << endl
            << "Outputs dimensions:" << endl;
        //cout << output_dimensions << endl;
        cout << "Outputs:" << endl;
        //cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
        cout << "Attention scores:" << endl;
        cout << attention_scores << endl;
    }


    void VitMultiheadAttentionLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
    {
        layer = new_layer;

        VitMultiheadAttentionLayer* vit_multihead_attention_layer = static_cast<VitMultiheadAttentionLayer*>(layer);

        batch_samples_number = new_batch_samples_number;

        const Index input_size = vit_multihead_attention_layer->get_input_size();
        const Index depth = vit_multihead_attention_layer->get_depth();
        const Index heads_number = vit_multihead_attention_layer->get_heads_number();
        const Index hidden_depth = vit_multihead_attention_layer->get_weights_depth();

        attention_scores_derivatives.resize(input_size, input_size, batch_samples_number, heads_number);
        attention_weights_derivatives.resize(input_size, input_size, batch_samples_number, heads_number);
        attention_output_derivatives.resize(input_size, hidden_depth, batch_samples_number, heads_number);

        sample_deltas.resize(input_size, depth);

        query_derivatives.resize(input_size, hidden_depth, batch_samples_number, heads_number);
        key_derivatives.resize(input_size, hidden_depth, batch_samples_number, heads_number);
        value_derivatives.resize(input_size, hidden_depth, batch_samples_number, heads_number);

        query_weights_derivatives.resize(depth, hidden_depth, heads_number);
        key_weights_derivatives.resize(depth, hidden_depth, heads_number);
        value_weights_derivatives.resize(depth, hidden_depth, heads_number);

        projection_weights_derivatives.resize(depth, depth);

        query_biases_derivatives.resize(hidden_depth, heads_number);
        key_biases_derivatives.resize(hidden_depth, heads_number);
        value_biases_derivatives.resize(hidden_depth, heads_number);

        projection_biases_derivatives.resize(depth);

        concatenate_derivatives.resize(batch_samples_number, input_size, depth);

        dropout_derivatives.resize(input_size, input_size, batch_samples_number, heads_number);

        aux_rows.resize(input_size);

        input_derivatives.resize(batch_samples_number, input_size, depth);
    }


    void VitMultiheadAttentionLayerBackPropagation::print() const
    {
    }


    VitMultiheadAttentionLayerBackPropagation::VitMultiheadAttentionLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    vector<pair<type*, dimensions>> VitMultiheadAttentionLayerBackPropagation::get_input_derivative_pairs() const
    {
        VitMultiheadAttentionLayer* vit_multihead_attention_layer = static_cast<VitMultiheadAttentionLayer*>(layer);

        const Index input_size = vit_multihead_attention_layer->get_input_size();
        const Index depth = vit_multihead_attention_layer->get_depth();

        return
        { {(type*)(input_derivatives.data()), {batch_samples_number, input_size, depth}} };
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
