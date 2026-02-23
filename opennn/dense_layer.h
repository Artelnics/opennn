//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "random_utilities.h"

namespace opennn
{

template<int Rank> class Dense;

template<int Rank>
struct DenseForwardPropagation final : LayerForwardPropagation
{
    DenseForwardPropagation(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    ~DenseForwardPropagation() override = default;

    void initialize() override
    {
        const auto* dense_layer = static_cast<const Dense<Rank>*>(layer);
        const Shape output_shape = dense_layer->get_output_shape();

        Shape full_output_dims = {batch_size};
        full_output_dims.insert(full_output_dims.end(), output_shape.begin(), output_shape.end());

        outputs.shape = full_output_dims;
        activation_derivatives.shape = full_output_dims;

        if (dense_layer->get_batch_normalization())
        {
            const Index outputs_number = dense_layer->get_outputs_number();
            means.shape = {outputs_number};
            standard_deviations.shape = {outputs_number};
            normalized_outputs.shape = full_output_dims;
        }
    }

    vector<TensorView*> get_workspace_views() override
    {
        vector<TensorView*> views = { &outputs, &activation_derivatives };

        if (means.size() > 0)
            views.insert(views.end(), { &means, &standard_deviations, &normalized_outputs });

        return views;
    }

    void print() const override
    {
        cout << "Outputs:" << endl
             << outputs.data << endl
             << "Activation derivatives:" << endl
             << activation_derivatives.data << endl;
    }

    TensorView means;
    TensorView standard_deviations;
    TensorView normalized_outputs;
    TensorView activation_derivatives;
};


template<int Rank>
struct DenseBackPropagation final : LayerBackPropagation
{
    DenseBackPropagation(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    ~DenseBackPropagation() override = default;

    void initialize() override
    {
        const auto* dense_layer = static_cast<const Dense<Rank>*>(layer);

        const Index outputs_number = layer->get_outputs_number();
        const Index inputs_number = layer->get_input_shape()[0];

        bias_gradients.shape = {outputs_number};
        weight_gradients.shape = {inputs_number, outputs_number};

        if (dense_layer->get_batch_normalization())
        {
            gamma_gradients.shape = {outputs_number};
            beta_gradients.shape = {outputs_number};
        }

        const Shape input_shape = dense_layer->get_input_shape();

        Shape full_input_shape = { batch_size };
        full_input_shape.insert(full_input_shape.end(), input_shape.begin(), input_shape.end());

        input_gradients_memory.resize(1);
        input_gradients_memory[0].resize(full_input_shape.count());
        input_gradients.resize(1);
        input_gradients[0].data = input_gradients_memory[0].data();
        input_gradients[0].shape = full_input_shape;
    }


    vector<TensorView*> get_gradient_views() override
    {
        vector<TensorView*> views = {&bias_gradients, &weight_gradients};

        const auto* dense_layer = static_cast<const Dense<Rank>*>(layer);

        if (dense_layer->get_batch_normalization())
            views.insert(views.end(), {&gamma_gradients, &beta_gradients});

        return views;
    }


    void print() const override
    {
        cout << "Bias output_gradients:" << endl
             << bias_gradients.data << endl
             << "Weight output_gradients:" << endl
             << weight_gradients.data << endl;
    }

    TensorView bias_gradients;
    TensorView weight_gradients;

    TensorView gamma_gradients;
    TensorView beta_gradients;
};


struct DenseBackPropagationLM final : LayerBackPropagationLM
{
    DenseBackPropagationLM(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    ~DenseBackPropagationLM() override = default;

    void initialize() override
    {
        const Index parameters_number = layer->get_parameters_number();
        const Shape layer_input_shape = layer->get_input_shape();

        Shape input_shape_vec = {batch_size};
        input_shape_vec.insert(input_shape_vec.end(), layer_input_shape.begin(), layer_input_shape.end());

        input_gradients_memory.resize(1);
        input_gradients_memory[0].resize(input_shape_vec.count());

        input_gradients.resize(1);
        input_gradients[0].data = input_gradients_memory[0].data();
        input_gradients[0].shape = input_shape_vec;

        squared_errors_Jacobian.shape = {batch_size, parameters_number};
    }

    vector<TensorView*> get_gradient_views() override
    {
        return {&squared_errors_Jacobian};
    }

    void print() const override
    {
        cout << "Squared errors Jacobian: " << endl;
        squared_errors_Jacobian.print();
        cout << "Input derivatives: " << endl;
        input_gradients[0].print();
    }

    TensorView squared_errors_Jacobian;
};


#ifdef OPENNN_CUDA

template<int Rank>
struct DenseForwardPropagationCuda : public LayerForwardPropagationCuda
{
    DenseForwardPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {
        const Index outputs_number = layer->get_output_shape().back();

        Index total_rows = batch_size;
        if constexpr (Rank == 3)
            total_rows *= layer->get_input_shape()[0];

        auto* dense_layer = static_cast<Dense<Rank>*>(this->layer);

        cudnnCreateTensorDescriptor(&biases_add_tensor_descriptor);
        cudnnSetTensor4dDescriptor(biases_add_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, (int)outputs_number, (int)total_rows, 1);

        cudnnCreateTensorDescriptor(&output_softmax_tensor_descriptor);
        cudnnSetTensor4dDescriptor(output_softmax_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, (int)outputs_number, (int)total_rows, 1);
        
        if (dense_layer->use_combinations)
            combinations.resize({ total_rows, outputs_number, 1, 1 });
        outputs.set_descriptor({ total_rows, outputs_number, 1, 1});

        if (dense_layer->get_dropout_rate() > 0)
        {
            cudnnCreateDropoutDescriptor(&dropout_descriptor);
            cudnnDropoutGetStatesSize(get_cudnn_handle(), &dropout_states_size);
            CHECK_CUDA(cudaMalloc(&dropout_states, dropout_states_size));
            cudnnSetDropoutDescriptor(dropout_descriptor, get_cudnn_handle(), (float)dense_layer->get_dropout_rate(), dropout_states, dropout_states_size, dropout_seed);
            cudnnDropoutGetReserveSpaceSize(outputs.get_descriptor(), &dropout_reserve_space_size);
            CHECK_CUDA(cudaMalloc(&dropout_reserve_space, dropout_reserve_space_size));
        }

        if (dense_layer->get_batch_normalization())
        {
            Shape batch_normalization_dims = { 1, outputs_number, 1, 1 };

            batch_means.resize(batch_normalization_dims);
            bn_saved_inv_variance.resize(batch_normalization_dims);
        }
    }

    void free() override
    {
        if (dropout_states) cudaFree(dropout_states);
        if (dropout_reserve_space) cudaFree(dropout_reserve_space);

        cudnnDestroyTensorDescriptor(output_softmax_tensor_descriptor);
        cudnnDestroyTensorDescriptor(biases_add_tensor_descriptor);

        if (dropout_descriptor) cudnnDestroyDropoutDescriptor(dropout_descriptor);
    }

    TensorCuda combinations;

    cudnnTensorDescriptor_t output_softmax_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t biases_add_tensor_descriptor = nullptr;

    cudnnDropoutDescriptor_t dropout_descriptor = nullptr;

    // Batch normalization

    TensorCuda batch_means;
    TensorCuda bn_saved_inv_variance;

    // Dropout

    void* dropout_states = nullptr;
    size_t dropout_states_size = 0;    
    unsigned long long dropout_seed;

    void* dropout_reserve_space = nullptr;
    size_t dropout_reserve_space_size = 0;
};


template<int Rank>
struct DenseBackPropagationCuda : public LayerBackPropagationCuda
{
    DenseBackPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const Index outputs_number = layer->get_output_shape().back();
        const Index inputs_number = layer->get_input_shape().back();

        Index total_rows = batch_size;
        if constexpr (Rank == 3)
            total_rows *= layer->get_input_shape()[0];

        CHECK_CUDA(cudaMalloc(&ones, total_rows * sizeof(float)));
        vector<float> ones_host(total_rows, 1.0f);
        CHECK_CUDA(cudaMemcpy(ones, ones_host.data(), total_rows * sizeof(float), cudaMemcpyHostToDevice));

        input_gradients.resize(1);
        input_gradients[0].resize({ 1, inputs_number, total_rows, 1 });

        bias_gradients.set_descriptor({ 1, outputs_number, 1, 1 });
        weight_gradients.set_descriptor({ 1, inputs_number * outputs_number, 1, 1 });

        cudnnCreateTensorDescriptor(&gradients_tensor_descriptor);
        cudnnSetTensor4dDescriptor(gradients_tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)total_rows, (int)outputs_number, 1, 1);

        const Dense<Rank>* dense_layer = static_cast<Dense<Rank>*>(this->layer);

        if (dense_layer->get_batch_normalization())
        {
            beta_gradients.set_descriptor({1, outputs_number,1,1});
            gamma_gradients.set_descriptor({1, outputs_number,1,1});
        }
    }

    vector<TensorViewCuda*> get_workspace_views() override
    {
        vector<TensorViewCuda*> views = { &bias_gradients, &weight_gradients};

        auto* dense_layer = static_cast<const Dense<Rank>*>(this->layer);

        if (dense_layer && dense_layer->get_batch_normalization())
            views.insert(views.end(), { &gamma_gradients, &beta_gradients});

        return views;
    }


    void free() override
    {
        cudaFree(ones);
        ones = nullptr;

        cudnnDestroyTensorDescriptor(gradients_tensor_descriptor);
        gradients_tensor_descriptor = nullptr;
    }

    TensorViewCuda bias_gradients;
    TensorViewCuda weight_gradients;

    float* ones = nullptr;

    TensorViewCuda gamma_gradients;
    TensorViewCuda beta_gradients;
    cudnnTensorDescriptor_t gradients_tensor_descriptor = nullptr;
};

#endif


struct Dense2dForwardPropagationLM;


template<int Rank>
class Dense final : public Layer
{

public:

    Dense(const Shape& new_input_shape = {0},
          const Shape& new_output_shape = {0},
          const string& new_activation_function = "HyperbolicTangent",
          bool new_batch_normalization = false,
          const string& new_label = "dense2d_layer")
    {
        set(new_input_shape, new_output_shape, new_activation_function, new_batch_normalization, new_label);
    }


    Dense(const Index input_sequence_length,
          Index embedding_dimension,
          Index feed_forward_dimension,
          const string& new_activation_function,
          const string& new_label)
    {
        set({input_sequence_length, embedding_dimension},
            {feed_forward_dimension},
            new_activation_function,
            false,
            new_label);
    }


    Shape get_input_shape() const override
    {
        return { weights.shape[0] };
    }


    Shape get_output_shape() const override
    {
        return { biases.size() };
    }


    vector<TensorView*> get_parameter_views() override
    {
        vector<TensorView*> views = {&biases, &weights};

        if (batch_normalization)
            views.insert(views.end(), {&gammas, &betas});

        return views;
    }


    type get_dropout_rate() const
    {
        return dropout_rate;
    }


    bool get_batch_normalization() const
    {
        return batch_normalization;
    }


    const string& get_activation_function() const
    {
        return activation_function;
    }


    void set(const Shape& new_input_shape = {},
             const Shape& new_output_shape = {},
             const string& new_activation_function = "HyperbolicTangent",
             bool new_batch_normalization = false,
             const string& new_label = "dense_layer")
    {
        if (new_input_shape.size() != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        if (new_output_shape.size() != 1)
            throw runtime_error("Output shape size is not 1");

        biases.shape = { new_output_shape[0] };
        weights.shape = { new_input_shape[0], new_output_shape[0] };

        set_activation_function(new_activation_function);

        set_batch_normalization(new_batch_normalization);

        const Index outputs_number = get_outputs_number();

        if (batch_normalization)
        {
            gammas.shape = {outputs_number};
            betas.shape = {outputs_number};
            running_means.resize(outputs_number);
            running_standard_deviations.resize(outputs_number);
        }

        set_label(new_label);

        name = "Dense" + to_string(Rank) + "d";

#ifdef OPENNN_CUDA

        biases_device.set_descriptor({1, outputs_number, 1, 1});
        weights_device.set_descriptor({ new_input_shape[0], outputs_number, 1, 1 });

        if (batch_normalization)
        {
            Shape batch_normalization_dims = { 1, outputs_number, 1, 1 };

            betas_device.set_descriptor(batch_normalization_dims);
            gammas_device.set_descriptor(batch_normalization_dims);
            running_means_device.resize(batch_normalization_dims);
            running_variances_device.resize(batch_normalization_dims);
        }

#endif
    }


    void set_parameters_glorot() override
    {
        const type limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));

        if(biases.size() > 0)
        {
            VectorMap biases_map(biases.data, biases.size());
            biases_map.setZero();
        }

        if(weights.size() > 0)
        {
            VectorMap weights_map(weights.data, weights.size());
            set_random_uniform(weights_map, -limit, limit);
        }

        if(batch_normalization)
        {
            if(gammas.size() > 0)
            {
                VectorMap scales_map(gammas.data, gammas.size());
                scales_map.setConstant(1.0);
            }
            if(betas.size() > 0)
            {
                VectorMap offsets_map(betas.data, betas.size());
                offsets_map.setZero();
            }
        }
    }

    void set_parameters_random() override
    {
        if(biases.size() > 0)
        {
            VectorMap biases_map(biases.data, biases.size());
            biases_map.setZero();
        }

        if(weights.size() > 0)
        {
            VectorMap weights_map(weights.data, weights.size());
            set_random_uniform(weights_map);
        }

        if (batch_normalization)
        {
            if(gammas.size() > 0)
                VectorMap(gammas.data, gammas.size()).setConstant(1.0);

            if(betas.size() > 0)
                VectorMap(betas.data, betas.size()).setZero();
        }
    }


    void set_input_shape(const Shape& new_input_shape) override
    {
        const Index inputs_number = new_input_shape[0];
        const Index outputs_number = get_outputs_number();

        biases.shape = { outputs_number };
        weights.shape = { inputs_number, outputs_number };
    }


    void set_output_shape(const Shape& new_output_shape) override
    {
        const Index inputs_number = get_inputs_number();
        const Index neurons_number = new_output_shape[0];

        biases.shape = { neurons_number };
        weights.shape = { inputs_number, neurons_number };
    }


    void set_activation_function(const string& new_activation_function)
    {
        static const unordered_set<string> activation_functions =
            {"Sigmoid", "HyperbolicTangent", "Linear", "RectifiedLinear", "ScaledExponentialLinear", "Softmax"};

        if (activation_functions.count(new_activation_function))
        {
            if (get_output_shape()[0] == 1 && new_activation_function == "Softmax")
                activation_function = "Sigmoid";
            else
                activation_function = new_activation_function;
        }
        else
            throw runtime_error("Unknown activation function: " + new_activation_function);

#ifdef OPENNN_CUDA

        if (activation_descriptor == nullptr && activation_function != "Softmax")
            cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY;
        double relu_ceiling = 0.0;

        if (activation_function == "Linear")
        {
            activation_mode = CUDNN_ACTIVATION_IDENTITY;
            use_combinations = false;
        }
        else if (activation_function == "Sigmoid")
        {
            activation_mode = CUDNN_ACTIVATION_SIGMOID;
            use_combinations = false;
        }
        else if (activation_function == "HyperbolicTangent")
        {
            activation_mode = CUDNN_ACTIVATION_TANH;
            use_combinations = false;
        }
        else if (activation_function == "RectifiedLinear")
        {
            activation_mode = CUDNN_ACTIVATION_RELU;
            use_combinations = false;
        }
        else if (activation_function == "ScaledExponentialLinear")
        {
            activation_mode = CUDNN_ACTIVATION_ELU;
            use_combinations = true;
        }
        else if (activation_function == "ClippedRelu")
        {
            activation_mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            use_combinations = true;
            relu_ceiling = 6.0;
        }
        else if (activation_function == "Swish")
        {
            activation_mode = CUDNN_ACTIVATION_SWISH;
            use_combinations = true;
        }
        else if (activation_function == "Softmax")
            use_combinations = true;

        if (activation_function != "Softmax")
            cudnnSetActivationDescriptor(activation_descriptor, activation_mode, CUDNN_PROPAGATE_NAN, relu_ceiling);

#endif
    }

    void set_dropout_rate(const type new_dropout_rate)
    {
        if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
            throw runtime_error("Dropout rate must be in [0,1).");

        dropout_rate = new_dropout_rate;
    }


    void set_batch_normalization(bool new_batch_normalization)
    {
        batch_normalization = new_batch_normalization;
    }

/*
    void normalization(VectorR& means, VectorR& standard_deviations, const Tensor2& inputs, Tensor2& outputs) const
    {
        const array<Index, 2> rows({outputs.dimension(0), 1});

        const array<int, 1> axis_x({0});

        means.device(get_device()) = outputs.mean(axis_x);

        standard_deviations.device(get_device())
            = (outputs - means.broadcast(rows)).square().mean(axis_x).sqrt();

        outputs = inputs;// -means.broadcast(array<Index, 2>({ outputs.dimension(0), 1 }));
            //shifts.broadcast(rows);
            //+ (outputs - means.broadcast(rows))*gammas.broadcast(rows)/standard_deviations.broadcast(rows);
    }
*/

    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           bool is_training) override
    {
        auto* dense_forward_propagation = static_cast<DenseForwardPropagation<Rank>*>(layer_forward_propagation.get());

        auto outputs = tensor_map<Rank>(dense_forward_propagation->outputs);

        calculate_combinations<Rank>(tensor_map<Rank>(input_views[0]),
                                     matrix_map(weights),
                                     vector_map(biases),
                                     outputs);

        if(batch_normalization)
        {
            auto normalized_outputs = tensor_map<Rank>(dense_forward_propagation->normalized_outputs);

            normalize_batch<Rank>(
                outputs,
                normalized_outputs,
                vector_map(dense_forward_propagation->means),
                vector_map(dense_forward_propagation->standard_deviations),
                running_means,
                running_standard_deviations,
                vector_map(gammas),
                vector_map(betas),
                is_training,
                momentum);
        }

        if(is_training)
        {
            auto activation_derivatives = tensor_map<Rank>(dense_forward_propagation->activation_derivatives);
            calculate_activations<Rank>(activation_function, outputs, activation_derivatives);
        }
        else
        {
            if constexpr(Rank == 2)
                calculate_activations<Rank>(activation_function, outputs, MatrixMap(empty_2.data(), empty_2.dimensions()));
            else if constexpr(Rank == 3)
                calculate_activations<Rank>(activation_function, outputs, TensorMap3(empty_3.data(), empty_3.dimensions()));
        }

        if(is_training && dropout_rate > type(0))
            dropout<Rank>(outputs, dropout_rate);
    }


    void back_propagate(const vector<TensorView>& input_views,
                        const vector<TensorView>& output_gradient_views,
                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        const MatrixMap inputs = matrix_map(input_views[0]);
        MatrixMap output_gradients = matrix_map(output_gradient_views[0]);

        // Forward propagation

        const DenseForwardPropagation<2>* dense_forward_propagation =
            static_cast<DenseForwardPropagation<2>*>(forward_propagation.get());

        const MatrixMap activation_derivatives = matrix_map(dense_forward_propagation->activation_derivatives);

        const MatrixMap weights_map = matrix_map(weights);

        // Back propagation

        DenseBackPropagation<2>* dense2d_back_propagation =
            static_cast<DenseBackPropagation<2>*>(back_propagation.get());

        if(activation_function != "Softmax")
            output_gradients.array() *= activation_derivatives.array();

        if (batch_normalization)
        {
            const MatrixMap normalized_outputs = matrix_map(dense_forward_propagation->normalized_outputs);

            VectorMap gamma_gradients = vector_map(dense2d_back_propagation->gamma_gradients);
            VectorMap beta_gradients = vector_map(dense2d_back_propagation->beta_gradients);

            beta_gradients.noalias() = output_gradients.colwise().sum();

            gamma_gradients = (output_gradients.array() * normalized_outputs.array()).colwise().sum().transpose();
        }

        MatrixMap weight_gradients = matrix_map(dense2d_back_propagation->weight_gradients);
        VectorMap bias_gradients = vector_map(dense2d_back_propagation->bias_gradients);
        MatrixMap input_gradients = matrix_map(back_propagation->input_gradients[0]);

        weight_gradients.noalias() = inputs.transpose() * output_gradients;

        bias_gradients.noalias() = output_gradients.colwise().sum();

        if(!dense2d_back_propagation->is_first_layer)
            input_gradients.noalias() = output_gradients * weights_map.transpose();
    }


    void back_propagate_lm(const vector<TensorView>& input_views,
                           const vector<TensorView>& output_gradient_views,
                           unique_ptr<LayerForwardPropagation>& forward_propagation,
                           unique_ptr<LayerBackPropagationLM>& back_propagation) const override
    {
        const MatrixMap inputs = matrix_map(input_views[0]);
        MatrixMap output_gradients = matrix_map(output_gradient_views[0]);

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();
        const Index biases_number = biases.size();

        // Forward propagation

        const DenseForwardPropagation<Rank>* dense_forward_propagation =
            static_cast<DenseForwardPropagation<Rank>*>(forward_propagation.get());

        const MatrixMap activation_derivatives = matrix_map(dense_forward_propagation->activation_derivatives);

        // Back propagation

        DenseBackPropagationLM* dense_back_propagation_lm =
            static_cast<DenseBackPropagationLM*>(back_propagation.get());

        MatrixMap squared_errors_Jacobian = matrix_map(dense_back_propagation_lm->squared_errors_Jacobian);

        if(activation_function != "Softmax")
            output_gradients.array() *= activation_derivatives.array();

        squared_errors_Jacobian.leftCols(biases_number).noalias() = output_gradients;

        for(Index j = 0; j < outputs_number; j++)
        {
            for(Index i = 0; i < inputs_number; i++)
            {
                const Index weight_column_index = biases_number + j * inputs_number + i;

                squared_errors_Jacobian.col(weight_column_index).array()
                    = output_gradients.col(j).array() * inputs.col(i).array();
            }
        }

        if(!dense_back_propagation_lm->is_first_layer)
        {
            MatrixMap input_gradients = matrix_map(dense_back_propagation_lm->input_gradients[0]);

            const MatrixMap weights_map = matrix_map(weights);

            input_gradients.noalias() = output_gradients * weights_map.transpose();
        }
    }


    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                           Index start_column_index,
                                           MatrixR& global_jacobian) const override
    {
        const Index alignment_elements = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
        const Index mask_elements = ~(alignment_elements - 1);
        const Index total_error_terms = global_jacobian.rows();

        Index global_offset = start_column_index;
        Index local_offset = 0;

        DenseBackPropagationLM* dense_lm = static_cast<DenseBackPropagationLM*>(back_propagation.get());

        MatrixMap layer_jacobian = matrix_map(dense_lm->squared_errors_Jacobian);

        const Index biases_size = biases.size();
        if(biases_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, biases_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, biases_size);

            local_offset += biases_size;
            global_offset += (biases_size + alignment_elements - 1) & mask_elements;
        }

        const Index weights_size = weights.size();

        if(weights_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, weights_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, weights_size);

            local_offset += weights_size;
            global_offset += (weights_size + alignment_elements - 1) & mask_elements;
        }

        if(!batch_normalization) return;

        const Index gammas_size = gammas.size();

        if(gammas_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, gammas_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, gammas_size);

            local_offset += gammas_size;
            global_offset += (gammas_size + alignment_elements - 1) & mask_elements;
        }

        const Index betas_size = betas.size();

        if(betas_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, betas_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, betas_size);

            local_offset += betas_size;
            global_offset += (betas_size + alignment_elements - 1) & mask_elements;
        }
    }


    string get_expression(const vector<string>& new_feature_names = vector<string>(),
                          const vector<string>& new_output_names = vector<string>()) const override
    {
        const vector<string> input_names = new_feature_names.empty()
            ? get_default_feature_names()
            : new_feature_names;

        const vector<string> output_names = new_output_names.empty()
            ? get_default_output_names()
            : new_output_names;

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        ostringstream buffer;
/*
        for(Index j = 0; j < outputs_number; j++)
        {
            const VectorMap weights_column = tensor_map(weights, j);

            buffer << output_names[j] << " = " << activation_function << "( " << biases(j) << " + ";

            for(Index i = 0; i < inputs_number - 1; i++)
                buffer << "(" << weights_column(i) << "*" << input_names[i] << ") + ";

            buffer << "(" << weights_column(inputs_number - 1) << "*" << input_names[inputs_number - 1] << ") );\n";
        }
*/
        return buffer.str();
    }


    void print() const override
    {
/*
        cout << "Dense layer" << endl
             << "Input shape: " << get_input_shape()[0] << endl
             << "Output shape: " << get_output_shape()[0] << endl
             << "Biases shape: " << biases.dimensions() << endl
             << "Weights shape: " << weights.dimensions() << endl;

        cout << "Activation function:" << endl;
        cout << activation_function << endl;
*/
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* dense2d_layer_element = document.FirstChildElement(name.c_str());

        if(!dense2d_layer_element)
            throw runtime_error(name + " element is nullptr.\n");

        set_label(read_xml_string(dense2d_layer_element, "Label"));

        const Index inputs_number = read_xml_index(dense2d_layer_element, "InputsNumber");
        const Index neurons_number = read_xml_index(dense2d_layer_element, "NeuronsNumber");

        set_input_shape({ inputs_number });
        set_output_shape({ neurons_number });

        set_activation_function(read_xml_string(dense2d_layer_element, "Activation"));

        bool use_batch_normalization = false;
        const XMLElement* bn_element = dense2d_layer_element->FirstChildElement("BatchNormalization");

        if (bn_element && bn_element->GetText())
            use_batch_normalization = (string(bn_element->GetText()) == "true");
        set_batch_normalization(use_batch_normalization);      

        if (batch_normalization)
        {
            running_means.resize(neurons_number);
            running_standard_deviations.resize(neurons_number);

            string_to_vector(read_xml_string(dense2d_layer_element, "RunningMeans"), running_means);
            string_to_vector(read_xml_string(dense2d_layer_element, "RunningStandardDeviations"), running_standard_deviations);
        }
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement(name.c_str());

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputsNumber", to_string(get_input_shape()[0]));
        add_xml_element(printer, "NeuronsNumber", to_string(get_output_shape()[0]));
        add_xml_element(printer, "Activation", activation_function);
        add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");

        if (batch_normalization)
        {
            add_xml_element(printer, "RunningMeans", vector_to_string(running_means));
            add_xml_element(printer, "RunningStandardDeviations", vector_to_string(running_standard_deviations));
        }

        printer.CloseElement();
    }


#ifdef OPENNN_CUDA

public:

    vector<TensorViewCuda*> get_parameter_views_device() override
    {
        vector<TensorViewCuda*> views = { &biases_device, &weights_device };

        if (batch_normalization)
            views.insert(views.end(), { &gammas_device, &betas_device });

        return views;
    }

    // @todo The following are not parameters

    void copy_parameters_device()
    {
        if (!batch_normalization) return;

        CHECK_CUDA(cudaMemcpy(running_means_device.data, running_means.data(), running_means.size() * sizeof(type), cudaMemcpyHostToDevice));
        VectorR moving_variances = running_standard_deviations.square();
        CHECK_CUDA(cudaMemcpy(running_variances_device.data, moving_variances.data(), moving_variances.size() * sizeof(type), cudaMemcpyHostToDevice));
    }


    void copy_parameters_host()
    {
        if (!batch_normalization) return;
        CHECK_CUDA(cudaMemcpy(running_means.data(), running_means_device.data, running_means.size() * sizeof(type), cudaMemcpyDeviceToHost));
        VectorR moving_variances(running_standard_deviations.size());
        CHECK_CUDA(cudaMemcpy(moving_variances.data(), running_variances_device.data, moving_variances.size() * sizeof(type), cudaMemcpyDeviceToHost));
        running_standard_deviations = moving_variances.sqrt();
    }


    void forward_propagate(const vector<TensorViewCuda>& inputs,
                                unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                bool is_training)
    {
        // Dense layer

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        // Forward propagation

        const Index batch_size = forward_propagation->batch_size;

        TensorViewCuda& outputs = forward_propagation->outputs;

        auto* dense_forward_propagation = static_cast<DenseForwardPropagationCuda<Rank>*>(forward_propagation.get());

        type* combinations = dense_forward_propagation->combinations.data;

        const cudnnTensorDescriptor_t output_softmax_tensor_descriptor = dense_forward_propagation->output_softmax_tensor_descriptor;

        const cudnnTensorDescriptor_t& biases_add_tensor_descriptor = dense_forward_propagation->biases_add_tensor_descriptor;

        type* outputs_buffer = use_combinations ? combinations : outputs.data;

        // Combinations

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    batch_size, outputs_number, inputs_number,
                    &alpha,
                    inputs[0].data,
                    batch_size,
                    weights_device.data,
                    inputs_number,
                    &beta,
                    outputs_buffer,
                    batch_size));

        CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                                   &alpha,
                                   biases_device.get_descriptor(),
                                   biases_device.data,
                                   &beta_add,
                                   biases_add_tensor_descriptor,
                                   outputs_buffer));

        // Batch Normalization

        if (batch_normalization && is_training)
                CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
                    get_cudnn_handle(),
                    CUDNN_BATCHNORM_PER_ACTIVATION,
                    &alpha,
                    &beta_add,
                    outputs.get_descriptor(),
                    outputs_buffer,
                    outputs.get_descriptor(),
                    outputs_buffer,
                    gammas_device.get_descriptor(),
                    gammas_device.data,
                    betas_device.data,
                    momentum,
                    running_means_device.data,
                    running_variances_device.data,
                    numeric_limits<type>::epsilon(),
                    dense_forward_propagation->batch_means.data,
                    dense_forward_propagation->bn_saved_inv_variance.data));
        else if (batch_normalization && !is_training)
                CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
                    get_cudnn_handle(),
                    CUDNN_BATCHNORM_PER_ACTIVATION,
                    &alpha, &beta_add,
                    outputs.get_descriptor(),
                    outputs_buffer,
                    outputs.get_descriptor(),
                    outputs_buffer,
                    gammas_device.get_descriptor(),
                    gammas_device.data,
                    betas_device.data,
                    running_means_device.data,
                    running_variances_device.data,
                    numeric_limits<type>::epsilon()));

        // Activations

        if (activation_function == "Linear")
            cudaMemcpy(outputs.data, combinations, batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
        else if (activation_function == "Softmax")
            cudnnSoftmaxForward(get_cudnn_handle(),
                                CUDNN_SOFTMAX_ACCURATE,
                                CUDNN_SOFTMAX_MODE_CHANNEL,
                                &alpha,
                                output_softmax_tensor_descriptor,
                                combinations,
                                &beta,
                                output_softmax_tensor_descriptor,
                                outputs.data);
        else
            cudnnActivationForward(get_cudnn_handle(),
                                   activation_descriptor,
                                   &alpha,
                                   outputs.get_descriptor(),
                                   outputs_buffer,
                                   &beta,
                                   outputs.get_descriptor(),
                                   outputs.data);

        // Droput

        if (is_training && activation_function != "Softmax" && get_dropout_rate() > type(0))
            CHECK_CUDNN(cudnnDropoutForward(get_cudnn_handle(),
                                            dense_forward_propagation->dropout_descriptor,
                                            outputs.get_descriptor(),
                                            outputs.data,
                                            outputs.get_descriptor(),
                                            outputs.data,
                                            dense_forward_propagation->dropout_reserve_space,
                                            dense_forward_propagation->dropout_reserve_space_size)); 
    }


    void back_propagate(const vector<TensorViewCuda>& inputs,
                             const vector<TensorViewCuda>& output_gradients,
                             unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                             unique_ptr<LayerBackPropagationCuda>& bp_cuda) const
    {
        // Dense layer

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        // Forward propagation

        const Index batch_size = forward_propagation->batch_size;

        const TensorViewCuda& outputs_view = forward_propagation->outputs;

        const auto* dense_forward_propagation = static_cast<DenseForwardPropagationCuda<Rank>*>(forward_propagation.get());

//        const Dense* dense_layer = static_cast<Dense*>(dense_forward_propagation->layer);

        type* combinations = dense_forward_propagation->combinations.data;

        // Back propagation

        float* input_gradients = bp_cuda->input_gradients[0].data;

        auto* dense_layer_back_propagation = static_cast<DenseBackPropagationCuda<Rank>*>(bp_cuda.get());

        float* ones = dense_layer_back_propagation->ones;

        float* bias_gradients = dense_layer_back_propagation->bias_gradients.data;
        float* weight_gradients = dense_layer_back_propagation->weight_gradients.data;

        const cudnnTensorDescriptor_t gradients_tensor_descriptor = dense_layer_back_propagation->gradients_tensor_descriptor;

        // Dropout

        if (get_dropout_rate() > type(0) && activation_function != "Softmax")
            CHECK_CUDNN(cudnnDropoutBackward(get_cudnn_handle(),
                                             dense_forward_propagation->dropout_descriptor,
                                             gradients_tensor_descriptor,
                                             output_gradients[0].data,
                                             gradients_tensor_descriptor,
                                             output_gradients[0].data,
                                             dense_forward_propagation->dropout_reserve_space,
                                             dense_forward_propagation->dropout_reserve_space_size));

        // Error combinations derivatives

        if (activation_function != "Linear" && activation_function != "Softmax" && use_combinations)
            CHECK_CUDNN(cudnnActivationBackward(get_cudnn_handle(),
                                                activation_descriptor,
                                                &alpha,
                                                gradients_tensor_descriptor,
                                                outputs_view.data,
                                                gradients_tensor_descriptor,
                                                output_gradients[0].data,
                                                gradients_tensor_descriptor,
                                                combinations,
                                                &beta,
                                                gradients_tensor_descriptor,
                                                output_gradients[0].data));
        else if (activation_function != "Linear" && activation_function != "Softmax" && !use_combinations)
            CHECK_CUDNN(cudnnActivationBackward(get_cudnn_handle(),
                                                activation_descriptor,
                                                &alpha,
                                                gradients_tensor_descriptor,
                                                outputs_view.data,
                                                gradients_tensor_descriptor,
                                                output_gradients[0].data,
                                                gradients_tensor_descriptor,
                                                outputs_view.data,
                                                &beta,
                                                gradients_tensor_descriptor,
                                                output_gradients[0].data));

        // Batch Normalization

        if (batch_normalization)
            CHECK_CUDNN(cudnnBatchNormalizationBackward(
                get_cudnn_handle(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha, &beta,
                &alpha, &beta,
                dense_forward_propagation->outputs.get_descriptor(),
                use_combinations ? combinations : outputs_view.data,
                gradients_tensor_descriptor,
                output_gradients[0].data,
                gradients_tensor_descriptor,
                output_gradients[0].data,
                gammas_device.get_descriptor(),
                gammas_device.data,
                dense_layer_back_propagation->gamma_gradients.data,
                dense_layer_back_propagation->beta_gradients.data,
                numeric_limits<type>::epsilon(),
                dense_forward_propagation->batch_means.data,
                dense_forward_propagation->bn_saved_inv_variance.data));

        // Bias derivatives

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    outputs_number,
                    1,
                    batch_size,
                    &alpha,
                    output_gradients[0].data,
                    batch_size,
                    ones,
                    batch_size,
                    &beta,
                    bias_gradients,
                    outputs_number));

        // Weight derivatives

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    inputs_number,
                    outputs_number,
                    batch_size,
                    &alpha,
                    inputs[0].data,
                    batch_size,
                    output_gradients[0].data,
                    batch_size,
                    &beta,
                    weight_gradients,
                    inputs_number));

        // Input derivatives

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                    batch_size,
                    inputs_number,
                    outputs_number,
                    &alpha,
                    output_gradients[0].data,
                    batch_size,
                    weights_device.data,
                    inputs_number,
                    &beta,
                    input_gradients,
                    batch_size));
    }

    bool use_combinations = true;

private:

    TensorViewCuda biases_device;
    TensorViewCuda weights_device;

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    TensorViewCuda gammas_device;
    TensorViewCuda betas_device;

    TensorCuda running_means_device;
    TensorCuda running_variances_device;

#endif

private:

    Index inputs_number;
    Index outputs_number;

    TensorView biases;
    TensorView weights;

    TensorView gammas;
    TensorView betas;

    VectorR running_means;
    VectorR running_standard_deviations;

    bool batch_normalization = false;

    type momentum = type(0.9);

    string activation_function = "HyperbolicTangent";

    type dropout_rate = type(0);
};

void reference_dense_layer();

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
