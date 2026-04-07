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
        const Dense<Rank>* dense_layer = static_cast<const Dense<Rank>*>(layer);

        const Shape full_output_shape = Shape{batch_size}.append(dense_layer->get_output_shape());

        outputs.shape = full_output_shape;
        activation_derivatives.shape = full_output_shape;

        if (dense_layer->get_batch_normalization())
        {
            const Index outputs_number = dense_layer->get_neurons_number();
            means.shape = {outputs_number};
            standard_deviations.shape = {outputs_number};
            normalized_outputs.shape = full_output_shape;
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
        cout << "Dense forward propagation" << endl;
        cout << "Outputs dimensions: " << outputs.shape << endl;
        cout << "Outputs data:" << endl;
        outputs.print();
        cout << "Activation derivatives dimensions: " << activation_derivatives.shape << endl;
        cout << "Activation derivatives data:" << endl;
        activation_derivatives.print();
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
        const Dense<Rank>* dense_layer = static_cast<const Dense<Rank>*>(layer);

        const Index outputs_number = dense_layer->get_neurons_number();

        const Index inputs_number = dense_layer->get_input_features_number();

        bias_gradients.shape = {outputs_number};
        weight_gradients.shape = {inputs_number, outputs_number};

        if (dense_layer->get_batch_normalization())
        {
            gamma_gradients.shape = {outputs_number};
            beta_gradients.shape = {outputs_number};
        }

        input_gradients = {{nullptr, Shape{batch_size}.append(dense_layer->get_input_shape())}};
    }


    vector<TensorView*> get_gradient_views() override
    {
        vector<TensorView*> views = {&bias_gradients, &weight_gradients};

        const Dense<Rank>* dense_layer = static_cast<const Dense<Rank>*>(layer);

        if (dense_layer->get_batch_normalization())
            views.insert(views.end(), {&gamma_gradients, &beta_gradients});

        return views;
    }


    void print() const override
    {
        cout << "Dense back propagation" << endl;
        cout << "Bias gradients dimensions: " << bias_gradients.shape << endl;
        cout << "Bias gradients data:" << endl;
        bias_gradients.print();
        cout << "Weight gradients dimensions: " << weight_gradients.shape << endl;
        cout << "Weight gradients data:" << endl;
        weight_gradients.print();
        cout << "Input gradients dimensions: " << input_gradients[0].shape << endl;
        cout << "Input gradients data:" << endl;
        input_gradients[0].print();
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
        input_gradients = {{nullptr, Shape{batch_size}.append(layer->get_input_shape())}};

        squared_errors_Jacobian.shape = {batch_size, layer->get_parameters_number()};
    }

    vector<TensorView*> get_gradient_views() override
    {
        return {&squared_errors_Jacobian};
    }

    void print() const override
    {
        cout << "Dense back propagation LM" << endl;
        cout << "Squared errors Jacobian dimensions: " << squared_errors_Jacobian.shape << endl;
        cout << "Squared errors Jacobian data: " << endl;
        squared_errors_Jacobian.print();
        cout << "Input derivatives data: " << endl;
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
        Dense<Rank>* dense_layer = static_cast<Dense<Rank>*>(this->layer);

        const Index outputs_number = dense_layer->get_neurons_number();
        const Shape full_output_shape = dense_layer->get_batch_output_shape(batch_size);

        if (dense_layer->use_combinations)
            combinations.set_descriptor(full_output_shape);

        outputs.set_descriptor(full_output_shape);

        if (dense_layer->get_dropout_rate() > 0)
        {
            dropout_seed = static_cast<unsigned long long>(get_seed());

            if (dropout_descriptor == nullptr) cudnnCreateDropoutDescriptor(&dropout_descriptor);

            cudnnDropoutGetStatesSize(get_cudnn_handle(), &dropout_states_size);

            if (dropout_states) cudaFree(dropout_states);
            CHECK_CUDA(cudaMalloc(&dropout_states, dropout_states_size));

            cudnnSetDropoutDescriptor(dropout_descriptor, get_cudnn_handle(), (float)dense_layer->get_dropout_rate(), dropout_states, dropout_states_size, dropout_seed);

            cudnnDropoutGetReserveSpaceSize(outputs.get_descriptor(), &dropout_reserve_space_size);

            if (dropout_reserve_space) cudaFree(dropout_reserve_space);
            CHECK_CUDA(cudaMalloc(&dropout_reserve_space, dropout_reserve_space_size));
        }

        if (dense_layer->get_batch_normalization())
        {
            const Shape batch_normalization_shape = { outputs_number };

            means.resize(batch_normalization_shape);
            inverse_variance.resize(batch_normalization_shape);
        }
    }

    void free() override
    {
        cudaFree(dropout_states);
        dropout_states = nullptr;

        cudaFree(dropout_reserve_space);
        dropout_reserve_space = nullptr;

        if (dropout_descriptor) cudnnDestroyDropoutDescriptor(dropout_descriptor);
    }


    vector<TensorViewCuda*> get_workspace_views() override
    {
        vector<TensorViewCuda*> views = { &outputs };

        Dense<Rank>* dense_layer = static_cast<Dense<Rank>*>(this->layer);

        if (dense_layer->use_combinations)
            views.push_back(&combinations);

        return views;
    }

    void print() const override
    {
        const Index outputs_number = layer->get_output_shape().back();

        const Index total_rows = static_cast<const Dense<Rank>*>(this->layer)->get_total_rows(batch_size);

        cout << "Dense forward propagation CUDA" << endl;
        cout << "Batch size: " << batch_size << endl;
        cout << "Outputs dimensions: " << total_rows << "x" << outputs_number << endl;
        cout << "Outputs data:" << endl;

        if (outputs.data)
            cout << matrix_from_device(outputs.data, total_rows, outputs_number) << endl;
        else
            cout << "Empty (nullptr)" << endl;
    }

    TensorViewCuda combinations;

    cudnnDropoutDescriptor_t dropout_descriptor = nullptr;

    // Batch normalization

    TensorCuda means;
    TensorCuda inverse_variance;

    // Dropout

    unsigned long long dropout_seed;

    void* dropout_states = nullptr;
    size_t dropout_states_size = 0;

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
        const Dense<Rank>* dense_layer = static_cast<Dense<Rank>*>(this->layer);

        const Index outputs_number = dense_layer->get_neurons_number();
        const Index inputs_number = dense_layer->get_input_features_number();
        const Index total_rows = dense_layer->get_total_rows(batch_size);

        ones.resize({total_rows});
        ones.fill(1.0f);

        const Shape full_input_shape = dense_layer->get_batch_input_shape(batch_size);
        const Shape full_output_shape = dense_layer->get_batch_output_shape(batch_size);

        input_gradients = {TensorViewCuda(full_input_shape)};

        bias_gradients.set_descriptor({ outputs_number });
        weight_gradients.set_descriptor({ inputs_number, outputs_number });

        if (gradients_tensor_descriptor == nullptr)
            cudnnCreateTensorDescriptor(&gradients_tensor_descriptor);

        int n = 1, c = 1, h = 1, w = 1;
        if (full_output_shape.size() == 3)
        {
            n = static_cast<int>(full_output_shape[0]);
            w = static_cast<int>(full_output_shape[1]);
            c = static_cast<int>(full_output_shape[2]);
        }
        else if (full_output_shape.size() == 2)
        {
            n = static_cast<int>(full_output_shape[0]);
            c = static_cast<int>(full_output_shape[1]);
        }
        else if (full_output_shape.size() == 1)
            c = static_cast<int>(full_output_shape[0]);

        cudnnSetTensor4dDescriptor(gradients_tensor_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w);

        if (dense_layer->get_batch_normalization())
        {
            beta_gradients.set_descriptor({outputs_number});
            gamma_gradients.set_descriptor({outputs_number});
        }
    }

    vector<TensorViewCuda*> get_gradient_views() override
    {
        vector<TensorViewCuda*> views = { &bias_gradients, &weight_gradients};

        const Dense<Rank>* dense_layer = static_cast<const Dense<Rank>*>(this->layer);

        if (dense_layer && dense_layer->get_batch_normalization())
            views.insert(views.end(), { &gamma_gradients, &beta_gradients});

        return views;
    }


    void free() override
    {
        cudnnDestroyTensorDescriptor(gradients_tensor_descriptor);
        gradients_tensor_descriptor = nullptr;
    }

    void print() const override
    {
        const Dense<Rank>* dense_layer = static_cast<const Dense<Rank>*>(this->layer);
        const Index outputs_number = dense_layer->get_neurons_number();
        const Index inputs_number = dense_layer->get_input_features_number();

        const Index total_rows = static_cast<const Dense<Rank>*>(this->layer)->get_total_rows(batch_size);

        cout << "Dense back propagation CUDA" << endl;
        cout << "Batch size: " << batch_size << endl;
        cout << "Bias gradients (" << outputs_number << "):" << endl;
        if (bias_gradients.data) cout << vector_from_device(bias_gradients.data, outputs_number) << endl;
        cout << "Weight gradients (" << inputs_number << "x" << outputs_number << "):" << endl;
        if (weight_gradients.data) cout << matrix_from_device(weight_gradients.data, inputs_number, outputs_number) << endl;
        cout << "Input gradients (" << total_rows << "x" << inputs_number << "):" << endl;
        if (!input_gradients.empty() && input_gradients[0].data)
            cout << matrix_from_device(input_gradients[0].data, total_rows, inputs_number) << endl;
    }

    TensorViewCuda bias_gradients;
    TensorViewCuda weight_gradients;

    TensorViewCuda gamma_gradients;
    TensorViewCuda beta_gradients;

    TensorCuda ones;

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
        return input_shape;
    }


    Shape get_output_shape() const override
    {
        if constexpr (Rank == 2)
            return { biases.size() };
        else
            return { input_shape[0], biases.size() };
    }


    Index get_input_features_number() const
    {
        return weights.shape[0];
    }


    Index get_neurons_number() const
    {
        return biases.size();
    }


    Index get_sequence_length() const
    {
        if constexpr (Rank == 3)
            return input_shape[0];
        else
            return 1;
    }


    Index get_total_rows(const Index batch_size) const
    {
        return batch_size * get_sequence_length();
    }


    Shape get_batch_input_shape(const Index batch_size) const
    {
        return Shape{batch_size}.append(get_input_shape());
    }


    Shape get_batch_output_shape(const Index batch_size) const
    {
        return Shape{batch_size}.append(get_output_shape());
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

        input_shape = new_input_shape;

        biases.shape = { new_output_shape[0] };
        weights.shape = { new_input_shape.back(), new_output_shape[0] };

        set_activation_function(new_activation_function);

        set_batch_normalization(new_batch_normalization);

        const Index outputs_number = get_neurons_number();

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

        biases_device.set_descriptor({outputs_number});
        weights_device.set_descriptor({ new_input_shape.back(), outputs_number });

        if (batch_normalization)
        {
            Shape batch_normalization_shape = { outputs_number };

            betas_device.set_descriptor(batch_normalization_shape);
            gammas_device.set_descriptor(batch_normalization_shape);
            running_means_device.resize(batch_normalization_shape);
            running_variances_device.resize(batch_normalization_shape);
        }

#endif
    }

    void set_parameters_glorot() override
    {
        const type limit = sqrt(6.0 / (get_input_features_number() + get_neurons_number()));

        if(biases.size() > 0)
            VectorMap(biases.data, biases.size()).setZero();

        if(weights.size() > 0)
            set_random_uniform(VectorMap(weights.data, weights.size()), -limit, limit);

        if(batch_normalization && gammas.size() > 0)
            VectorMap(gammas.data, gammas.size()).setConstant(1.0);

        if(batch_normalization && betas.size() > 0)
            VectorMap(betas.data, betas.size()).setZero();
    }

    void set_parameters_random() override
    {
        if(biases.size() > 0)
            VectorMap(biases.data, biases.size()).setZero();

        if(weights.size() > 0)
            set_random_uniform(VectorMap(weights.data, weights.size()));

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
        if (new_input_shape.size() != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        input_shape = new_input_shape;

        const Index inputs_number = new_input_shape.back();
        const Index outputs_number = get_neurons_number();

        biases.shape = { outputs_number };
        weights.shape = { inputs_number, outputs_number };
    }


    void set_output_shape(const Shape& new_output_shape) override
    {
        if (new_output_shape.size() != 1)
            throw runtime_error("Output shape size is not 1");

        const Index inputs_number = get_input_features_number();
        const Index neurons_number = new_output_shape[0];

        biases.shape = { neurons_number };
        weights.shape = { inputs_number, neurons_number };
    }


    void set_activation_function(const string& new_activation_function)
    {
        static const unordered_set<string> activation_functions =
            {"Sigmoid", "HyperbolicTangent", "Linear", "RectifiedLinear", "ScaledExponentialLinear", "Softmax","Logistic"};

        if (activation_functions.count(new_activation_function))
            if (get_output_shape()[0] == 1 && new_activation_function == "Softmax")
                activation_function = "Sigmoid";
            else
                activation_function = new_activation_function;
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

        if (activation_function == "Softmax")
            use_combinations = batch_normalization;
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

#ifdef OPENNN_CUDA

        if (activation_function == "Softmax")
            use_combinations = batch_normalization;
#endif
    }


    void normalization(Tensor1& means, Tensor1& standard_deviations, const Tensor2& inputs, Tensor2& outputs) const
    {
        const Index batch_size = inputs.dimension(0);
        const Index features_number = inputs.dimension(1);

        const array<int, 1> reduction_axis({0});

        const array<Index, 2> reshape_dims({1, features_number});
        const array<Index, 2> broadcast_dims({batch_size, 1});

        means.device(get_device()) = inputs.mean(reduction_axis);

        standard_deviations.device(get_device()) = (inputs - means.reshape(reshape_dims).broadcast(broadcast_dims))
                                                    .square()
                                                    .mean(reduction_axis)
                                                    .sqrt();

        outputs.device(get_device()) = (inputs - means.reshape(reshape_dims).broadcast(broadcast_dims)) /
                                       (standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims) + EPSILON);

        if (batch_normalization && gammas.data != nullptr && betas.data != nullptr)
        {
            const MatrixMap gammas_map(gammas.data, 1, features_number);
            const MatrixMap betas_map(betas.data, 1, features_number);

            TensorMap2 g(gammas.data, 1, features_number);
            TensorMap2 b(betas.data, 1, features_number);

            outputs.device(get_device()) = outputs * g.broadcast(broadcast_dims) + b.broadcast(broadcast_dims);
        }
    }


    void forward_propagate(unique_ptr<LayerForwardPropagation>& forward_propagation,
                           bool is_training) override
    {
        DenseForwardPropagation<Rank>* dense_forward_propagation = static_cast<DenseForwardPropagation<Rank>*>(forward_propagation.get());

        TensorMapR<Rank> outputs = tensor_map<Rank>(dense_forward_propagation->outputs);

        calculate_combinations<Rank>(tensor_map<Rank>(forward_propagation->inputs[0]),
                                     matrix_map(weights),
                                     vector_map(biases),
                                     outputs);
        if(batch_normalization)
        {
            TensorMapR<Rank> normalized_outputs = tensor_map<Rank>(dense_forward_propagation->normalized_outputs);

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

            outputs = normalized_outputs;
        }

        TensorMapR<Rank> derivatives = [&]()
        {
            if constexpr (Rank == 2)
                return is_training
                   ? tensor_map<Rank>(dense_forward_propagation->activation_derivatives)
                   : TensorMap2(nullptr, 0, 0);
            else
                return is_training
                   ? tensor_map<Rank>(dense_forward_propagation->activation_derivatives)
                   : TensorMap3(nullptr, 0, 0, 0);
        }();

        calculate_activations<Rank>(activation_function, outputs, derivatives);

        if(is_training && dropout_rate > type(0))
            dropout<Rank>(outputs, dropout_rate);
    }


    void back_propagate(unique_ptr<LayerForwardPropagation>& forward_propagation,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        const Index inputs_number = get_input_features_number();
        const Index outputs_number = get_neurons_number();
        const Index total_rows = forward_propagation->inputs[0].size() / inputs_number;

        const MatrixMap inputs(forward_propagation->inputs[0].data, total_rows, inputs_number);
        const MatrixMap output_gradients(back_propagation->output_gradients[0].data, total_rows, outputs_number);

        const DenseForwardPropagation<Rank>* dense_forward_propagation = static_cast<const DenseForwardPropagation<Rank>*>(forward_propagation.get());

        DenseBackPropagation<Rank>* dense_back_propagation = static_cast<DenseBackPropagation<Rank>*>(back_propagation.get());

        // @todo generating memory here

        MatrixR delta;

        if(activation_function != "Softmax")
        {
            const MatrixMap activation_derivatives(dense_forward_propagation->activation_derivatives.data, total_rows, outputs_number);
            delta = output_gradients.array() * activation_derivatives.array();
        }
        else
            delta = output_gradients;

        if(batch_normalization)
        {
            const MatrixMap normalized_outputs(dense_forward_propagation->normalized_outputs.data, total_rows, outputs_number);

            VectorMap gamma_gradients(dense_back_propagation->gamma_gradients.data, dense_back_propagation->gamma_gradients.size());
            VectorMap beta_gradients(dense_back_propagation->beta_gradients.data, dense_back_propagation->beta_gradients.size());

            beta_gradients.array() = delta.colwise().sum();
            gamma_gradients.array() = (delta.array() * normalized_outputs.array()).colwise().sum();
        }

        MatrixMap weight_gradients(dense_back_propagation->weight_gradients.data, inputs_number, outputs_number);
        VectorMap bias_gradients(dense_back_propagation->bias_gradients.data, outputs_number);

        weight_gradients.noalias() = inputs.transpose() * delta;
        bias_gradients.noalias() = delta.colwise().sum();

        if(!is_first_layer)
        {
            MatrixMap input_gradients(back_propagation->input_gradients[0].data, total_rows, inputs_number);
            const MatrixMap weights_map(weights.data, inputs_number, outputs_number);

            input_gradients.noalias() = delta * weights_map.transpose();
        }
    }


    void back_propagate(unique_ptr<LayerForwardPropagation>& forward_propagation,
                        unique_ptr<LayerBackPropagationLM>& back_propagation) const override
    {
        const Index inputs_number = get_input_features_number();
        const Index outputs_number = get_neurons_number();
        const Index batch_size = forward_propagation->inputs[0].size() / inputs_number;
        const Index biases_number = biases.size();

        const MatrixMap inputs(forward_propagation->inputs[0].data, batch_size, inputs_number);
        MatrixMap output_gradients(back_propagation->output_gradients[0].data, batch_size, outputs_number);

        DenseForwardPropagation<Rank>* dense_fp = static_cast<DenseForwardPropagation<Rank>*>(forward_propagation.get());
        DenseBackPropagationLM* dense_bp_lm = static_cast<DenseBackPropagationLM*>(back_propagation.get());

        MatrixMap jacobian = matrix_map(dense_bp_lm->squared_errors_Jacobian);
        jacobian.setZero();

        if(activation_function != "Softmax")
        {
            const MatrixMap activation_derivatives(dense_fp->activation_derivatives.data, batch_size, outputs_number);
            output_gradients.array() *= activation_derivatives.array();
        }

        const Index total_error_terms = jacobian.rows();
        const Index network_outputs = (batch_size > 0) ? total_error_terms / batch_size : 1;

        for(Index j = 0; j < outputs_number; j++)
            for(Index k = 0; k < network_outputs; ++k)
                for(Index s = 0; s < batch_size; ++s)
                    jacobian(s * network_outputs + k, j) = output_gradients(s, j);

        for(Index i = 0; i < inputs_number; i++)
        {
            for(Index j = 0; j < outputs_number; j++)
            {
                const Index weight_col = biases_number + i * outputs_number + j;

                const VectorR interaction = output_gradients.col(j).array() * inputs.col(i).array();

                for(Index k = 0; k < network_outputs; ++k)
                    for(Index s = 0; s < batch_size; ++s)
                        jacobian(s * network_outputs + k, weight_col) = interaction(s);
            }
        }

        if(!is_first_layer)
        {
            MatrixMap input_gradients(dense_bp_lm->input_gradients[0].data, batch_size, inputs_number);
            const MatrixMap weights_map(weights.data, inputs_number, outputs_number);

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


    string get_expression(const vector<string>& new_input_names = vector<string>(),
                          const vector<string>& new_output_names = vector<string>()) const override
    {
        const vector<string> input_names = new_input_names.empty()
        ? get_default_feature_names()
        : new_input_names;

        const vector<string> output_names = new_output_names.empty()
                                                ? get_default_output_names()
                                                : new_output_names;

        const Index inputs_number = get_input_features_number();
        const Index outputs_number = get_neurons_number();

        if (biases.data == nullptr || weights.data == nullptr) return "";

        ostringstream buffer;

        for(Index j = 0; j < outputs_number; j++)
        {
            buffer << output_names[j] << " = " << activation_function << "( " << biases.data[j] << " + ";

            for(Index i = 0; i < inputs_number; i++)
            {
                const Index weight_index = i * outputs_number + j;

                buffer << "(" << weights.data[weight_index] << "*" << input_names[i] << ")";

                if (i < inputs_number - 1) buffer << " + ";
            }

            buffer << " );\n";
        }

        return buffer.str();
    }


    void print() const override
    {
        cout << "Dense layer" << endl
             << "Input shape: " << get_input_shape() << endl
             << "Output shape: " << get_output_shape() << endl
             << "Biases shape: " << biases.shape << endl
             << "Weights shape: " << weights.shape << endl
             << "Activation function: " << activation_function << endl
             << "Batch normalization: " << (batch_normalization ? "True" : "False") << endl
             << "Dropout rate: " << dropout_rate << endl;
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* dense2d_layer_element = document.FirstChildElement(name.c_str());

        if(!dense2d_layer_element)
            throw runtime_error(name + " element is nullptr.\n");

        set_label(read_xml_string(dense2d_layer_element, "Label"));

        const Index inputs_number = read_xml_index(dense2d_layer_element, "InputsNumber");
        const Index neurons_number = read_xml_index(dense2d_layer_element, "NeuronsNumber");

        if constexpr (Rank == 3)
        {
            const Index input_sequence_length = read_xml_index(dense2d_layer_element, "InputSequenceLength");
            set_input_shape({ input_sequence_length, inputs_number });
        }
        else
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
        if constexpr (Rank == 3)
        {
            add_xml_element(printer, "InputSequenceLength", to_string(get_input_shape()[0]));
            add_xml_element(printer, "InputsNumber", to_string(get_input_shape()[1]));
            add_xml_element(printer, "NeuronsNumber", to_string(get_output_shape()[1]));
        }
        else
        {
            add_xml_element(printer, "InputsNumber", to_string(get_input_shape()[0]));
            add_xml_element(printer, "NeuronsNumber", to_string(get_output_shape()[0]));
        }
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


    void forward_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation, bool is_training)
    {
        // Dense layer

        const Index inputs_number = get_input_features_number();
        const Index outputs_number = get_neurons_number();

        // Forward propagation

        const Index total_rows = forward_propagation->inputs[0].size() / inputs_number;

        TensorViewCuda& outputs = forward_propagation->outputs;

        DenseForwardPropagationCuda<Rank>* dense_forward_propagation = static_cast<DenseForwardPropagationCuda<Rank>*>(forward_propagation.get());

        type* combinations = dense_forward_propagation->combinations.data;
        type* outputs_buffer = use_combinations ? combinations : outputs.data;

        // Combinations

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 outputs_number, total_rows, inputs_number,
                                 &alpha,
                                 weights_device.data, outputs_number,
                                 forward_propagation->inputs[0].data, inputs_number,
                                 &beta,
                                 outputs_buffer, outputs_number));

        CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                                   &alpha,
                                   biases_device.get_descriptor(),
                                   biases_device.data,
                                   &beta_add,
                                   outputs.get_descriptor(),
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
                    EPSILON,
                    dense_forward_propagation->means.data,
                    dense_forward_propagation->inverse_variance.data));
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
                    EPSILON));

        // Activations

        if (activation_function == "Linear")
        {
            // Nothing
        }
        else if (activation_function == "Softmax")
        {
            CHECK_CUDNN(cudnnSoftmaxForward(get_cudnn_handle(),
                                            CUDNN_SOFTMAX_ACCURATE,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &alpha,
                                            outputs.get_descriptor(),
                                            outputs_buffer,
                                            &beta,
                                            outputs.get_descriptor(),
                                            outputs.data));
        }
        else
        {
            CHECK_CUDNN(cudnnActivationForward(get_cudnn_handle(),
                                               activation_descriptor,
                                               &alpha,
                                               outputs.get_descriptor(),
                                               outputs_buffer,
                                               &beta,
                                               outputs.get_descriptor(),
                                               outputs.data));
        }

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


    void back_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                        unique_ptr<LayerBackPropagationCuda>& bp_cuda) const
    {
        // Dense layer

        const Index inputs_number = get_input_features_number();
        const Index outputs_number = get_neurons_number();

        // Forward propagation

        const Index total_rows = forward_propagation->inputs[0].size() / inputs_number;

        const TensorViewCuda& outputs_view = forward_propagation->outputs;

        const DenseForwardPropagationCuda<Rank>* dense_forward_propagation = static_cast<DenseForwardPropagationCuda<Rank>*>(forward_propagation.get());

        type* combinations = dense_forward_propagation->combinations.data;

        // Back propagation

        float* output_gradients_data = bp_cuda->output_gradients[0].data;

        DenseBackPropagationCuda<Rank>* dense_layer_back_propagation = static_cast<DenseBackPropagationCuda<Rank>*>(bp_cuda.get());

        float* ones = dense_layer_back_propagation->ones.data;

        float* bias_gradients = dense_layer_back_propagation->bias_gradients.data;
        float* weight_gradients = dense_layer_back_propagation->weight_gradients.data;

        const cudnnTensorDescriptor_t gradients_tensor_descriptor = dense_layer_back_propagation->gradients_tensor_descriptor;

        // Dropout

        if (get_dropout_rate() > type(0) && activation_function != "Softmax")
        {
            CHECK_CUDNN(cudnnDropoutBackward(get_cudnn_handle(),
                                             dense_forward_propagation->dropout_descriptor,
                                             gradients_tensor_descriptor,
                                             output_gradients_data,
                                             gradients_tensor_descriptor,
                                             output_gradients_data,
                                             dense_forward_propagation->dropout_reserve_space,
                                             dense_forward_propagation->dropout_reserve_space_size));
        }

        // Error combinations derivatives

        if (activation_function != "Linear" && activation_function != "Softmax" && use_combinations)
        {
            CHECK_CUDNN(cudnnActivationBackward(get_cudnn_handle(),
                                                activation_descriptor,
                                                &alpha,
                                                gradients_tensor_descriptor,
                                                outputs_view.data,
                                                gradients_tensor_descriptor,
                                                output_gradients_data,
                                                gradients_tensor_descriptor,
                                                combinations,
                                                &beta,
                                                gradients_tensor_descriptor,
                                                output_gradients_data));
        }
        else if (activation_function != "Linear" && activation_function != "Softmax" && !use_combinations)
        {
            CHECK_CUDNN(cudnnActivationBackward(get_cudnn_handle(),
                                                activation_descriptor,
                                                &alpha,
                                                gradients_tensor_descriptor,
                                                outputs_view.data,
                                                gradients_tensor_descriptor,
                                                output_gradients_data,
                                                gradients_tensor_descriptor,
                                                outputs_view.data,
                                                &beta,
                                                gradients_tensor_descriptor,
                                                output_gradients_data));
        }

        // Batch Normalization

        if (batch_normalization)
        {
            CHECK_CUDNN(cudnnBatchNormalizationBackward(
                get_cudnn_handle(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha, &beta,
                &alpha, &beta,
                dense_forward_propagation->outputs.get_descriptor(),
                use_combinations ? combinations : outputs_view.data,
                gradients_tensor_descriptor,
                output_gradients_data,
                gradients_tensor_descriptor,
                output_gradients_data,
                gammas_device.get_descriptor(),
                gammas_device.data,
                dense_layer_back_propagation->gamma_gradients.data,
                dense_layer_back_propagation->beta_gradients.data,
                EPSILON,
                dense_forward_propagation->means.data,
                dense_forward_propagation->inverse_variance.data));
        }

        // Bias derivatives

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 outputs_number, 1, total_rows,
                                 &alpha,
                                 output_gradients_data, outputs_number,
                                 ones, total_rows,
                                 &beta,
                                 bias_gradients, outputs_number));

        // Weight derivatives

        CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 outputs_number, inputs_number, total_rows,
                                 &alpha,
                                 output_gradients_data, outputs_number,
                                 forward_propagation->inputs[0].data, inputs_number,
                                 &beta,
                                 weight_gradients, outputs_number));

        // Input derivatives

        if (!is_first_layer)
        {
            float* input_gradients_data = bp_cuda->input_gradients[0].data;

            CHECK_CUBLAS(cublasSgemm(get_cublas_handle(),
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     inputs_number, total_rows, outputs_number,
                                     &alpha,
                                     weights_device.data, outputs_number,
                                     output_gradients_data, outputs_number,
                                     &beta,
                                     input_gradients_data, inputs_number));
        }
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

    Shape input_shape;

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
