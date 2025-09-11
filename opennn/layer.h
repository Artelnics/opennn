//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LAYER_H
#define LAYER_H

#include "tinyxml2.h"
#include "tensors.h"

using namespace tinyxml2;

namespace opennn
{

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

struct LayerForwardPropagationCuda;
struct LayerBackPropagationCuda;

class Layer
{

public:

    Layer();

    const string& get_label() const;

    const bool& get_display() const;

    const string& get_name() const;

    virtual void set_input_dimensions(const dimensions&);
    virtual void set_output_dimensions(const dimensions&);

    void set_label(const string&);

    void set_display(const bool&);

    virtual void set_parameters_random();

    virtual void set_parameters_glorot();

    Index get_parameters_number() const;

    virtual vector<ParameterView> get_parameter_views() const;

    //virtual pair

    virtual dimensions get_input_dimensions() const = 0;
    virtual dimensions get_output_dimensions() const = 0;

    Index get_inputs_number() const;

    Index get_outputs_number() const;

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const vector<TensorView>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   const bool&) = 0;

    // Back propagation

    virtual void back_propagate(const vector<TensorView>&,
                                const vector<TensorView>&,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>&) const {}

    virtual void back_propagate_lm(const vector<TensorView>&,
                                   const vector<TensorView>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   unique_ptr<LayerBackPropagationLM>&) const {}

    // virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
    //                                                   unique_ptr<LayerForwardPropagation>&,
    //                                                   unique_ptr<LayerBackPropagationLM>&) {}

    virtual void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void to_XML(tinyxml2::XMLPrinter&) const {}

    virtual string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const;

    virtual void print() const {}

    vector<string> get_default_input_names() const;

    vector<string> get_default_output_names() const;

    bool get_is_trainable() const;

protected:

    unique_ptr<ThreadPool> thread_pool = nullptr;
    unique_ptr<ThreadPoolDevice> thread_pool_device = nullptr;

    string label = "my_layer";

    string name = "layer";

    bool is_trainable = true;

    Tensor<type, 2> empty_2;
    Tensor<type, 3> empty_3;
    Tensor<type, 4> empty_4;

    bool display = true;

    template <int Rank>
    void calculate_activations(const string& activation_function,
                               Tensor<type, Rank>& activations,
                               Tensor<type, Rank>& activation_derivatives) const
    {
        if (activation_function == "Linear")
            linear(activations, activation_derivatives);
        else if (activation_function == "Logistic")
            logistic(activations, activation_derivatives);
        else if (activation_function == "Softmax")
            softmax(activations);
        else if (activation_function == "HyperbolicTangent")
            hyperbolic_tangent(activations, activation_derivatives);
        else if (activation_function == "RectifiedLinear")
            rectified_linear(activations, activation_derivatives);
        else if (activation_function == "ScaledExponentialLinear")
            exponential_linear(activations, activation_derivatives);
        else
            throw runtime_error("Unknown activation: " + activation_function);
    }


    template <int Rank>
    void binary(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx, type threshold) const
    {
        y.device(*thread_pool_device) = (y < threshold).select(type(0), type(1));

        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(0));
    }


    template <int Rank>
    void linear(Tensor<type, Rank>&, Tensor<type, Rank>& dy_dx) const
    {
        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(1));
    }


    template <int Rank>
    void exponential_linear(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        const type alpha = type(1);

        y.device(*thread_pool_device) = (y > type(0)).select(y, alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(type(1)), y + alpha);
    }


    template <int Rank>
    void hyperbolic_tangent(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = y.tanh();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (type(1) - y.square()).eval();
    }


    template <int Rank>
    void logistic(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (type(1) + (-y).exp()).inverse();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y * (type(1) - y)).eval();
    }


    template <int Rank>
    void rectified_linear(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = y.cwiseMax(type(0));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(0)));
    }


    template <int Rank>
    void leaky_rectified_linear(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx, type slope) const
    {
        y.device(*thread_pool_device) = (y > type(0)).select(y, slope * y);

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(slope)));
    }


    template <int Rank>
    void scaled_exponential_linear(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1.6733);

        y.device(*thread_pool_device) = (y > type(0)).select(lambda * y, lambda * alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(lambda), y + alpha * lambda);
    }

    void softmax(Tensor<type, 2>&) const;
    void softmax(Tensor<type, 3>&) const;
    void softmax(Tensor<type, 4>&) const;

    //void softmax_derivatives_times_tensor(const Tensor<type, 3>&, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, Tensor<type, 1>&) const;
    void softmax_derivatives_times_tensor(const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, Tensor<type, 1>&) const;

    void add_deltas(const vector<TensorView>& delta_views) const;

    template <int Rank>
    void dropout(Tensor<type, Rank>& tensor, const type& dropout_rate) const
    {
        const type scaling_factor = type(1) / (type(1) - dropout_rate);

#pragma omp parallel
        {
            mt19937 gen(random_device{}() + omp_get_thread_num());  // thread-local RNG
            uniform_real_distribution<float> dis(0.0f, 1.0f);

#pragma omp for
            for (Index i = 0; i < tensor.size(); i++)
                tensor(i) = (dis(gen) < dropout_rate)
                                ? 0
                                : tensor(i) * scaling_factor;
        }
    }

#ifdef OPENNN_CUDA

public:

    void create_cuda();
    void destroy_cuda();

    cudnnHandle_t get_cudnn_handle();

    virtual void forward_propagate_cuda(const vector<float*>&,
                                        unique_ptr<LayerForwardPropagationCuda>&,
                                        const bool&)
    {
        throw runtime_error("CUDA forward propagation not implemented for layer type: " + get_name());
    }

    virtual void back_propagate_cuda(const vector<float*>&,
                                     const vector<float*>&,
                                     unique_ptr<LayerForwardPropagationCuda>&,
                                     unique_ptr<LayerBackPropagationCuda>&) const {}

    virtual vector<ParameterView> get_parameter_views_device() const;

    virtual void copy_parameters_host() {}

    virtual void copy_parameters_device() {}

    virtual void allocate_parameters_device() {}

    virtual void free_parameters_device() {}

    virtual void print_parameters_cuda() {}

protected:

    cublasHandle_t cublas_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;

    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;

#endif

};


struct LayerForwardPropagation
{
    LayerForwardPropagation() {}

    virtual void set(const Index& = 0, Layer* = nullptr) = 0;

    virtual TensorView get_output_pair() const = 0;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;
};


struct LayerBackPropagation
{
    LayerBackPropagation() {}

    virtual void set(const Index& = 0, Layer* = nullptr) = 0;

    virtual vector<TensorView> get_input_derivative_views() const = 0;

    virtual vector<ParameterView> get_parameter_delta_views() const
    {
        return vector<ParameterView>();
    }

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;
};


struct LayerBackPropagationLM
{
    LayerBackPropagationLM() {}

    virtual vector<TensorView> get_input_derivative_views() const = 0;

    virtual void set(const Index& = 0, Layer* = nullptr) = 0;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;
};


#ifdef OPENNN_CUDA

struct LayerForwardPropagationCuda
{
    explicit LayerForwardPropagationCuda() {}

    virtual ~LayerForwardPropagationCuda() {}

    virtual void set(const Index& = 0, Layer* = nullptr) = 0;

    virtual void print() const {}

    virtual void free() {}

    virtual float* get_output_device() { return outputs; }

    Index batch_size = 0;

    Layer* layer = nullptr;

    float* outputs = nullptr;

    cudnnTensorDescriptor_t output_tensor_descriptor = nullptr;
};


struct LayerBackPropagationCuda
{
    LayerBackPropagationCuda() {}

    virtual void set(const Index& = 0, Layer* = nullptr) = 0;

    virtual void print() const {}

    virtual void free() {}

    virtual vector<float*> get_input_derivatives_device() { return {input_deltas}; }

    virtual vector<ParameterView> get_parameter_delta_views_device() const
    {
        return vector<ParameterView>();
    }

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    float* input_deltas = nullptr;

    cudnnTensorDescriptor_t input_derivatives_tensor_descriptor = nullptr;
};

#endif

}

#endif // LAYER_H
