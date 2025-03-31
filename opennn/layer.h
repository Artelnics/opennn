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
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

using namespace tinyxml2;

namespace opennn
{

//#ifdef OPENNN_CUDA
//struct LayerForwardPropagationCuda;
//struct LayerBackPropagationCuda;
//#endif

class Layer
{

public:

    enum class Type{None,
                    Scaling2D,
                    Scaling4D,
                    Addition3D,
                    Normalization3D,
                    Convolutional,
                    Perceptron,
                    Perceptron3d,
                    Pooling,
                    Probabilistic,
                    Probabilistic3d,
                    LongShortTermMemory,
                    Recurrent,
                    Unscaling,
                    Bounding,
                    Flatten,
                    Flatten3D,
                    NonMaxSuppression,
                    MultiheadAttention,
                    Embedding};

    Layer();

    virtual ~Layer() {}

    string get_name() const;

    const bool& get_display() const;

    string layer_type_to_string(const Layer::Type&);
    Type string_to_layer_type(const string&);

    Type get_type() const;

    string get_type_string() const;

    virtual void set_input_dimensions(const dimensions&);
    virtual void set_output_dimensions(const dimensions&);

    void set_name(const string&);

    void set_display(const bool&);

    virtual void set_parameters_constant(const type&);
    virtual void set_parameters_random();

    virtual Index get_parameters_number() const;
    virtual Tensor<type, 1> get_parameters() const;

    virtual dimensions get_input_dimensions() const = 0;
    virtual dimensions get_output_dimensions() const = 0;

    Index get_inputs_number() const;

    Index get_outputs_number() const;

    virtual void set_parameters(const Tensor<type, 1>&, Index&);

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const vector<pair<type*, dimensions>>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   const bool&) = 0;

    // Back propagation

    virtual void back_propagate(const vector<pair<type*, dimensions>>&,
                                const vector<pair<type*, dimensions>>&,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>&) const {}

    virtual void back_propagate_lm(const vector<pair<type*, dimensions>>&,
                                   const vector<pair<type*, dimensions>>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   unique_ptr<LayerBackPropagationLM>&) const {}

    virtual void insert_gradient(unique_ptr<LayerBackPropagation>&,
                                 Index&,
                                 Tensor<type, 1>&) const {}

    virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                                      unique_ptr<LayerForwardPropagation>&,
                                                      unique_ptr<LayerBackPropagationLM>&) {}

    virtual void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void to_XML(tinyxml2::XMLPrinter&) const {}

    virtual string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const;

    virtual void print() const {}

    vector<string> get_default_input_names() const;

    vector<string> get_default_output_names() const;

protected:

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    string name = "layer";

    Type layer_type = Type::None;

    Tensor<type, 2> empty_2;
    Tensor<type, 3> empty_3;
    Tensor<type, 4> empty_4;


    bool display = true;

    template <int Rank>
    void binary(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx, type threshold) const
    {
        y.device(*thread_pool_device) = (y < threshold).select(type(0), type(1));

        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(0));
    }


    template <int Rank>
    void linear(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
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
    void hard_sigmoid(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = ((y*type(0.2) + type(0.5)).cwiseMin(type(2.5)).cwiseMax(type(-2.5))).eval();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device)
            = (y > type(0) && y < type(1)).select(dy_dx.constant(type(0.2)), dy_dx.constant(type(0)));
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


    template <int Rank>
    void soft_plus(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (type(1) + y.exp()).log();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = type(1) - (-y).exp();
    }


    template <int Rank>
    void soft_sign(Tensor<type, Rank>& y, Tensor<type, Rank>& dy_dx) const
    {
        Tensor<type, Rank> x = y;

        y.device(*thread_pool_device) = (x / (1 + x.abs())).eval();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (type(1)/ (type(1) + x.abs()).pow(type(2)));
    }

    void competitive(Tensor<type, 2>&) const;

    void softmax(Tensor<type, 2>&) const;
    void softmax(Tensor<type, 3>&) const;
    void softmax(Tensor<type, 4>&) const;

    void softmax_derivatives_times_tensor(const Tensor<type, 3>&, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, Tensor<type, 1>&) const;
    void softmax_derivatives_times_tensor(const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, Tensor<type, 1>&) const;

    void add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
    {
        TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

        for (Index i = 1; i < Index(delta_pairs.size()); i++)
            deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
    }


    template <int Rank>
    void dropout(Tensor<type, Rank>& tensor, const type& dropout_rate) const
    {
        const type scaling_factor = type(1) / (type(1) - dropout_rate);

        #pragma omp parallel
        {
            mt19937 gen(random_device{}() + omp_get_thread_num());  // thread-local RNG
            uniform_real_distribution<float> dis(0.0f, 1.0f);

            #pragma omp parallel for
            for (Index i = 0; i < tensor.size(); i++)
                tensor(i) = (dis(gen) < dropout_rate)
                ? 0
                : tensor(i) * scaling_factor;
        }
    }

    const Eigen::array<IndexPair<Index>, 1> A_B = { IndexPair<Index>(1, 0) };
    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};


#ifdef OPENNN_CUDA

protected:

        cublasHandle_t cublas_handle = nullptr;
        cudnnHandle_t cudnn_handle = nullptr;

        cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
        cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;

public:

    virtual ~Layer() = default;

    void create_cuda();
    void destroy_cuda();

    cudnnHandle_t get_cudnn_handle();

    virtual void allocate_parameters_device();
    virtual void free_parameters_device();
    virtual void copy_parameters_device();

    virtual void forward_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                                        LayerForwardPropagationCuda*,
                                        const bool&);

    virtual void copy_parameters_host();
    virtual void print_parameters_cuda();

    virtual void back_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                                     const Tensor<pair<type*, dimensions>, 1>&,
                                     LayerForwardPropagationCuda*,
                                     LayerBackPropagationCuda*) const;

    virtual void insert_gradient_cuda(LayerBackPropagationCuda*, const Index&, float*) const;

    virtual void set_parameters_cuda(const float*, const Index&);

    virtual void get_parameters_cuda(Tensor<type, 1>&, const Index&);

    virtual string get_type_string() const = 0;
#endif

};


#ifdef OPENNN_CUDA

struct LayerForwardPropagationCuda
{
    explicit LayerForwardPropagationCuda();

    explicit LayerForwardPropagationCuda(const Index&, Layer*);

    virtual ~LayerForwardPropagationCuda();

    virtual void set(const Index&, Layer*) = 0;

    virtual void free() = 0;

    virtual void print() const;

    virtual pair<type*, dimensions> get_outputs_pair() const = 0;

    Index batch_samples_number = 0;

    Layer* layer = nullptr;

    type* outputs = nullptr;

    cudnnTensorDescriptor_t outputs_tensor_descriptor = nullptr;
};


struct LayerBackPropagationCuda
{
    explicit LayerBackPropagationCuda();

    explicit LayerBackPropagationCuda(const Index&, Layer*);

    virtual ~LayerBackPropagationCuda();

    virtual void set(const Index&, Layer*) = 0;

    virtual void free() = 0;

    virtual void print() const;

    Tensor<pair<type*, dimensions>, 1>& get_inputs_derivatives_pair_device();

    Index batch_samples_number = 0;

    Layer* layer = nullptr;

    type* inputs_derivatives = nullptr;

    Tensor<pair<type*, dimensions>, 1> inputs_derivatives_pair_device;

    cudnnTensorDescriptor_t inputs_derivatives_tensor_descriptor = nullptr;
};

#endif

}

#endif // LAYER_H
