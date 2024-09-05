//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LAYER_H
#define LAYER_H

// System includes

#include <string>
#include <memory>

// OpenNN includes

#include "config.h"
#include "tinyxml2.h"

namespace opennn
{

class Layer;

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

#ifdef OPENNN_CUDA
struct LayerForwardPropagationCuda;
struct LayerBackPropagationCuda;
#endif


class Layer
{

public:

    enum class Type{Scaling2D,
                    Scaling4D,
                    Addition3D,
                    Normalization3D,
                    Convolutional,
                    Perceptron,
                    Perceptron3D,
                    Pooling,
                    Probabilistic,
                    Probabilistic3D,
                    LongShortTermMemory,
                    Recurrent,
                    Unscaling,
                    Bounding,
                    Flatten,
                    RegionProposal,
                    NonMaxSuppression,
                    MultiheadAttention,
                    Embedding};

    // Constructor

    explicit Layer()
    {
        //layer_type = Layer::Type::Perceptron;

        const int n = omp_get_max_threads();

        thread_pool = new ThreadPool(n);
        thread_pool_device = new ThreadPoolDevice(thread_pool, n);
    }

    // Destructor

    ~Layer()
    {
        delete thread_pool;
        delete thread_pool_device;
    }

    string get_name() const;

    // Get neurons number

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;

    virtual void set_inputs_number(const Index&);
    virtual void set_neurons_number(const Index&);

    // Layer type

    Type get_type() const;

    string get_type_string() const;

    // Parameters initialization

    virtual void set_parameters_constant(const type&);
    virtual void set_parameters_random();

    // Architecture

    virtual Index get_parameters_number() const;
    virtual Tensor<type, 1> get_parameters() const;

    virtual dimensions get_output_dimensions() const;

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                                   LayerForwardPropagation*,
                                   const bool&) = 0;

    // Back propagation

    virtual void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                                const Tensor<pair<type*, dimensions>, 1>&,
                                LayerForwardPropagation*,
                                LayerBackPropagation*) const {}

    virtual void back_propagate_lm(const Tensor<pair<type*, dimensions>, 1>&,
                                   const Tensor<pair<type*, dimensions>, 1>&,
                                   LayerForwardPropagation*,
                                   LayerBackPropagationLM*) const {}

    virtual void insert_gradient(LayerBackPropagation*,
                                 const Index&,
                                 Tensor<type, 1>&) const {}

    virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                                      LayerForwardPropagation*,
                                                      LayerBackPropagationLM*) {}

    virtual void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    // Serialization

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void to_XML(tinyxml2::XMLPrinter&) const {}

    // Expression

    virtual string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const 
    {
        return string();
    }

protected:

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    string name = "layer";

    Type layer_type;

    template <int rank>
    void linear(Tensor<type, rank>&) const
    {
        // Do nothing
    }


    template <int rank>
    void binary(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {     
        const Tensor<bool, rank> if_sentence = x < x.constant(type(0.5));

        const Tensor<type, rank> f_1 = x.constant(type(false));

        const Tensor<type, rank> f_2 = x.constant(type(true));

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }


    template <int rank>
    void exponential_linear(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = (x > 0).select(x, x.exp() - type(1));
    }


    template <int rank>
    void hard_sigmoid(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = ((x*type(0.2) + type(0.5)).cwiseMin(type(2.5)).cwiseMax(type(-2.5))).eval();
    }


    template <int rank>
    void hyperbolic_tangent(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = x.tanh();
    }


    template <int rank>
    void logistic(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = (type(1) + (-x).exp()).inverse();
    }


    template <int rank>
    void rectified_linear(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = x.cwiseMax(type(0));
    }


    template <int rank>
    void leaky_rectified_linear(Tensor<type, rank>& x) const
    {

    }


    template <int rank>
    void scaled_exponential_linear(Tensor<type, rank>& x) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1);

        x.device(*thread_pool_device) = (x > 0).select(lambda * x, lambda * alpha * (x.exp() - type(1)));
    }


    template <int rank>
    void soft_plus(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = (type(1) + x.exp()).log();
    }


    template <int rank>
    void soft_sign(Tensor<type, rank>& x) const
    {
        x.device(*thread_pool_device) = x / (1 + x.abs());
    }


    void competitive(const Tensor<type, 2>&, Tensor<type, 2>&) const;
//    void competitive(const Tensor<type, 3>&, Tensor<type, 3>&) const;


    void softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 1>&) const;
    void softmax(const Tensor<type, 3>& x, Tensor<type, 3>& y) const;
    void softmax(const Tensor<type, 4>& x, Tensor<type, 4>& y) const;


    template <int rank>
    void linear_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        dy_dx.setConstant(type(1));
    }


    template <int rank>
    void exponential_linear_derivatives(Tensor<type, rank>& x, Tensor<type, rank>& dy_dx) const
    {
/*
        const type alpha = type(1);

        const Tensor<bool, rank> if_sentence = x < x.constant(type(0));

        Tensor<type, rank> f_1 = alpha*(x.exp() - type(1));

        // Activations

        y.device(*thread_pool_device) = if_sentence.select(f_1, x);

        // Activations Derivatives

        f_1 = alpha * x.exp();

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, x.constant(type(1)));
*/
    }


    template <int rank>
    void hard_sigmoid_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (y*type(0.2) + type(0.5)).cwiseMax(type(0)).cwiseMin(type(1));

        dy_dx.device(*thread_pool_device) = (y > type(0) && y < type(1)).select(dy_dx.constant(type(0.2)), dy_dx.constant(type(0)));
    }


    template <int rank>
    void hyperbolic_tangent_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = y.tanh();

        dy_dx.device(*thread_pool_device) = (type(1) - y.square()).eval();
    }


    template <int rank>
    void logistic_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        logistic(y);

        dy_dx.device(*thread_pool_device) = (y*(type(1) - y)).eval();
    }


    template <int rank>
    void rectified_linear_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {

        y.device(*thread_pool_device) = y.cwiseMax(type(0));

        dy_dx.device(*thread_pool_device) =  (y > type(0)).cast<type>();

    }


    template <int rank>
    void leaky_rectified_linear_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {

    }


    template <int rank>
    void scaled_exponential_linear_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
/*
        const type lambda = type(1.0507);

        const type alpha = type(1.67326);

        const Tensor<bool, rank> if_sentence = x < x.constant(type(0));

        Tensor<type, rank> f_1 = lambda*alpha*(x.exp()-type(1));

        Tensor<type, rank> f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        f_1 = lambda*alpha*x.exp();

        f_2 = x.constant(type(1))*lambda;

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }


    template <int rank>
    void soft_plus_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
/*
        y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();

        dy_dx.device(*thread_pool_device) = type(1) / (type(1) + x.exp().inverse());
*/
    }


    template <int rank>
    void soft_sign_derivatives(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
/*
        const Tensor<bool, rank> if_sentence = x < x.constant(type(0));

        Tensor<type, rank> f_1 = x / (type(1) - x);

        Tensor<type, rank> f_2 = x / (type(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = type(1) / (type(1) - x).pow(type(2));

        f_2 = type(1) / (type(1) + x).pow(type(2));

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }

    void softmax_derivatives_times_tensor(const Tensor<type, 3>&, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, Tensor<type, 1>&) const;

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/layer_cuda.h"
#endif

};


#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/layer_forward_propagation_cuda.h"
#include "../../opennn_cuda/opennn_cuda/layer_back_propagation_cuda.h"
#endif


}

#endif // LAYER_H
