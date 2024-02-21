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

#include <iostream>
#include <iostream>
#include <sstream>
#include <string>

// OpenNN includes

#include "config.h"
#include "tensors.h"


namespace opennn {

class Layer;

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_layer_cuda.h"
#endif


/// This abstract class represents the concept of layer of neurons in OpenNN.

/// A layer is a group of neurons having connections to the same inputs and sending outputs to the same destinations.

class Layer
{

public:

    /// This enumeration represents the possible types of layers.

    enum class Type{Scaling2D,
                    Scaling4D,
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
        layer_type = Layer::Type::Perceptron;

        const int n = omp_get_max_threads();

        thread_pool = new ThreadPool(n);
        thread_pool_device = new ThreadPoolDevice(thread_pool, n);
    }

    // Destructor

    virtual ~Layer();

    string get_name() const
    {
        return layer_name;
    }

    // Get neurons number

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;

    virtual void set_inputs_number(const Index&);
    virtual void set_neurons_number(const Index&);

    // Layer type

    Type get_type() const;

    string get_type_string() const;

    // Parameters initialization methods

    virtual void set_parameters_constant(const type&);

    virtual void set_parameters_random();

    // Architecture

    virtual Index get_parameters_number() const;
    virtual Tensor<type, 1> get_parameters() const;

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const pair<type*, dimensions>&,
                                   LayerForwardPropagation*,
                                   const bool&) = 0;

    // Back propagation

    virtual void calculate_hidden_delta(LayerForwardPropagation*,
                                        LayerBackPropagation*,
                                        LayerBackPropagation*) const {}

    virtual void calculate_hidden_delta_lm(LayerForwardPropagation*,
                                           LayerBackPropagationLM*,
                                           LayerBackPropagationLM*) const {}

    virtual void calculate_error_gradient(const pair<type*, dimensions>&,
                                          LayerForwardPropagation*,
                                          LayerBackPropagation*) const {}

    virtual void insert_gradient(LayerBackPropagation*,
                                 const Index&,
                                 Tensor<type, 1>&) const {}

    virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                                      LayerForwardPropagation*,
                                                      LayerBackPropagationLM*) {}

    virtual void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    // Serialization methods

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void write_XML(tinyxml2::XMLPrinter&) const {}

    // Expression methods

    virtual string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const {return string();}

protected:

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    /// Layer name.

    string layer_name = "layer";

    /// Layer type.

    Type layer_type;

    /// Activation functions

    template <int rank>
    void linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x;
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
    void exponential_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {

        y.device(*thread_pool_device) = (x > 0).select(x, x.exp() - type(1));
    }


    template <int rank>
    void hard_sigmoid(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = (x*type(0.2) + type(0.5)).cwiseMin(type(2.5)).cwiseMax(type(-2.5));
    }


    template <int rank>
    void hyperbolic_tangent(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x.tanh();
    }


    template <int rank>
    void logistic(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = type(1)/(type(1) + x.exp().inverse());
    }


    template <int rank>
    void rectified_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x.cwiseMax(type(0));
    }


    template <int rank>
    void leaky_rectified_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {

    }


    template <int rank>
    void scaled_exponential_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1);

        y.device(*thread_pool_device) = (x > 0).select(lambda * x, lambda * alpha * (x.exp() - type(1)));
    }


    template <int rank>
    void soft_plus(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = (type(1) + x.exp()).log();
    }


    template <int rank>
    void soft_sign(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x / (1 + x.abs());
    }


    template <int rank>
    void symmetric_threshold(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        Tensor<type, 1> ones(x.dimension(0));
        ones.setConstant(type(1));

        y.device(*thread_pool_device) = (x > 0).select(ones, -ones);
    }


    template <int rank>
    void threshold(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        Tensor<type, 1> ones(x.dimension(0));
        ones.setConstant(type(1));

        Tensor<type, 1> zeros(x.dimension(0));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = (x >= 0).select(ones, zeros);
    }


    void competitive(const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y) const;

    void softmax(const Tensor<type, 3>& x, Tensor<type, 3>& y) const;

    void softmax(const Tensor<type, 4>& x, Tensor<type, 4>& y) const;

    template <int rank>
    void linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = x;

        dy_dx.setConstant(type(1));
    }


    template <int rank>
    void exponential_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        const type alpha = type(1);

        const Tensor<bool, rank> if_sentence = x < x.constant(type(0));

        Tensor<type, rank> f_1 = alpha*(x.exp() - type(1));

        // Activations

        y.device(*thread_pool_device) = if_sentence.select(f_1, x);

        // Activations Derivatives

        f_1 = alpha * x.exp();

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, x.constant(type(1)));
    }


    template <int rank>
    void hard_sigmoid_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (x*type(0.2) + type(0.5)).cwiseMax(type(0)).cwiseMin(type(1));

        dy_dx.device(*thread_pool_device) = (y > type(0) && y < type(1)).select(dy_dx.constant(type(0.2)), dy_dx.constant(type(0)));
    }


    template <int rank>
    void hyperbolic_tangent_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = x.tanh();

        dy_dx.device(*thread_pool_device) = type(1) - y.square();
    }


    template <int rank>
    void logistic_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        logistic(x, y);

        dy_dx.device(*thread_pool_device) = y*(type(1) - y);
    }


    template <int rank>
    void rectified_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = x.cwiseMax(type(0));

        dy_dx.device(*thread_pool_device) = (y > 0).select(x.constant(type(1)), x.constant(type(0)));
    }


    template <int rank>
    void leaky_rectified_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {

    }


    template <int rank>
    void scaled_exponential_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1.67326);

        const Tensor<bool, rank> if_sentence = x < x.constant(type(0));

        Tensor<type, rank> f_1 = lambda*alpha*(x.exp()-type(1));

        Tensor<type, rank> f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        f_1 = lambda*alpha*x.exp();

        f_2 = x.constant(type(1))*lambda;

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }


    template <int rank>
    void soft_plus_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();

        dy_dx.device(*thread_pool_device) = type(1) / (type(1) + x.exp().inverse());
    }


    template <int rank>
    void soft_sign_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        const Tensor<bool, rank> if_sentence = x < x.constant(type(0));

        Tensor<type, rank> f_1 = x / (type(1) - x);

        Tensor<type, rank> f_2 = x / (type(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = type(1) / (type(1) - x).pow(type(2));

        f_2 = type(1) / (type(1) + x).pow(type(2));

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }


    /// @todo inefficient code

    void softmax_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 3>& dy_dx) const;

    void softmax_derivatives(const Tensor<type, 3>& x, Tensor<type, 3>& y, Tensor<type, 4>& dy_dx) const;

    void softmax_derivatives(const Tensor<type, 3>& y, Tensor<type, 4>& dy_dx) const;
    

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/layer_cuda.h"
#endif

};

}

#endif // LAYER_H
