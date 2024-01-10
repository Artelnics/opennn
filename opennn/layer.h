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
#include <string>
#include <sstream>
#include <iostream>

// OpenNN includes

#include "config.h"


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
                    Pooling,
                    Probabilistic,
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

    virtual void forward_propagate(const pair<type*, dimensions>&,
                                   Tensor<type, 1>&,
                                   LayerForwardPropagation*);

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

    void binary(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void exponential_linear(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void hard_sigmoid(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void hyperbolic_tangent(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void linear(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void logistic(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void rectified_linear(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void leaky_rectified_linear(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void scaled_exponential_linear(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void soft_plus(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void soft_sign(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void symmetric_threshold(const Tensor<type, 1>&, Tensor<type, 1>&) const;
    void threshold(const Tensor<type, 1>&, Tensor<type, 1>&) const;

    void exponential_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void hard_sigmoid_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void logistic_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void rectified_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void leaky_rectified_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void soft_plus_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void soft_sign_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;

    void binary(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void exponential_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hard_sigmoid(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hyperbolic_tangent(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void logistic(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void rectified_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void leaky_rectified_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void scaled_exponential_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_plus(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_sign(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void symmetric_threshold(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void threshold(const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hard_sigmoid_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void rectified_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void leaky_rectified_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_plus_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_sign_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;

    void binary(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void exponential_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hard_sigmoid(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hyperbolic_tangent(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void logistic(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void rectified_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void leaky_rectified_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void scaled_exponential_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_plus(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_sign(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void symmetric_threshold(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void threshold(const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void exponential_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hard_sigmoid_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void logistic_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void rectified_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void leaky_rectified_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_plus_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_sign_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/layer_cuda.h"
#else
};
#endif
}

#endif // LAYER_H
