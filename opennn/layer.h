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

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <ctype.h>
#include <iostream>
#include <vector>

// OpenNN includes

#include "config.h"
//#include "tensor_utilities.h"
#include "statistics.h"
#include "data_set.h"

using namespace std;
using namespace Eigen;

namespace OpenNN {

class Layer;

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/struct_layer_cuda.h"
#endif


/// This abstract class represents the concept of layer of neurons in OpenNN.

/// Layer is a group of neurons having connections to the same inputs and sending outputs to the same destinations.
/// Also is used to store information about the layers of the different architectures of NeuralNetworks.

class Layer
{

public:

    // Enumerations

    /// This enumeration represents the possible types of layers.

    enum Type{Scaling, Convolutional, Perceptron, Pooling, Probabilistic,
              LongShortTermMemory,Recurrent, Unscaling, Bounding};

    // Constructor

    explicit Layer()   
    {
        const int n = omp_get_max_threads();

        non_blocking_thread_pool = new NonBlockingThreadPool(n);
        thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);
    }


    // Destructor

    virtual ~Layer();

    string get_name() const
    {
        return layer_name;
    }

    // Parameters initialization methods

    virtual void set_parameters_constant(const type&);

    virtual void set_parameters_random();

    virtual void set_synaptic_weights_glorot();

    // Architecture

    virtual Tensor<type, 1> get_parameters() const;
    virtual Index get_parameters_number() const;

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_threads_number(const int&);

    virtual void insert_gradient(LayerBackPropagation*, const Index&, Tensor<type, 1>&) const {}

    // Outputs

    virtual Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&); // Cannot be const because of Recurrent and LSTM layers

    virtual Tensor<type, 2> calculate_outputs_from4D(const Tensor<type, 4>&) {return Tensor<type, 2>();}

    virtual Tensor<type, 4> calculate_outputs_4D(const Tensor<type, 4>&) {return Tensor<type, 4>();}

    virtual void forward_propagate(const Tensor<type, 2>&, LayerForwardPropagation*) {} // Cannot be const because of Recurrent and LSTM layers
    virtual void forward_propagate(const Tensor<type, 4>&, LayerForwardPropagation*) {}

    virtual void forward_propagate(const Tensor<type, 4>&, Tensor<type, 1>, LayerForwardPropagation*) {}
    virtual void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>, LayerForwardPropagation*) {} // Cannot be const because of Recurrent and LSTM layers

    // Deltas

    virtual void calculate_hidden_delta(LayerForwardPropagation*,
                                        LayerBackPropagation*,
                                        LayerBackPropagation*) const {}

    virtual void calculate_hidden_delta(LayerForwardPropagation*,
                                        LayerBackPropagationLM*,
                                        LayerBackPropagationLM*) const {}

    // Error gradient

    virtual void calculate_error_gradient(const Tensor<type, 2>&,
                                          LayerForwardPropagation*,
                                          LayerBackPropagation*) const {}

    virtual void calculate_error_gradient(const Tensor<type, 4>&,
                                          LayerForwardPropagation*,
                                          LayerBackPropagation*) const {}

    // Squared errors

    virtual void calculate_squared_errors_Jacobian(const Tensor<type, 2>&,
                                                   LayerForwardPropagation*,
                                                   LayerBackPropagationLM*) {}

    virtual void insert_squared_errors_Jacobian(LayerBackPropagationLM*,
                                                const Index&,
                                                Tensor<type, 2>&) const {}

    // Get neurons number

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;
    virtual Index get_synaptic_weights_number() const;
    virtual void set_inputs_number(const Index&);
    virtual void set_neurons_number(const Index&);

    // Layer type

    Type get_type() const;

    string get_type_string() const;

    // Serialization methods

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void write_XML(tinyxml2::XMLPrinter&) const {}

    // Expression methods

    virtual string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const {return string();}

    virtual string write_expression_c() const {return string();}

    virtual string write_expression_python() const {return string();}

protected:

    NonBlockingThreadPool* non_blocking_thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    /// Layer name.

    string layer_name = "layer";

    /// Layer type.

    Type layer_type = Perceptron;

    // activations 1d (Time Series)

    void hard_sigmoid(const Tensor<type,1>&, Tensor<type,1>&) const;
    void hyperbolic_tangent(const Tensor<type,1>&, Tensor<type,1>&) const;
    void logistic(const Tensor<type,1>&, Tensor<type,1>&) const;
    void linear(const Tensor<type,1>&, Tensor<type,1>&) const;
    void threshold(const Tensor<type,1>&, Tensor<type,1>&) const;
    void symmetric_threshold(const Tensor<type,1>&, Tensor<type,1>&) const;
    void rectified_linear(const Tensor<type,1>&, Tensor<type,1>&) const;
    void scaled_exponential_linear(const Tensor<type,1>&, Tensor<type,1>&) const;
    void soft_plus(const Tensor<type,1>&, Tensor<type,1>&) const;
    void soft_sign(const Tensor<type,1>&, Tensor<type,1>&) const;
    void exponential_linear(const Tensor<type,1>&, Tensor<type,1>&) const;
    void softmax(const Tensor<type,1>&, Tensor<type,1>&) const;
    void binary(const Tensor<type,1>&, Tensor<type,1>&) const;
    void competitive(const Tensor<type,1>&, Tensor<type,1>&) const;

    void hard_sigmoid_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void logistic_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void threshold_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void symmetric_threshold_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void rectified_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void soft_plus_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void soft_sign_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
    void exponential_linear_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;

    // activations 2d

    void hard_sigmoid(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hyperbolic_tangent(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void logistic(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void threshold(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void symmetric_threshold(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void rectified_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void scaled_exponential_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_plus(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_sign(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void exponential_linear(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void softmax(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void binary(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void competitive(const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void hard_sigmoid_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void symmetric_threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void rectified_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_plus_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_sign_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;

    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 3>&) const;
    void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 3>&) const;

    // activations 4d

    void linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void logistic(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hyperbolic_tangent(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void threshold(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void symmetric_threshold(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void rectified_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void scaled_exponential_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_plus(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_sign(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hard_sigmoid(const Tensor<type, 4>&, Tensor<type, 4>&) const;
    void exponential_linear(const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void logistic_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void threshold_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void symmetric_threshold_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void rectified_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_plus_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void soft_sign_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void hard_sigmoid_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;
    void exponential_linear_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/layer_cuda.h"
#else
};
#endif
struct LayerForwardPropagation
{
    /// Default constructor.

    explicit LayerForwardPropagation()
    {
    }

    virtual ~LayerForwardPropagation() {}

    virtual void set(const Index&, Layer*) {}

    virtual void print() const = 0;

    Index batch_samples_number = 0;

    Layer* layer_pointer = nullptr;
};


struct LayerBackPropagation
{
    /// Default constructor.

    explicit LayerBackPropagation() {}

    virtual ~LayerBackPropagation() {}

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer_pointer = nullptr;
};


struct LayerBackPropagationLM
{
    /// Default constructor.

    explicit LayerBackPropagationLM() {}

    virtual ~LayerBackPropagationLM() {}

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer_pointer = nullptr;
};

}

#endif // LAYER_H
