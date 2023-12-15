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
#include "tensor_utilities.h"
#include "dynamic_tensor.h"
#include "statistics.h"
#include "scaling.h"
//#include "data_set.h"

#include <tuple>


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

    // Enumerations

    /// This enumeration represents the possible types of layers.

    enum class Type{Scaling,
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

    string get_output_shape() const
    {
        Tensor<Index, 1> output_dimensions = get_outputs_dimensions();

        stringstream output_shape_string;

        output_shape_string << "(";

        for(Index i = 0; i < output_dimensions.size(); i++)
        {
            output_shape_string << output_dimensions[i];
            if(i != output_dimensions.size() - 1)
            {
                output_shape_string << ", ";
            }
        }
        output_shape_string << ")";

        return output_shape_string.str();
    }

    virtual Tensor<Index,  1> get_inputs_dimensions() const {return Tensor<Index,1>();}
    virtual Tensor<Index,  1> get_outputs_dimensions() const {return Tensor<Index,1>();}

    // Parameters initialization methods

    virtual void set_parameters_constant(const type&);

    virtual void set_parameters_random();

    // Architecture

    virtual Index get_parameters_number() const;
    virtual Tensor<type, 1> get_parameters() const;

    virtual Tensor< TensorMap< Tensor<type, 1>>*, 1> get_layer_parameters();

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_threads_number(const int&);

    virtual void insert_gradient(LayerBackPropagation*, const Index&, Tensor<type, 1>&) const {}

    // Outputs

    virtual void forward_propagate(const Tensor<DynamicTensor<type>, 1>&,
                                   LayerForwardPropagation*, const bool&) = 0;

    virtual void forward_propagate(const Tensor<DynamicTensor<type>, 1>&,
                                   Tensor<type, 1>&, LayerForwardPropagation*);

    // Deltas

    virtual void calculate_hidden_delta(LayerForwardPropagation*,
                                        LayerBackPropagation*,
                                        LayerBackPropagation*) const {}

    virtual void calculate_hidden_delta_lm(LayerForwardPropagation*,
                                           LayerBackPropagationLM*,
                                           LayerBackPropagationLM*) const {}

    // Jacobian

    virtual void calculate_inputs_outputs_derivatives(LayerForwardPropagation*) const {}


    // Error gradient

    virtual void calculate_error_gradient(type*,
                                          LayerForwardPropagation*,
                                          LayerBackPropagation*) const {}

    // Squared errors

    virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                                      LayerForwardPropagation*,
                                                      LayerBackPropagationLM*) {}

    virtual void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    // Get neurons number

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;

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

protected:

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    /// Layer name.

    string layer_name = "layer";

    /// Layer type.

    Type layer_type;

    /// Activation functions

    void binary(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void competitive(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void exponential_linear(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void hard_sigmoid(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void hyperbolic_tangent(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void linear(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void logistic(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void rectified_linear(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void leaky_rectified_linear(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void scaled_exponential_linear(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void softmax(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void soft_plus(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void soft_sign(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void symmetric_threshold(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void threshold(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;

    void exponential_linear_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void hard_sigmoid_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void hyperbolic_tangent_derivatives(type*, const Tensor<Index, 1>&,type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void linear_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void logistic_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void rectified_linear_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void leaky_rectified_linear_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void scaled_exponential_linear_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void softmax_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void soft_plus_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void soft_sign_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void symmetric_threshold_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;
    void threshold_derivatives(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) const;

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/layer_cuda.h"
#else
};
#endif

struct LayerBackPropagationLM
{
    /// Default constructor.

    explicit LayerBackPropagationLM() {}

    virtual ~LayerBackPropagationLM() {}

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number;

    Layer* layer_pointer = nullptr;

    Tensor<type, 2> deltas;
};

}

#endif // LAYER_H
