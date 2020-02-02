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
#include "device.h"
#include "tinyxml2.h"

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
//#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

using namespace std;
using namespace Eigen;

namespace OpenNN {

/// This abstract class represents the concept of layer of neurons in OpenNN.

/// Layer is a group of neurons having connections to the same inputs and sending outputs to the same destinations.
/// Also is used to store information about the layers of the different architectures of NeuralNetworks.

class Layer
{

public:

    // Enumerations

    /// This enumeration represents the possible types of layers.

    enum Type{Scaling, Convolutional, Perceptron, Pooling, Probabilistic, LongShortTermMemory,Recurrent, Unscaling, Bounding, PrincipalComponents};

    /// This structure represents the first order activaions of layers.

    struct ForwardPropagation
    {
        /// Default constructor.

        explicit ForwardPropagation()
        {
        }

        virtual ~ForwardPropagation()
        {
        }

        void print() const
        {
            cout << "Combinations:" << endl;
            cout << combinations << endl;

            cout << "Activations:" << endl;
            cout << activations << endl;

            cout << "Activation derivatives:" << endl;
            cout << activations_derivatives << endl;
        }

        Tensor<type, 2> combinations;

        Tensor<type, 2> activations;

        Tensor<type, 2> activations_derivatives;

        Tensor<type, 3> activations_derivatives_3d;

    };

    // Constructor

    explicit Layer()
    {
    }

    // Destructor

    virtual ~Layer() {}

    // Parameters initialization methods

    virtual void set_parameters_constant(const type&);

    virtual void set_parameters_random();

    // Architecture

    virtual Tensor<type, 1> get_parameters() const;
    virtual Index get_parameters_number() const;

    virtual void set_parameters(const Tensor<type, 1>&);

    void set_device_pointer(Device*);

    // Outputs

    virtual Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
    virtual Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

    virtual Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

    virtual void calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, Tensor<type, 1>&) {}

    virtual ForwardPropagation calculate_forward_propagation(const Tensor<type, 2>&);

    virtual void calculate_forward_propagation(const Tensor<type, 2>&, ForwardPropagation&) {}

    // Deltas

    virtual Tensor<type, 2> calculate_output_delta(const Tensor<type, 2>& activations_derivatives, const Tensor<type, 2>& output_gradient) const
    {

        return activations_derivatives*output_gradient;
    }

    void calculate_output_delta(const Tensor<type, 2>& activations_derivatives,
                                const Tensor<type, 2>& output_gradient,
                                Tensor<type, 2>& output_delta) const
    {
        switch(device_pointer->get_type())
        {
             case Device::EigenDefault:
             {
                 DefaultDevice* default_device = device_pointer->get_eigen_default_device();

                 output_delta.device(*default_device) = activations_derivatives*output_gradient;

                 return;
             }

             case Device::EigenSimpleThreadPool:
             {
                ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

                output_delta.device(*thread_pool_device) = activations_derivatives*output_gradient;

                return;
             }

            case Device::EigenGpu:
            {
//                 GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

                 break;
            }

             default:
             {
                ostringstream buffer;

                buffer << "OpenNN Exception: Layer class.\n"
                       << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                       << "Unknown device.\n";

                throw logic_error(buffer.str());
            }
        }
    }


    virtual Tensor<type, 2> calculate_hidden_delta(Layer*,
                                                  const Tensor<type, 2>&,
                                                  const Tensor<type, 2>&,
                                                  const Tensor<type, 2>&) const;

    virtual void calculate_hidden_delta(Layer*,
                                        const Tensor<type, 2>&,
                                        const Tensor<type, 2>&,
                                        const Tensor<type, 2>&,
                                        Tensor<type, 2>&) const {}

    // Get neurons number

    virtual Tensor<Index, 1> get_input_variables_dimensions() const;

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;

    virtual void set_inputs_number(const Index&);
    virtual void set_neurons_number(const Index&);

    virtual string object_to_string() const;

    // Layer type

    Type get_type() const;

    string get_type_string() const;

    // Serialization methods

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void write_XML(tinyxml2::XMLPrinter&) const {}

protected:

    Device* device_pointer = nullptr;

    /// Layer type object.

    Type layer_type = Perceptron;

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

    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void symmetric_threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void rectified_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_plus_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_sign_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hard_sigmoid_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;
    void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const;

    const Eigen::array<IndexPair<Index>, 1> product_dimensions = {IndexPair<Index>(1, 0)};
    const Eigen::array<IndexPair<Index>, 1> transposed_product_dimensions = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> product_vector_vector = {IndexPair<Index>(0, 0)}; // Vector product, (0,0) first vector is transpose
    const Eigen::array<IndexPair<Index>, 1> product_matrix_transpose_vector = {IndexPair<Index>(0, 0) }; // Matrix times vector, (0,0) matrix is transpose
    const Eigen::array<IndexPair<Index>, 1> product_matrix_transpose_matrix = {IndexPair<Index>(0, 0) }; // Matrix times matrix, (0,0) first matrix is transpose
    const Eigen::array<IndexPair<Index>, 1> product_matrix_matrix = {IndexPair<Index>(1, 0)};
    const Eigen::array<IndexPair<Index>, 1> dimensions = {IndexPair<Index>(0, 0)};

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/layer_cuda.h"
#endif


};

}

#endif // __LAYER_H
