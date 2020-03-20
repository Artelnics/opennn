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

//Eigen includes

#include "../eigen/unsupported/Eigen/CXX11/Tensor"

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

    enum Type{Scaling, Convolutional, Perceptron, Pooling, Probabilistic,
              LongShortTermMemory,Recurrent, Unscaling, Bounding, PrincipalComponents};

    /// This structure represents the first order activaions of layers.

    struct ForwardPropagation
    {
        /// Default constructor.

        explicit ForwardPropagation()
        {
        }

        explicit ForwardPropagation(const Index& new_batch_instances_number, Layer* new_layer_pointer)
        {
            set(new_batch_instances_number, new_layer_pointer);
        }


        virtual ~ForwardPropagation() {}


        void set(const Index& new_batch_instances_number, Layer* new_layer_pointer)
        {
            batch_instances_number = new_batch_instances_number;

            layer_pointer = new_layer_pointer;

            const Index neurons_number = layer_pointer->get_neurons_number();

            combinations_2d.resize(batch_instances_number, neurons_number);

            activations_2d.resize(batch_instances_number, neurons_number);

            if(layer_pointer->get_type() == Perceptron)
            {
                activations_derivatives_2d.resize(batch_instances_number, neurons_number);
            }
            else
            {
                activations_derivatives_3d.resize(neurons_number, neurons_number, batch_instances_number);
            }
        }


        void print() const
        {
            cout << "Combinations: " << endl;
            cout << combinations_2d << endl;

            cout << "Activations: " << endl;
            cout << activations_2d << endl;

            if(layer_pointer->get_type() == Perceptron)
            {
                cout << "Activations derivatives: " << endl;
                cout << activations_derivatives_2d << endl;
            }
            else
            {
                cout << "Activations derivatives 3d:" << endl;
                cout << activations_derivatives_3d << endl;
            }

        }

        Index batch_instances_number = 0;

        Layer* layer_pointer;

        Tensor<type, 2> combinations_2d;
        Tensor<type, 2> activations_2d;
        Tensor<type, 2> activations_derivatives_2d;

        Tensor<type, 3> activations_derivatives_3d;

        Tensor<type, 4> combinations_4d;
        Tensor<type, 4> activations_4d;
        Tensor<type, 4> activations_derivatives_4d;
    };


    struct BackPropagation
    {
        /// Default constructor.

        explicit BackPropagation() {}

        explicit BackPropagation(const Index& new_batch_instances_number, Layer* new_layer_pointer)
        {
            set(new_batch_instances_number, new_layer_pointer);
        }


        virtual ~BackPropagation() {}


        void set(const Index& new_batch_instances_number, Layer* new_layer_pointer)
        {
            batch_instances_number = new_batch_instances_number;

            layer_pointer = new_layer_pointer;

            const Index neurons_number = layer_pointer->get_neurons_number();
            const Index inputs_number = layer_pointer->get_inputs_number();

            biases_derivatives.resize(neurons_number);

            synaptic_weights_derivatives.resize(inputs_number, neurons_number);

            delta.resize(batch_instances_number, neurons_number);
        }

        virtual void print() const {}

        Index batch_instances_number = 0;

        Layer* layer_pointer = nullptr;

        Tensor<type, 2> delta;

        Tensor<type, 1> biases_derivatives;

        Tensor<type, 2> synaptic_weights_derivatives;
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

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_device_pointer(Device*);

    virtual void insert_gradient(const BackPropagation&, const Index&, Tensor<type, 1>&) const {}

    // Outputs

    virtual Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
    virtual Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

    virtual Tensor<type, 4> calculate_outputs(const Tensor<type, 4>&) {return Tensor<type, 4>();}
    virtual Tensor<type, 4> calculate_outputs(const Tensor<type, 4>&, const Tensor<type, 1>&) {return Tensor<type, 4>();}

    virtual void calculate_error_gradient(const Tensor<type, 2>&,
                                          const Layer::ForwardPropagation&, Layer::BackPropagation&) const {}

    virtual void forward_propagate(const Tensor<type, 2>&, ForwardPropagation&) const {}
    virtual void forward_propagate(const Tensor<type, 4>&, ForwardPropagation&) const {}

    virtual void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>, ForwardPropagation&) const {}

    // Deltas

    virtual void calculate_output_delta(ForwardPropagation&,
                                const Tensor<type, 2>&,
                                Tensor<type, 2>&) const {}

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

    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 3>&) const;
    void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 3>&) const;

    void hard_sigmoid_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void hyperbolic_tangent_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void symmetric_threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void rectified_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void scaled_exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_plus_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void soft_sign_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
    void exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;

    void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0) };
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/layer_cuda.h"
#endif

};

}

#endif // LAYER_H
