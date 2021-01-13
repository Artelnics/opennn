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
#include <omp.h>

// OpenNN includes

#include "config.h"
#include "statistics.h"

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

        explicit ForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        {
            set(new_batch_samples_number, new_layer_pointer);
        }


        virtual ~ForwardPropagation() {}

        void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        {
            batch_samples_number = new_batch_samples_number;

            layer_pointer = new_layer_pointer;

            const Index neurons_number = layer_pointer->get_neurons_number();

            ones_2d.resize(batch_samples_number, neurons_number);
            ones_2d.setConstant(1);

            combinations_2d.resize(batch_samples_number, neurons_number);

            activations_2d.resize(batch_samples_number, neurons_number);

            if(layer_pointer->get_type() == Perceptron) // Perceptron
            {
                activations_derivatives_2d.resize(batch_samples_number, neurons_number);
            }
            else if(layer_pointer->get_type() == Recurrent ) // Recurrent
            {
                activations_derivatives_2d.resize(batch_samples_number, neurons_number);
            }
            else if(layer_pointer->get_type() == LongShortTermMemory) // LSTM
            {
                combinations_1d.resize(neurons_number);

                activations_1d.resize(neurons_number);

                activations_3d.resize(batch_samples_number, neurons_number, 6);

                activations_derivatives_3d.resize(batch_samples_number, neurons_number, 6);
            }
            else if(layer_pointer->get_type() == Probabilistic) // Probabilistic
            {
                activations_derivatives_3d.resize(batch_samples_number, neurons_number, neurons_number);
            }
            else // Convolutional
            {
                activations_4d.resize(batch_samples_number, neurons_number, neurons_number, neurons_number);
                activations_derivatives_4d.resize(batch_samples_number, neurons_number, neurons_number, neurons_number);// @todo
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

        Index batch_samples_number = 0;

        Layer* layer_pointer;

        Tensor<type, 1> combinations_1d;
        Tensor<type, 1> activations_1d;

        Tensor<type, 2> combinations_2d;
        Tensor<type, 2> activations_2d;
        Tensor<type, 2> activations_derivatives_2d;

        Tensor<type, 2> ones_2d;

        Tensor<type, 3> activations_3d;
        Tensor<type, 3> activations_derivatives_3d;

        Tensor<type, 4> combinations_4d;
        Tensor<type, 4> activations_4d;
        Tensor<type, 4> activations_derivatives_4d;
    };


    struct BackPropagation
    {
        /// Default constructor.

        explicit BackPropagation() {}

        explicit BackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        {
            set(new_batch_samples_number, new_layer_pointer);
        }


        virtual ~BackPropagation() {}


        void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        {
            batch_samples_number = new_batch_samples_number;

            layer_pointer = new_layer_pointer;

            const Index neurons_number = layer_pointer->get_neurons_number();
            const Index inputs_number = layer_pointer->get_inputs_number();

            biases_derivatives.resize(neurons_number);

            synaptic_weights_derivatives.resize(inputs_number, neurons_number);

            delta.resize(batch_samples_number, neurons_number);
        }

        virtual void print() const {}

        Index batch_samples_number = 0;

        Layer* layer_pointer = nullptr;

        Tensor<type, 2> delta;

        Tensor<type, 4> delta_4d;

        Tensor<type, 1> biases_derivatives;

        Tensor<type, 2> synaptic_weights_derivatives;

        Tensor<type, 4> synaptic_weights_derivatives_4d;
    };


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

    virtual void insert_gradient(const BackPropagation&, const Index&, Tensor<type, 1>&) const {}

    // Outputs

    virtual Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

    virtual Tensor<type, 2> calculate_outputs_from4D(const Tensor<type, 4>&) {return Tensor<type, 2>();}

    virtual Tensor<type, 4> calculate_outputs_4D(const Tensor<type, 4>&) {return Tensor<type, 4>();}

    virtual void calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, Layer::BackPropagation&) const {}
    virtual void calculate_error_gradient(const Tensor<type, 4>&, const Layer::ForwardPropagation&, Layer::BackPropagation&) const {}

    virtual void forward_propagate(const Tensor<type, 2>&, ForwardPropagation&) const {}
    virtual void forward_propagate(const Tensor<type, 4>&, ForwardPropagation&) const {}

    virtual void forward_propagate(const Tensor<type, 4>&, Tensor<type, 1>, ForwardPropagation&) const {}
    virtual void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>, ForwardPropagation&) const {}

    // Deltas

    virtual void calculate_output_delta(ForwardPropagation&,
                                const Tensor<type, 2>&,
                                Tensor<type, 2>&) const {}

    virtual void calculate_hidden_delta(Layer*,
                                        const Tensor<type, 2>&,
                                        ForwardPropagation&,
                                        const Tensor<type, 2>&,
                                        Tensor<type, 2>&) const {}

    // Get neurons number

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;
    virtual Index get_synaptic_weights_number() const;


    virtual void set_inputs_number(const Index&);
    virtual void set_neurons_number(const Index&);

    virtual 

    // Layer type

    Type get_type() const;

    string get_type_string() const;

    // Serialization methods

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void write_XML(tinyxml2::XMLPrinter&) const {}

    // Expression methods

    virtual string write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const {return string();}

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
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/layer_mkl.h"
#endif

};

}

#endif // LAYER_H
