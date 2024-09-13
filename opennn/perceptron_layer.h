//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H

// System includes

#include <string>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

namespace opennn
{

struct LayerForwardPropagation;
struct LayerBackPropagation;

struct PerceptronLayerForwardPropagation;
struct PerceptronLayerBackPropagation;
struct PerceptronLayerBackPropagationLM;

#ifdef OPENNN_CUDA
    struct PerceptronLayerForwardPropagationCuda;
    struct PerceptronLayerBackPropagationCuda;
#endif


class PerceptronLayer : public Layer
{

public:

    enum class ActivationFunction {
        Logistic,
        HyperbolicTangent,
        Linear,
        RectifiedLinear,
        ExponentialLinear,
        ScaledExponentialLinear,
        SoftPlus,
        SoftSign,
        HardSigmoid
    };

    // Constructors

    explicit PerceptronLayer();

    explicit PerceptronLayer(const Index&,
        const Index&,
        const ActivationFunction & = PerceptronLayer::ActivationFunction::HyperbolicTangent);

    // Get

    Index get_inputs_number() const final;
    Index get_neurons_number() const final;

    // Parameters

    const Tensor<type, 1>& get_biases() const;
    const Tensor<type, 2>& get_synaptic_weights() const;
    Tensor<type, 1> get_parameters() const final;

    Index get_biases_number() const;
    Index get_synaptic_weights_number() const;
    Index get_parameters_number() const final;
    type get_dropout_rate() const;

    dimensions get_output_dimensions() const final;

    // Activation functions

    const PerceptronLayer::ActivationFunction& get_activation_function() const;

    string write_activation_function() const;

    // Display messages

    const bool& get_display() const;

    // Set

    void set();

    void set(const Index&,
        const Index&,
        const PerceptronLayer::ActivationFunction & = PerceptronLayer::ActivationFunction::HyperbolicTangent);

    void set_default();
    void set_name(const string&);

    // Architecture

    void set_inputs_number(const Index&) final;
    void set_neurons_number(const Index&) final;

    // Parameters

    void set_biases(const Tensor<type, 1>&);
    void set_synaptic_weights(const Tensor<type, 2>&);

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;

    // Activation functions

    void set_activation_function(const ActivationFunction&);
    void set_activation_function(const string&);
    void set_dropout_rate(const type&);

    // Display messages

    void set_display(const bool&);

    // Parameters initialization

    
    

    void set_parameters_constant(const type&) final;

    void set_parameters_random() final;

    // Forward propagation

    void calculate_combinations(const Tensor<type, 2>&,
        Tensor<type, 2>&) const;

    void dropout(Tensor<type, 2>&) const;

    void calculate_activations(Tensor<type, 2>&,
                               Tensor<type, 2>& = Tensor<type, 2>()) const;

    void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
        LayerForwardPropagation*,
        const bool&) final;

    // Gradient

    void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                        const Tensor<pair<type*, dimensions>, 1>&,
                        LayerForwardPropagation*,
                        LayerBackPropagation*) const final;

    void back_propagate_lm(const Tensor<pair<type*, dimensions>, 1>&,
                           const Tensor<pair<type*, dimensions>, 1>&,
                           LayerForwardPropagation*,
                           LayerBackPropagationLM*) const final;

    void insert_gradient(LayerBackPropagation*,
                         const Index&,
                         Tensor<type, 1>&) const final;

    void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                           const Index&,
                                           Tensor<type, 2>&) const final;

    // Expression   

    string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

    string write_activation_function_expression() const;

    // Serialization

    void from_XML(const tinyxml2::XMLDocument&) final;
    void to_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/perceptron_layer_cuda.h"
    #endif


protected:

    Tensor<type, 1> biases;

    Tensor<type, 2> synaptic_weights;

    ActivationFunction activation_function;

    type dropout_rate = type(0);

    bool display = true;

};


struct PerceptronLayerForwardPropagation : LayerForwardPropagation
{
    explicit PerceptronLayerForwardPropagation();

    explicit PerceptronLayerForwardPropagation(const Index&, Layer*);

    virtual ~PerceptronLayerForwardPropagation();

    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 2> outputs;

    Tensor<type, 2> activations_derivatives;
};


struct PerceptronLayerBackPropagation : LayerBackPropagation
{
    // Default constructor

    explicit PerceptronLayerBackPropagation();

    explicit PerceptronLayerBackPropagation(const Index&, Layer*);

    virtual ~PerceptronLayerBackPropagation();

    void set(const Index&, Layer*) final;

    void print() const;

    //Tensor<type, 2> deltas;

    Tensor<type, 2> error_combinations_derivatives;
    Tensor<type, 2> input_derivatives;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;
};


struct PerceptronLayerBackPropagationLM : LayerBackPropagationLM
{
    // Default constructor

    explicit PerceptronLayerBackPropagationLM();

    explicit PerceptronLayerBackPropagationLM(const Index&, Layer*);

    virtual ~PerceptronLayerBackPropagationLM();

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 2> error_combinations_derivatives;
    Tensor<type, 2> input_derivatives;

    Tensor<type, 2> squared_errors_Jacobian;

};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_back_propagation_cuda.h"
#endif


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
