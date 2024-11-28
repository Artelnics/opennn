//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H


#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

namespace opennn
{

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

    PerceptronLayer(const dimensions& = {0},
                    const dimensions& = {0},
                    const ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent,
                    const string& = "perceptron_layer");

    dimensions get_input_dimensions() const final;
    dimensions get_output_dimensions() const final;

    Tensor<type, 1> get_parameters() const final;

    Index get_parameters_number() const final;
    type get_dropout_rate() const;

    const PerceptronLayer::ActivationFunction& get_activation_function() const;

    string get_activation_function_string() const;

    void set(const dimensions& = {0},
             const dimensions& = {0},
             const PerceptronLayer::ActivationFunction & = PerceptronLayer::ActivationFunction::HyperbolicTangent,
             const string& = "perceptron_layer");

    void set_input_dimensions(const dimensions&) final;
    void set_output_dimensions(const dimensions&) final;

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;
    void set_parameters_constant(const type&) final;
    void set_parameters_random() final;

    void set_activation_function(const ActivationFunction&);
    void set_activation_function(const string&);
    void set_dropout_rate(const type&);

    void calculate_combinations(const Tensor<type, 2>&,
                                Tensor<type, 2>&) const;

    void dropout(Tensor<type, 2>&) const;

    void calculate_activations(Tensor<type, 2>&,
                               Tensor<type, 2>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) final;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const final;

    void back_propagate_lm(const vector<pair<type*, dimensions>>&,
                           const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           unique_ptr<LayerBackPropagationLM>&) const final;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         const Index&,
                         Tensor<type, 1>&) const final;

    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                           const Index&,
                                           Tensor<type, 2>&) const final;

    string get_expression(const vector<string>&, const vector<string>&) const final;

    string get_activation_function_string_expression() const;

    void print() const;

    void from_XML(const XMLDocument&) final;
    void to_XML(XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/perceptron_layer_cuda.h"
    #endif

private:

    Tensor<type, 1> biases;

    Tensor<type, 2> synaptic_weights;

    ActivationFunction activation_function = ActivationFunction::HyperbolicTangent;

    type dropout_rate = type(0);

    const Eigen::array<Index, 1> sum_dimensions_1 = {0};
};


struct PerceptronLayerForwardPropagation : LayerForwardPropagation
{
    explicit PerceptronLayerForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 2> outputs;

    Tensor<type, 2> activation_derivatives;
};


struct PerceptronLayerBackPropagation : LayerBackPropagation
{
    explicit PerceptronLayerBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 2> combinations_derivatives;
    Tensor<type, 2> input_derivatives;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;
};


struct PerceptronLayerBackPropagationLM : LayerBackPropagationLM
{
    explicit PerceptronLayerBackPropagationLM(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 2> combinations_derivatives;
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
