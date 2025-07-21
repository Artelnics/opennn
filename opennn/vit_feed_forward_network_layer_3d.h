//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I T  F E E D  F O R W A R D  N E T W O R K   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef VIT_FEED_FORWARD_NETWORK_LAYER_3D_H
#define VIT_FEED_FORWARD_NETWORK_LAYER_3D_H

#include "layer.h"

namespace opennn
{

#ifdef OPENNN_CUDA
    struct PerceptronLayer3DForwardPropagationCuda;
    struct PerceptronLayer3DBackPropagationCuda;
#endif

    class VitFeedForwardNetworkLayer3D : public Layer
    {

    public:

        enum class ActivationFunction {
            HyperbolicTangent,
            Linear,
            RectifiedLinear
        };

        VitFeedForwardNetworkLayer3D(const Index & = 0,
            const Index & = 0,
            const Index & = 0,
            const ActivationFunction & = VitFeedForwardNetworkLayer3D::ActivationFunction::HyperbolicTangent,
            const string & = "vit_feed_forward_network_layer_3d");

        Index get_tokens_number() const;
        Index get_inputs_depth() const;
        Index get_neurons_number() const;

        // @todo
        dimensions get_input_dimensions() const override
        {
            throw runtime_error("XXX");
        }

        dimensions get_output_dimensions() const override;

        Index get_parameters_number() const override;
        type get_dropout_rate() const;
        Tensor<type, 1> get_parameters() const override;

        const VitFeedForwardNetworkLayer3D::ActivationFunction& get_activation_function() const;

        string get_activation_function_string() const;

        void set(const Index & = 0,
            const Index & = 0,
            const Index & = 0,
            const VitFeedForwardNetworkLayer3D::ActivationFunction & = VitFeedForwardNetworkLayer3D::ActivationFunction::HyperbolicTangent,
            const string & = "vit_feed_forward_network_layer_3d");

        void set_tokens_number(Index);
        void set_input_dimensions(const dimensions&) override;

        void set_inputs_depth(const Index&);
        void set_output_dimensions(const dimensions&) override;

        void set_parameters(const Tensor<type, 1>&, const Index& index = 0) override;

        void set_activation_function(const ActivationFunction&);
        void set_activation_function(const string&);
        void set_dropout_rate(const type&);

        void set_parameters_constant(const type&) override;
        void set_parameters_random() override;
        void set_parameters_glorot();

        void calculate_combinations(const Tensor<type, 3>&,
            Tensor<type, 3>&) const;

        void dropout(Tensor<type, 3>&) const;

        void calculate_activations(Tensor<type, 3>&,
            Tensor<type, 3>&) const;

        void forward_propagate(const vector<pair<type*, dimensions>>&,
            unique_ptr<LayerForwardPropagation>&,
            const bool&) override;

        void back_propagate(const vector<pair<type*, dimensions>>&,
            const vector<pair<type*, dimensions>>&,
            unique_ptr<LayerForwardPropagation>&,
            unique_ptr<LayerBackPropagation>&) const override;

        void add_deltas(const vector<pair<type*, dimensions>>&) const;

        void insert_gradient(unique_ptr<LayerBackPropagation>&,
            const Index&,
            Tensor<type, 1>&) const override;

        void from_XML(const XMLDocument&) override;
        void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_cuda.h"
#endif

    private:

        Index tokens_number;

        Tensor<type, 1> biases1;
        Tensor<type, 1> biases2;

        Tensor<type, 2> synaptic_weights1;
        Tensor<type, 2> synaptic_weights2;

        ActivationFunction activation_function;

        type dropout_rate = type(0);

        Tensor<type, 3> empty;

        const Eigen::array<Index, 2> sum_dimensions = { 0, 1 };

        const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 0) };

        const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };
        const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
    };


    struct VitFeedForwardNetworkLayer3DForwardPropagation : LayerForwardPropagation
    {
        VitFeedForwardNetworkLayer3DForwardPropagation(const Index & = 0, Layer* = nullptr);

        pair<type*, dimensions> get_outputs_pair() const override;

        void set(const Index & = 0, Layer* = nullptr);

        void print() const override;

        void set_transformed_inputs(const Tensor<type, 3>&);

        Tensor<type, 3> outputs;

        Tensor<type, 3> activation_derivatives;

        Tensor<type, 3> transformed_inputs;

    };


    struct VitFeedForwardNetworkLayer3DBackPropagation : LayerBackPropagation
    {
        VitFeedForwardNetworkLayer3DBackPropagation(const Index & = 0, Layer* = 0);

        vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

        void set(const Index & = 0, Layer* = nullptr);

        void print() const override;

        Tensor<type, 1> bias_derivatives1;
        Tensor<type, 2> synaptic_weight_derivatives1;

        Tensor<type, 1> bias_derivatives2;
        Tensor<type, 2> synaptic_weight_derivatives2;

        Tensor<type, 3> combination_derivatives1;
        Tensor<type, 3> combination_derivatives2;
        Tensor<type, 3> input_derivatives;

        Tensor<type, 3> activation_deltas;
    };


#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_forward_propagation_cuda.h"
#include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_back_propagation_cuda.h"
#endif


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
