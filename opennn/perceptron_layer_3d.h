//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER3D_H
#define PERCEPTRONLAYER3D_H

#include "layer.h"

namespace opennn
{

class Dense3d : public Layer
{

public:

    enum class Activation{HyperbolicTangent,
                          Logistic,
                          Linear,
                          RectifiedLinear,
                          Softmax};

    Dense3d(const Index& = 0,
            const Index& = 0,
            const Index& = 0,
            const Activation& = Dense3d::Activation::HyperbolicTangent,
            const string& = "dense3d_layer");

    Index get_sequence_length() const;
    Index get_input_embedding() const;
    Index get_output_embedding() const;

   // @todo

   dimensions get_input_dimensions() const override;
   dimensions get_output_dimensions() const override;

   Index get_parameters_number() const override;
   type get_dropout_rate() const;
   void get_parameters(Tensor<type, 1>&) const override;

   const Dense3d::Activation& get_activation_function() const;

   string get_activation_function_string() const;

   void set(const Index& = 0,
            const Index& = 0,
            const Index& = 0,
            const Dense3d::Activation& = Dense3d::Activation::HyperbolicTangent,
            const string & = "dense3d_layer");

   void set_parameters(const Tensor<type, 1>&, Index&) override;

   void set_activation_function(const Activation&);
   void set_activation_function(const string&);
   void set_dropout_rate(const type&);

   void set_parameters_random() override;
   void set_parameters_glorot();

   void calculate_combinations(const Tensor<type, 3>&,
                               Tensor<type, 3>&) const;

   void calculate_activations(Tensor<type, 3>&,
                              Tensor<type, 3>&) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) override;

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const override;

   void insert_gradient(unique_ptr<LayerBackPropagation>&,
                        Index&,
                        Tensor<type, 1>&) const override;

   void from_XML(const XMLDocument&) override;
   void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
        // @todo
    #endif

private:

   Index sequence_length;

   Tensor<type, 1> biases;

   Tensor<type, 2> weights;

   Activation activation_function;

   type dropout_rate = type(0);
};


struct Dense3dForwardPropagation : LayerForwardPropagation
{
    Dense3dForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 3> outputs;

    Tensor<type, 3> activation_derivatives;
};


struct Dense3dBackPropagation : LayerBackPropagation
{
    Dense3dBackPropagation(const Index& = 0, Layer* = 0);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 1> bias_deltas;
    Tensor<type, 2> weight_deltas;

    Tensor<type, 3> input_deltas;
};


#ifdef OPENNN_CUDA
    // @todo
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
