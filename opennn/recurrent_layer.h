//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RECURRENTLAYER_H
#define RECURRENTLAYER_H

#include "layer.h"

namespace opennn
{

class Recurrent : public Layer
{

public:

   Recurrent(const dimensions & = {0}, const dimensions& = {0});

   dimensions get_input_dimensions() const override;
   dimensions get_output_dimensions() const override;

   Index get_timesteps() const;

   vector<pair<type*, Index>> get_parameter_pairs() const override;

   string get_activation_function() const;

   void set(const dimensions& = {}, const dimensions& = {});

   void set_input_dimensions(const dimensions&) override;
   void set_output_dimensions(const dimensions&) override;

   void set_timesteps(const Index&);

   void set_activation_function(const string&);

   void calculate_combinations(const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               Tensor<type, 2>&) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) override;

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const override;

   string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

   void print() const override;

   void from_XML(const XMLDocument&) override;
   void to_XML(XMLPrinter&) const override;

private:

    Index time_steps = 2;

    Index batch_size = 0;

    Tensor<type, 1> biases;

    Tensor<type, 2> input_weights;
    Tensor<type, 2> recurrent_weights;

    string activation_function = "Linear";

#ifdef OPENNN_CUDA
    // @todo
#endif

};


struct RecurrentForwardPropagation : LayerForwardPropagation
{
    RecurrentForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> outputs;

    Tensor<type, 3> current_inputs;
    Tensor<type, 2> current_activation_derivatives;

    Tensor<type, 3> activation_derivatives;

    Tensor<type, 3> hidden_states;
};


struct RecurrentBackPropagation : LayerBackPropagation
{
    RecurrentBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    vector<pair<type*, Index>> get_parameter_delta_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> current_deltas;
    Tensor<type, 2> current_targets;

    Tensor<type, 2> combination_deltas;
    Tensor<type, 2> current_combination_deltas;

    Tensor<type, 2> combinations_bias_deltas;
    Tensor<type, 3> combinations_input_weight_deltas;
    Tensor<type, 3> combinations_recurrent_weight_deltas;

    Tensor<type, 1> bias_deltas;

    Tensor<type, 2> input_weight_deltas;

    Tensor<type, 2> recurrent_weight_deltas;

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
