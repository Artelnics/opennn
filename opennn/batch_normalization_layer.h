//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M A L I Z A T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BATCHNORMALIZATIONLAYER_H
#define BATCHNORMALIZATIONLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "opennn_strings.h"

namespace opennn
{
class BatchNormalizationLayer : public Layer

{

public:
   // Constructors

   explicit BatchNormalizationLayer();

   explicit BatchNormalizationLayer(const Index&);

//   explicit BatchNormalizationLayer(const Index&, const Index&, const ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent);

   // Get methods

//   Index get_inputs_number() const override;
//   Index get_neurons_number() const final;

   // Parameters

//   const Tensor<type, 2>& get_biases() const;
//   const Tensor<type, 2>& get_synaptic_weights() const;

//   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
//   Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

//   Index get_biases_number() const;
//   Index get_synaptic_weights_number() const;
//   Index get_parameters_number() const final;
//   Tensor<type, 1> get_parameters() const final;

//   Tensor< TensorMap< Tensor<type, 1>>*, 1> get_layer_parameters() final;

//   string write_activation_function() const;

   // Display messages

//   const bool& get_display() const;

   // Set methods

   void set(const Index&);
//   void set(const Index&, const Index&, const PerceptronLayer::ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent);

   void set_default();
//   void set_name(const string&);

   // Architecture

//   void set_inputs_number(const Index&) final;
//   void set_neurons_number(const Index&) final;

   // Parameters

//   void set_parameters(const Tensor<type, 1>&, const Index& index=0) final;

//   // Display messages

//   void set_display(const bool&);

//   // Parameters initialization methods
//   void set_biases_constant(const type&);
//   void set_synaptic_weights_constant(const type&);
   
//   void set_parameters_constant(const type&) final;

   void set_parameters_random();

   // Perceptron layer combinations

   void calculate_combinations(const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               Tensor<type, 2>&);


   // Perceptron layer outputs


//   void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) final;

//   void forward_propagate(type*, const Tensor<Index, 1>&,
//                          LayerForwardPropagation*) final;

//   void forward_propagate(type*,
//                          const Tensor<Index, 1>&,
//                          Tensor<type, 1>&,
//                          LayerForwardPropagation*) final;


   // Gradient methods

//   void calculate_error_gradient(type*,
//                                 LayerForwardPropagation*,
//                                 LayerBackPropagation*) const final;

//   void insert_gradient(LayerBackPropagation*,
//                        const Index&,
//                        Tensor<type, 1>&) const final;



   // Serialization methods

//   void from_XML(const tinyxml2::XMLDocument&) final;
//   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   // MEMBERS

   /// Inputs

   Tensor<type, 2> inputs;

   /// Fixed parameters

   Tensor<type, 2> mean;
   Tensor<type, 2> std;

   /// Outputs

   Tensor<type, 2> outputs;

   /// learneable_parameters

   Tensor<type, 2> normalization_weights;

   /// Display messages to screen. 

   bool display = true;
};

#endif
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
