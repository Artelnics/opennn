//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G H T E D   S Q U A R E D   E R R O R    C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef WEIGHTEDSQUAREDERROR_H
#define WEIGHTEDSQUAREDERROR_H

#include <string>
#include <math.h>

#include "config.h"
#include "loss_index.h"
#include "data_set.h"

namespace opennn
{

class WeightedSquaredError : public LossIndex
{

public:

   // Constructors

   explicit WeightedSquaredError();

   explicit WeightedSquaredError(NeuralNetwork*, DataSet*); 

   // Get

   type get_positives_weight() const;
   type get_negatives_weight() const;

   type get_normalizaton_coefficient() const;

   // Set

   void set_default();

   void set_positives_weight(const type&);
   void set_negatives_weight(const type&);

   void set_weights(const type&, const type&);

   void set_weights();

   void set_normalization_coefficient() final;

   void set_data_set(DataSet*) final;

   string get_error_type() const final;

   string get_error_type_text() const final;

   // Back propagation

   void calculate_error(const Batch&,
                        const ForwardPropagation&,
                        BackPropagation&) const final;

   void calculate_output_delta(const Batch&,
                               ForwardPropagation&,
                               BackPropagation&) const final;

   // Back propagation LM

   void calculate_squared_errors_lm(const Batch&,
                                    const ForwardPropagation&,
                                    BackPropagationLM&) const final;

   void calculate_error_lm(const Batch&,
                           const ForwardPropagation&,
                           BackPropagationLM&) const final;

   void calculate_error_gradient_lm(const Batch&,
                                    BackPropagationLM&) const final;

   void calculate_error_hessian_lm(const Batch&,
                                   BackPropagationLM&) const final;

   // Serialization

   void from_XML(const tinyxml2::XMLDocument&);

   void to_XML(tinyxml2::XMLPrinter&) const final;

private:

   type positives_weight = type(NAN);

   type negatives_weight = type(NAN);

   type normalization_coefficient;

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/weighted_squared_error_cuda.h"
#endif

};

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
