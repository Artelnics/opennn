//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZEDSQUAREDERROR_H
#define NORMALIZEDSQUAREDERROR_H

// System includes

#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "data_set.h"

namespace opennn
{

/// This class represents the normalized squared error term. 

///
/// This error term is used in data modeling problems.
/// If it has a value of unity then the neural network is predicting the data "in the mean",
/// A value of zero means perfect prediction of data.

class NormalizedSquaredError : public LossIndex
{

public:

    // Constructors

   explicit NormalizedSquaredError(NeuralNetwork*, DataSet*);

   explicit NormalizedSquaredError();   

   // Get methods

    type get_normalization_coefficient() const;
    type get_selection_normalization_coefficient() const;

   // Set methods

    void set_normalization_coefficient() override;
    void set_normalization_coefficient(const type&);

    void set_time_series_normalization_coefficient();

    void set_selection_normalization_coefficient();
    void set_selection_normalization_coefficient(const type&);

    virtual void set_default();

    void set_data_set_pointer(DataSet* new_data_set_pointer) final;

   // Normalization coefficients 

   type calculate_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

   type calculate_time_series_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Back propagation
     
   void calculate_error(const DataSetBatch&,
                        const NeuralNetworkForwardPropagation&,
                        LossIndexBackPropagation&) const final;

   void calculate_output_delta(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagation&) const final;

    // Back propagation LM

   void calculate_error_lm(const DataSetBatch&,
                           const NeuralNetworkForwardPropagation&,
                           LossIndexBackPropagationLM&) const final;

   void calculate_output_delta_lm(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagationLM&) const final;

   void calculate_error_gradient_lm(const DataSetBatch&,
                              LossIndexBackPropagationLM&) const final;

   void calculate_error_hessian_lm(const DataSetBatch&,
                                        LossIndexBackPropagationLM&) const final;

   // Serialization methods

   string get_error_type() const final;
   string get_error_type_text() const final;

   virtual void from_XML(const tinyxml2::XMLDocument&) const;

   void write_XML(tinyxml2::XMLPrinter&) const final;

private:

   /// Coefficient of normalization for the calculation of the training error.

   type normalization_coefficient = type(NAN);

   type selection_normalization_coefficient = type(NAN);

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/normalized_squared_error_cuda.h"
#endif

};

}

#endif


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
