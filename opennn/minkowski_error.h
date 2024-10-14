//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MINKOWSKIERROR_H
#define MINKOWSKIERROR_H



#include <string>



#include "config.h"
#include "loss_index.h"
#include "data_set.h"

namespace opennn
{

class MinkowskiError : public LossIndex
{

public:

   // Constructors

   explicit MinkowskiError();

   explicit MinkowskiError(NeuralNetwork*, DataSet*);

   // Get

   type get_Minkowski_parameter() const;

   // Set

   void set_default();

   void set_Minkowski_parameter(const type&);

   // loss

   void calculate_error(const Batch& batch,
                        const ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation) const final;

   void calculate_output_delta(const Batch&,
                               ForwardPropagation&,
                               BackPropagation&) const final;

   // Serialization

   string get_loss_method() const final;
   string get_error_type_text() const final;

   virtual void from_XML(const tinyxml2::XMLDocument&);

   void to_XML(tinyxml2::XMLPrinter&) const final;

private:

   type minkowski_parameter;

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/minkowski_error_cuda.h"
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
