//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N K O W S K I   E R R O R   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MINKOWSKIERROR_H
#define MINKOWSKIERROR_H

#include "loss_index.h"

namespace opennn
{

class MinkowskiError : public LossIndex
{

public:

   MinkowskiError(const NeuralNetwork* = nullptr, const Dataset* = nullptr);

   type get_Minkowski_parameter() const override;

   void set_default();

   void set_Minkowski_parameter(const type&);

   void calculate_error(const Batch& batch,
                        const ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation) const override;

   void calculate_output_delta(const Batch&,
                               ForwardPropagation&,
                               BackPropagation&) const override;

   string get_name() const override;

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

private:

   type minkowski_parameter;

    const type epsilon = numeric_limits<type>::epsilon();

#ifdef OPENNN_CUDA

   // Error

   void calculate_error_cuda(const BatchCuda&,
                             const ForwardPropagationCuda&,
                             BackPropagationCuda&) const override;

   // Gradient

   void calculate_output_delta_cuda(const BatchCuda&,
                                    ForwardPropagationCuda&,
                                    BackPropagationCuda&) const override;

#endif

};

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
