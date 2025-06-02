//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CROSSENTROPYERROR3D_H
#define CROSSENTROPYERROR3D_H

#include "loss_index.h"

namespace opennn
{

class CrossEntropyError3d : public LossIndex
{

public:

    CrossEntropyError3d(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const override;

    void calculate_binary_error(const Batch&,
                                const ForwardPropagation&,
                                BackPropagation&) const;

    void calculate_multiple_error(const Batch&,
                                  const ForwardPropagation&,
                                  BackPropagation&) const;

    void calculate_output_delta(const Batch&,
                                ForwardPropagation&,
                                BackPropagation&) const override;

    string get_loss_method() const override;
    string get_error_type_text() const override;

    virtual void from_XML(const XMLDocument&);

    void to_XML(XMLPrinter&) const override;

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
