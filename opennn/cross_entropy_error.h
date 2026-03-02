//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   2 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "loss.h"

namespace opennn
{

class CrossEntropyError2d final : public Loss
{

public:

    CrossEntropyError2d(const NeuralNetwork* = nullptr, const Dataset* = nullptr);

    // Error

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const override;

    void calculate_binary_error(const Batch&,
                                const ForwardPropagation&,
                                BackPropagation&) const;

    void calculate_multiple_error(const Batch&,
                                  const ForwardPropagation&,
                                  BackPropagation&) const;

    // Gradient

    void calculate_output_gradients(const Batch&,
                                    ForwardPropagation&,
                                    BackPropagation&) const override;

    void calculate_binary_output_gradients(const Batch&,
                                           ForwardPropagation&,
                                           BackPropagation&) const;

    void calculate_multiple_output_gradients(const Batch&,
                                             ForwardPropagation&,
                                             BackPropagation&) const;

    // Serialization

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

#ifdef OPENNN_CUDA

public:

    // Error

    void calculate_error(const BatchCuda&,
                              const ForwardPropagationCuda&,
                              BackPropagationCuda&) const override;

    void calculate_binary_error(const BatchCuda&,
                                     const ForwardPropagationCuda&,
                                     BackPropagationCuda&) const;

    void calculate_multiple_error(const BatchCuda&,
                                       const ForwardPropagationCuda&,
                                       BackPropagationCuda&) const;

    // Gradient

    void calculate_output_gradients(const BatchCuda&,
                                     ForwardPropagationCuda&,
                                     BackPropagationCuda&) const override;

    void calculate_binary_output_gradients(const BatchCuda&,
                                            ForwardPropagationCuda&,
                                            BackPropagationCuda&) const;

    void calculate_multiple_output_gradients(const BatchCuda&,
                                              ForwardPropagationCuda&,
                                              BackPropagationCuda&) const;

#endif

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
