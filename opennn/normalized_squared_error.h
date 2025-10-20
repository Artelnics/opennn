//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZEDSQUAREDERROR_H
#define NORMALIZEDSQUAREDERROR_H

#include "loss_index.h"

namespace opennn
{

class NormalizedSquaredError final : public LossIndex
{

public:

    NormalizedSquaredError(const NeuralNetwork* = nullptr, const Dataset* = nullptr);

    void set_normalization_coefficient() override;

    void set_time_series_normalization_coefficient();

    void set_default();

    void set_dataset(const Dataset* new_dataset) override;

    type calculate_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

    type calculate_time_series_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const override;

    void calculate_output_delta(const Batch&,
                                ForwardPropagation&,
                                BackPropagation&) const override;

    void calculate_error_lm(const Batch&,
                            const ForwardPropagation&,
                            BackPropagationLM&) const override;

    void calculate_output_delta_lm(const Batch&,
                                   ForwardPropagation&,
                                   BackPropagationLM&) const override;

    void calculate_error_gradient_lm(const Batch&,
                                     BackPropagationLM&) const override;

    void calculate_error_hessian_lm(const Batch&,
                                    BackPropagationLM&) const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

protected:

    type normalization_coefficient = type(1);


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
