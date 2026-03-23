//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Normalization3d final : public Layer
{

public:

    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    vector<TensorView*> get_parameter_views() override;

    void set(const Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

    vector<TensorViewCuda*> get_parameter_views_device() override;

    void forward_propagate(unique_ptr<LayerForwardPropagationCuda>&,
                           bool) override;

    void back_propagate(const vector<TensorViewCuda>&,
                        const vector<TensorViewCuda>&,
                        unique_ptr<LayerForwardPropagationCuda>&,
                        unique_ptr<LayerBackPropagationCuda>&) const override;

protected:

    TensorViewCuda gammas_device;
    TensorViewCuda betas_device;

#endif

private:

    Index sequence_length;

    TensorView gammas;
    TensorView betas;
};


struct Normalization3dForwardPropagation final : LayerForwardPropagation
{
    Normalization3dForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;

    Tensor2 means;
    Tensor2 standard_deviations;
    Tensor3 normalized_inputs;
};


struct Normalization3dBackPropagation final : LayerBackPropagation
{
    Normalization3dBackPropagation(const Index new_batch_size = 0,
                                   Layer* new_layer = nullptr);

    vector<TensorView*> get_gradient_views() override;

    void initialize() override;

    void print() const override;

    TensorView gamma_gradients;
    TensorView beta_gradients;
};


#ifdef OPENNN_CUDA

struct Normalization3dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    Normalization3dForwardPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void free() override {}

    TensorCuda means_device;
    TensorCuda inv_variances_device;
};


struct Normalization3dBackPropagationCuda : public LayerBackPropagationCuda
{
    Normalization3dBackPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr);

    vector<TensorViewCuda*> get_gradient_views() override;

    void initialize() override;

    TensorViewCuda gamma_gradients;
    TensorViewCuda beta_gradients;
};

#endif

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
