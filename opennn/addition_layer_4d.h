//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   4 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
/*
#ifndef ADDITIONLAYER4D_H
#define ADDITIONLAYER4D_H

#include "layer.h"

namespace opennn
{

    class Addition4d : public Layer
    {

    public:

        Addition4d(const dimensions& new_input_dimensions = { 0, 0, 0 },
                   const string& new_name = "addition_layer_4d");

        Index get_input_height() const;
        Index get_input_width() const;
        Index get_input_channels() const;

        dimensions get_input_dimensions() const override;
        dimensions get_output_dimensions() const override;

        void set(const dimensions& new_input_dimensions,
                 const string& new_name = "addition_layer_4d");

        void set_input_dimensions(const dimensions&) override;

        void forward_propagate(const vector<pair<type*, dimensions>>&,
                               unique_ptr<LayerForwardPropagation>&,
                               const bool&) override;

        void back_propagate(const vector<pair<type*, dimensions>>&,
                            const vector<pair<type*, dimensions>>&,
                            unique_ptr<LayerForwardPropagation>&,
                            unique_ptr<LayerBackPropagation>&) const override;

        void from_XML(const XMLDocument&) override;
        void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA
    public:

        void forward_propagate_cuda(const vector<float*>&,
                                    unique_ptr<LayerForwardPropagationCuda>&,
                                    const bool&) override;

        void back_propagate_cuda(const vector<float*>&,
                                 const vector<float*>&,
                                 unique_ptr<LayerForwardPropagationCuda>&,
                                 unique_ptr<LayerBackPropagationCuda>&) const override;
#endif

    private:

        dimensions input_dimensions;
    };


    struct Addition4dForwardPropagation : LayerForwardPropagation
    {
        Addition4dForwardPropagation(const Index& = 0, Layer* = nullptr);

        pair<type*, dimensions> get_output_pair() const override;

        void set(const Index& = 0, Layer* = nullptr) override;

        void print() const override;

        Tensor<type, 4> outputs;
    };


    struct Addition4dBackPropagation : LayerBackPropagation
    {
        Addition4dBackPropagation(const Index& = 0, Layer* = nullptr);

        vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

        void set(const Index& = 0, Layer* = nullptr) override;

        void print() const override;

        Tensor<type, 4> input_1_derivatives;
        Tensor<type, 4> input_2_derivatives;
    };


#ifdef OPENNN_CUDA

    struct Addition4dForwardPropagationCuda : public LayerForwardPropagationCuda
    {
        Addition4dForwardPropagationCuda(const Index& = 0, Layer* = nullptr);

        void set(const Index& = 0, Layer* = nullptr) override;

        void print() const override;

        void free() override;
    };

    struct Addition4dBackPropagationCuda : public LayerBackPropagationCuda
    {
        Addition4dBackPropagationCuda(const Index& = 0, Layer* = nullptr);

        vector<float*> get_input_derivatives_device() override;

        void set(const Index& = 0, Layer* = nullptr) override;

        void print() const override;

        void free() override;

        float* inputs_1_derivatives = nullptr;
        float* inputs_2_derivatives = nullptr;
    };

#endif // OPENNN_CUDA

} // namespace opennn

#endif // ADDITIONLAYER4D_H
*/
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
