//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Convolutional;

class Pooling final : public Layer
{

public:

    Pooling(const Shape& = {2, 2, 1}, // Input shape {height,width,channels}
            const Shape& = { 2, 2 },  // Pool shape {pool_height,pool_width}
            const Shape& = { 2, 2 },  // Stride shape {row_stride, column_stride}
            const Shape& = { 0, 0 },  // Padding shape {padding_height, padding_width}
            const string& = "MaxPooling",
            const string& = "pooling_layer");

    void set(const Shape& = { 0, 0, 0 },
             const Shape& = { 1, 1 },
             const Shape& = { 1, 1 },
             const Shape& = { 0, 0 },
             const string & = "MaxPooling",
             const string & = "pooling_layer");

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    Index get_input_height() const;
    Index get_input_width() const;

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_channels_number() const;

    Index get_padding_height() const;
    Index get_padding_width() const;

    Index get_row_stride() const;
    Index get_column_stride() const;

    Index get_pool_height() const;
    Index get_pool_width() const;

    string get_pooling_method() const;

    void set_input_shape(const Shape&) override;

    void set_padding_height(const Index);
    void set_padding_width(const Index);

    void set_row_stride(const Index);
    void set_column_stride(const Index);

    void set_pool_size(const Index, Index);

    void set_pooling_method(const string&);

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    void forward_propagate_max_pooling(const Tensor4&,
                                       unique_ptr<LayerForwardPropagation>&,
                                       bool) const;

    void forward_propagate_average_pooling(const Tensor4&,
                                           unique_ptr<LayerForwardPropagation>&,
                                           bool) const;

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void back_propagate_max_pooling(const Tensor4&,
                                    const Tensor4&,
                                    unique_ptr<LayerForwardPropagation>&,
                                    unique_ptr<LayerBackPropagation>&) const;

    void back_propagate_average_pooling(const Tensor4&,
                                        const Tensor4&,
                                        unique_ptr<LayerBackPropagation>&) const;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

#ifdef OPENNN_CUDA

public:

    void forward_propagate(const vector<TensorViewCuda>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                bool) override;

    void back_propagate(const vector<TensorViewCuda>&,
                        const vector<TensorViewCuda>&,
                        unique_ptr<LayerForwardPropagationCuda>&,
                        unique_ptr<LayerBackPropagationCuda>&) const override;

#endif

private:

    Shape input_shape;

    Index pool_height = 1;

    Index pool_width = 1;

    Index padding_height = 0;

    Index padding_width = 0;

    Index row_stride = 1;

    Index column_stride = 1;

    string pooling_method = "MaxPooling";

#ifdef OPENNN_CUDA

    cudnnPoolingMode_t pooling_mode = cudnnPoolingMode_t::CUDNN_POOLING_MAX;

    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;

#endif

};


struct PoolingForwardPropagation final : LayerForwardPropagation
{
    PoolingForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorView*> get_workspace_views() override;

    void print() const override;

    Tensor<Index, 4> maximal_indices;
};


struct PoolingBackPropagation final : LayerBackPropagation
{
    PoolingBackPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
};


#ifdef OPENNN_CUDA

struct PoolingForwardPropagationCuda : public LayerForwardPropagationCuda
{
    PoolingForwardPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorViewCuda*> get_workspace_views() override;

    void print() const override;

    void free() override;

    cudnnTensorDescriptor_t input_tensor_descriptor = nullptr;
};


struct PoolingBackPropagationCuda : public LayerBackPropagationCuda
{
    PoolingBackPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
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
