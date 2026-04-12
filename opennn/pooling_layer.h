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

class Pooling final : public Layer
{

public:

    Pooling(const Shape& = {2, 2, 1}, // Input shape {height,width,channels}
            const Shape& = { 2, 2 },  // Pool shape {pool_height,pool_width}
            const Shape& = { 2, 2 },  // Stride shape {row_stride, column_stride}
            const Shape& = { 0, 0 },  // Padding shape {padding_height, padding_width}
            const string& = "MaxPooling",
            const string& = "pooling_layer");

#ifdef CUDA
    ~Pooling()
    {
        if(pooling_descriptor) cudnnDestroyPoolingDescriptor(pooling_descriptor);
    }
#endif

    void set(const Shape& = { 0, 0, 0 },
             const Shape& = { 1, 1 },
             const Shape& = { 1, 1 },
             const Shape& = { 0, 0 },
             const string & = "MaxPooling",
             const string & = "pooling_layer");

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Shape out_shape = get_output_shape(); // {out_h, out_w, channels}

        vector<Shape> shapes;

        if (pooling_method == "MaxPooling")
            shapes.push_back(Shape{batch_size}.append(out_shape)); // MaximalIndices

        shapes.push_back(Shape{batch_size}.append(out_shape)); // Outputs (must be last for wiring)

        return shapes;
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{batch_size, get_input_height(), get_input_width(), get_channels_number()}};
    }

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override;

    Index get_input_height() const { return input_shape[0]; }
    Index get_input_width() const { return input_shape[1]; }

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_channels_number() const { return input_shape[2]; }

    Index get_padding_height() const { return padding_height; }
    Index get_padding_width() const { return padding_width; }

    Index get_row_stride() const { return row_stride; }
    Index get_column_stride() const { return column_stride; }

    Index get_pool_height() const { return pool_height; }
    Index get_pool_width() const { return pool_width; }

    string get_pooling_method() const { return pooling_method; }

    void set_input_shape(const Shape&) override;

    void set_padding_height(const Index h) { padding_height = h; }
    void set_padding_width(const Index w) { padding_width = w; }

    void set_row_stride(const Index s) { row_stride = s; }
    void set_column_stride(const Index s) { column_stride = s; }

    void set_pool_size(const Index, Index);

    void set_pooling_method(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

private:

    enum Forward {Inputs, MaximalIndices, Outputs};
    enum Backward {OutputGradients, InputGradients};

    Index pool_height = 1;
    Index pool_width = 1;

    Index padding_height = 0;
    Index padding_width = 0;

    Index row_stride = 1;
    Index column_stride = 1;

    string pooling_method = "MaxPooling";

    PoolingArguments cached_pool_args;

#ifdef CUDA

    cudnnPoolingMode_t pooling_mode = cudnnPoolingMode_t::CUDNN_POOLING_MAX;

    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;

#endif

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
