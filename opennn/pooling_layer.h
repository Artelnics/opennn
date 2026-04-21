//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

enum class PoolingMethod
{
    MaxPooling,
    AveragePooling
};

inline const EnumMap<PoolingMethod>& pooling_method_map()
{
    static const vector<pair<PoolingMethod, string>> entries = {
        {PoolingMethod::MaxPooling,     "MaxPooling"},
        {PoolingMethod::AveragePooling, "AveragePooling"}
    };
    static const EnumMap<PoolingMethod> map{entries};
    return map;
}

inline const string& pooling_method_to_string(PoolingMethod method)
{
    return pooling_method_map().to_string(method);
}

inline PoolingMethod string_to_pooling_method(const string& name)
{
    return pooling_method_map().from_string(name);
}

class Pooling final : public Layer
{

public:

    Pooling(const Shape& = {2, 2, 1},
            const Shape& = { 2, 2 },
            const Shape& = { 2, 2 },
            const Shape& = { 0, 0 },
            const string& = "MaxPooling",
            const string& = "pooling_layer");

    ~Pooling()
    {
#ifdef OPENNN_WITH_CUDA
        destroy_cuda();
#endif
    }

    void destroy_cuda();

    // Getters

    Shape get_input_shape() const override { return {input_height, input_width, input_channels}; }
    Shape get_output_shape() const override;

    Index get_input_height() const { return input_height; }
    Index get_input_width() const { return input_width; }
    Index get_channels_number() const { return input_channels; }

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_pool_height() const { return pool_height; }
    Index get_pool_width() const { return pool_width; }

    Index get_row_stride() const { return row_stride; }
    Index get_column_stride() const { return column_stride; }

    Index get_padding_height() const { return padding_height; }
    Index get_padding_width() const { return padding_width; }

    PoolingMethod get_pooling_method() const { return pooling_method; }

    // Setters

    void set(const Shape& = { 0, 0, 0 },
             const Shape& = { 1, 1 },
             const Shape& = { 1, 1 },
             const Shape& = { 0, 0 },
             const string & = "MaxPooling",
             const string & = "pooling_layer");

    void set_input_shape(const Shape&) override;

    void set_pool_size(const Index, Index);

    void set_row_stride(const Index s) { row_stride = s; }
    void set_column_stride(const Index s) { column_stride = s; }

    void set_padding_height(const Index h) { padding_height = h; }
    void set_padding_width(const Index w) { padding_width = w; }

    void set_pooling_method(const string&);

    // Forward / back propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

private:

    enum Forward {Input, MaximalIndices, Output};
    enum Backward {OutputGradient, InputGradient};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Shape out_shape = get_output_shape();

        vector<Shape> shapes;

        if (pooling_method == PoolingMethod::MaxPooling)
            shapes.push_back(Shape{batch_size}.append(out_shape)); // MaximalIndices

        shapes.push_back(Shape{batch_size}.append(out_shape)); // Output (must be last)

        return shapes;
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{batch_size, input_height, input_width, input_channels}};
    }

    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index pool_height = 1;
    Index pool_width = 1;

    Index padding_height = 0;
    Index padding_width = 0;

    Index row_stride = 1;
    Index column_stride = 1;

    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    PoolingArguments cached_pool_args;

    cudnnPoolingMode_t pooling_mode = cudnnPoolingMode_t::CUDNN_POOLING_MAX;
    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
