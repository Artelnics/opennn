//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/// @brief Pooling reduction method used by Pooling and Pooling3d layers.
enum class PoolingMethod
{
    MaxPooling,
    AveragePooling
};

inline const string& pooling_method_to_string(PoolingMethod method)
{
    static const string max_str = "MaxPooling";
    static const string avg_str = "AveragePooling";
    return method == PoolingMethod::MaxPooling ? max_str : avg_str;
}

inline PoolingMethod string_to_pooling_method(const string& name)
{
    if (name == "MaxPooling")     return PoolingMethod::MaxPooling;
    if (name == "AveragePooling") return PoolingMethod::AveragePooling;
    throw runtime_error(format("Unknown pooling method: {}", name));
}

/// @brief 2D spatial pooling layer supporting max and average reduction.
class Pooling final : public Layer
{
public:

    /// @brief Constructs a pooling layer with given input, pool, stride and padding shapes.
    /// @param input_shape Shape of the input tensor (height, width, channels).
    /// @param pool_shape Pooling window size (height, width).
    /// @param stride_shape Stride along height and width.
    /// @param padding_shape Padding along height and width.
    /// @param pooling_method "MaxPooling" or "AveragePooling".
    /// @param name Layer name used for serialization.
    Pooling(const Shape& = {2, 2, 1},
            const Shape& = { 2, 2 },
            const Shape& = { 2, 2 },
            const Shape& = { 0, 0 },
            const string& = "MaxPooling",
            const string& = "pooling_layer");

    /// @brief Returns the input tensor shape (height, width, channels).
    Shape get_input_shape() const override { return {input_height, input_width, input_channels}; }

    /// @brief Returns the output tensor shape after pooling.
    Shape get_output_shape() const override;

    /// @brief Returns the output feature map height.
    Index get_output_height() const;

    /// @brief Returns the output feature map width.
    Index get_output_width() const;

    Index get_input_height() const { return input_height; }
    Index get_input_width() const { return input_width; }
    Index get_input_channels() const { return input_channels; }

    Index get_pool_height() const { return pool_height; }
    Index get_pool_width() const { return pool_width; }

    Index get_row_stride() const { return row_stride; }
    Index get_column_stride() const { return column_stride; }

    Index get_padding_height() const { return padding_height; }
    Index get_padding_width() const { return padding_width; }

    PoolingMethod get_pooling_method() const { return pooling_method; }

    /// @brief Returns the tensor specifications used during forward propagation.
    vector<TensorSpec> get_forward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with new shapes and pooling method.
    void set(const Shape& = { 0, 0, 0 },
             const Shape& = { 1, 1 },
             const Shape& = { 1, 1 },
             const Shape& = { 0, 0 },
             const string & = "MaxPooling",
             const string & = "pooling_layer");

    /// @brief Updates the layer for a new input shape.
    void set_input_shape(const Shape&) override;

    /// @brief Sets the pooling window height and width.
    void set_pool_size(Index, Index);

    void set_row_stride(Index);
    void set_column_stride(Index);
    void set_padding_height(Index);
    void set_padding_width(Index);

    /// @brief Sets the pooling method by name ("MaxPooling" or "AveragePooling").
    void set_pooling_method(const string&);

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

    /// @brief Writes the layer configuration to a JSON writer.
    void write_JSON_body(JsonWriter&) const override;

private:

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

    PoolOp pool;

    enum Forward {Input, MaximalIndices, Output};

    void update_pool_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
