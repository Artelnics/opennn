//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operator.h"

namespace opennn
{

struct PoolOperator : Operator
{
    enum Method { Max, Average };

    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index pool_height = 1;
    Index pool_width = 1;
    Index row_stride = 1;
    Index column_stride = 1;
    Index padding_height = 0;
    Index padding_width = 0;

    Method method = Max;

    void set(Index, Index, Index,
             Index, Index,
             Index, Index,
             Index, Index,
             Method);

    PoolOperator() = default;
    PoolOperator(const PoolOperator&) = delete;
    PoolOperator& operator=(const PoolOperator&) = delete;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

#ifdef OPENNN_HAS_CUDA
    cudnnPoolingDescriptor_t get_pooling_descriptor() const;

private:

    mutable shared_ptr<cudnnPoolingStruct> pooling_descriptor;
#endif
};

enum class PoolingMethod
{
    MaxPooling,
    AveragePooling,
    FirstToken
};

const string& pooling_method_to_string(PoolingMethod);
PoolingMethod string_to_pooling_method(const string&);

class Pooling final : public Layer
{
public:

    Pooling(const Shape& = {2, 2, 1},
            const Shape& = { 2, 2 },
            const Shape& = { 2, 2 },
            const Shape& = { 0, 0 },
            const string& = "MaxPooling",
            const string& = "pooling_layer");

    Shape get_input_shape() const noexcept override { return {input_height, input_width, input_channels}; }
    Shape get_output_shape() const override;

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_pool_height() const noexcept { return pool_height; }
    Index get_pool_width() const noexcept { return pool_width; }

    Index get_row_stride() const noexcept { return row_stride; }
    Index get_column_stride() const noexcept { return column_stride; }

    Index get_padding_height() const noexcept { return padding_height; }
    Index get_padding_width() const noexcept { return padding_width; }

    PoolingMethod get_pooling_method() const noexcept { return pooling_method; }

    bool is_passthrough() const noexcept;

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void set(const Shape& = { 0, 0, 0 },
             const Shape& = { 1, 1 },
             const Shape& = { 1, 1 },
             const Shape& = { 0, 0 },
             const string & = "MaxPooling",
             const string & = "pooling_layer");

    void set_input_shape(const Shape&) override;
    void set_pooling_method(const string&);

    void read_JSON_body(const Json*) override;
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

    PoolOperator pool;

    enum Forward {Input, MaximalIndices, Output};

    void update_pool_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
