//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "pooling_layer_3d.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Pooling3d::Pooling3d(const Shape& new_input_shape,
                     const PoolingMethod& new_pooling_method,
                     const string& new_name)
    : Layer("Pooling3d", LayerType::Pooling3d)
{
    operators = {&pool3d};
    set(new_input_shape, new_pooling_method, new_name);
}

Shape Pooling3d::get_output_shape() const
{
    return {input_features};
}

string Pooling3d::write_pooling_method() const
{
    return pooling_method_to_string(pooling_method);
}

vector<pair<Shape, Type>> Pooling3d::get_forward_specs(Index batch_size) const
{
    return {
        {(pooling_method == PoolingMethod::MaxPooling)
            ? Shape{batch_size, input_features}
            : Shape{},
         Type::FP32},                                  // MaximalIndices
        {{batch_size, input_features}, compute_dtype}, // Output (must be last)
    };
}

void Pooling3d::set(const Shape& new_input_shape,
                    const PoolingMethod& new_pooling_method,
                    const string& new_label)
{
    sequence_length = new_input_shape.dim_or_zero(0);
    input_features  = new_input_shape.dim_or_zero(1);

    pooling_method = new_pooling_method;

    set_label(new_label);

    pool3d.method = (pooling_method == PoolingMethod::MaxPooling) ? Pool3dOp::Max : Pool3dOp::Average;
    pool3d.input_slots  = {Input};
    pool3d.output_slots = {Output, MaximalIndices};

    pool3d.output_delta_slots = {OutputDelta};
    pool3d.input_delta_slots  = {InputDelta};
}

void Pooling3d::set_pooling_method(const string& new_pooling_method)
{
    pooling_method = string_to_pooling_method(new_pooling_method);
    pool3d.method = (pooling_method == PoolingMethod::MaxPooling) ? Pool3dOp::Max : Pool3dOp::Average;
}

void Pooling3d::read_JSON_body(const Json* element)
{
    set_pooling_method(read_json_string(element, "PoolingMethod"));
}

void Pooling3d::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "PoolingMethod", write_pooling_method());
}

REGISTER(Layer, Pooling3d, "Pooling3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
