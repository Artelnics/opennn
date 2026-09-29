// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/network/onnx_export.h"

#include <cstdint>
#include <cstring>
#include <numeric>

#include "opennn/network/layers/layer_registry.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/scaling_layer.h"

#ifndef OPENNN_NO_VISION
#include "opennn/network/layers/convolutional_layer.h"
#include "opennn/network/layers/detection_v8_layer.h"
#include "opennn/network/layers/pooling_layer.h"
#include "opennn/network/layers/upsampling_layer.h"
#endif
#include "opennn/core/tensor_operations.h"

namespace opennn
{

namespace
{

// Minimal protobuf encoder: only the wire types ONNX uses.

class Message
{
public:

    void add_int(int field, int64_t value)
    {
        write_key(field, 0);
        write_varint(uint64_t(value));
    }

    void add_float(int field, float value)
    {
        write_key(field, 5);
        char bytes[4];
        memcpy(bytes, &value, 4);
        buffer.append(bytes, 4);
    }

    void add_bytes(int field, const string& value)
    {
        write_key(field, 2);
        write_varint(value.size());
        buffer += value;
    }

    void add_message(int field, const Message& message) { add_bytes(field, message.buffer); }

    const string& bytes() const { return buffer; }

private:

    void write_key(int field, int wire_type) { write_varint(uint64_t(field) << 3 | uint64_t(wire_type)); }

    void write_varint(uint64_t value)
    {
        while (value >= 0x80)
        {
            buffer += char((value & 0x7F) | 0x80);
            value >>= 7;
        }
        buffer += char(value);
    }

    string buffer;
};

// Field numbers and enum values from onnx.proto.

enum TensorElementType { ElementFloat = 1, ElementInt64 = 7, ElementBool = 9 };
enum AttributeType { AttributeFloat = 1, AttributeInt = 2, AttributeString = 3, AttributeInts = 7 };

constexpr int64_t IR_VERSION = 7;
constexpr int64_t OPSET_VERSION = 13;

struct Attribute
{
    string name;
    AttributeType type = AttributeInt;
    float f = 0.0f;
    int64_t i = 0;
    string s;
    vector<int64_t> ints;
};

Attribute int_attribute(const string& name, int64_t value) { return {name, AttributeInt, 0.0f, value, {}, {}}; }
Attribute float_attribute(const string& name, float value) { return {name, AttributeFloat, value, 0, {}, {}}; }
Attribute string_attribute(const string& name, const string& value) { return {name, AttributeString, 0.0f, 0, value, {}}; }
Attribute ints_attribute(const string& name, const vector<int64_t>& values) { return {name, AttributeInts, 0.0f, 0, {}, values}; }

struct Dimension
{
    string symbol;
    int64_t value = 0;
};

class GraphBuilder
{
public:

    // A float tensor with the given dimensions, stored as raw little-endian data.
    string add_initializer(const string& hint, const vector<int64_t>& dims, const float* data, Index size)
    {
        const string name = unique_name(hint);

        Message tensor;
        for (const int64_t dim : dims) tensor.add_int(1, dim);
        tensor.add_int(2, ElementFloat);
        tensor.add_bytes(8, name);
        tensor.add_bytes(9, string(reinterpret_cast<const char*>(data), size_t(size) * sizeof(float)));

        initializers.push_back(tensor);
        return name;
    }

    string add_initializer(const string& hint, const vector<float>& values)
    {
        return add_initializer(hint, {int64_t(values.size())}, values.data(), Index(values.size()));
    }

    string add_scalar(const string& hint, float value) { return add_initializer(hint, {1}, &value, 1); }

    string add_int64_initializer(const string& hint, const vector<int64_t>& values)
    {
        const string name = unique_name(hint);

        Message tensor;
        tensor.add_int(1, int64_t(values.size()));
        tensor.add_int(2, ElementInt64);
        tensor.add_bytes(8, name);
        tensor.add_bytes(9, string(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(int64_t)));

        initializers.push_back(tensor);
        return name;
    }

    string add_mask(const string& hint, const vector<bool>& mask)
    {
        const string name = unique_name(hint);

        string data;
        for (const bool value : mask) data += char(value ? 1 : 0);

        Message tensor;
        tensor.add_int(1, int64_t(mask.size()));
        tensor.add_int(2, ElementBool);
        tensor.add_bytes(8, name);
        tensor.add_bytes(9, data);

        initializers.push_back(tensor);
        return name;
    }

    string add_node(const string& op_type, const vector<string>& inputs, const vector<Attribute>& attributes = {},
                    const string& output = {})
    {
        const string output_name = output.empty() ? unique_name(op_type) : output;

        Message node;
        for (const string& input : inputs) node.add_bytes(1, input);
        node.add_bytes(2, output_name);
        node.add_bytes(3, unique_name("node_" + op_type));
        node.add_bytes(4, op_type);

        for (const Attribute& attribute : attributes)
        {
            Message encoded;
            encoded.add_bytes(1, attribute.name);
            switch (attribute.type)
            {
            case AttributeFloat:  encoded.add_float(2, attribute.f); break;
            case AttributeInt:    encoded.add_int(3, attribute.i); break;
            case AttributeString: encoded.add_bytes(4, attribute.s); break;
            case AttributeInts:   for (const int64_t value : attribute.ints) encoded.add_int(8, value); break;
            }
            encoded.add_int(20, attribute.type);
            node.add_message(5, encoded);
        }

        nodes.push_back(node);
        return output_name;
    }

    Message build(const Message& input, const Message& output) const
    {
        Message graph;
        for (const Message& node : nodes) graph.add_message(1, node);
        graph.add_bytes(2, "opennn_network");
        for (const Message& initializer : initializers) graph.add_message(5, initializer);
        graph.add_message(11, input);
        graph.add_message(12, output);
        return graph;
    }

    Index nodes_number() const { return Index(nodes.size()); }

    static Message value_info(const string& name, const vector<Dimension>& dimensions)
    {
        Message shape;
        for (const Dimension& dimension : dimensions)
        {
            Message encoded;
            if (dimension.symbol.empty()) encoded.add_int(1, dimension.value);
            else                          encoded.add_bytes(2, dimension.symbol);
            shape.add_message(1, encoded);
        }

        Message tensor_type;
        tensor_type.add_int(1, ElementFloat);
        tensor_type.add_message(2, shape);

        Message type;
        type.add_message(1, tensor_type);

        Message info;
        info.add_bytes(1, name);
        info.add_message(2, type);
        return info;
    }

private:

    string unique_name(const string& hint) { return hint + "_" + to_string(counter++); }

    vector<Message> nodes;
    vector<Message> initializers;
    int counter = 0;
};

vector<float> to_vector(const VectorR& values)
{
    return vector<float>(values.data(), values.data() + values.size());
}

// Scaling and Unscaling: an affine map per feature, or log / exp for the
// Logarithm scaler, selected per feature with a constant mask.
string add_scaling(GraphBuilder& graph, const Scaling& layer, const string& input)
{
    const bool inverse = layer.is_inverse();
    const vector<Descriptives>& descriptives = layer.get_descriptives();
    const vector<ScalerMethod>& scalers = layer.get_scalers();
    const Index features = layer.get_outputs_number();

    throw_if(ssize(scalers) != features || ssize(descriptives) != features,
             "ONNX export: layer '{}' has {} scalers for {} features.", layer.get_label(), scalers.size(), features);

    vector<float> slopes(size_t(features), 1.0f);
    vector<float> offsets(size_t(features), 0.0f);
    vector<bool> logarithmic(size_t(features), false);
    bool any_affine = false;
    bool any_logarithmic = false;

    for (size_t i = 0; i < size_t(features); ++i)
    {
        if (scalers[i] == ScalerMethod::Logarithm)
        {
            logarithmic[i] = any_logarithmic = true;
            continue;
        }

        const AffineMap affine = inverse
            ? unscaling_affine(scalers[i], descriptives[i], layer.get_min_range(), layer.get_max_range())
            : scaling_affine(scalers[i], descriptives[i], layer.get_min_range(), layer.get_max_range());

        slopes[i] = affine.slope;
        offsets[i] = affine.offset;
        any_affine = any_affine || affine.slope != 1.0f || affine.offset != 0.0f;
    }

    string output = input;

    if (any_affine)
    {
        output = graph.add_node("Mul", {input, graph.add_initializer(layer.get_label() + "_slope", slopes)});
        output = graph.add_node("Add", {output, graph.add_initializer(layer.get_label() + "_offset", offsets)});
    }

    if (any_logarithmic)
    {
        const string logarithm = inverse
            ? graph.add_node("Exp", {input})
            : graph.add_node("Log", {graph.add_node("Max", {input, graph.add_scalar("epsilon", EPSILON)})});

        output = graph.add_node("Where", {graph.add_mask(layer.get_label() + "_logarithmic", logarithmic),
                                          logarithm, output});
    }

    return output;
}

string add_activation(GraphBuilder& graph, ActivationFunction activation, const string& input,
                      int64_t softmax_axis = -1)
{
    switch (activation)
    {
    case ActivationFunction::Identity:  return input;
    case ActivationFunction::Sigmoid:   return graph.add_node("Sigmoid", {input});
    case ActivationFunction::Tanh:      return graph.add_node("Tanh", {input});
    case ActivationFunction::ReLU:      return graph.add_node("Relu", {input});
    case ActivationFunction::Softmax:   return graph.add_node("Softmax", {input}, {int_attribute("axis", softmax_axis)});
    case ActivationFunction::LeakyReLU:
        return graph.add_node("LeakyRelu", {input}, {float_attribute("alpha", LEAKY_RELU_SLOPE)});
    case ActivationFunction::SiLU:
        return graph.add_node("Mul", {input, graph.add_node("Sigmoid", {input})});
    case ActivationFunction::GELU:
    {
        // 0.5 x (1 + erf(x / sqrt(2)))
        const string erf = graph.add_node("Erf", {graph.add_node("Mul", {input, graph.add_scalar("inv_sqrt_2", INV_SQRT_2)})});
        const string one_plus = graph.add_node("Add", {erf, graph.add_scalar("one", 1.0f)});
        return graph.add_node("Mul", {graph.add_node("Mul", {input, one_plus}), graph.add_scalar("half", 0.5f)});
    }
    case ActivationFunction::GELUTanh:
    {
        // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
        const string cube = graph.add_node("Mul", {input, graph.add_node("Mul", {input, input})});
        const string inner = graph.add_node("Add", {input, graph.add_node("Mul", {cube, graph.add_scalar("gelu_cubic", GELU_TANH_CUBIC)})});
        const string hyperbolic = graph.add_node("Tanh", {graph.add_node("Mul", {inner, graph.add_scalar("sqrt_2_over_pi", SQRT_2_OVER_PI)})});
        const string one_plus = graph.add_node("Add", {hyperbolic, graph.add_scalar("one", 1.0f)});
        return graph.add_node("Mul", {graph.add_node("Mul", {input, one_plus}), graph.add_scalar("half", 0.5f)});
    }
    }

    throw runtime_error("ONNX export: unknown activation function.");
}

// Dense: y = activation(x W + b), W stored row-major as [inputs, outputs] and
// the bias first when the layer has one (CombinationOperator::parameter_specs).
string add_dense(GraphBuilder& graph, const Dense& layer, const string& input)
{
    const string& label = layer.get_label();

    throw_if(layer.get_batch_normalization(),
             "ONNX export: layer '{}' uses batch normalization, which is not supported yet.", label);
    throw_if(layer.get_gated(),
             "ONNX export: layer '{}' is gated (SwiGLU), which is not supported yet.", label);
    throw_if(layer.get_tied_weight().source != nullptr,
             "ONNX export: layer '{}' shares its weights with another layer, which is not supported.", label);

    const vector<TensorView>& views = layer.get_parameter_views();
    const bool use_bias = layer.get_use_bias();

    throw_if(views.size() != (use_bias ? 2u : 1u), "ONNX export: layer '{}' is not configured.", label);

    for (const TensorView& view : views)
        throw_if(!view.get_data() || view.get_type() != Type::FP32,
                 "ONNX export: layer '{}' does not hold FP32 parameters.", label);

    const Index inputs_number = layer.get_inputs_number();
    const Index outputs_number = layer.get_outputs_number();
    const TensorView& weights = views.back();

    throw_if(weights.size() != inputs_number * outputs_number,
             "ONNX export: layer '{}' has an unexpected weight shape.", label);

    const string weights_name = graph.add_initializer(label + "_weights", {inputs_number, outputs_number},
                                                      weights.as<float>(), weights.size());

    const string combination = use_bias
        ? graph.add_node("Gemm", {input, weights_name,
                                  graph.add_initializer(label + "_bias", {outputs_number},
                                                        views.front().as<float>(), outputs_number)})
        : graph.add_node("MatMul", {input, weights_name});

    return add_activation(graph, layer.get_activation_function(), combination);
}

string add_clamping(GraphBuilder& graph, const Clamping& layer, const string& input)
{
    if (layer.get_clamping_method() == Clamping::ClampingMethod::NoClamping)
        return input;

    const string lower = graph.add_initializer(layer.get_label() + "_lower", to_vector(layer.get_lower_bounds()));
    const string upper = graph.add_initializer(layer.get_label() + "_upper", to_vector(layer.get_upper_bounds()));

    return graph.add_node("Min", {graph.add_node("Max", {input, lower}), upper});
}


#ifndef OPENNN_NO_VISION

string add_slice(GraphBuilder& graph, const string& input, int64_t axis, int64_t start, int64_t end)
{
    return graph.add_node("Slice", {input,
                                    graph.add_int64_initializer("starts", {start}),
                                    graph.add_int64_initializer("ends", {end}),
                                    graph.add_int64_initializer("axes", {axis})});
}

string add_reshape(GraphBuilder& graph, const string& input, const vector<int64_t>& shape)
{
    return graph.add_node("Reshape", {input, graph.add_int64_initializer("shape", shape)});
}

string add_convolution(GraphBuilder& graph, const Convolutional& layer, const string& input)
{
    const string& label = layer.get_label();

    throw_if(layer.get_residual(),
             "ONNX export: layer '{}' has a residual input, which is not supported yet.", label);

    const Shape input_shape = layer.get_input_shape();
    const Index kernel_height = layer.get_kernel_height();
    const Index kernel_width = layer.get_kernel_width();
    const Index row_stride = layer.get_row_stride();
    const Index column_stride = layer.get_column_stride();
    const Index padding_height = layer.get_padding_height();
    const Index padding_width = layer.get_padding_width();

    throw_if((input_shape[0] + 2 * padding_height - kernel_height) / row_stride + 1 != layer.get_output_height()
             || (input_shape[1] + 2 * padding_width - kernel_width) / column_stride + 1 != layer.get_output_width(),
             "ONNX export: the padding of layer '{}' cannot be represented.", label);

    vector<float> kernel;
    vector<float> bias;
    layer.get_folded_parameters(kernel, bias);

    const string convolution = graph.add_node("Conv",
        {input,
         graph.add_initializer(label + "_kernel",
                               {layer.get_kernels_number(), layer.get_kernel_channels(), kernel_height, kernel_width},
                               kernel.data(), ssize(kernel)),
         graph.add_initializer(label + "_bias", bias)},
        {ints_attribute("kernel_shape", {kernel_height, kernel_width}),
         ints_attribute("strides", {row_stride, column_stride}),
         ints_attribute("pads", {padding_height, padding_width, padding_height, padding_width})});

    return add_activation(graph, layer.get_activation_function(), convolution, 1);
}

string add_pooling(GraphBuilder& graph, const Pooling& layer, const string& input)
{
    throw_if(layer.get_pooling_method() != PoolingMethod::MaxPooling,
             "ONNX export: layer '{}' uses average pooling, which is not supported yet.", layer.get_label());

    const Index padding_height = layer.get_padding_height();
    const Index padding_width = layer.get_padding_width();

    return graph.add_node("MaxPool", {input},
        {ints_attribute("kernel_shape", {layer.get_pool_height(), layer.get_pool_width()}),
         ints_attribute("strides", {layer.get_row_stride(), layer.get_column_stride()}),
         ints_attribute("pads", {padding_height, padding_width, padding_height, padding_width})});
}

string add_upsampling(GraphBuilder& graph, const Upsampling& layer, const string& input)
{
    const float scale = float(layer.get_output_shape()[0]) / float(layer.get_input_shape()[0]);

    return graph.add_node("Resize",
        {input, "", graph.add_initializer(layer.get_label() + "_scales", {1.0f, 1.0f, scale, scale})},
        {string_attribute("mode", "nearest"),
         string_attribute("coordinate_transformation_mode", "asymmetric"),
         string_attribute("nearest_mode", "floor")});
}

string add_detection_v8(GraphBuilder& graph, const DetectionV8& layer, const string& input, const Shape& image_shape)
{
    const string& label = layer.get_label();
    const DetectionHeadMetadata metadata = layer.get_detection_head_metadata();
    const Index regression_bins = metadata.regression_bins;
    const Index classes_number = metadata.classes_number;
    const Index grid_height = layer.get_input_shape()[0];
    const Index grid_width = layer.get_input_shape()[1];
    const Index box_channels = 4 * regression_bins;

    throw_if(regression_bins <= 1,
             "ONNX export: layer '{}' does not regress box distributions (reg_max <= 1), which is not supported.",
             label);

    vector<float> bins(static_cast<size_t>(regression_bins));
    iota(bins.begin(), bins.end(), 0.0f);

    const string distribution = graph.add_node("Softmax",
        {add_reshape(graph, add_slice(graph, input, 1, 0, box_channels), {0, 4, regression_bins, -1})},
        {int_attribute("axis", 2)});

    const string distances = graph.add_node("ReduceSum",
        {graph.add_node("Mul", {distribution,
                                graph.add_initializer(label + "_bins", {1, 1, regression_bins, 1},
                                                      bins.data(), regression_bins)}),
         graph.add_int64_initializer("axes", {2})},
        {int_attribute("keepdims", 0)});

    const string left_top = add_slice(graph, distances, 1, 0, 2);
    const string right_bottom = add_slice(graph, distances, 1, 2, 4);

    const Index cells = grid_height * grid_width;
    vector<float> centres(size_t(2 * cells));
    for (Index row = 0; row < grid_height; ++row)
        for (Index column = 0; column < grid_width; ++column)
        {
            centres[size_t(row * grid_width + column)] = float(column) + 0.5f;
            centres[size_t(cells + row * grid_width + column)] = float(row) + 0.5f;
        }

    const string centre = graph.add_node("Add",
        {graph.add_initializer(label + "_centres", {1, 2, cells}, centres.data(), 2 * cells),
         graph.add_node("Mul", {graph.add_node("Sub", {right_bottom, left_top}), graph.add_scalar("half", 0.5f)})});

    const string size = graph.add_node("Add", {left_top, right_bottom});

    const float stride_x = float(image_shape[1]) / float(grid_width);
    const float stride_y = float(image_shape[0]) / float(grid_height);
    const vector<float> strides = {stride_x, stride_y, stride_x, stride_y};

    const string boxes = graph.add_node("Mul",
        {graph.add_node("Concat", {centre, size}, {int_attribute("axis", 1)}),
         graph.add_initializer(label + "_strides", {1, 4, 1}, strides.data(), 4)});

    const string classes = graph.add_node("Sigmoid",
        {add_reshape(graph, add_slice(graph, input, 1, box_channels, box_channels + classes_number),
                     {0, classes_number, -1})});

    return graph.add_node("Concat", {boxes, classes}, {int_attribute("axis", 1)});
}

#endif

runtime_error unsupported_layer(const Layer& layer)
{
    return runtime_error(format("ONNX export: layer '{}' ({}) cannot be exported to ONNX yet. "
                                "Supported layers: Scaling, Dense, Unscaling and Clamping.",
                                layer.get_label(), layer_type_to_string(layer.get_type())));
}

Message metadata_entry(const string& key, const string& value)
{
    Message entry;
    entry.add_bytes(1, key);
    entry.add_bytes(2, value);
    return entry;
}

Message string_entry(const string& key, const vector<string>& values)
{
    string joined;
    for (size_t i = 0; i < values.size(); ++i)
        joined += (i ? "," : "") + values[i];

    return metadata_entry(key, joined);
}

Message encode_model(const GraphBuilder& graph, const Message& input, const Message& output,
                     const vector<Message>& metadata)
{
    Message opset;
    opset.add_bytes(1, "");
    opset.add_int(2, OPSET_VERSION);

    Message encoded;
    encoded.add_int(1, IR_VERSION);
    encoded.add_bytes(2, "OpenNN");
    encoded.add_message(7, graph.build(input, output));
    encoded.add_message(8, opset);
    for (const Message& entry : metadata)
        encoded.add_message(14, entry);

    return encoded;
}

#ifndef OPENNN_NO_VISION

runtime_error unsupported_detection_layer(const Layer& layer)
{
    return runtime_error(format("ONNX export: layer '{}' ({}) cannot be exported to ONNX yet. "
                                "Object detection networks support Convolutional, Activation, Addition, "
                                "Concatenation, Upsampling, max Pooling and DetectionV8 layers.",
                                layer.get_label(), layer_type_to_string(layer.get_type())));
}

string python_names(const vector<string>& names)
{
    string text = "{";
    for (size_t i = 0; i < names.size(); ++i)
    {
        text += (i ? ", " : "") + to_string(i) + ": '";
        for (const char character : names[i])
        {
            if (character == '\\' || character == '\'') text += '\\';
            text += character;
        }
        text += "'";
    }
    return text + "}";
}

OnnxModel build_onnx_detection_model(const Network& network, const vector<string>& class_names)
{
    const vector<unique_ptr<Layer>>& layers = network.get_layers();
    const vector<vector<Index>>& source_layers = network.get_source_layers();
    const Shape image_shape = network.get_input_shape();

    throw_if(image_shape.get_rank() != 3, "ONNX export: object detection networks take images as inputs.");

    GraphBuilder graph;
    OnnxModel model;

    const string input = "images";
    vector<string> outputs(layers.size());
    vector<string> heads;
    Index classes_number = 0;
    Index anchors_number = 0;
    Index largest_stride = 0;

    for (size_t i = 0; i < layers.size(); ++i)
    {
        const Layer& layer = *layers[i];

        vector<string> inputs;
        for (const Index source : source_layers[i])
            inputs.push_back(source < 0 ? input : outputs[size_t(source)]);

        switch (layer.get_type())
        {
        case LayerType::Convolutional:
            outputs[i] = add_convolution(graph, static_cast<const Convolutional&>(layer), inputs[0]);
            break;
        case LayerType::Activation:
            outputs[i] = add_activation(graph, layer.get_output_activation(), inputs[0], 1);
            break;
        case LayerType::Addition:
            outputs[i] = graph.add_node("Sum", inputs);
            break;
        case LayerType::Concatenation:
            outputs[i] = graph.add_node("Concat", inputs, {int_attribute("axis", 1)});
            break;
        case LayerType::Upsampling:
            outputs[i] = add_upsampling(graph, static_cast<const Upsampling&>(layer), inputs[0]);
            break;
        case LayerType::Pooling:
            outputs[i] = add_pooling(graph, static_cast<const Pooling&>(layer), inputs[0]);
            break;
        case LayerType::DetectionV8:
        {
            const DetectionV8& head = static_cast<const DetectionV8&>(layer);
            const Shape grid = head.get_input_shape();
            heads.push_back(add_detection_v8(graph, head, inputs[0], image_shape));
            classes_number = head.get_classes_number();
            anchors_number += grid[0] * grid[1];
            largest_stride = max(largest_stride, image_shape[0] / grid[0]);
            break;
        }
        default:
            throw unsupported_detection_layer(layer);
        }

        model.layers.push_back(layer.get_label() + " (" + layer_type_to_string(layer.get_type()) + ")");
    }

    const string output = graph.add_node("Concat", heads, {int_attribute("axis", 2)}, "output0");

    vector<Message> metadata = {
        metadata_entry("task", "detect"),
        metadata_entry("stride", to_string(largest_stride)),
        metadata_entry("batch", "1"),
        metadata_entry("imgsz", format("[{}, {}]", image_shape[0], image_shape[1]))};

    if (ssize(class_names) == classes_number)
        metadata.push_back(metadata_entry("names", python_names(class_names)));

    const Message encoded = encode_model(graph,
        GraphBuilder::value_info(input, {{"batch"}, {"", image_shape[2]}, {"", image_shape[0]}, {"", image_shape[1]}}),
        GraphBuilder::value_info(output, {{"batch"}, {"", 4 + classes_number}, {"", anchors_number}}),
        metadata);

    model.bytes = encoded.bytes();
    model.nodes_number = graph.nodes_number();
    return model;
}

#endif

}

OnnxModel build_onnx_model(const Network& network, const vector<string>& class_names)
{
    const vector<unique_ptr<Layer>>& layers = network.get_layers();
    const vector<vector<Index>>& source_layers = network.get_source_layers();

    throw_if(layers.empty(), "ONNX export: the neural network has no layers.");

#ifndef OPENNN_NO_VISION
    if (network.get_first(LayerType::DetectionV8))
        return build_onnx_detection_model(network, class_names);
#else
    (void)class_names;
#endif

    for (const unique_ptr<Layer>& layer : layers)
        if (!is_one_of(layer->get_type(), LayerType::Scaling, LayerType::Unscaling,
                       LayerType::Dense, LayerType::Clamping))
            throw unsupported_layer(*layer);

    throw_if(network.get_input_shape().get_rank() != 1,
             "ONNX export: only networks whose inputs are a flat list of variables are supported.");

    GraphBuilder graph;
    OnnxModel model;

    const string input = "input";
    string current = input;

    for (size_t i = 0; i < layers.size(); ++i)
    {
        const Layer& layer = *layers[i];

        // The graph is emitted as a chain, so every layer must read exactly the previous one.
        throw_if(i < source_layers.size()
                 && (source_layers[i].size() != 1 || source_layers[i][0] != Index(i) - 1),
                 "ONNX export: layer '{}' does not follow the previous layer; only sequential networks "
                 "are supported.", layer.get_label());

        switch (layer.get_type())
        {
        case LayerType::Scaling:
        case LayerType::Unscaling:
            current = add_scaling(graph, static_cast<const Scaling&>(layer), current);
            break;
        case LayerType::Dense:
            current = add_dense(graph, static_cast<const Dense&>(layer), current);
            break;
        case LayerType::Clamping:
            current = add_clamping(graph, static_cast<const Clamping&>(layer), current);
            break;
        default:
            throw unsupported_layer(layer);
        }

        model.layers.push_back(layer.get_label() + " (" + layer_type_to_string(layer.get_type()) + ")");
    }

    const string output = graph.add_node("Identity", {current}, {}, "output");

    const Message encoded = encode_model(graph,
        GraphBuilder::value_info(input, {{"batch_size"}, {"", network.get_inputs_number()}}),
        GraphBuilder::value_info(output, {{"batch_size"}, {"", network.get_outputs_number()}}),
        {string_entry("input_names", network.get_input_feature_names()),
         string_entry("output_names", network.get_output_feature_names())});

    model.bytes = encoded.bytes();
    model.nodes_number = graph.nodes_number();
    return model;
}

void save_onnx_model(const Network& network, const filesystem::path& file_path, const vector<string>& class_names)
{
    const OnnxModel model = build_onnx_model(network, class_names);

    ofstream file(file_path, ios::binary | ios::trunc);
    throw_if(!file, "ONNX export: cannot open {} for writing.", file_path.string());

    file.write(model.bytes.data(), streamsize(model.bytes.size()));
    file.close();
    throw_if(!file, "ONNX export: cannot write {}.", file_path.string());
}

}
