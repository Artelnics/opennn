// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/network/onnx_export.h"

#include <cstdint>
#include <cstring>

#include "opennn/registry.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/scaling_layer.h"
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

enum TensorElementType { ElementFloat = 1, ElementBool = 9 };
enum AttributeType { AttributeFloat = 1, AttributeInt = 2 };

constexpr int64_t IR_VERSION = 7;
constexpr int64_t OPSET_VERSION = 13;

struct Attribute
{
    string name;
    bool is_float = false;
    float f = 0.0f;
    int64_t i = 0;
};

Attribute int_attribute(const string& name, int64_t value) { return {name, false, 0.0f, value}; }
Attribute float_attribute(const string& name, float value) { return {name, true, value, 0}; }

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
            if (attribute.is_float) encoded.add_float(2, attribute.f);
            else                    encoded.add_int(3, attribute.i);
            encoded.add_int(20, attribute.is_float ? AttributeFloat : AttributeInt);
            node.add_message(5, encoded);
        }

        nodes.push_back(node);
        return output_name;
    }

    Message build(const string& input, Index inputs_number, const string& output, Index outputs_number) const
    {
        Message graph;
        for (const Message& node : nodes) graph.add_message(1, node);
        graph.add_bytes(2, "opennn_network");
        for (const Message& initializer : initializers) graph.add_message(5, initializer);
        graph.add_message(11, value_info(input, inputs_number));
        graph.add_message(12, value_info(output, outputs_number));
        return graph;
    }

    Index nodes_number() const { return Index(nodes.size()); }

private:

    static Message value_info(const string& name, Index features)
    {
        Message batch;
        batch.add_bytes(2, "batch_size");

        Message width;
        width.add_int(1, features);

        Message shape;
        shape.add_message(1, batch);
        shape.add_message(1, width);

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

string add_activation(GraphBuilder& graph, ActivationFunction activation, const string& input)
{
    switch (activation)
    {
    case ActivationFunction::Identity:  return input;
    case ActivationFunction::Sigmoid:   return graph.add_node("Sigmoid", {input});
    case ActivationFunction::Tanh:      return graph.add_node("Tanh", {input});
    case ActivationFunction::ReLU:      return graph.add_node("Relu", {input});
    case ActivationFunction::Softmax:   return graph.add_node("Softmax", {input}, {int_attribute("axis", -1)});
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

runtime_error unsupported_layer(const Layer& layer)
{
    return runtime_error(format("ONNX export: layer '{}' ({}) cannot be exported to ONNX yet. "
                                "Supported layers: Scaling, Dense, Unscaling and Clamping.",
                                layer.get_label(), layer_type_to_string(layer.get_type())));
}

Message string_entry(const string& key, const vector<string>& values)
{
    string joined;
    for (size_t i = 0; i < values.size(); ++i)
        joined += (i ? "," : "") + values[i];

    Message entry;
    entry.add_bytes(1, key);
    entry.add_bytes(2, joined);
    return entry;
}

}

OnnxModel build_onnx_model(const Network& network)
{
    const vector<unique_ptr<Layer>>& layers = network.get_layers();
    const vector<vector<Index>>& source_layers = network.get_source_layers();

    throw_if(layers.empty(), "ONNX export: the neural network has no layers.");

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

    Message opset;
    opset.add_bytes(1, "");
    opset.add_int(2, OPSET_VERSION);

    Message encoded;
    encoded.add_int(1, IR_VERSION);
    encoded.add_bytes(2, "OpenNN");
    encoded.add_message(7, graph.build(input, network.get_inputs_number(), output, network.get_outputs_number()));
    encoded.add_message(8, opset);
    encoded.add_message(14, string_entry("input_names", network.get_input_feature_names()));
    encoded.add_message(14, string_entry("output_names", network.get_output_feature_names()));

    model.bytes = encoded.bytes();
    model.nodes_number = graph.nodes_number();
    return model;
}

void save_onnx_model(const Network& network, const filesystem::path& file_path)
{
    const OnnxModel model = build_onnx_model(network);

    ofstream file(file_path, ios::binary | ios::trunc);
    throw_if(!file, "ONNX export: cannot open {} for writing.", file_path.string());

    file.write(model.bytes.data(), streamsize(model.bytes.size()));
    file.close();
    throw_if(!file, "ONNX export: cannot write {}.", file_path.string());
}

}
