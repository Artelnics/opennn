// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "tests/pch.h"

#include "opennn/registry.h"
#include "opennn/models/models.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/convolutional_layer.h"
#include "opennn/network/layers/detection_v8_layer.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>

using namespace opennn;

namespace
{

class ProtoWriter
{
public:

    void add_varint(int field, uint64_t value)
    {
        write_varint(uint64_t(field) << 3);
        write_varint(value);
    }

    void add_bytes(int field, const string& value)
    {
        write_varint(uint64_t(field) << 3 | 2);
        write_varint(value.size());
        buffer += value;
    }

    const string& bytes() const { return buffer; }

private:

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

struct Tensor
{
    vector<int64_t> dims;
    vector<float> values;
};

vector<pair<string, string>> yolov8s_keys()
{
    vector<pair<string, string>> keys;

    const auto c2f = [&](const string& prefix, const string& key, int blocks)
    {
        keys.push_back({prefix + "_cv1a", key + ".cv1"});
        keys.push_back({prefix + "_cv1b", key + ".cv1"});
        for (int block = 0; block < blocks; ++block)
        {
            const string label = prefix + "_b" + to_string(block + 1);
            const string block_key = key + ".m." + to_string(block);
            keys.push_back({label + "_cv1", block_key + ".cv1"});
            keys.push_back({label + "_cv2", block_key + ".cv2"});
        }
        keys.push_back({prefix + "_cv2", key + ".cv2"});
    };

    keys.push_back({"c8_stem", "model.0"});
    keys.push_back({"c8_s1_down", "model.1"});
    c2f("c8_s1", "model.2", 1);
    keys.push_back({"c8_s2_down", "model.3"});
    c2f("c8_s2", "model.4", 2);
    keys.push_back({"c8_s3_down", "model.5"});
    c2f("c8_s3", "model.6", 2);
    keys.push_back({"c8_s4_down", "model.7"});
    c2f("c8_s4", "model.8", 1);
    keys.push_back({"c8_sppf_in", "model.9.cv1"});
    keys.push_back({"c8_sppf_out", "model.9.cv2"});
    c2f("c8_n12", "model.12", 1);
    c2f("c8_n15", "model.15", 1);
    keys.push_back({"c8_pan_n4_down", "model.16"});
    c2f("c8_n18", "model.18", 1);
    keys.push_back({"c8_pan_n5_down", "model.19"});
    c2f("c8_n21", "model.21", 1);

    const vector<string> heads = {"small", "medium", "large"};
    for (size_t head = 0; head < heads.size(); ++head)
    {
        const string label = "c8_" + heads[head];
        const string index = to_string(head);
        keys.push_back({label + "_box_c1", "model.22.cv2." + index + ".0"});
        keys.push_back({label + "_box_c2", "model.22.cv2." + index + ".1"});
        keys.push_back({label + "_box_out", "model.22.cv2." + index + ".2"});
        keys.push_back({label + "_cls_c1", "model.22.cv3." + index + ".0"});
        keys.push_back({label + "_cls_c2", "model.22.cv3." + index + ".1"});
    }

    return keys;
}

void append(Tensor& tensor, const vector<float>& values, const vector<int64_t>& dims)
{
    if (tensor.dims.empty()) tensor.dims = dims;
    else tensor.dims[0] += dims[0];
    tensor.values.insert(tensor.values.end(), values.begin(), values.end());
}

void write_ultralytics_onnx(const Network& network, const filesystem::path& path)
{
    map<string, Tensor> tensors;

    for (const auto& [label, key] : yolov8s_keys())
    {
        const auto& convolution = static_cast<const Convolutional&>(*network.get_layer(label));

        vector<float> kernel;
        vector<float> bias;
        convolution.get_folded_parameters(kernel, bias);

        const string prefix = convolution.get_batch_normalization() ? key + ".conv" : key;
        append(tensors[prefix + ".weight"], kernel,
               {convolution.get_kernels_number(), convolution.get_kernel_channels(),
                convolution.get_kernel_height(), convolution.get_kernel_width()});
        append(tensors[prefix + ".bias"], bias, {convolution.get_kernels_number()});
    }

    ProtoWriter graph;
    for (const auto& [name, tensor] : tensors)
    {
        ProtoWriter encoded;
        for (const int64_t dim : tensor.dims) encoded.add_varint(1, uint64_t(dim));
        encoded.add_varint(2, 1);
        encoded.add_bytes(8, name);
        encoded.add_bytes(9, string(reinterpret_cast<const char*>(tensor.values.data()),
                                    tensor.values.size() * sizeof(float)));
        graph.add_bytes(5, encoded.bytes());
    }

    ProtoWriter model;
    model.add_varint(1, 7);
    model.add_bytes(7, graph.bytes());

    ofstream(path, ios::binary) << model.bytes();
}

unique_ptr<Yolo> build_yolov8s(unsigned seed)
{
    auto network = make_unique<Yolo>(Shape{64, 64, 3}, 2, vector<std::array<float, 2>>(9, {0.1f, 0.1f}), 2,
                                     Yolo::Backbone::CSPDarknet53v11, Yolo::ClassActivation::Sigmoid,
                                     Yolo::HeadStyle::FPNv8, Yolo::BodyActivation::SiLU, true, 16,
                                     Yolo::ModelSize::s);

    mt19937 generator(seed);
    uniform_real_distribution<float> statistics(0.5f, 1.5f);
    float* states = network->get_states_data();
    for (Index i = 0; i < network->get_states_buffer_size(); ++i)
        states[i] = statistics(generator);

    return network;
}

vector<vector<float>> head_outputs(Network& network, vector<float>& image)
{
    ForwardPropagation forward_propagation(1, &network);
    network.forward_propagate({TensorView(image.data(), {1, 64, 64, 3}, Type::FP32)},
                              forward_propagation, ForwardPropagationMode::Inference);

    vector<vector<float>> outputs;
    for (size_t layer = 0; layer < network.get_layers().size(); ++layer)
        if (network.get_layer(Index(layer))->get_type() == LayerType::DetectionV8)
        {
            const TensorView& output = forward_propagation.slots[layer].back();
            outputs.emplace_back(output.as<float>(), output.as<float>() + output.size());
        }

    return outputs;
}

}

TEST(YoloOnnxLoader, LoadsUltralyticsWeightsAndResetsTheClassOutputs)
{
    const unique_ptr<Yolo> source = build_yolov8s(3);
    const unique_ptr<Yolo> target = build_yolov8s(5);

    const filesystem::path path = filesystem::temp_directory_path()
        / ("opennn_yolov8s_" + to_string(random_device{}()) + ".onnx");
    write_ultralytics_onnx(*source, path);

    const Index loaded = load_yolov8s_onnx(*target, path, 2);
    error_code error;
    filesystem::remove(path, error);

    EXPECT_EQ(loaded, Index(yolov8s_keys().size()) + 3);

    mt19937 generator(9);
    uniform_real_distribution<float> pixel(0.0f, 1.0f);
    vector<float> image(64 * 64 * 3);
    for (float& value : image) value = pixel(generator);

    const vector<vector<float>> expected = head_outputs(*source, image);
    const vector<vector<float>> actual = head_outputs(*target, image);
    ASSERT_EQ(expected.size(), 3u);
    ASSERT_EQ(actual.size(), 3u);

    const Index box_channels = 4 * 16;
    const Index channels = box_channels + 2;
    const float prior = 1.0f / (1.0f + exp(4.5951f));

    for (size_t head = 0; head < expected.size(); ++head)
        for (size_t i = 0; i < expected[head].size(); ++i)
        {
            if (Index(i) % channels < box_channels)
                EXPECT_NEAR(actual[head][i], expected[head][i], 1e-3f * max(1.0f, abs(expected[head][i])))
                    << "head " << head << ", value " << i;
            else
                EXPECT_NEAR(actual[head][i], prior, 1e-5f) << "head " << head << ", value " << i;
        }
}

TEST(YoloOnnxLoader, RejectsAMissingFile)
{
    const unique_ptr<Yolo> network = build_yolov8s(3);

    EXPECT_THROW(load_yolov8s_onnx(*network, filesystem::temp_directory_path() / "opennn_missing_yolov8s.onnx", 2),
                 runtime_error);
}
