// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

// The ONNX export is checked structurally everywhere, and numerically - the saved
// model run by onnxruntime against Network::calculate_outputs - wherever Python
// has onnxruntime installed.

#include "tests/pch.h"

#include "opennn/network/onnx_export.h"
#include "opennn/network/network.h"
#include "opennn/models/models.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/network/layers/unscaling_layer.h"
#include "opennn/core/statistics.h"

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <random>
#include <sstream>

using namespace opennn;

namespace
{

string quoted_path(const filesystem::path& path)
{
    return "\"" + path.string() + "\"";
}

string read_file(const filesystem::path& path)
{
    ifstream file(path, ios::binary);
    return string(istreambuf_iterator<char>(file), istreambuf_iterator<char>());
}

bool run(const string& command, const filesystem::path& output_path)
{
    string line = command + " > " + quoted_path(output_path) + " 2>&1";
#ifdef _WIN32
    // cmd.exe strips the outer quotes of a line that begins with one.
    if (!line.empty() && line.front() == '"') line = "\"" + line + "\"";
#endif
    return system(line.c_str()) == 0;
}

filesystem::path temporary_directory(const string& name)
{
    const filesystem::path directory = filesystem::temp_directory_path()
        / (name + "_" + to_string(random_device{}()));
    filesystem::create_directories(directory);
    return directory;
}

bool onnxruntime_is_available()
{
    static const bool available = []
    {
        const filesystem::path directory = temporary_directory("opennn_onnxruntime_probe");
        const bool found = run("python -c \"import numpy, onnxruntime\"", directory / "probe.txt");
        error_code error;
        filesystem::remove_all(directory, error);
        return found;
    }();

    return available;
}

// Saves the network, runs it with onnxruntime and compares every output with
// the network's own.
void expect_onnxruntime_matches(const Network& network, const MatrixR& inputs, const string& name)
{
    const MatrixR expected = const_cast<Network&>(network).calculate_outputs(inputs);

    const filesystem::path directory = temporary_directory("opennn_onnx_" + name);
    const filesystem::path model_path = directory / "model.onnx";
    save_onnx_model(network, model_path);

    ostringstream script;
    script.precision(9);
    script << "import numpy as np, onnxruntime as ort\n"
           << "x = np.array([";
    for (Index row = 0; row < inputs.rows(); ++row)
    {
        script << (row ? "," : "") << "[";
        for (Index column = 0; column < inputs.cols(); ++column)
            script << (column ? "," : "") << inputs(row, column);
        script << "]";
    }
    // The values go to their own file: onnxruntime may log warnings to the console
    // (on CI runners it reports hardware it cannot identify).
    const filesystem::path values_path = directory / "values.txt";

    script << "], dtype=np.float32)\n"
           << "ort.set_default_logger_severity(3)\n"
           << "y = ort.InferenceSession(r'" << model_path.string() << "').run(None, {'input': x})[0]\n"
           << "np.savetxt(r'" << values_path.string() << "', y, fmt='%.9g')\n";

    const filesystem::path script_path = directory / "run.py";
    ofstream(script_path, ios::binary) << script.str();

    const filesystem::path output_path = directory / "output.txt";
    const bool ran = run("python " + quoted_path(script_path), output_path);
    const string output = read_file(output_path);
    ASSERT_TRUE(ran) << output;

    istringstream lines(read_file(values_path));
    for (Index row = 0; row < expected.rows(); ++row)
        for (Index column = 0; column < expected.cols(); ++column)
        {
            double value = 0.0;
            ASSERT_TRUE(lines >> value) << "onnxruntime returned too few values:\n" << output;
            const float reference = expected(row, column);
            EXPECT_NEAR(reference, float(value), 1e-4f * max(1.0f, abs(reference)))
                << name << ": row " << row << ", output " << column;
        }

    error_code error;
    filesystem::remove_all(directory, error);
}

unique_ptr<ApproximationNetwork> build_scaled_network(const string& activation)
{
    auto network = make_unique<ApproximationNetwork>(Shape{3}, Shape{4}, Shape{2}, activation);

    network->set_input_variables(vector<Variable>(network->get_inputs_number()));
    network->set_output_variables(vector<Variable>(network->get_outputs_number()));
    network->set_input_names({"positive", "ranged", "centered"});
    network->set_output_names({"first", "second"});

    Scaling* scaling = static_cast<Scaling*>(network->get_first("Scaling"));
    scaling->set_scalers(vector<string>{"Logarithm", "MinimumMaximum", "MeanStandardDeviation"});
    scaling->set_descriptives({Descriptives(0.1f, 100.0f, 10.0f, 20.0f),
                               Descriptives(-1.0f, 1.0f, 0.0f, 1.0f),
                               Descriptives(-5.0f, 5.0f, 1.0f, 2.0f)});

    Unscaling* unscaling = static_cast<Unscaling*>(network->get_first("Unscaling"));
    unscaling->set_scalers(vector<string>{"MinimumMaximum", "Logarithm"});
    unscaling->set_descriptives({Descriptives(-10.0f, 10.0f, 0.0f, 5.0f),
                                 Descriptives(0.5f, 50.0f, 5.0f, 10.0f)});

    network->set_parameters_random();

    return network;
}

MatrixR scaled_inputs()
{
    MatrixR inputs(3, 3);
    inputs << 0.5f, -0.5f,  2.0f,
              7.0f,  0.25f, -3.0f,
             80.0f,  0.9f,  0.5f;
    return inputs;
}

}

TEST(OnnxExport, WritesAModelProtoWithOneNodeChainPerLayer)
{
    const unique_ptr<ApproximationNetwork> network = build_scaled_network("Tanh");

    const OnnxModel model = build_onnx_model(*network);

    ASSERT_GE(model.bytes.size(), 2u);
    EXPECT_EQ(model.bytes[0], '\x08') << "field 1 (ir_version) comes first";
    EXPECT_EQ(model.bytes[1], '\x07');
    EXPECT_EQ(ssize(model.layers), network->get_layers_number());
    EXPECT_GT(model.nodes_number, network->get_layers_number());
    EXPECT_NE(model.bytes.find("input_names"), string::npos);
    EXPECT_NE(model.bytes.find("positive,ranged,centered"), string::npos);
}

TEST(OnnxExport, RejectsLayersItCannotRepresent)
{
    ForecastingNetwork network(Shape{2, 3}, Shape{4}, Shape{1});

    try
    {
        build_onnx_model(network);
        FAIL() << "a recurrent network was exported";
    }
    catch (const exception& e)
    {
        EXPECT_NE(string(e.what()).find("Recurrent"), string::npos) << e.what();
    }
}

TEST(OnnxExport, EveryDenseActivationMatchesOnnxruntime)
{
    if (!onnxruntime_is_available()) GTEST_SKIP() << "python with numpy and onnxruntime is not on PATH.";

    for (const char* activation : {"Tanh", "Sigmoid", "ReLU", "LeakyReLU", "GELU", "GELUTanh", "SiLU"})
    {
        SCOPED_TRACE(activation);
        const unique_ptr<ApproximationNetwork> network = build_scaled_network(activation);
        expect_onnxruntime_matches(*network, scaled_inputs(), activation);
    }
}

TEST(OnnxExport, ClampingAndSoftmaxMatchOnnxruntime)
{
    if (!onnxruntime_is_available()) GTEST_SKIP() << "python with numpy and onnxruntime is not on PATH.";

    const unique_ptr<ApproximationNetwork> approximation = build_scaled_network("Tanh");
    Clamping* clamping = static_cast<Clamping*>(approximation->get_first("Clamping"));
    clamping->set_clamping_method(Clamping::ClampingMethod::Clamping);
    clamping->set_lower_bound(0, -1.0f);
    clamping->set_upper_bound(0, 1.0f);
    clamping->set_lower_bound(1, 1.0f);
    clamping->set_upper_bound(1, 2.0f);
    expect_onnxruntime_matches(*approximation, scaled_inputs(), "clamping");

    ClassificationNetwork classification(Shape{3}, Shape{5}, Shape{3});
    classification.set_input_variables(vector<Variable>(3));
    classification.set_output_variables(vector<Variable>(3));
    classification.set_parameters_random();
    expect_onnxruntime_matches(classification, scaled_inputs(), "softmax");
}
