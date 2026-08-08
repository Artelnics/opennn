//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E X P R E S S I O N   E X E C U T I O N   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The exported model is a second implementation of the network: every layer
// writes its own arithmetic as source text in write_expression, and nothing
// forces that text to agree with forward_propagate. The other expression tests
// check that the emitted source contains the right tokens, which cannot catch a
// formula that is merely wrong.
//
// This one runs it. The emitted Python is executed and its outputs compared
// against calculate_outputs on the same network and the same inputs.
//
// The comparison is deliberately loose: the emitter writes constants at stream
// precision, so the exported model carries about six significant digits. The
// target is a structurally wrong formula, not the last ulp.

#include "pch.h"

#include "opennn/model_expression.h"
#include "opennn/neural_network.h"
#include "opennn/standard_networks.h"
#include "opennn/scaling_layer.h"
#include "opennn/unscaling_layer.h"
#include "opennn/statistics.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace opennn;

namespace
{

// Quoted so a temporary directory containing spaces still works.
string quoted_path(const filesystem::path& path)
{
    return "\"" + path.string() + "\"";
}

bool python_is_available()
{
    static const bool available = []
    {
        const filesystem::path probe = filesystem::temp_directory_path() / "opennn_python_probe.txt";
        const string command = "python --version > " + quoted_path(probe) + " 2>&1";
        const bool ran = system(command.c_str()) == 0;
        error_code error;
        filesystem::remove(probe, error);
        return ran;
    }();

    return available;
}

void write_file(const filesystem::path& path, const string& text)
{
    ofstream file(path, ios::binary);
    file << text;
}

string read_file(const filesystem::path& path)
{
    ifstream file(path);
    stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

// Feeds each row to the exported model and returns whatever it printed.
// The emitted module guards its own main with __name__, so importing it runs
// nothing; the driver drives it explicitly.
string run_exported_model(const filesystem::path& directory,
                          const ModelExpression& model_expression,
                          const MatrixR& inputs)
{
    // Exactly the call a user makes to export a model.
    model_expression.save(directory / "opennn_exported_model.py",
                          ModelExpression::ProgrammingLanguage::Python);

    ostringstream driver;
    driver << "import sys\n"
           << "sys.path.insert(0, " << "r'" << directory.string() << "'" << ")\n"
           << "from opennn_exported_model import NeuralNetwork\n"
           << "network = NeuralNetwork()\n"
           << "rows = [\n";

    for (Index row = 0; row < inputs.rows(); ++row)
    {
        driver << "    [";
        for (Index column = 0; column < inputs.cols(); ++column)
            driver << (column ? ", " : "") << inputs(row, column);
        driver << "],\n";
    }

    driver << "]\n"
           << "for row in rows:\n"
           << "    values = network.calculate_outputs(list(row))\n"
           << "    print(' '.join('%.9g' % float(v) for v in values))\n";

    const filesystem::path driver_path = directory / "opennn_exported_driver.py";
    const filesystem::path output_path = directory / "opennn_exported_output.txt";
    write_file(driver_path, driver.str());

    const string command = "python " + quoted_path(driver_path)
                         + " > " + quoted_path(output_path) + " 2>&1";

    if (system(command.c_str()) != 0)
        return "PYTHON FAILED: " + read_file(output_path);

    return read_file(output_path);
}

MatrixR parse_output(const string& text, Index rows, Index columns)
{
    MatrixR values(rows, columns);
    values.setConstant(numeric_limits<float>::quiet_NaN());

    istringstream stream(text);
    string line;

    for (Index row = 0; row < rows && getline(stream, line); ++row)
    {
        istringstream row_stream(line);
        for (Index column = 0; column < columns; ++column)
            row_stream >> values(row, column);
    }

    return values;
}

// Scaling covers a healthy feature per method plus one with no spread, since
// that is where the scalers have historically drifted apart.
unique_ptr<ApproximationNetwork> build_network()
{
    auto network = make_unique<ApproximationNetwork>(Shape{4}, Shape{5}, Shape{2});

    network->set_input_variables(vector<Variable>(network->get_inputs_number()));
    network->set_output_variables(vector<Variable>(network->get_outputs_number()));
    network->set_input_names({"alpha", "beta", "gamma", "delta"});
    network->set_output_names({"first", "second"});

    Scaling* scaling = static_cast<Scaling*>(network->get_first("Scaling"));
    scaling->set_scalers(vector<string>{"MinimumMaximum",
                                        "MeanStandardDeviation",
                                        "StandardDeviation",
                                        "None"});
    scaling->set_descriptives({Descriptives(-2.0f, 6.0f, 1.0f, 2.0f),
                               Descriptives(0.0f, 9.0f, 3.0f, 1.5f),
                               Descriptives(-4.0f, 4.0f, 0.0f, 2.0f),
                               Descriptives(-1.0f, 1.0f, 0.0f, 1.0f)});

    Unscaling* unscaling = static_cast<Unscaling*>(network->get_first("Unscaling"));
    unscaling->set_scalers(vector<string>{"MinimumMaximum", "MeanStandardDeviation"});
    unscaling->set_descriptives({Descriptives(-5.0f, 5.0f, 0.0f, 2.5f),
                                 Descriptives(1.0f, 7.0f, 4.0f, 1.25f)});

    network->set_parameters_random();

    return network;
}

MatrixR sample_inputs()
{
    MatrixR inputs(4, 4);
    inputs << 0.0f,  1.0f, -1.0f,  2.0f,
              2.5f, -3.0f,  4.0f,  0.5f,
             -7.0f,  8.0f,  0.0f, -2.0f,
              1.0f,  0.0f,  3.0f,  1.5f;
    return inputs;
}

}

TEST(ExpressionExecution, PythonModelMatchesTheNetworkItCameFrom)
{
    if (!python_is_available()) GTEST_SKIP() << "python is not on PATH.";

    const unique_ptr<ApproximationNetwork> network = build_network();
    const MatrixR inputs = sample_inputs();

    const MatrixR expected = network->calculate_outputs(inputs);

    const ModelExpression model_expression(network.get());

    const filesystem::path directory =
        filesystem::temp_directory_path() / "opennn_expression_execution";
    filesystem::create_directories(directory);

    const string output = run_exported_model(directory, model_expression, inputs);

    ASSERT_EQ(output.find("PYTHON FAILED"), string::npos) << output;
    ASSERT_FALSE(output.empty()) << "the exported model printed nothing";

    const MatrixR actual = parse_output(output, expected.rows(), expected.cols());

    for (Index row = 0; row < expected.rows(); ++row)
        for (Index column = 0; column < expected.cols(); ++column)
        {
            const float reference = expected(row, column);
            EXPECT_NEAR(reference, actual(row, column), 1e-3f * max(1.0f, abs(reference)))
                << "exported model disagrees at row " << row << ", output " << column
                << "\n--- it printed ---\n" << output;
        }

    error_code error;
    filesystem::remove_all(directory, error);
}

// A feature with no spread is the case the scalers kept getting wrong, so pin
// it through the exported model too: it must come back as the constant.
TEST(ExpressionExecution, PythonModelReproducesDegenerateScaling)
{
    if (!python_is_available()) GTEST_SKIP() << "python is not on PATH.";

    auto network = make_unique<ApproximationNetwork>(Shape{2}, Shape{3}, Shape{1});

    network->set_input_variables(vector<Variable>(network->get_inputs_number()));
    network->set_output_variables(vector<Variable>(network->get_outputs_number()));
    network->set_input_names({"flat", "spread"});
    network->set_output_names({"result"});

    Scaling* scaling = static_cast<Scaling*>(network->get_first("Scaling"));
    scaling->set_scalers(vector<string>{"StandardDeviation", "MinimumMaximum"});
    scaling->set_descriptives({Descriptives(3.0f, 3.0f, 3.0f, 0.0f),
                               Descriptives(-1.0f, 1.0f, 0.0f, 1.0f)});

    Unscaling* unscaling = static_cast<Unscaling*>(network->get_first("Unscaling"));
    unscaling->set_scalers(vector<string>{"StandardDeviation"});
    unscaling->set_descriptives({Descriptives(7.0f, 7.0f, 7.0f, 0.0f)});

    network->set_parameters_random();

    MatrixR inputs(2, 2);
    inputs << 5.0f,  0.5f,
             -2.0f, -0.5f;

    const MatrixR expected = network->calculate_outputs(inputs);

    const ModelExpression model_expression(network.get());

    const filesystem::path directory =
        filesystem::temp_directory_path() / "opennn_expression_degenerate";
    filesystem::create_directories(directory);

    const string output = run_exported_model(directory, model_expression, inputs);

    ASSERT_EQ(output.find("PYTHON FAILED"), string::npos) << output;

    const MatrixR actual = parse_output(output, expected.rows(), expected.cols());

    // The output feature had no spread, so both paths owe the same constant.
    for (Index row = 0; row < expected.rows(); ++row)
    {
        EXPECT_NEAR(7.0f, expected(row, 0), 1e-4f) << "the library lost the constant";
        EXPECT_NEAR(expected(row, 0), actual(row, 0), 1e-3f)
            << "exported model disagrees\n--- it printed ---\n" << output;
    }

    error_code error;
    filesystem::remove_all(directory, error);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
