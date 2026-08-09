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

#include "opennn/neural_network/model_expression.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/core/statistics.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
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

// Runs a command, returning whatever it printed. Failures come back tagged so
// the assertion that spots them can show the compiler or interpreter output
// rather than an empty string.
string run_capturing(const string& command, const filesystem::path& output_path,
                     const char* failure_tag)
{
    string line = command + " > " + quoted_path(output_path) + " 2>&1";
#ifdef _WIN32
    // cmd.exe strips the outer quotes of a command line that begins with one,
    // which breaks the redirect. Wrapping the whole line gives it a pair to eat.
    if (!line.empty() && line.front() == '"') line = "\"" + line + "\"";
#endif

    const string output = system(line.c_str()) == 0 ? string() : string(failure_tag) + ": ";

    return output + read_file(output_path);
}

// Feeds each row to the exported model and returns whatever it printed.
// The emitted module guards its own main with __name__, so importing it runs
// nothing; the driver drives it explicitly.
string run_exported_python_model(const filesystem::path& directory,
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

    return run_capturing("python " + quoted_path(driver_path),
                         output_path, "PYTHON FAILED");
}

// clang and gcc both link the emitted C without any MSVC environment set up, so
// this does not depend on the toolchain that built the library being reachable.
string find_c_compiler()
{
    static const string compiler = []() -> string
    {
        for (const char* candidate : {"clang", "gcc", "cc"})
        {
            const filesystem::path probe =
                filesystem::temp_directory_path() / "opennn_c_compiler_probe.txt";
            const string command =
                string(candidate) + " --version > " + quoted_path(probe) + " 2>&1";

            const bool ran = system(command.c_str()) == 0;

            error_code error;
            filesystem::remove(probe, error);

            if (ran) return candidate;
        }

        return {};
    }();

    return compiler;
}

// Same idea as the Python path, but the emitted C has to be compiled first.
// OPENNN_EXPORT_NO_MAIN is the emitter's own switch for dropping its
// placeholder main, which exists so the code can be linked into something else.
string run_exported_c_model(const filesystem::path& directory,
                            const ModelExpression& model_expression,
                            const MatrixR& inputs,
                            Index outputs_number,
                            ModelExpression::ProgrammingLanguage language)
{
    const filesystem::path model_path = directory / "opennn_exported_model.c";
    model_expression.save(model_path, language);

    ostringstream driver;
    driver << setprecision(9);
    driver << "#include <stdio.h>\n\n"
           << "float* calculate_outputs(const float* inputs);\n\n"
           << "int main(void)\n{\n"
           << "\tstatic const float rows[" << inputs.rows()
           << "][" << inputs.cols() << "] = {\n";

    for (Index row = 0; row < inputs.rows(); ++row)
    {
        driver << "\t\t{";
        for (Index column = 0; column < inputs.cols(); ++column)
            driver << (column ? ", " : "") << inputs(row, column);
        driver << "},\n";
    }

    driver << "\t};\n\n"
           << "\tfor (int row = 0; row < " << inputs.rows() << "; ++row)\n\t{\n"
           << "\t\tconst float* outputs = calculate_outputs(rows[row]);\n"
           << "\t\tfor (int i = 0; i < " << outputs_number << "; ++i)\n"
           << "\t\t\tprintf(\"%s%.9g\", i ? \" \" : \"\", (double)outputs[i]);\n"
           << "\t\tprintf(\"\\n\");\n"
           << "\t}\n\n\treturn 0;\n}\n";

    const filesystem::path driver_path = directory / "opennn_exported_driver.c";
    write_file(driver_path, driver.str());

    const filesystem::path program_path = directory / "opennn_exported_model.exe";
    const filesystem::path build_log = directory / "opennn_exported_build.txt";

    string build = find_c_compiler() + " -DOPENNN_EXPORT_NO_MAIN "
                 + quoted_path(model_path) + " " + quoted_path(driver_path)
                 + " -o " + quoted_path(program_path);
#ifndef _WIN32
    build += " -lm";
#endif
    // A successful compile still prints warnings, so test the tag rather than
    // whether anything was written.
    const string build_output = run_capturing(build, build_log, "COMPILE FAILED");
    if (build_output.starts_with("COMPILE FAILED")) return build_output;

    const filesystem::path output_path = directory / "opennn_exported_output.txt";

    return run_capturing(quoted_path(program_path), output_path, "RUN FAILED");
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

// A feature with no spread is the case the scalers kept getting wrong, so the
// exported model has to reproduce that constant too.
unique_ptr<ApproximationNetwork> build_degenerate_network()
{
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

    return network;
}

MatrixR degenerate_inputs()
{
    MatrixR inputs(2, 2);
    inputs << 5.0f,  0.5f,
             -2.0f, -0.5f;
    return inputs;
}

// The targets differ only in how the export is run, so every check below is
// written once and pointed at any of them.
enum class Target { Python, C, CEmbedded };

bool target_available(Target target)
{
    return target == Target::Python ? python_is_available() : !find_c_compiler().empty();
}

const char* target_missing(Target target)
{
    return target == Target::Python ? "python is not on PATH." : "no C compiler on PATH.";
}

ModelExpression::ProgrammingLanguage target_language(Target target)
{
    using enum ModelExpression::ProgrammingLanguage;
    switch (target)
    {
    case Target::Python:    return Python;
    case Target::C:         return C;
    case Target::CEmbedded: return CEmbedded;
    }
    return C;
}

// Exports the network, runs it, and compares row by row. The tolerance is loose
// on purpose: the emitters write constants at stream precision, so the target is
// a structurally wrong formula, not the last ulp.
void expect_export_matches(Target target, const string& directory_name,
                           const NeuralNetwork& network, const MatrixR& inputs,
                           const MatrixR& expected)
{
    const ModelExpression model_expression(&network);

    const filesystem::path directory = filesystem::temp_directory_path() / directory_name;
    filesystem::create_directories(directory);

    const string output = target == Target::Python
        ? run_exported_python_model(directory, model_expression, inputs)
        : run_exported_c_model(directory, model_expression, inputs, expected.cols(),
                               target_language(target));

    for (const char* failure : {"PYTHON FAILED", "COMPILE FAILED", "RUN FAILED"})
        ASSERT_EQ(output.find(failure), string::npos) << output;

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

}

TEST(ExpressionExecution, PythonModelMatchesTheNetworkItCameFrom)
{
    if (!target_available(Target::Python)) GTEST_SKIP() << target_missing(Target::Python);

    const unique_ptr<ApproximationNetwork> network = build_network();
    const MatrixR inputs = sample_inputs();

    expect_export_matches(Target::Python, "opennn_expression_python",
                          *network, inputs, network->calculate_outputs(inputs));
}

// C is the language customers embed, and it does not share the Python emitter's
// prelude, activation table or float-literal formatting - only the expression
// body underneath. Running it is what covers that half.
TEST(ExpressionExecution, CModelMatchesTheNetworkItCameFrom)
{
    if (!target_available(Target::C)) GTEST_SKIP() << target_missing(Target::C);

    const unique_ptr<ApproximationNetwork> network = build_network();
    const MatrixR inputs = sample_inputs();

    expect_export_matches(Target::C, "opennn_expression_c",
                          *network, inputs, network->calculate_outputs(inputs));
}

// The degenerate scaler rule has to survive into both exports: the emitters are
// separate code from the numeric paths and from each other.
TEST(ExpressionExecution, PythonModelReproducesDegenerateScaling)
{
    if (!target_available(Target::Python)) GTEST_SKIP() << target_missing(Target::Python);

    const unique_ptr<ApproximationNetwork> network = build_degenerate_network();
    const MatrixR inputs = degenerate_inputs();
    const MatrixR expected = network->calculate_outputs(inputs);

    for (Index row = 0; row < expected.rows(); ++row)
        EXPECT_NEAR(7.0f, expected(row, 0), 1e-4f) << "the library lost the constant";

    expect_export_matches(Target::Python, "opennn_degenerate_python",
                          *network, inputs, expected);
}

TEST(ExpressionExecution, CModelReproducesDegenerateScaling)
{
    if (!target_available(Target::C)) GTEST_SKIP() << target_missing(Target::C);

    const unique_ptr<ApproximationNetwork> network = build_degenerate_network();
    const MatrixR inputs = degenerate_inputs();
    const MatrixR expected = network->calculate_outputs(inputs);

    for (Index row = 0; row < expected.rows(); ++row)
        EXPECT_NEAR(7.0f, expected(row, 0), 1e-4f) << "the library lost the constant";

    expect_export_matches(Target::C, "opennn_degenerate_c",
                          *network, inputs, expected);
}

// The embedded export is a separate emitter again - weight tables and its own
// nn_dense_forward rather than one expression per neuron - and it is the one
// documented as going into firmware, where a wrong number is hardest to recall.
TEST(ExpressionExecution, EmbeddedModelMatchesTheNetworkItCameFrom)
{
    if (!target_available(Target::CEmbedded)) GTEST_SKIP() << target_missing(Target::CEmbedded);

    const unique_ptr<ApproximationNetwork> network = build_network();
    const MatrixR inputs = sample_inputs();

    // Both C dialects compile and run the same way, so confirm this really is
    // the embedded one: a silent fall back to plain C would pass identically.
    const filesystem::path probe =
        filesystem::temp_directory_path() / "opennn_embedded_probe.c";
    ModelExpression(network.get()).save(probe, ModelExpression::ProgrammingLanguage::CEmbedded);
    const string emitted = read_file(probe);
    EXPECT_NE(emitted.find("NN_FLASH"), string::npos) << "not the embedded emitter";
    EXPECT_NE(emitted.find("nn_dense_forward"), string::npos) << "not the embedded emitter";
    filesystem::remove(probe);

    expect_export_matches(Target::CEmbedded, "opennn_expression_embedded",
                          *network, inputs, network->calculate_outputs(inputs));
}

TEST(ExpressionExecution, EmbeddedModelReproducesDegenerateScaling)
{
    if (!target_available(Target::CEmbedded)) GTEST_SKIP() << target_missing(Target::CEmbedded);

    const unique_ptr<ApproximationNetwork> network = build_degenerate_network();
    const MatrixR inputs = degenerate_inputs();
    const MatrixR expected = network->calculate_outputs(inputs);

    for (Index row = 0; row < expected.rows(); ++row)
        EXPECT_NEAR(7.0f, expected(row, 0), 1e-4f) << "the library lost the constant";

    expect_export_matches(Target::CEmbedded, "opennn_degenerate_embedded",
                          *network, inputs, expected);
}

// The layer emitters are reached through the executed exports above, but this
// one runs without an interpreter or a compiler - so the degenerate-scaler rule
// stays covered on a machine that has neither.
TEST(ExpressionExecution, UnscalingEmitsTheDegenerateConstantWithoutRunningAnything)
{
    const float constant = 7.0f;

    Unscaling layer(Shape{1});
    layer.set_scalers("StandardDeviation");
    layer.set_descriptives({Descriptives(constant, constant, constant, 0.0f)});

    const string expression = layer.write_expression({"x"}, {"y"});

    EXPECT_NE(expression.find(to_string(int(constant))), string::npos)
        << "expected the constant in: " << expression;
    EXPECT_EQ(expression.find("x*0"), string::npos)
        << "must not multiply by a zero standard deviation: " << expression;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
