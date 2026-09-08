//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   E X P R E S S I O N   G O L D E N   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The five language emitters produce text that the rest of the suite only
// spot-checks with substring assertions, so a refactor can reword the output
// without any test noticing. This dumps every emitter's output for a set of
// fixed networks into OPENNN_EXPRESSION_DUMP_DIR, which lets a refactor be
// verified by diffing the whole generated corpus before and after.
//
//     set OPENNN_EXPRESSION_DUMP_DIR=C:\some\dir  &&  opennn_tests.exe
//         --gtest_filter=ModelExpressionGolden.*
//
// The test skips when the variable is unset, so it costs nothing in CI.

#include "tests/pch.h"

#include "opennn/network/model_expression.h"
#include "opennn/network/network.h"
#include "opennn/models/models.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>

using namespace opennn;

namespace
{

using Language = ModelExpression::ProgrammingLanguage;

struct LanguageCase
{
    Language language;
    const char* extension;
};

const LanguageCase language_cases[] = {
    {Language::C,          "c"},
    {Language::CEmbedded,  "embedded.c"},
    {Language::Python,     "py"},
    {Language::JavaScript, "html"},
    {Language::PHP,        "php"}
};


// Parameters must not be random: the dump is compared byte for byte across
// builds, so every network is filled with the same deterministic ramp.
void set_deterministic_parameters(Network& network)
{
    // Buffer size, not get_parameters_number(): set_parameters wants the padded
    // buffer, and the logical count is smaller wherever a layer is aligned.
    VectorR parameters(network.get_parameters_buffer_size());

    for (Index i = 0; i < parameters.size(); ++i)
        parameters(i) = float((i % 17) - 8) * 0.125f;

    network.set_parameters(parameters);
}


filesystem::path dump_directory()
{
    const char* const directory = getenv("OPENNN_EXPRESSION_DUMP_DIR");
    return directory ? filesystem::path(directory) : filesystem::path();
}


void dump_every_language(Network& network, const string& case_name)
{
    set_deterministic_parameters(network);

    const ModelExpression model_expression(&network);

    for (const auto& [language, extension] : language_cases)
        model_expression.save(dump_directory() / (case_name + "." + extension), language);
}

}


class ModelExpressionGolden : public ::testing::Test
{
protected:

    void SetUp() override
    {
        if (dump_directory().empty())
            GTEST_SKIP() << "OPENNN_EXPRESSION_DUMP_DIR is not set.";

        filesystem::create_directories(dump_directory());
    }
};


TEST_F(ModelExpressionGolden, Approximation)
{
    ApproximationNetwork network(Shape{3}, Shape{4}, Shape{2});

    network.set_input_variables(vector<Variable>(network.get_inputs_number()));
    network.set_output_variables(vector<Variable>(network.get_outputs_number()));
    network.set_input_names({"alpha", "beta", "Temp (C)"});
    network.set_output_names({"gamma", "Humidity (%)"});

    dump_every_language(network, "approximation");
}


TEST_F(ModelExpressionGolden, Classification)
{
    ClassificationNetwork network(Shape{4}, Shape{5}, Shape{3});

    network.set_input_variables(vector<Variable>(network.get_inputs_number()));
    network.set_output_variables(vector<Variable>(network.get_outputs_number()));

    dump_every_language(network, "classification");
}


TEST_F(ModelExpressionGolden, Forecasting)
{
    ForecastingNetwork network(Shape{2, 3}, Shape{4}, Shape{1});

    network.set_input_variables(vector<Variable>(network.get_inputs_number()));
    network.set_output_variables(vector<Variable>(network.get_outputs_number()));

    dump_every_language(network, "forecasting");
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
