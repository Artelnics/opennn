#include "pch.h"
#include "../opennn/model_expression.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"

#include <fstream>
#include <sstream>
#include <filesystem>

using namespace opennn;

namespace
{

string read_whole_file(const filesystem::path& path)
{
    ifstream file(path);
    stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

bool contains_token(const string& text, const string& token)
{
    return text.find(token) != string::npos;
}

}

class ModelExpressionTest : public ::testing::Test
{
protected:

    unique_ptr<ApproximationNetwork> neural_network;

    const vector<string> input_names = { "alpha", "beta" };
    const vector<string> output_names = { "gamma" };

    void SetUp() override
    {
        neural_network = make_unique<ApproximationNetwork>(Shape{ 2 }, Shape{ 3 }, Shape{ 1 });
        neural_network->set_input_variables(vector<Variable>(neural_network->get_inputs_number()));
        neural_network->set_output_variables(vector<Variable>(neural_network->get_outputs_number()));
        neural_network->set_input_names(input_names);
        neural_network->set_output_names(output_names);
    }
};


TEST_F(ModelExpressionTest, BuildExpressionNotEmpty)
{
    const ModelExpression model_expression(neural_network.get());

    const string expression = model_expression.build_expression();

    EXPECT_FALSE(expression.empty());
}


TEST_F(ModelExpressionTest, BuildExpressionContainsVariableNames)
{
    const ModelExpression model_expression(neural_network.get());

    const string expression = model_expression.build_expression();

    EXPECT_TRUE(contains_token(expression, "alpha"));
    EXPECT_TRUE(contains_token(expression, "beta"));
    EXPECT_TRUE(contains_token(expression, "gamma"));
}


TEST_F(ModelExpressionTest, BuildExpressionContainsScalingAndActivations)
{
    const ModelExpression model_expression(neural_network.get());

    const string expression = model_expression.build_expression();

    EXPECT_TRUE(contains_token(expression, "scaled_alpha"));
    EXPECT_TRUE(contains_token(expression, "scaled_beta"));
    EXPECT_TRUE(contains_token(expression, "Tanh"));
    EXPECT_TRUE(contains_token(expression, "Identity"));
}


TEST_F(ModelExpressionTest, BuildExpressionHasOneLinePerNonInputNeuron)
{
    const ModelExpression model_expression(neural_network.get());

    const string expression = model_expression.build_expression();

    const Index assignments = ranges::count(expression, '=');

    EXPECT_GT(assignments, 0);
}


TEST_F(ModelExpressionTest, SaveCExpression)
{
    const ModelExpression model_expression(neural_network.get());

    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_model_expression_test.c";

    model_expression.save(path, ModelExpression::ProgrammingLanguage::C);

    ASSERT_TRUE(filesystem::exists(path));

    const string source = read_whole_file(path);

    EXPECT_FALSE(source.empty());
    EXPECT_TRUE(contains_token(source, "calculate_outputs"));
    EXPECT_TRUE(contains_token(source, "int main"));
    EXPECT_TRUE(contains_token(source, "float Identity"));
    EXPECT_TRUE(contains_token(source, "gamma"));

    filesystem::remove(path);
}


TEST_F(ModelExpressionTest, SaveCEmbeddedExpression)
{
    const ModelExpression model_expression(neural_network.get());

    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_model_expression_test_embedded.c";

    model_expression.save(path, ModelExpression::ProgrammingLanguage::CEmbedded);

    ASSERT_TRUE(filesystem::exists(path));

    const string source = read_whole_file(path);

    EXPECT_FALSE(source.empty());
    EXPECT_TRUE(contains_token(source, "static const float"));
    EXPECT_TRUE(contains_token(source, "nn_dense_forward"));
    EXPECT_TRUE(contains_token(source, "nn_affine_forward"));
    EXPECT_TRUE(contains_token(source, "float* calculate_outputs"));
    EXPECT_TRUE(contains_token(source, "int main"));
    EXPECT_TRUE(contains_token(source, "OPENNN_EXPORT_NO_MAIN"));

    filesystem::remove(path);
}


TEST_F(ModelExpressionTest, SavePythonExpression)
{
    const ModelExpression model_expression(neural_network.get());

    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_model_expression_test.py";

    model_expression.save(path, ModelExpression::ProgrammingLanguage::Python);

    ASSERT_TRUE(filesystem::exists(path));

    const string source = read_whole_file(path);

    EXPECT_FALSE(source.empty());
    EXPECT_TRUE(contains_token(source, "import numpy as np"));
    EXPECT_TRUE(contains_token(source, "class NeuralNetwork"));
    EXPECT_TRUE(contains_token(source, "def calculate_outputs"));
    EXPECT_TRUE(contains_token(source, "def main"));

    filesystem::remove(path);
}


TEST_F(ModelExpressionTest, SaveJavaScriptExpression)
{
    const ModelExpression model_expression(neural_network.get());

    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_model_expression_test.html";

    model_expression.save(path, ModelExpression::ProgrammingLanguage::JavaScript);

    ASSERT_TRUE(filesystem::exists(path));

    const string source = read_whole_file(path);

    EXPECT_FALSE(source.empty());
    EXPECT_TRUE(contains_token(source, "function neuralNetwork"));
    EXPECT_TRUE(contains_token(source, "function calculate_outputs"));
    EXPECT_TRUE(contains_token(source, "<script>"));

    filesystem::remove(path);
}


TEST_F(ModelExpressionTest, SavePhpExpression)
{
    const ModelExpression model_expression(neural_network.get());

    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_model_expression_test.php";

    model_expression.save(path, ModelExpression::ProgrammingLanguage::PHP);

    ASSERT_TRUE(filesystem::exists(path));

    const string source = read_whole_file(path);

    EXPECT_FALSE(source.empty());
    EXPECT_TRUE(contains_token(source, "<?php"));
    EXPECT_TRUE(contains_token(source, "session_start"));
    EXPECT_TRUE(contains_token(source, "json_encode"));

    filesystem::remove(path);
}


TEST_F(ModelExpressionTest, SaveThrowsOnUnwritablePath)
{
    const ModelExpression model_expression(neural_network.get());

    const filesystem::path path = "/this_directory_should_not_exist_12345/model.c";

    EXPECT_ANY_THROW(model_expression.save(path, ModelExpression::ProgrammingLanguage::C));
}
