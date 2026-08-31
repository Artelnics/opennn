//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   T E S T S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tests/pch.h"

#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/expression_evaluator.h"
#include "opennn/response_optimization/genetic_response.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/random_utilities.h"
#include "opennn/response_optimization/response_optimization.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/core/statistics.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/core/variable.h"

using namespace opennn;

using Sense = ResponseOptimization::Objective::Sense;
using Condition = ResponseOptimization::Constraint::Condition;

namespace
{

vector<pair<string, Index>> make_named_columns(const vector<string>& names)
{
    vector<pair<string, Index>> columns;

    columns.reserve(names.size());

    for (Index i = 0; i < Index(names.size()); i++)
        columns.emplace_back(names[size_t(i)], i);

    return columns;
}


float lookup_coefficient(const vector<pair<Index, float>>& terms, const Index column)
{
    for (const auto& [term_column, coefficient] : terms)
        if (term_column == column) return coefficient;

    return 0.0f;
}


vector<Descriptives> make_descriptives(const Index count, const float minimum, const float maximum)
{
    return vector<Descriptives>(size_t(count),
                                Descriptives(minimum,
                                             maximum,
                                             0.5f*(minimum + maximum),
                                             0.25f*(maximum - minimum)));
}


// A compiled, Glorot-initialized approximation network with named variables and a box-scaled input
// domain. Nothing is trained: the tests that touch the network only need a smooth finite response
// and the input box that calculate_domain() reads off the scaling layer.

struct MinimalApproximation
{
    unique_ptr<ApproximationNetwork> network;

    MinimalApproximation(const vector<string>& input_names,
                         const vector<string>& output_names,
                         const float input_minimum = 0.0f,
                         const float input_maximum = 10.0f,
                         const float output_minimum = -1.0f,
                         const float output_maximum = 1.0f)
    {
        const Index inputs_number = Index(input_names.size());
        const Index outputs_number = Index(output_names.size());

        network = make_unique<ApproximationNetwork>(Shape{inputs_number}, Shape{4}, Shape{outputs_number});

        vector<Variable> input_variables(static_cast<size_t>(inputs_number));

        for (Index i = 0; i < inputs_number; i++)
        {
            input_variables[size_t(i)].name = input_names[size_t(i)];
            input_variables[size_t(i)].set_role("Input");
            input_variables[size_t(i)].type = VariableType::Numeric;
        }

        network->set_input_variables(input_variables);

        vector<Variable> output_variables(static_cast<size_t>(outputs_number));

        for (Index i = 0; i < outputs_number; i++)
        {
            output_variables[size_t(i)].name = output_names[size_t(i)];
            output_variables[size_t(i)].set_role("Target");
            output_variables[size_t(i)].type = VariableType::Numeric;
        }

        network->set_output_variables(output_variables);

        static_cast<Scaling*>(network->get_first("Scaling"))
            ->set_descriptives(make_descriptives(inputs_number, input_minimum, input_maximum));

        static_cast<Unscaling*>(network->get_first("Unscaling"))
            ->set_descriptives(make_descriptives(outputs_number, output_minimum, output_maximum));
    }
};


struct CategoricalApproximation
{
    unique_ptr<ApproximationNetwork> network;

    CategoricalApproximation(const vector<string>& numeric_names,
                             const string& categorical_name,
                             const vector<string>& categories,
                             const Index outputs_number = 1,
                             const float input_minimum = 0.0f,
                             const float input_maximum = 10.0f)
    {
        const Index numeric_number = Index(numeric_names.size());
        const Index categories_number = Index(categories.size());

        network = make_unique<ApproximationNetwork>(Shape{numeric_number + categories_number},
                                                    Shape{4},
                                                    Shape{outputs_number});

        vector<Variable> input_variables(size_t(numeric_number) + 1);

        for (Index i = 0; i < numeric_number; i++)
        {
            input_variables[size_t(i)].name = numeric_names[size_t(i)];
            input_variables[size_t(i)].set_role("Input");
            input_variables[size_t(i)].type = VariableType::Numeric;
        }

        input_variables.back().name = categorical_name;
        input_variables.back().set_role("Input");
        input_variables.back().type = VariableType::Categorical;
        input_variables.back().set_categories(categories);

        network->set_input_variables(input_variables);

        vector<Variable> output_variables(static_cast<size_t>(outputs_number));

        for (Index i = 0; i < outputs_number; i++)
        {
            output_variables[size_t(i)].name = "y" + to_string(i + 1);
            output_variables[size_t(i)].set_role("Target");
            output_variables[size_t(i)].type = VariableType::Numeric;
        }

        network->set_output_variables(output_variables);

        vector<Descriptives> input_descriptives = make_descriptives(numeric_number,
                                                                    input_minimum,
                                                                    input_maximum);

        const vector<Descriptives> category_descriptives = make_descriptives(categories_number,
                                                                             0.0f,
                                                                             1.0f);

        input_descriptives.insert(input_descriptives.end(),
                                  category_descriptives.begin(),
                                  category_descriptives.end());

        static_cast<Scaling*>(network->get_first("Scaling"))->set_descriptives(input_descriptives);

        static_cast<Unscaling*>(network->get_first("Unscaling"))
            ->set_descriptives(make_descriptives(outputs_number, -1.0f, 1.0f));
    }
};


vector<float> scan_categories(NeuralNetwork& network,
                              const Index numeric_number,
                              const Index categories_number,
                              const float input_minimum,
                              const float input_maximum,
                              const Index samples_number = 4096)
{
    const Index features_number = numeric_number + categories_number;

    vector<float> best_values(size_t(categories_number), -numeric_limits<float>::max());

    MatrixR inputs(samples_number, features_number);

    for (Index category = 0; category < categories_number; category++)
    {
        set_random_uniform(inputs, input_minimum, input_maximum);

        inputs.rightCols(categories_number).setZero();
        inputs.col(numeric_number + category).setOnes();

        const MatrixR outputs = network.calculate_outputs(inputs);

        for (Index i = 0; i < samples_number; i++)
            best_values[size_t(category)] = max(best_values[size_t(category)], -outputs(i, 0));
    }

    return best_values;
}


Index read_category(const MatrixR& results,
                    const Index row,
                    const Index numeric_number,
                    const Index categories_number)
{
    Index category = -1;

    for (Index j = 0; j < categories_number; j++)
    {
        const float value = results(row, numeric_number + j);

        if (value == 0.0f) continue;

        if (value != 1.0f || category >= 0) return -1;

        category = j;
    }

    return category;
}


// The median and the spread of what the untrained network actually produces over its input box, so
// Fixed-objective and output-constraint tests aim at a value that is provably attainable.

pair<float, float> sample_response(NeuralNetwork& network,
                                   const Index inputs_number,
                                   const float input_minimum,
                                   const float input_maximum,
                                   const Index samples_number = 512)
{
    MatrixR inputs(samples_number, inputs_number);

    set_random_uniform(inputs, input_minimum, input_maximum);

    const MatrixR outputs = network.calculate_outputs(inputs);

    vector<float> values(static_cast<size_t>(samples_number));

    for (Index i = 0; i < samples_number; i++)
        values[size_t(i)] = outputs(i, 0);

    ranges::sort(values);

    return {values[values.size()/2], values.back() - values.front()};
}


enum class Driver { Contraction, Genetic };

unique_ptr<ResponseOptimization> make_driver(const Driver driver, NeuralNetwork* network)
{
    if (driver == Driver::Genetic) return make_unique<GeneticResponse>(network);

    return make_unique<DomainContraction>(network);
}


string driver_name(const testing::TestParamInfo<Driver>& info)
{
    return info.param == Driver::Genetic ? "Genetic" : "Contraction";
}


// Both drivers sample at random; a fixed seed keeps the end-to-end assertions reproducible.

class ResponseDriver : public testing::TestWithParam<Driver>
{
protected:

    void SetUp() override { set_seed(1234); }
};

}


// -----------------------------------------------------------------------------
// Expression compiler: linearity
// -----------------------------------------------------------------------------

TEST(Expression, LinearSumKeepsSignedCoefficients)
{
    const CompiledExpression expression =
        compile_expression("x1 + 2*x2 - 3", make_named_columns({"x1", "x2"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Linear);
    EXPECT_EQ(expression.involvement, ExpressionInvolvement::InputsOnly);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(expression.linear_constant, -3.0f, 1e-6f);
}


TEST(Expression, UnaryNegationFlipsCoefficients)
{
    const CompiledExpression expression =
        compile_expression("-x1 + x2", make_named_columns({"x1", "x2"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Linear);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 0), -1.0f, 1e-6f);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 1), 1.0f, 1e-6f);
}


TEST(Expression, ConstantScalingDistributesOverSum)
{
    const CompiledExpression expression =
        compile_expression("3*(x1 + x2)", make_named_columns({"x1", "x2"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Linear);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 1), 3.0f, 1e-6f);
}


TEST(Expression, DivisionByConstantIsLinear)
{
    const CompiledExpression expression = compile_expression("x1 / 4", make_named_columns({"x1"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Linear);
    EXPECT_NEAR(lookup_coefficient(expression.linear_input_terms, 0), 0.25f, 1e-6f);
}


TEST(Expression, ProductOfVariablesIsNonlinear)
{
    const CompiledExpression expression =
        compile_expression("x1 * x2", make_named_columns({"x1", "x2"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Nonlinear);
}


TEST(Expression, DivisionByVariableIsNonlinear)
{
    const CompiledExpression expression =
        compile_expression("x1 / x2", make_named_columns({"x1", "x2"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Nonlinear);
}


TEST(Expression, SqrtIsNonlinear)
{
    const CompiledExpression expression = compile_expression("sqrt(x1) + 1", make_named_columns({"x1"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Nonlinear);
}


TEST(Expression, PowerWithNonUnitExponentIsNonlinear)
{
    const CompiledExpression expression = compile_expression("x1 ^ 2", make_named_columns({"x1"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Nonlinear);
}


TEST(Expression, SingleColumnExpressionIsUnivariate)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    EXPECT_EQ(compile_expression("2*x1", inputs, {}).complexity, ExpressionComplexity::Univariate);
    EXPECT_EQ(compile_expression("x1 + x2", inputs, {}).complexity, ExpressionComplexity::Multivariate);
}


// -----------------------------------------------------------------------------
// Expression compiler: involvement
// -----------------------------------------------------------------------------

TEST(Expression, InvolvementInputsOnly)
{
    const CompiledExpression expression = compile_expression("x1 + x2",
                                                             make_named_columns({"x1", "x2"}),
                                                             make_named_columns({"y1"}));

    EXPECT_EQ(expression.involvement, ExpressionInvolvement::InputsOnly);
    EXPECT_FALSE(is_output_coupled(expression));
    EXPECT_EQ(expression.input_indices.size(), 2u);
    EXPECT_TRUE(expression.output_indices.empty());
}


TEST(Expression, InvolvementOutputsOnly)
{
    const CompiledExpression expression = compile_expression("y1",
                                                             make_named_columns({"x1"}),
                                                             make_named_columns({"y1"}));

    EXPECT_EQ(expression.involvement, ExpressionInvolvement::OutputsOnly);
    EXPECT_TRUE(is_output_coupled(expression));
    EXPECT_TRUE(expression.input_indices.empty());
    EXPECT_EQ(expression.output_indices.size(), 1u);
}


TEST(Expression, InvolvementMixed)
{
    const CompiledExpression expression = compile_expression("x1 + y1",
                                                             make_named_columns({"x1"}),
                                                             make_named_columns({"y1"}));

    EXPECT_EQ(expression.involvement, ExpressionInvolvement::Mixed);
    EXPECT_TRUE(is_output_coupled(expression));
}


// -----------------------------------------------------------------------------
// Expression compiler: evaluation
// -----------------------------------------------------------------------------

TEST(Expression, EvaluateLinearRespectsSignedCoefficients)
{
    const CompiledExpression expression =
        compile_expression("-x1 + 2*x2 + 1", make_named_columns({"x1", "x2"}), {});

    VectorR input(2); input << 3.0f, 5.0f;
    const VectorR output(0);

    EXPECT_NEAR(expression.evaluate(input, output), 8.0f, 1e-5f);
}


TEST(Expression, EvaluateNonlinearExpression)
{
    const CompiledExpression expression =
        compile_expression("sqrt(x1) + x2^2", make_named_columns({"x1", "x2"}), {});

    VectorR input(2); input << 9.0f, 3.0f;
    const VectorR output(0);

    EXPECT_NEAR(expression.evaluate(input, output), 12.0f, 1e-5f);
}


TEST(Expression, EvaluateUsesOutputsWhenMixed)
{
    const CompiledExpression expression = compile_expression("x1 + 2*y1",
                                                             make_named_columns({"x1"}),
                                                             make_named_columns({"y1"}));

    VectorR input(1); input << 1.0f;
    VectorR output(1); output << 4.0f;

    EXPECT_NEAR(expression.evaluate(input, output), 9.0f, 1e-5f);
}


TEST(Expression, ParenthesesOverridePrecedence)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    const CompiledExpression without = compile_expression("2 * x1 + x2", inputs, {});
    const CompiledExpression with = compile_expression("2 * (x1 + x2)", inputs, {});

    VectorR input(2); input << 3.0f, 5.0f;
    const VectorR output(0);

    EXPECT_NEAR(without.evaluate(input, output), 11.0f, 1e-5f);
    EXPECT_NEAR(with.evaluate(input, output), 16.0f, 1e-5f);
}


TEST(Expression, MinMaxAreNonSmooth)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    const CompiledExpression smallest = compile_expression("min(x1, x2)", inputs, {});
    const CompiledExpression largest = compile_expression("max(x1, x2)", inputs, {});

    VectorR input(2); input << 2.0f, 7.0f;
    const VectorR output(0);

    EXPECT_NEAR(smallest.evaluate(input, output), 2.0f, 1e-5f);
    EXPECT_NEAR(largest.evaluate(input, output), 7.0f, 1e-5f);
    EXPECT_EQ(smallest.smoothness, ExpressionSmoothness::NonSmooth);
    EXPECT_EQ(smallest.linearity, ExpressionLinearity::Nonlinear);
}


// -----------------------------------------------------------------------------
// Expression compiler: rejected text
// -----------------------------------------------------------------------------

TEST(Expression, UnknownIdentifierThrows)
{
    EXPECT_THROW(compile_expression("x1 + z9", make_named_columns({"x1"}), {}), runtime_error);
}


TEST(Expression, UnknownFunctionThrows)
{
    EXPECT_THROW(compile_expression("bogus(x1)", make_named_columns({"x1"}), {}), runtime_error);
}


TEST(Expression, EmptyExpressionThrows)
{
    EXPECT_THROW(compile_expression("", {}, {}), runtime_error);
}


TEST(Expression, ExpressionWithoutVariablesThrows)
{
    EXPECT_THROW(compile_expression("1 + 2", make_named_columns({"x1"}), {}), runtime_error);
}


TEST(Expression, WrongFunctionArityThrows)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    EXPECT_THROW(compile_expression("sqrt(x1, x2)", inputs, {}), runtime_error);
    EXPECT_THROW(compile_expression("min(x1)", inputs, {}), runtime_error);
}


TEST(Expression, MismatchedParenthesesThrow)
{
    EXPECT_THROW(compile_expression("(x1 + 1", make_named_columns({"x1"}), {}), runtime_error);
}


TEST(Expression, ComparisonSymbolsAreRejectedAgainstANetwork)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    EXPECT_THROW(compile_expression("x1 <= 3", setup.network.get(), "Constraint"), runtime_error);
}


TEST(Expression, CompilingWithoutANetworkThrows)
{
    EXPECT_THROW(compile_expression("x1", nullptr, "Objective"), runtime_error);
}


// -----------------------------------------------------------------------------
// Expression helpers used by the feasibility gate
// -----------------------------------------------------------------------------

TEST(ConstraintResidual, SilentInsideAndSignedOutside)
{
    ResponseOptimization::Constraint constraint;

    constraint.expression = compile_expression("x1", make_named_columns({"x1"}), {});

    constraint.condition = Condition::Between;
    constraint.values = {2.0f, 6.0f};

    const VectorR output;

    EXPECT_FALSE(isfinite(constraint.calculate_residual(VectorR::Constant(1, 4.0f), output)));

    EXPECT_NEAR(constraint.calculate_residual(VectorR::Constant(1, 1.0f), output), -1.0f, 1e-6f);

    EXPECT_NEAR(constraint.calculate_residual(VectorR::Constant(1, 8.0f), output), 2.0f, 1e-6f);
}


TEST(ConstraintResidual, EqualityIsSilentOnTargetAndSignedOutside)
{
    ResponseOptimization::Constraint constraint;

    constraint.expression = compile_expression("x1", make_named_columns({"x1"}), {});

    constraint.condition = Condition::Equal;
    constraint.values = {5.0f};

    const VectorR output;

    EXPECT_FALSE(isfinite(constraint.calculate_residual(VectorR::Constant(1, 5.0f), output)));

    EXPECT_NEAR(constraint.calculate_residual(VectorR::Constant(1, 7.0f), output), 2.0f, 1e-6f);
}


TEST(ExpressionHelpers, SameExpressionComparesTheCompiledForm)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    // Spelling and spacing do not matter, and a linear expression is compared by its terms rather
    // than by its operations.
    EXPECT_TRUE(same_expression(compile_expression("x1 + 2*x2", inputs, {}),
                                compile_expression("x1+2*x2", inputs, {})));

    EXPECT_TRUE(same_expression(compile_expression("2*(x1 + x2)", inputs, {}),
                                compile_expression("2*x1 + 2*x2", inputs, {})));

    EXPECT_FALSE(same_expression(compile_expression("x1", inputs, {}),
                                 compile_expression("2*x1", inputs, {})));

    EXPECT_FALSE(same_expression(compile_expression("x1 + x2", inputs, {}),
                                 compile_expression("x1 * x2", inputs, {})));
}


TEST(ExpressionHelpers, SameExpressionIsSensitiveToTermOrder)
{
    // Known limitation, pinned so a change is deliberate: linear terms are compared as an ordered
    // vector, so a commutative rewrite is not recognized as the same expression. The duplicate
    // objective warning and the contradictory constraint check both miss such a pair.
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    EXPECT_FALSE(same_expression(compile_expression("x1 + x2", inputs, {}),
                                 compile_expression("x2 + x1", inputs, {})));
}


TEST(ExpressionHelpers, BareVariableIsAPlainUnscaledColumn)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    EXPECT_TRUE(is_bare_variable(compile_expression("x1", inputs, {})));
    EXPECT_FALSE(is_bare_variable(compile_expression("2*x1", inputs, {})));
    EXPECT_FALSE(is_bare_variable(compile_expression("x1 + 1", inputs, {})));
    EXPECT_FALSE(is_bare_variable(compile_expression("x1 + x2", inputs, {})));
}


TEST(ExpressionHelpers, InputGradientMatchesTheCoefficients)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    VectorR point(2); point << 3.0f, 4.0f;
    const VectorR output(0);

    const VectorR linear_gradient =
        evaluate_input_gradient(compile_expression("-x1 + 2*x2", inputs, {}), point, output);

    EXPECT_NEAR(linear_gradient(0), -1.0f, 1e-5f);
    EXPECT_NEAR(linear_gradient(1), 2.0f, 1e-5f);

    // d(x1^2 + x2^2) = (2*x1, 2*x2)
    const VectorR nonlinear_gradient =
        evaluate_input_gradient(compile_expression("x1^2 + x2^2", inputs, {}), point, output);

    EXPECT_NEAR(nonlinear_gradient(0), 6.0f, 1e-4f);
    EXPECT_NEAR(nonlinear_gradient(1), 8.0f, 1e-4f);
}


// -----------------------------------------------------------------------------
// Selector conjunctions: min/max/abs split into smooth pieces
// -----------------------------------------------------------------------------

TEST(DrawKHot, DrawsExactlyKHonouringPins)
{
    const Index count = 8;
    const Index k = 3;

    vector<char> force_on(size_t(count), 0);
    vector<char> force_off(size_t(count), 0);

    force_on[1] = 1;
    force_off[5] = 1;
    force_off[6] = 1;

    for (Index trial = 0; trial < 200; trial++)
    {
        vector<float> selection;

        ASSERT_TRUE(draw_k_hot(count, k, force_on, force_off, selection));
        ASSERT_EQ(Index(selection.size()), count);

        float ones = 0.0f;

        for (const float value : selection)
        {
            EXPECT_TRUE(value == 0.0f || value == 1.0f);
            ones += value;
        }

        EXPECT_EQ(ones, float(k));
        EXPECT_EQ(selection[1], 1.0f);
        EXPECT_EQ(selection[5], 0.0f);
        EXPECT_EQ(selection[6], 0.0f);
    }
}


TEST(DrawKHot, ReportsInfeasiblePins)
{
    const Index count = 4;

    vector<float> selection;

    {
        vector<char> force_on(size_t(count), 0);
        vector<char> force_off(size_t(count), 0);

        force_on[0] = force_on[1] = force_on[2] = 1;

        EXPECT_FALSE(draw_k_hot(count, 2, force_on, force_off, selection));
    }

    {
        vector<char> force_on(size_t(count), 0);
        vector<char> force_off(size_t(count), 0);

        force_off[0] = force_off[1] = force_off[2] = 1;

        EXPECT_FALSE(draw_k_hot(count, 2, force_on, force_off, selection));
    }

    {
        vector<char> force_on(size_t(count), 0);
        vector<char> force_off(size_t(count), 0);

        force_on[0] = 1;
        force_off[0] = 1;

        EXPECT_FALSE(draw_k_hot(count, 1, force_on, force_off, selection));
    }
}


// -----------------------------------------------------------------------------
// ResponseOptimization: what the setup refuses
// -----------------------------------------------------------------------------

TEST(ResponseOptimizationSetup, NoNeuralNetworkThrows)
{
    DomainContraction optimization;

    EXPECT_THROW(optimization.add_objective("y", Sense::Minimize), runtime_error);
}


TEST(ResponseOptimizationSetup, NoObjectiveThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationSetup, CardinalityConditionIsRefused)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::Cardinality, {2.0f}), runtime_error);
}


TEST(ResponseOptimizationSetup, NonFiniteConstraintValueThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::LessEqual,
                                             {numeric_limits<float>::infinity()}),
                 runtime_error);
}


TEST(ResponseOptimizationSetup, AllowedSetNeedsAtLeastOneValue)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::AllowedSet, {}), runtime_error);
}


TEST(ResponseOptimizationSetup, IntegerConditionOnlyAppliesToASingleVariable)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1 + x2", Condition::Integer), runtime_error);
    EXPECT_NO_THROW(optimization.add_constraint("x1", Condition::Integer));
}


TEST(ResponseOptimizationSetup, MissingConditionValuesThrow)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::Between, {5.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1", Condition::LessEqual, {}), runtime_error);
}


TEST(ResponseOptimizationSetup, EmptyBetweenIntervalThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::Between, {5.0f, 2.0f}), runtime_error);
}


TEST(ResponseOptimizationSetup, ConstraintsThatEmptyAColumnThrowWhenTheDomainIsBuilt)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    DomainContraction optimization(setup.network.get());

    optimization.add_objective("y", Sense::Minimize);

    // Written differently, so neither is recognized as constraining the same expression as the
    // other, yet together they leave x1 with the empty range [8, 2].
    optimization.add_constraint("x1", Condition::GreaterEqual, {8.0f});
    optimization.add_constraint("2 * x1", Condition::LessEqual, {4.0f});

    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


// -----------------------------------------------------------------------------
// ResponseOptimization: both drivers honour the feasible set
// -----------------------------------------------------------------------------

TEST_P(ResponseDriver, ResultsStayInsideTheInputBox)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    ASSERT_EQ(results.cols(), 3);

    for (Index i = 0; i < results.rows(); i++)
        for (Index j = 0; j < 2; j++)
        {
            EXPECT_GE(results(i, j), -1e-3f);
            EXPECT_LE(results(i, j), 10.0f + 1e-3f);
        }
}


TEST_P(ResponseDriver, LinearInputConstraintHolds)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);
    optimization->add_constraint("x1 + x2", Condition::LessEqual, {4.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_LE(results(i, 0) + results(i, 1), 4.0f + 1e-2f)
            << "row " << i << " x1=" << results(i, 0) << " x2=" << results(i, 1);
}


TEST_P(ResponseDriver, EqualityLandsOnTheHyperplane)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);
    optimization->add_constraint("x1 + x2", Condition::Equal, {5.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_NEAR(results(i, 0) + results(i, 1), 5.0f, 5e-2f)
            << "row " << i << " x1=" << results(i, 0) << " x2=" << results(i, 1);
}


TEST_P(ResponseDriver, NonlinearInputConstraintHolds)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, -5.0f, 5.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);

    // Inside the disk of radius 2.
    optimization->add_constraint("x1^2 + x2^2", Condition::LessEqual, {4.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_LE(results(i, 0)*results(i, 0) + results(i, 1)*results(i, 1), 4.0f + 5e-2f)
            << "row " << i;
}


TEST_P(ResponseDriver, SingleVariableConstraintIsFoldedIntoTheBox)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);

    // 2*x1 - 6 <= 0 is one linear term, so calculate_domain() turns it into x1 <= 3.
    optimization->add_constraint("2 * x1 - 6", Condition::LessEqual, {0.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_LE(results(i, 0), 3.0f + 1e-3f) << "row " << i;
}


TEST_P(ResponseDriver, OutputConstraintHolds)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    // Half the box already satisfies y >= median, so the constraint is attainable by construction.
    const auto [median, span] = sample_response(*setup.network, 2, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("x1", Sense::Minimize);
    optimization->add_constraint("y", Condition::GreaterEqual, {median});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_GE(results(i, 2), median - 1e-2f*span) << "row " << i << " y=" << results(i, 2);
}


TEST_P(ResponseDriver, AllowedSetKeepsResultsOnTheListedValues)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);
    optimization->add_constraint("x1", Condition::AllowedSet, {1.0f, 5.0f, 9.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        const float x1 = results(i, 0);

        EXPECT_LT(min(min(abs(x1 - 1.0f), abs(x1 - 5.0f)), abs(x1 - 9.0f)), 1e-2f)
            << "row " << i << " x1=" << x1 << " is not in {1, 5, 9}";
    }
}


TEST_P(ResponseDriver, FixedObjectiveApproachesAReachableTarget)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const auto [target, span] = sample_response(*setup.network, 2, 0.0f, 10.0f);

    ASSERT_GT(span, 0.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Fixed, target);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    EXPECT_LE(abs(results(0, 2) - target), 0.1f*span)
        << "y=" << results(0, 2) << " target=" << target << " span=" << span;
}


TEST_P(ResponseDriver, MultipleObjectivesReturnAFeasibleFront)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Maximize);
    optimization->add_objective("x1", Sense::Minimize);
    optimization->add_constraint("x1 + x2", Condition::LessEqual, {8.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    ASSERT_EQ(results.cols(), 3);

    for (Index i = 0; i < results.rows(); i++)
    {
        EXPECT_LE(results(i, 0) + results(i, 1), 8.0f + 1e-2f) << "row " << i;

        for (Index j = 0; j < 2; j++)
        {
            EXPECT_GE(results(i, j), -1e-3f);
            EXPECT_LE(results(i, j), 10.0f + 1e-3f);
        }
    }
}


TEST_P(ResponseDriver, ConflictingObjectivesFillTheRequestedFront)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("x1", Sense::Maximize);
    optimization->add_objective("x2", Sense::Maximize);
    optimization->add_constraint("x1 + x2", Condition::LessEqual, {8.0f});

    const MatrixR results = optimization->perform_response_optimization();

    EXPECT_EQ(results.rows(), 100);

    for (Index i = 0; i < results.rows(); i++)
    {
        EXPECT_LE(results(i, 0) + results(i, 1), 8.0f + 1e-2f) << "row " << i << " is not feasible";

        EXPECT_GE(results(i, 0) + results(i, 1), 8.0f - 0.15f)
            << "row " << i << " sits behind the front: x1=" << results(i, 0) << " x2=" << results(i, 1);
    }

    EXPECT_GT(results.col(0).maxCoeff() - results.col(0).minCoeff(), 4.0f);
}


// -----------------------------------------------------------------------------
// Categorical inputs: one-hot blocks stay on the lattice
// -----------------------------------------------------------------------------

TEST(CategoricalBlocks, ReportsOneBlockPerCategoricalVariable)
{
    vector<Variable> variables(3);

    variables[0].name = "x1";
    variables[0].type = VariableType::Numeric;

    variables[1].name = "material";
    variables[1].type = VariableType::Categorical;
    variables[1].set_categories({"steel", "copper", "brass"});

    variables[2].name = "x2";
    variables[2].type = VariableType::Numeric;

    const vector<pair<Index, Index>> blocks = get_categorical_blocks(variables);

    ASSERT_EQ(blocks.size(), size_t(1));
    EXPECT_EQ(blocks[0].first, 1);
    EXPECT_EQ(blocks[0].second, 3);

    EXPECT_TRUE(get_categorical_blocks({variables[0], variables[2]}).empty());
}


TEST_P(ResponseDriver, CategoricalResultsAreOneHot)
{
    CategoricalApproximation setup({"x1", "x2"}, "material", {"steel", "copper", "brass"});

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y1", Sense::Minimize);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_GE(read_category(results, i, 2, 3), 0)
            << "row " << i << " holds "
            << results(i, 2) << ", " << results(i, 3) << ", " << results(i, 4);
}


TEST_P(ResponseDriver, CategoricalResultsSurviveAConstraint)
{
    CategoricalApproximation setup({"x1", "x2"}, "material", {"steel", "copper", "brass"});

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y1", Sense::Minimize);
    optimization->add_constraint("x1 + x2", Condition::Equal, {5.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        EXPECT_GE(read_category(results, i, 2, 3), 0) << "row " << i;

        EXPECT_NEAR(results(i, 0) + results(i, 1), 5.0f, 5e-2f) << "row " << i;
    }
}


TEST_P(ResponseDriver, CategoricalMultiObjectiveResultsAreOneHot)
{
    CategoricalApproximation setup({"x1", "x2"}, "material", {"steel", "copper", "brass"}, 2);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y1", Sense::Minimize);
    optimization->add_objective("y2", Sense::Maximize);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_GE(read_category(results, i, 2, 3), 0) << "row " << i;
}


TEST_P(ResponseDriver, CategoricalSearchMatchesAScanOverCategories)
{
    CategoricalApproximation setup({"x1", "x2"}, "material", {"steel", "copper", "brass"});

    const vector<float> scan_values = scan_categories(*setup.network, 2, 3, 0.0f, 10.0f);

    const float best_scan_value = ranges::max(scan_values);

    ASSERT_GT(best_scan_value - ranges::min(scan_values), 0.5f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y1", Sense::Minimize);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_EQ(results.rows(), 1);

    const Index category = read_category(results, 0, 2, 3);

    ASSERT_GE(category, 0);

    EXPECT_GE(-results(0, 5), best_scan_value - 1e-2f)
        << "kept category " << category << " worth " << -results(0, 5)
        << " against a scan best of " << best_scan_value;
}


INSTANTIATE_TEST_SUITE_P(Drivers,
                         ResponseDriver,
                         testing::Values(Driver::Contraction, Driver::Genetic),
                         driver_name);

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
