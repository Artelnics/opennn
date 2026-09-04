//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E X P R E S S I O N   T E S T S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The expression language a constraint or an objective is written in, on its own: what
// compiling a string yields, what the compiled form says about itself, what it evaluates
// to, and which strings are refused.
//
// Most cases here compile against a bare list of named columns, with no network at all,
// so what they check is the parser and nothing else. The residual cases follow, since a
// residual is an expression read through a condition.
//
// What the optimizers then do with those expressions is checked in conditions_test.cpp.

#include "tests/pch.h"

#include "tests/response_optimization/synthetic_fixture.h"

#include "opennn/response_optimization/expression_evaluator.h"

namespace
{

// The columns an expression is compiled against: a name and the position it stands for.

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

}


TEST(Expression, LinearSumKeepsSignedCoefficients)
{
    const CompiledExpression expression =
        compile_expression("x1 + 2*x2 - 3", make_named_columns({"x1", "x2"}), {});

    EXPECT_EQ(expression.linearity, ExpressionLinearity::Linear);
    EXPECT_FALSE(is_output_coupled(expression));
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

    EXPECT_TRUE(is_univariate(compile_expression("2*x1", inputs, {})));
    EXPECT_FALSE(is_univariate(compile_expression("x1 + x2", inputs, {})));
}


TEST(Expression, InvolvementInputsOnly)
{
    const CompiledExpression expression = compile_expression("x1 + x2",
                                                             make_named_columns({"x1", "x2"}),
                                                             make_named_columns({"y1"}));

    EXPECT_FALSE(is_output_coupled(expression));
    EXPECT_EQ(expression.input_indices.size(), 2u);
    EXPECT_TRUE(expression.output_indices.empty());
}


TEST(Expression, InvolvementOutputsOnly)
{
    const CompiledExpression expression = compile_expression("y1",
                                                             make_named_columns({"x1"}),
                                                             make_named_columns({"y1"}));

    EXPECT_TRUE(is_output_coupled(expression));
    EXPECT_TRUE(expression.input_indices.empty());
    EXPECT_EQ(expression.output_indices.size(), 1u);
}


TEST(Expression, InvolvementMixed)
{
    const CompiledExpression expression = compile_expression("x1 + y1",
                                                             make_named_columns({"x1"}),
                                                             make_named_columns({"y1"}));

    EXPECT_TRUE(is_output_coupled(expression));
    EXPECT_FALSE(expression.input_indices.empty());
    EXPECT_FALSE(expression.output_indices.empty());
}


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


TEST(Expression, MinMaxSelectABranch)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    const CompiledExpression smallest = compile_expression("min(x1, x2)", inputs, {});
    const CompiledExpression largest = compile_expression("max(x1, x2)", inputs, {});

    VectorR input(2); input << 2.0f, 7.0f;
    const VectorR output(0);

    EXPECT_NEAR(smallest.evaluate(input, output), 2.0f, 1e-5f);
    EXPECT_NEAR(largest.evaluate(input, output), 7.0f, 1e-5f);
    EXPECT_EQ(smallest.linearity, ExpressionLinearity::Nonlinear);
}


// Away from the kink a selector is differentiable, and its derivative is the derivative of
// whichever branch wins. On the kink it is not, and the gradient says so by not being finite,
// which is how the solver knows to fall back to probing that row.

TEST(Expression, SelectorGradientFollowsTheWinningBranch)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    const VectorR output(0);

    VectorR point(2); point << 2.0f, 7.0f;

    const VectorR smallest_gradient =
        evaluate_input_gradient(compile_expression("min(x1, x2)", inputs, {}), point, output);

    EXPECT_NEAR(smallest_gradient(0), 1.0f, 1e-4f);
    EXPECT_NEAR(smallest_gradient(1), 0.0f, 1e-4f);

    const VectorR largest_gradient =
        evaluate_input_gradient(compile_expression("max(x1, x2)", inputs, {}), point, output);

    EXPECT_NEAR(largest_gradient(0), 0.0f, 1e-4f);
    EXPECT_NEAR(largest_gradient(1), 1.0f, 1e-4f);

    const VectorR magnitude_gradient =
        evaluate_input_gradient(compile_expression("abs(x1 - 5)", inputs, {}), point, output);

    EXPECT_NEAR(magnitude_gradient(0), -1.0f, 1e-4f);
}


TEST(Expression, SelectorGradientIsNotFiniteOnTheKink)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    const VectorR output(0);

    VectorR tied(2); tied << 5.0f, 5.0f;

    EXPECT_FALSE(evaluate_input_gradient(compile_expression("min(x1, x2)", inputs, {}),
                                         tied, output).allFinite());

    EXPECT_FALSE(evaluate_input_gradient(compile_expression("abs(x1 - 5)", inputs, {}),
                                         tied, output).allFinite());
}


TEST(Expression, PowerCallIsTheSameAsThePowerOperator)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

    const CompiledExpression called = compile_expression("pow(x1, 2) + x2", inputs, {});
    const CompiledExpression written = compile_expression("x1^2 + x2", inputs, {});

    EXPECT_TRUE(same_expression(called, written));

    VectorR point(2); point << 3.0f, 4.0f;
    const VectorR output(0);

    EXPECT_NEAR(called.evaluate(point, output), 13.0f, 1e-5f);

    const VectorR gradient = evaluate_input_gradient(called, point, output);

    EXPECT_NEAR(gradient(0), 6.0f, 1e-4f);
    EXPECT_NEAR(gradient(1), 1.0f, 1e-4f);
}


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


TEST(ConstraintResidual, SilentInsideAndSignedOutside)
{
    ResponseOptimization::Constraint constraint;

    constraint.expression = compile_expression("x1", make_named_columns({"x1"}), {});

    constraint.condition = Condition::Between;
    constraint.values = {2.0f, 6.0f};

    EXPECT_FALSE(isfinite(constraint.calculate_residual(4.0f)));

    EXPECT_NEAR(constraint.calculate_residual(1.0f), -1.0f, 1e-6f);

    EXPECT_NEAR(constraint.calculate_residual(8.0f), 2.0f, 1e-6f);
}


TEST(ConstraintResidual, EqualityIsSilentOnTargetAndSignedOutside)
{
    ResponseOptimization::Constraint constraint;

    constraint.expression = compile_expression("x1", make_named_columns({"x1"}), {});

    constraint.condition = Condition::Equal;
    constraint.values = {5.0f};

    EXPECT_FALSE(isfinite(constraint.calculate_residual(5.0f)));

    EXPECT_NEAR(constraint.calculate_residual(7.0f), 2.0f, 1e-6f);
}


TEST(ExpressionHelpers, SameExpressionComparesTheCompiledForm)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1", "x2"});

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

    const VectorR nonlinear_gradient =
        evaluate_input_gradient(compile_expression("x1^2 + x2^2", inputs, {}), point, output);

    EXPECT_NEAR(nonlinear_gradient(0), 6.0f, 1e-4f);
    EXPECT_NEAR(nonlinear_gradient(1), 8.0f, 1e-4f);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
