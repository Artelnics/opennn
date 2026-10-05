// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "tests/common/pch.h"

#include <bit>
#include <random>

#include "tests/response_optimization/synthetic_fixture.h"

#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/expression_evaluator.h"

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


double rooted_mean_by_subsets(const vector<double>& scaled, const Index order)
{
    const size_t variables_number = scaled.size();

    double sum = 0.0;
    double subsets = 0.0;

    for (uint32_t mask = 0; mask < (uint32_t(1) << variables_number); mask++)
    {
        if (popcount(mask) != int(order)) continue;

        double product = 1.0;

        for (size_t i = 0; i < variables_number; i++)
            if ((mask >> i) & 1u) product *= scaled[i]*scaled[i];

        sum += product;
        subsets += 1.0;
    }

    return sqrt(sum/subsets);
}


struct CardinalityCase
{
    vector<Index> columns;
    vector<float> spans;
    VectorR point;

    CompiledExpression row(const Index order, const float tolerance) const
    {
        return compile_elementary_symmetric(columns, spans, order, tolerance);
    }

    vector<double> scaled(const VectorR& at) const
    {
        vector<double> values(columns.size());

        for (size_t i = 0; i < columns.size(); i++)
            values[i] = double(at(columns[i]))/double(spans[i]);

        return values;
    }
};


struct ConstraintProbe : DomainContraction
{
    using DomainContraction::DomainContraction;

    using ResponseOptimization::constraints;
};


CardinalityCase draw_cardinality_case(mt19937& generator, const Index variables_number,
                                      const double smallest_magnitude)
{
    uniform_real_distribution<double> span(0.5, 50.0);
    uniform_real_distribution<double> magnitude(smallest_magnitude, 1.0);
    bernoulli_distribution negative(0.3);

    CardinalityCase c;

    c.point.resize(variables_number);

    for (Index i = 0; i < variables_number; i++)
    {
        c.columns.push_back(variables_number - 1 - i);
        c.spans.push_back(float(span(generator)));
    }

    for (Index i = 0; i < variables_number; i++)
        c.point(c.columns[size_t(i)]) = float((negative(generator) ? -1.0 : 1.0)
                                              *magnitude(generator)*double(c.spans[size_t(i)]));

    return c;
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


TEST(Expression, NonlinearOperationsAreClassified)
{
    const auto inputs = make_named_columns({"x1", "x2"});

    for (const string& text : {"x1 * x2", "x1 / x2", "sqrt(x1) + 1", "x1 ^ 2"})
    {
        SCOPED_TRACE(text);
        EXPECT_EQ(compile_expression(text, inputs, {}).linearity, ExpressionLinearity::Nonlinear);
    }
}


TEST(Expression, QuotedNamesPreservePunctuationAndRemainDistinct)
{
    const auto inputs = make_named_columns({"Flow rate (m3/s)", "Flow/rate", "Flow_rate", "a`b", "1st;input", "sqrt"});
    const auto outputs = make_named_columns({"Pressure (Pa)"});
    const auto expression = compile_expression(
        "`Flow rate (m3/s)` + 2*`Flow/rate` + Flow_rate + `a``b` + `1st;input` + sqrt(`sqrt`) + `Pressure (Pa)`",
        inputs, outputs);
    VectorR input(6); input << 1, 2, 3, 4, 5, 9;
    VectorR output(1); output << 7;
    EXPECT_FLOAT_EQ(expression.evaluate(input, output), 27.0f);
    const string unicode_name = "Flow (m\xC2\xB3/s)";
    const auto unicode_expression = compile_expression("`" + unicode_name + "`", {{unicode_name, 0}}, {});
    EXPECT_FLOAT_EQ(unicode_expression.evaluate(input, {}), 1.0f);
    EXPECT_TRUE(is_bare_variable(compile_expression("`Flow/rate`", inputs, outputs)));
    EXPECT_THROW(compile_expression("`missing`", inputs, outputs), runtime_error);
    EXPECT_THROW(compile_expression("`Flow/rate", inputs, outputs), runtime_error);
    EXPECT_THROW(compile_expression("``", inputs, outputs), runtime_error);
    EXPECT_THROW(compile_expression("`sqrt`(4)", inputs, outputs), runtime_error);
}


TEST(ConstraintCompilation, CardinalityAcceptsQuotedNamesAndEmbeddedSeparators)
{
    MinimalApproximation setup({"Flow (m3/s)", "a;b", "a`b"}, {"y"});
    ConstraintProbe problem(setup.network.get());
    problem.add_constraint("`Flow (m3/s)`; `a;b`; `a``b`", Condition::Cardinality, {1.0f});
    ASSERT_EQ(problem.constraints.size(), 1u);
    EXPECT_EQ(problem.constraints[0].equation.input_indices, vector<Index>({0, 1, 2}));
    EXPECT_THROW(problem.add_constraint("`a;b`; `a;b`", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(problem.add_constraint("`a;b`;", Condition::Cardinality, {1.0f}), runtime_error);
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

    EXPECT_EQ(expression.program.stack_depth, 3);

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

    EXPECT_EQ(smallest.program.stack_depth, 2);
    EXPECT_EQ(largest.program.stack_depth, 2);

    VectorR input(2); input << 2.0f, 7.0f;
    const VectorR output(0);

    EXPECT_NEAR(smallest.evaluate(input, output), 2.0f, 1e-5f);
    EXPECT_NEAR(largest.evaluate(input, output), 7.0f, 1e-5f);
    EXPECT_EQ(smallest.linearity, ExpressionLinearity::Nonlinear);
}


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

    EXPECT_TRUE(ranges::equal(called.program.operations, written.program.operations));
    EXPECT_EQ(called.program.stack_depth, 2);
    ASSERT_EQ(called.input_gradient.size(), 2u);
    EXPECT_EQ(called.input_gradient[1].first, 1);
    EXPECT_EQ(called.input_gradient[1].second.stack_depth, 1);

    VectorR point(2); point << 3.0f, 4.0f;
    const VectorR output(0);

    EXPECT_NEAR(called.evaluate(point, output), 13.0f, 1e-5f);

    const VectorR gradient = evaluate_input_gradient(called, point, output);

    EXPECT_NEAR(gradient(0), 6.0f, 1e-4f);
    EXPECT_NEAR(gradient(1), 1.0f, 1e-4f);
}


TEST(Expression, InvalidExpressionsThrow)
{
    const auto inputs = make_named_columns({"x1", "x2"});

    for (const string& text : {"x1 + z9", "bogus(x1)", "", "1 + 2",
                               "sqrt(x1, x2)", "min(x1)", "(x1 + 1"})
    {
        SCOPED_TRACE(text);
        EXPECT_THROW(compile_expression(text, inputs, {}), runtime_error);
    }
}


TEST(Expression, NestingBeyondTheLimitThrows)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1"});

    const auto nested = [](const size_t depth)
    { return string(depth, '(') + "x1" + string(depth, ')'); };

    EXPECT_NO_THROW(compile_expression(nested(250), inputs, {}));
    EXPECT_THROW(compile_expression(nested(300), inputs, {}), runtime_error);
    EXPECT_THROW(compile_expression(nested(5000), inputs, {}), runtime_error);
}


TEST(Expression, ChainedOperationsBeyondTheLimitThrow)
{
    const vector<pair<string, Index>> inputs = make_named_columns({"x1"});

    const auto chained = [](const size_t terms)
    {
        string expression = "x1";

        for (size_t term = 1; term < terms; term++)
            expression += " + x1";

        return expression;
    };

    EXPECT_NO_THROW(compile_expression(chained(500), inputs, {}));
    EXPECT_THROW(compile_expression(chained(600), inputs, {}), runtime_error);
    EXPECT_THROW(compile_expression(chained(5000), inputs, {}), runtime_error);
}


TEST(Expression, ARefusedExpressionIsQuotedBackAbbreviated)
{
    MinimalApproximation setup({"x1"}, {"y"});

    try
    {
        compile_expression(string(5000, '(') + "x1" + string(5000, ')'),
                           setup.network.get(), "Objective");

        FAIL() << "an expression nested five thousand levels deep must be refused";
    }
    catch (const runtime_error& error)
    {
        EXPECT_LT(string(error.what()).size(), size_t(400));
    }
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


TEST(ConstraintCompilation, AnIntervalConditionBecomesOneEquationAndItsBand)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    ConstraintProbe problem(setup.network.get());

    problem.add_constraint("x1", Condition::Between, {2.0f, 6.0f});
    problem.add_constraint("x1", Condition::Equal, {5.0f});

    ASSERT_EQ(problem.constraints.size(), 2u);

    const ResponseOptimization::Constraint& between = problem.constraints[0];
    const ResponseOptimization::Constraint& equal = problem.constraints[1];

    EXPECT_TRUE(is_bare_variable(between.equation));
    EXPECT_EQ(between.equation.text, "x1");

    EXPECT_EQ(between.values, vector<float>({2.0f, 6.0f}));
    EXPECT_EQ(equal.values, vector<float>({5.0f}));
}


TEST(ConstraintCompilation, DiscreteConditionsKeepTheirOriginalExpression)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    ConstraintProbe problem(setup.network.get());

    problem.add_constraint("x1", Condition::Integer);
    problem.add_constraint("x1", Condition::AllowedSet, {9.0f, 1.0f, 5.0f, 5.0f});

    ASSERT_EQ(problem.constraints.size(), 2u);

    const ResponseOptimization::Constraint& whole = problem.constraints[0];
    const ResponseOptimization::Constraint& listed = problem.constraints[1];

    EXPECT_EQ(whole.equation.input_indices, vector<Index>({0}));
    EXPECT_EQ(listed.equation.input_indices, vector<Index>({0}));
    EXPECT_EQ(listed.values, vector<float>({1.0f, 5.0f, 9.0f})) << "allowed values are kept sorted and unique";

    VectorR point(2);
    point << 3.5f, 0.0f;

    EXPECT_FLOAT_EQ(whole.equation.evaluate(point, {}), point(0));
    EXPECT_FLOAT_EQ(listed.equation.evaluate(point, {}), point(0));
}


TEST(ConstraintCompilation, CardinalityBecomesOneRowHeldToTheUnitBand)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"});

    ConstraintProbe problem(setup.network.get());

    problem.add_constraint("x1; x2; x3", Condition::Cardinality, {1.0f});
    problem.add_constraint("x1; x2; x3", Condition::Cardinality, {3.0f});

    ASSERT_EQ(problem.constraints.size(), 1u) << "a budget of every counted variable restricts nothing";

    const ResponseOptimization::Constraint& budget = problem.constraints[0];

    EXPECT_EQ(budget.equation.input_indices, vector<Index>({0, 1, 2}));
    EXPECT_EQ(budget.equation.text, "x1; x2; x3");
    EXPECT_EQ(budget.equation.symmetric_order, 2);

    VectorR point(3);

    point << 4.0f, 0.0f, 0.0f;
    EXPECT_EQ(budget.equation.evaluate(point, {}), 0.0f);


    point << 4.0f, 3.0f, 0.0f;
    EXPECT_NEAR(budget.equation.evaluate(point, {}), 0.12/sqrt(3.0)/1e-3, 1e-2);
}


TEST(CardinalityRow, ValueMatchesSubsetEnumeration)
{
    mt19937 generator(7);

    constexpr float tolerance = 1e-3f;

    for (Index variables_number = 2; variables_number <= 8; variables_number++)
        for (Index order = 1; order <= variables_number; order++)
            for (int draw = 0; draw < 5; draw++)
            {
                const CardinalityCase c = draw_cardinality_case(generator, variables_number, 0.0);

                const double expected = rooted_mean_by_subsets(c.scaled(c.point), order)/double(tolerance);

                EXPECT_NEAR(c.row(order, tolerance).evaluate(c.point, {}), expected, 1e-5*max(1.0, expected))
                    << "n=" << variables_number << " order=" << order << " draw=" << draw;
            }
}


TEST(CardinalityRow, GradientMatchesCentralDifferences)
{
    mt19937 generator(11);

    constexpr float tolerance = 1e-3f;

    for (Index variables_number = 2; variables_number <= 8; variables_number++)
        for (Index order = 1; order <= variables_number; order++)
            for (int draw = 0; draw < 5; draw++)
            {
                const CardinalityCase c = draw_cardinality_case(generator, variables_number, 0.1);

                const CompiledExpression row = c.row(order, tolerance);

                const VectorR gradient = evaluate_input_gradient(row, c.point, {});

                for (size_t i = 0; i < c.columns.size(); i++)
                {
                    const Index column = c.columns[i];

                    const double step = 1e-6*double(c.spans[i]);

                    vector<double> plus = c.scaled(c.point);
                    vector<double> minus = plus;

                    plus[i] += step/double(c.spans[i]);
                    minus[i] -= step/double(c.spans[i]);

                    const double difference = (rooted_mean_by_subsets(plus, order)
                                             - rooted_mean_by_subsets(minus, order))/(2.0*step*double(tolerance));

                    EXPECT_NEAR(gradient(column), difference, 1e-4*max(1.0, abs(difference)))
                        << "n=" << variables_number << " order=" << order << " draw=" << draw << " column=" << column;
                }
            }
}


TEST(CardinalityRow, VanishesExactlyOnTheKSparsePoints)
{
    mt19937 generator(13);

    constexpr Index variables_number = 6;

    for (Index kept = 0; kept < variables_number; kept++)
        for (Index support = 0; support <= variables_number; support++)
            for (int draw = 0; draw < 10; draw++)
            {
                CardinalityCase c = draw_cardinality_case(generator, variables_number, 1e-2);

                vector<Index> order_of_columns = c.columns;

                ranges::shuffle(order_of_columns, generator);

                for (Index i = support; i < variables_number; i++)
                    c.point(order_of_columns[size_t(i)]) = 0.0f;

                const CompiledExpression row = c.row(kept + 1, 1e-3f);

                const float value = row.evaluate(c.point, {});

                if (support <= kept)
                {
                    EXPECT_EQ(value, 0.0f) << "k=" << kept << " support=" << support << " draw=" << draw;

                    const VectorR gradient = evaluate_input_gradient(row, c.point, {});

                    EXPECT_TRUE(gradient.allFinite());
                    EXPECT_EQ(gradient.cwiseAbs().maxCoeff(), 0.0f);
                }
                else
                    EXPECT_GT(value, 0.0f) << "k=" << kept << " support=" << support << " draw=" << draw;
            }
}


TEST(CardinalityRow, BuilderRefusesWhatItCannotWrite)
{
    const vector<Index> columns = {0, 1, 2};
    const vector<float> spans = {1.0f, 1.0f, 1.0f};

    EXPECT_THROW(compile_elementary_symmetric(columns, spans, 0, 1e-3f), runtime_error);
    EXPECT_THROW(compile_elementary_symmetric(columns, spans, 4, 1e-3f), runtime_error);
    EXPECT_THROW(compile_elementary_symmetric(columns, {1.0f, 1.0f}, 2, 1e-3f), runtime_error);
    EXPECT_THROW(compile_elementary_symmetric(columns, spans, 2, 0.0f), runtime_error);
    EXPECT_NO_THROW(compile_elementary_symmetric(columns, spans, 3, 1e-3f));
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


TEST(ExpressionHelpers, OutputGradientKeepsInputsIndependentAndClearsUnusedColumns)
{
    const auto inputs = make_named_columns({"x1", "x2"});
    const auto outputs = make_named_columns({"y1", "y2", "y3"});
    VectorR point(2); point << 3.0f, 4.0f;
    VectorR output(3); output << 5.0f, 6.0f, 7.0f;
    VectorR gradient = VectorR::Constant(1, 9.0f);

    const auto mixed = compile_expression("-x1 + 2*y1 - 3*y2", inputs, outputs);
    evaluate_output_gradient(mixed, point, output, gradient);

    ASSERT_EQ(gradient.size(), output.size());
    EXPECT_FLOAT_EQ(gradient(0), 2.0f);
    EXPECT_FLOAT_EQ(gradient(1), -3.0f);
    EXPECT_FLOAT_EQ(gradient(2), 0.0f);
    EXPECT_FLOAT_EQ(evaluate_input_gradient(mixed, point, output)(0), -1.0f);

    for (const auto& expression : {compile_expression("x1 + x2", inputs, outputs),
                                   compile_expression("x1*x2", inputs, outputs),
                                   compile_elementary_symmetric({0, 1}, {1.0f, 1.0f}, 2, 1e-3f)})
    {
        gradient.setConstant(9.0f);
        evaluate_output_gradient(expression, point, output, gradient);
        EXPECT_TRUE(gradient.isZero());
    }
}


TEST(ExpressionHelpers, MixedPartialsComposeWithAnOutputJacobian)
{
    const auto inputs = make_named_columns({"x1", "x2"});
    const auto outputs = make_named_columns({"y1", "y2"});
    const auto expression = compile_expression("x1*y1 + exp(y2) + y1^x2", inputs, outputs);
    VectorR point(2); point << 1.0f, 2.0f;
    VectorR output(2); output << 3.0f, -1.0f;
    VectorR output_gradient;
    evaluate_output_gradient(expression, point, output, output_gradient);
    const VectorR input_gradient = evaluate_input_gradient(expression, point, output);

    EXPECT_NEAR(input_gradient(0), 3.0f, 1e-5f);
    EXPECT_NEAR(input_gradient(1), 9.0f*log(3.0f), 1e-5f);
    EXPECT_NEAR(output_gradient(0), 7.0f, 1e-5f);
    EXPECT_NEAR(output_gradient(1), exp(-1.0f), 1e-5f);

    MatrixR output_jacobian(2, 2);
    output_jacobian << 2.0f, 1.0f, 1.0f, -1.0f;
    const VectorR total_gradient = input_gradient + output_jacobian.transpose()*output_gradient;
    const auto composed = [&](const VectorR& value)
    {
        VectorR prediction(2);
        prediction << value(0)*value(0) + value(1), value(0) - value(1);
        return expression.evaluate(value, prediction);
    };

    for (Index i = 0; i < point.size(); ++i)
    {
        VectorR plus = point, minus = point;
        plus(i) += 1e-3f;
        minus(i) -= 1e-3f;
        EXPECT_NEAR(total_gradient(i), (composed(plus) - composed(minus))/2e-3f, 3e-3f);
    }
}


TEST(ExpressionHelpers, OutputGradientMatchesNumericalPartialsForSupportedOperations)
{
    const auto inputs = make_named_columns({"x1", "x2"});
    const auto outputs = make_named_columns({"y1", "y2"});
    VectorR point(2); point << 0.3f, 0.7f;
    VectorR output(2); output << 1.2f, -0.4f;

    for (const string& text : {"-(x1-y1)/(x2+y2)", "y1^3 + 2^y2",
                               "sqrt(y1) + log(y1) + abs(y2)",
                               "sin(y1) + cos(y2) + tan(y2)",
                               "min(x1,y1) + max(y1,y2)", "min(y2,x1) + max(y2,x2)"})
    {
        SCOPED_TRACE(text);
        const auto expression = compile_expression(text, inputs, outputs);
        VectorR gradient;
        evaluate_output_gradient(expression, point, output, gradient);
        ASSERT_TRUE(gradient.allFinite());

        for (Index i = 0; i < output.size(); ++i)
        {
            VectorR plus = output, minus = output;
            plus(i) += 1e-3f;
            minus(i) -= 1e-3f;
            EXPECT_NEAR(gradient(i),
                        (expression.evaluate(point, plus) - expression.evaluate(point, minus))/2e-3f,
                        2e-3f);
        }
    }
}


TEST(ExpressionHelpers, OutputGradientSignalsKinksAndSingularitiesWithoutThrowing)
{
    const auto inputs = make_named_columns({"x1"});
    const auto outputs = make_named_columns({"y1"});
    const VectorR point = VectorR::Constant(1, 2.0f);
    const VectorR output = VectorR::Constant(1, 2.0f);

    for (const string& text : {"min(x1,y1)", "max(x1,y1)", "abs(y1-x1)", "sqrt(y1-x1)"})
    {
        SCOPED_TRACE(text);
        const auto expression = compile_expression(text, inputs, outputs);
        ASSERT_TRUE(isfinite(expression.evaluate(point, output)));
        VectorR gradient;
        EXPECT_NO_THROW(evaluate_output_gradient(expression, point, output, gradient));
        EXPECT_FALSE(gradient.allFinite());
    }
}
