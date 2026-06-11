//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   T E S T S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#include "../opennn/constraint_formulas.h"
#include "../opennn/neural_network.h"
#include "../opennn/response_optimization.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/standard_networks.h"
#include "../opennn/statistics.h"
#include "../opennn/unscaling_layer.h"
#include "../opennn/variable.h"

using namespace opennn;

namespace
{

vector<NamedColumn> make_named_columns(const vector<string>& names)
{
    vector<NamedColumn> out;
    out.reserve(names.size());
    for (Index i = 0; i < static_cast<Index>(names.size()); ++i)
        out.push_back({ names[i], i });
    return out;
}


float lookup_coeff(const vector<pair<Index, float>>& terms, Index column)
{
    for (const auto& [col, coeff] : terms)
        if (col == column) return coeff;
    return float(0);
}


struct MinimalApproximation
{
    unique_ptr<ApproximationNetwork> network;

    MinimalApproximation(const vector<string>& input_names,
                         const vector<string>& output_names,
                         float input_min = float(0),
                         float input_max = float(10),
                         float output_min = float(-1),
                         float output_max = float(1))
    {
        const Index n_inputs = static_cast<Index>(input_names.size());
        const Index n_outputs = static_cast<Index>(output_names.size());

        network = make_unique<ApproximationNetwork>(Shape{ n_inputs },
                                                    Shape{ 4 },
                                                    Shape{ n_outputs });

        vector<Variable> input_vars(n_inputs);
        for (Index i = 0; i < n_inputs; ++i)
        {
            input_vars[i].name = input_names[i];
            input_vars[i].set_role("Input");
            input_vars[i].type = VariableType::Numeric;
        }
        network->set_input_variables(input_vars);

        vector<Variable> output_vars(n_outputs);
        for (Index i = 0; i < n_outputs; ++i)
        {
            output_vars[i].name = output_names[i];
            output_vars[i].set_role("Target");
            output_vars[i].type = VariableType::Numeric;
        }
        network->set_output_variables(output_vars);

        vector<Descriptives> in_desc(n_inputs);
        for (Descriptives& d : in_desc)
        {
            d.minimum = input_min;
            d.maximum = input_max;
            d.mean = (input_min + input_max) * float(0.5);
            d.standard_deviation = (input_max - input_min) * float(0.25);
        }
        auto* scaling = static_cast<Scaling*>(network->get_first("Scaling"));
        scaling->set_descriptives(in_desc);

        vector<Descriptives> out_desc(n_outputs);
        for (Descriptives& d : out_desc)
        {
            d.minimum = output_min;
            d.maximum = output_max;
            d.mean = (output_min + output_max) * float(0.5);
            d.standard_deviation = (output_max - output_min) * float(0.25);
        }
        auto* unscaling = static_cast<Unscaling*>(network->get_first("Unscaling"));
        unscaling->set_descriptives(out_desc);
    }
};

} // namespace


// -----------------------------------------------------------------------------
// Formula parser: shape classification
// -----------------------------------------------------------------------------

TEST(FormulaExpression, LinearSumIsAffine)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const vector<NamedColumn> outputs = make_named_columns({});

    const CompiledFormula f = compile_formula("x1 + 2*x2 - 3", inputs, outputs);

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_EQ(f.scope, FormulaScope::InputsOnly);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(1), float(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), float(2), float(1e-6));
    EXPECT_NEAR(f.affine_constant, float(-3), float(1e-6));
}


TEST(FormulaExpression, UnaryNegationFlipsCoefficients)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("-x1 + x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(-1), float(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), float(1), float(1e-6));
}


TEST(FormulaExpression, ConstantScalingDistributesOverSum)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("3*(x1 + x2)", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(3), float(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), float(3), float(1e-6));
}


TEST(FormulaExpression, DivisionByConstantIsAffine)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });
    const CompiledFormula f = compile_formula("x1 / 4", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(0.25), float(1e-6));
}


TEST(FormulaExpression, ProductOfVariablesIsNonlinear)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("x1 * x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, DivisionByVariableIsNonlinear)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("x1 / x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, SqrtFunctionIsNonlinear)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });
    const CompiledFormula f = compile_formula("sqrt(x1) + 1", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, PowerWithNonUnitExponentIsNonlinear)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });
    const CompiledFormula f = compile_formula("x1 ^ 2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, ScopeInputsOnly)
{
    const CompiledFormula f = compile_formula("x1 + x2",
                                              make_named_columns({ "x1", "x2" }),
                                              make_named_columns({ "y1" }));

    EXPECT_EQ(f.scope, FormulaScope::InputsOnly);
    EXPECT_EQ(f.input_indices.size(), 2u);
    EXPECT_TRUE(f.output_indices.empty());
}


TEST(FormulaExpression, ScopeOutputsOnly)
{
    const CompiledFormula f = compile_formula("y1",
                                              make_named_columns({ "x1" }),
                                              make_named_columns({ "y1" }));

    EXPECT_EQ(f.scope, FormulaScope::OutputsOnly);
    EXPECT_TRUE(f.input_indices.empty());
    EXPECT_EQ(f.output_indices.size(), 1u);
}


TEST(FormulaExpression, ScopeMixed)
{
    const CompiledFormula f = compile_formula("x1 + y1",
                                              make_named_columns({ "x1" }),
                                              make_named_columns({ "y1" }));

    EXPECT_EQ(f.scope, FormulaScope::Mixed);
}


// -----------------------------------------------------------------------------
// Formula parser: evaluation
// -----------------------------------------------------------------------------

TEST(FormulaExpression, EvaluateAffineRespectsSignedCoefficients)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("-x1 + 2*x2 + 1", inputs, {});

    VectorR in(2); in << float(3), float(5);
    VectorR out(0);

    // Expected: -3 + 10 + 1 = 8
    EXPECT_NEAR(f.evaluate(in, out), float(8), float(1e-5));
}


TEST(FormulaExpression, EvaluateNonlinearExpression)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("sqrt(x1) + x2^2", inputs, {});

    VectorR in(2); in << float(9), float(3);
    VectorR out(0);

    // Expected: 3 + 9 = 12
    EXPECT_NEAR(f.evaluate(in, out), float(12), float(1e-5));
}


TEST(FormulaExpression, EvaluateUsesOutputsForMixedScope)
{
    const CompiledFormula f = compile_formula("x1 + 2*y1",
                                              make_named_columns({ "x1" }),
                                              make_named_columns({ "y1" }));

    VectorR in(1); in << float(1);
    VectorR out(1); out << float(4);

    EXPECT_NEAR(f.evaluate(in, out), float(9), float(1e-5));
}


TEST(FormulaExpression, ParenthesesOverridePrecedence)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });

    const CompiledFormula a = compile_formula("2 * x1 + x2", inputs, {});
    const CompiledFormula b = compile_formula("2 * (x1 + x2)", inputs, {});

    VectorR in(2); in << float(3), float(5);
    VectorR out(0);

    EXPECT_NEAR(a.evaluate(in, out), float(11), float(1e-5));
    EXPECT_NEAR(b.evaluate(in, out), float(16), float(1e-5));
}


TEST(FormulaExpression, MinMaxFunctions)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula fmin = compile_formula("min(x1, x2)", inputs, {});
    const CompiledFormula fmax = compile_formula("max(x1, x2)", inputs, {});

    VectorR in(2); in << float(2), float(7);
    VectorR out(0);

    EXPECT_NEAR(fmin.evaluate(in, out), float(2), float(1e-5));
    EXPECT_NEAR(fmax.evaluate(in, out), float(7), float(1e-5));
    EXPECT_EQ(fmin.shape, FormulaShape::Nonlinear);
}


// -----------------------------------------------------------------------------
// Formula parser: error handling
// -----------------------------------------------------------------------------

TEST(FormulaExpression, UnknownIdentifierThrows)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });

    EXPECT_THROW(compile_formula("x1 + z9", inputs, {}), runtime_error);
}


TEST(FormulaExpression, UnknownFunctionThrows)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });

    EXPECT_THROW(compile_formula("bogus(x1)", inputs, {}), runtime_error);
}


TEST(FormulaExpression, EmptyExpressionThrows)
{
    EXPECT_THROW(compile_formula("", {}, {}), runtime_error);
}


TEST(FormulaExpression, ExpressionWithoutVariablesThrows)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });
    EXPECT_THROW(compile_formula("1 + 2", inputs, {}), runtime_error);
}


TEST(FormulaExpression, WrongFunctionArityThrows)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });

    EXPECT_THROW(compile_formula("sqrt(x1, x2)", inputs, {}), runtime_error);
    EXPECT_THROW(compile_formula("min(x1)", inputs, {}), runtime_error);
}


TEST(FormulaExpression, MismatchedParenthesesThrow)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });

    EXPECT_THROW(compile_formula("(x1 + 1", inputs, {}), runtime_error);
}


// -----------------------------------------------------------------------------
// ResponseOptimization: formula constraint integration
// -----------------------------------------------------------------------------

TEST(ResponseOptimizationFormula, AffineInputConstraintFiltersResults)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // Require x1 + x2 <= 1.0 over a domain with each variable in [0, 10]
    opt.set_formula_constraint("x1 + x2",
                               ComparisonOperator::LessEqualTo,
                               float(0), float(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float sum = results(i, 0) + results(i, 1);
        EXPECT_LE(sum, float(1) + float(1e-3))
            << "row " << i << " x1=" << results(i,0) << " x2=" << results(i,1);
    }
}


TEST(ResponseOptimizationFormula, AffineEqualityLandsOnHyperplane)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // Force x1 + x2 = 5
    opt.set_formula_constraint("x1 + x2",
                               ComparisonOperator::EqualTo,
                               float(5));

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float sum = results(i, 0) + results(i, 1);
        EXPECT_NEAR(sum, float(5), float(5e-2))
            << "row " << i << " x1=" << results(i,0) << " x2=" << results(i,1);
    }
}


TEST(ResponseOptimizationFormula, NonlinearInputConstraintFilters)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(-5), float(5),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // Require x1^2 + x2^2 <= 4 (points inside circle of radius 2).
    opt.set_formula_constraint("x1^2 + x2^2",
                               ComparisonOperator::LessEqualTo,
                               float(0), float(4));

    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float r2 = results(i, 0) * results(i, 0) + results(i, 1) * results(i, 1);
        EXPECT_LE(r2, float(4) + float(1e-2));
    }
}


TEST(ResponseOptimizationFormula, CallbackConstraintFilters)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // Callback: |x1 - x2| <= 1
    opt.set_formula_constraint(
        [](const VectorR& in, const VectorR& /*out*/) { return abs(in(0) - in(1)); },
        ComparisonOperator::LessEqualTo,
        float(0), float(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_LE(abs(results(i, 0) - results(i, 1)), float(1) + float(1e-3));
}


TEST(ResponseOptimizationFormula, InfeasibleConstraintThrows)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(1),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // Unreachable: inputs live in [0,1] but we demand x1 + x2 ∈ [90, 100]
    opt.set_formula_constraint("x1 + x2",
                               ComparisonOperator::Between,
                               float(90), float(100));

    opt.set_iterations(2);
    opt.set_evaluations_number(200);
    opt.set_max_oversample_factor(2);

    EXPECT_THROW(opt.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationFormula, NoFormulaConstraintsPreservesBaseline)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    opt.set_iterations(2);
    opt.set_evaluations_number(300);

    const MatrixR results = opt.perform_response_optimization();
    ASSERT_GT(results.rows(), 0);

    // Each sampled input must live inside the declared domain.
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_GE(results(i, 0), float(0) - float(1e-3));
        EXPECT_LE(results(i, 0), float(10) + float(1e-3));
        EXPECT_GE(results(i, 1), float(0) - float(1e-3));
        EXPECT_LE(results(i, 1), float(10) + float(1e-3));
    }
}


TEST(ResponseOptimizationFormula, ConstraintAndObjectiveOnSameVariable)
{
    // "Allow both": x1 is simultaneously a Minimize objective and box-constrained
    // to [2, 4]. The objective is honored within the constrained sub-box, so every
    // sampled x1 stays in [2, 4] and the optimum drives x1 toward the lower bound.
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());

    opt.set_objective("x1", ResponseOptimization::Sense::Minimize);
    opt.set_constraint("x1", ComparisonOperator::Between, float(2), float(4));

    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.set_iterations(5);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_GE(results(i, 0), float(2) - float(1e-2));
        EXPECT_LE(results(i, 0), float(4) + float(1e-2));
    }

    EXPECT_LT(results(0, 0), float(3))
        << "best x1 = " << results(0, 0) << " (should approach the lower bound 2)";
}


TEST(ResponseOptimizationFormula, TimeRoleScaffoldIsAvailable)
{
    // The forecasting time-role API is scaffolded now; the runtime forecasting path
    // is left untouched, so this only exercises the set/clear surface and the enum.
    MinimalApproximation setup({ "x1", "x2" }, { "y" });

    ResponseOptimization opt(setup.network.get());

    opt.set_time_role("x2", ResponseOptimization::TimeType::PastBatch);
    opt.clear_constraints("x2");

    SUCCEED();
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
