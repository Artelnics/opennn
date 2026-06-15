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
    opt.clear_time_roles("x2");
    opt.clear_time_roles();

    SUCCEED();
}


TEST(ResponseOptimizationClear, GranularClearsResetOnlyTheirOwnState)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" });

    ResponseOptimization opt(setup.network.get());

    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_objective("x1", ResponseOptimization::Sense::Maximize);
    opt.set_constraint("x2", ComparisonOperator::Between, float(2), float(4));
    opt.set_time_role("x2", ResponseOptimization::TimeType::PastBatch);

    EXPECT_EQ(opt.get_objectives_number(), 2);

    // Clearing one collection (by name or wholesale) must leave the others intact.
    opt.clear_objectives("x1");
    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.clear_objectives();
    EXPECT_EQ(opt.get_objectives_number(), 0);

    opt.clear_constraints();
    opt.clear_time_roles();
    SUCCEED();
}


// -----------------------------------------------------------------------------
// ResponseOptimization: integer decision variables
// -----------------------------------------------------------------------------

TEST(ResponseOptimizationInteger, IntegerVariableYieldsIntegralResults)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    input_variables[0].type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_NEAR(results(i, 0), round(results(i, 0)), float(1e-3))
            << "integer x1 must be integral, got " << results(i, 0);
        EXPECT_GE(results(i, 0), float(0) - float(1e-3));
        EXPECT_LE(results(i, 0), float(10) + float(1e-3));
    }
}


TEST(ResponseOptimizationInteger, IntegerBoxConstraintStaysIntegral)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    input_variables[0].type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_constraint("x1", ComparisonOperator::Between, float(2), float(8));

    opt.set_iterations(3);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_NEAR(results(i, 0), round(results(i, 0)), float(1e-3));
        EXPECT_GE(results(i, 0), float(2) - float(1e-3));
        EXPECT_LE(results(i, 0), float(8) + float(1e-3));
    }
}


TEST(ResponseOptimizationInteger, IntegerStaysIntegralAfterAffineRepair)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    input_variables[0].type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // The affine repair projects points onto x1 + x2 <= 5; the post-repair
    // re-round must keep the integer variable x1 on the lattice.
    opt.set_formula_constraint("x1 + x2",
                               ComparisonOperator::LessEqualTo,
                               float(0), float(5));

    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_NEAR(results(i, 0), round(results(i, 0)), float(1e-3))
            << "integer x1 not integral after repair at row " << i;
        EXPECT_LE(results(i, 0) + results(i, 1), float(5) + float(1e-2));
    }
}


TEST(ResponseOptimizationInteger, IntegerStaysIntegralAfterOutputRepair)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    input_variables[0].type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    // The mixed constraint routes through repair_output_constraints, which moves
    // points continuously; the post-repair re-snap must keep x1 on the lattice.
    opt.set_formula_constraint("y + 0.1*x2",
                               ComparisonOperator::LessEqualTo,
                               float(0), float(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_NEAR(results(i, 0), round(results(i, 0)), float(1e-3))
            << "integer x1 not integral after output repair at row " << i;
}


// -----------------------------------------------------------------------------
// AllowedSet: membership constraints (x in {values})
// -----------------------------------------------------------------------------

TEST(ResponseOptimizationAllowedSet, FreeInputIsDrawnFromTheSet)
{
    // x1 may only take {1, 5, 9}; it is referenced by no formula, so it is sampled
    // directly from the set within a single solve (no branching). Every returned x1
    // must be one of the allowed values.
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_constraint("x1", vector<float>{ float(1), float(5), float(9) });

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float x1 = results(i, 0);
        const float distance = min(min(abs(x1 - float(1)), abs(x1 - float(5))), abs(x1 - float(9)));
        EXPECT_LT(distance, float(1e-2)) << "x1 = " << x1 << " is not in {1,5,9}";
    }
}


TEST(ResponseOptimizationAllowedSet, FormulaMembershipBranchesToEachValue)
{
    // x1 + x2 must equal one of {3, 7}. This expression membership branches into two
    // EqualTo equality subproblems; the aggregated result must satisfy one of them.
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("x1 + x2", vector<float>{ float(3), float(7) });

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float sum = results(i, 0) + results(i, 1);
        const float distance = min(abs(sum - float(3)), abs(sum - float(7)));
        EXPECT_LT(distance, float(5e-2)) << "x1+x2 = " << sum << " is not in {3,7}";
    }
}


TEST(ResponseOptimizationAllowedSet, EntangledInputBranchesAndSkipsInfeasibleValue)
{
    // x1 in {2, 8} AND x1 + x2 <= 5, with x2 in [0,10]. x1 is referenced by the
    // formula, so it branches: the x1=2 branch is feasible (x2 <= 3) while the x1=8
    // branch is infeasible (needs x2 <= -3) and is skipped. Every result must come
    // from the surviving branch.
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_constraint("x1", vector<float>{ float(2), float(8) });
    opt.set_formula_constraint("x1 + x2",
                               ComparisonOperator::LessEqualTo,
                               float(0), float(5));

    opt.set_iterations(3);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_NEAR(results(i, 0), float(2), float(1e-2)) << "row " << i << " x1 not on the feasible branch";
        EXPECT_LE(results(i, 0) + results(i, 1), float(5) + float(1e-2));
    }
}


TEST(ResponseOptimizationAllowedSet, BudgetedPruningPreservesMembershipWithinBudget)
{
    // Four-way membership x1+x2 in {2,4,6,8} under a capped budget with pruning on
    // (default): successive halving probes all four cheaply, races them down, and
    // finishes the winner. The result must still satisfy membership, and the total
    // evaluations must stay near the cap (a per-branch sampling overshoot aside).
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("x1 + x2", vector<float>{ float(2), float(4), float(6), float(8) });

    opt.set_iterations(3);
    opt.set_evaluations_number(500);
    const Index budget = 8000;
    opt.set_max_total_evaluations(budget);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float sum = results(i, 0) + results(i, 1);
        const float distance = min(min(abs(sum - float(2)), abs(sum - float(4))),
                                   min(abs(sum - float(6)), abs(sum - float(8))));
        EXPECT_LT(distance, float(5e-2)) << "x1+x2 = " << sum << " is not in {2,4,6,8}";
    }

    EXPECT_GT(opt.get_evaluations_used(), 0);
    EXPECT_LE(opt.get_evaluations_used(), budget + 4 * 500)
        << "successive-halving overspent its budget";
}


TEST(ResponseOptimizationAllowedSet, ExhaustiveSwitchPreservesMembership)
{
    // The same four-way membership with pruning switched OFF: every branch is solved to
    // completion and the global front aggregated. Membership must still hold.
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("x1 + x2", vector<float>{ float(2), float(4), float(6), float(8) });
    opt.set_branch_pruning(false);

    opt.set_iterations(3);
    opt.set_evaluations_number(400);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float sum = results(i, 0) + results(i, 1);
        const float distance = min(min(abs(sum - float(2)), abs(sum - float(4))),
                                   min(abs(sum - float(6)), abs(sum - float(8))));
        EXPECT_LT(distance, float(5e-2)) << "x1+x2 = " << sum << " is not in {2,4,6,8}";
    }
}


// -----------------------------------------------------------------------------
// ResponseOptimization: categorical sample-frequency tracking + under-used exploration
// -----------------------------------------------------------------------------

TEST(ResponseOptimizationCategory, ExplorationSamplesEveryCategoryAndTracksFrequency)
{
    // One categorical input "color" with 4 categories. calculate_random_inputs runs
    // no network forward pass, so the input shape is set to the variable count (1) to
    // satisfy get_variables_and_descriptives; the categorical still expands to 4
    // one-hot sampling columns internally.
    auto network = make_unique<ApproximationNetwork>(Shape{ 1 }, Shape{ 3 }, Shape{ 1 });

    Variable color;
    color.name = "color";
    color.set_role("Input");
    color.type = VariableType::Categorical;
    color.set_categories({ "red", "green", "blue", "yellow" });
    network->set_input_variables({ color });

    Variable target;
    target.name = "y";
    target.set_role("Target");
    target.type = VariableType::Numeric;
    network->set_output_variables({ target });

    ResponseOptimization opt(network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    const ResponseOptimization::Domain input_domain = opt.get_original_domain("Input");
    opt.calculate_random_inputs(input_domain, 1000);

    const map<string, vector<Index>>& frequencies = opt.get_category_frequencies();
    ASSERT_EQ(frequencies.count("color"), 1u);

    const vector<Index>& color_frequencies = frequencies.at("color");
    ASSERT_EQ(Index(color_frequencies.size()), 4);

    Index total = 0;
    for (Index i = 0; i < 4; ++i)
    {
        EXPECT_GT(color_frequencies[i], 0) << "category " << i << " was never sampled";
        total += color_frequencies[i];
    }
    EXPECT_EQ(total, 1000);
}


TEST(ResponseOptimizationCategory, OneHotForwardRespectsAllowedSet)
{
    // 1 numeric input "x" + 1 categorical "cat" (3 levels) => 4 one-hot features fed to a
    // real network forward pass. The Scaling layer holds one descriptive PER FEATURE (4)
    // while the optimizer works per logical variable (2); this is the production layout
    // that get_variables_and_descriptives collapses (before that, a 4-vs-2 size mismatch
    // threw). An AllowedSet over category indices excludes the middle level "B".
    ApproximationNetwork network(Shape{ 4 }, Shape{ 4 }, Shape{ 1 });

    Variable x; x.name = "x"; x.set_role("Input"); x.type = VariableType::Numeric;
    Variable cat; cat.name = "cat"; cat.set_role("Input"); cat.type = VariableType::Categorical;
    cat.set_categories({ "A", "B", "C" });
    network.set_input_variables({ x, cat });

    Variable y; y.name = "y"; y.set_role("Target"); y.type = VariableType::Numeric;
    network.set_output_variables({ y });

    vector<Descriptives> in_desc(4);
    in_desc[0] = Descriptives(float(0), float(10), float(5), float(2.5));        // x
    for (Index j = 1; j < 4; ++j)
        in_desc[j] = Descriptives(float(0), float(1), float(0.5), float(0.5));   // one-hot column
    static_cast<Scaling*>(network.get_first("Scaling"))->set_descriptives(in_desc);

    static_cast<Unscaling*>(network.get_first("Unscaling"))
        ->set_descriptives({ Descriptives(float(-1), float(1), float(0), float(0.5)) });

    ResponseOptimization opt(&network);
    opt.set_objective("y", ResponseOptimization::Sense::Maximize);
    opt.set_constraint("cat", vector<float>{ float(0), float(2) });   // allow A and C, exclude B

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    ASSERT_EQ(results.cols(), 5);                                     // x, A, B, C, y
    for (Index i = 0; i < results.rows(); ++i)
    {
        const float a = results(i, 1), b = results(i, 2), c = results(i, 3);
        EXPECT_NEAR(a + b + c, float(1), float(1e-3)) << "row " << i << " not one-hot";
        EXPECT_LT(b, float(1e-3)) << "row " << i << " category B should be excluded";
        const bool one_hot = (a > float(0.99) && c < float(0.01))
                          || (c > float(0.99) && a < float(0.01));
        EXPECT_TRUE(one_hot) << "row " << i << " a=" << a << " c=" << c;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
