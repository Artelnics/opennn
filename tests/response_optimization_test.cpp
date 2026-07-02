//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   T E S T S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#include "opennn/response_optimization.h"
#include "opennn/response_constraints.h"
#include "opennn/neural_network.h"
#include "opennn/random_utilities.h"
#include "opennn/scaling_layer.h"
#include "opennn/standard_networks.h"
#include "opennn/statistics.h"
#include "opennn/unscaling_layer.h"
#include "opennn/variable.h"

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


// -----------------------------------------------------------------------------
// Variable-wise block decomposition of the input-repair router
// -----------------------------------------------------------------------------

TEST(RepairBlockDecomposition, DisjointAffineBlockStaysExactBesideNonlinearBlock)
{
    // Two constraint subsystems with no shared variable: an affine EQUALITY on {x0, x1}
    // and a nonlinear disk on {x2, x3}. The router partitions them by connected component,
    // so the affine block goes to the exact single-affine projector (equality met tightly)
    // while only the nonlinear block goes to Gauss-Newton -- rather than the whole input
    // vector being dragged through GN because one constraint happens to be nonlinear.
    const vector<NamedColumn> inputs = make_named_columns({ "x0", "x1", "x2", "x3" });

    auto make_fc = [&](const string& expression, ComparisonOperator op, float low, float up)
    {
        MultivariateConstraint constraint;
        constraint.expression = expression;
        constraint.comparison_operator = op;
        constraint.low_bound = low;
        constraint.up_bound = up;
        constraint.compiled = compile_formula(expression, inputs, {});
        constraint.kind = classify(constraint);
        return constraint;
    };

    vector<MultivariateConstraint> constraints;
    constraints.push_back(make_fc("x0 + x1", ComparisonOperator::EqualTo, float(5), float(5)));
    constraints.push_back(make_fc("x2^2 + x3^2", ComparisonOperator::LessEqualTo, float(0), float(1)));

    ASSERT_EQ(constraints[0].kind, ConstraintKind::AffineInput);
    ASSERT_EQ(constraints[1].kind, ConstraintKind::NonlinearInput);

    const Index n = 4;
    const VectorR inferior = VectorR::Constant(n, float(-5));
    const VectorR superior = VectorR::Constant(n, float(5));

    const Index rows = 200;
    MatrixR points(rows, n);
    set_random_uniform(points, float(-5), float(5));

    repair_inputs(points, inferior, superior, constraints);

    for (Index r = 0; r < rows; ++r)
        EXPECT_NEAR(points(r, 0) + points(r, 1), float(5), float(5e-3))
            << "affine equality block not met exactly at row " << r
            << " (it must be projected, not dragged through Gauss-Newton)";
}


TEST(ResponseOptimizationFormula, DisjointAffineAndNonlinearBlocksCoexist)
{
    // End-to-end: x0 + x1 == 5 (affine block) AND x2^2 + x3^2 <= 1 (nonlinear block), no
    // shared variable. Every surviving result must satisfy both subsystems.
    MinimalApproximation setup({ "x0", "x1", "x2", "x3" }, { "y" },
                                float(-5), float(5),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("x0 + x1", ComparisonOperator::EqualTo, float(5));
    opt.set_formula_constraint("x2^2 + x3^2", ComparisonOperator::LessEqualTo, float(0), float(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(1500);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_NEAR(results(i, 0) + results(i, 1), float(5), float(5e-2))
            << "affine block violated at row " << i;
        EXPECT_LE(results(i, 2) * results(i, 2) + results(i, 3) * results(i, 3), float(1) + float(2e-2))
            << "nonlinear block violated at row " << i;
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


// -----------------------------------------------------------------------------
// Non-smooth (min/max/abs) constraint expansion
// -----------------------------------------------------------------------------

TEST(NonSmoothExpand, SmoothExpressionIsSingleBranch)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("x1 + x2", ComparisonOperator::LessEqualTo, float(0), float(3), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    ASSERT_EQ(branches[0].size(), size_t(1));
    EXPECT_EQ(branches[0][0].compiled.shape, FormulaShape::Affine);
}


TEST(NonSmoothExpand, MinGreaterEqualIsAndIntersection)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("min(x1, x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));       // AND -> no disjunction
    EXPECT_EQ(branches[0].size(), size_t(2));    // x1 >= 1 AND x2 >= 1
    for (const auto& c : branches[0])
        EXPECT_EQ(c.comparison_operator, ComparisonOperator::GreaterEqualTo);
}


TEST(NonSmoothExpand, MaxLessEqualIsAndIntersection)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("max(x1, x2)", ComparisonOperator::LessEqualTo, float(0), float(2), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    EXPECT_EQ(branches[0].size(), size_t(2));
}


TEST(NonSmoothExpand, AbsLessEqualIsInterval)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("abs(x1 - x2)", ComparisonOperator::LessEqualTo, float(0), float(1), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    ASSERT_EQ(branches[0].size(), size_t(1));
    EXPECT_EQ(branches[0][0].comparison_operator, ComparisonOperator::Between);
    EXPECT_NEAR(branches[0][0].low_bound, float(-1), float(1e-6));
    EXPECT_NEAR(branches[0][0].up_bound, float(1), float(1e-6));
}


TEST(NonSmoothExpand, OrCasesBranch)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });

    EXPECT_EQ(expand_constraint("min(x1, x2)", ComparisonOperator::LessEqualTo, float(0), float(1), inputs, {}).size(), size_t(2));
    EXPECT_EQ(expand_constraint("max(x1, x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0), inputs, {}).size(), size_t(2));
    EXPECT_EQ(expand_constraint("abs(x1 - x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0), inputs, {}).size(), size_t(2));
}


TEST(NonSmoothExpand, NestedYieldsRegionProduct)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    // two selectors (max + abs) -> 2^2 = 4 regions
    const auto branches = expand_constraint("max(x1, abs(x2))", ComparisonOperator::LessEqualTo, float(0), float(1), inputs, {});
    EXPECT_EQ(branches.size(), size_t(4));
}


TEST(ResponseOptimizationNonSmooth, MinGreaterEqualKeepsBothAboveBound)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(-5), float(5), float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("min(x1, x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0));
    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_GE(results(i, 0), float(1) - float(1e-2));
        EXPECT_GE(results(i, 1), float(1) - float(1e-2));
    }
}


TEST(ResponseOptimizationNonSmooth, MaxGreaterEqualBranchesIntoUnion)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(-5), float(5), float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("max(x1, x2)", ComparisonOperator::GreaterEqualTo, float(3), float(0));
    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_TRUE(results(i, 0) >= float(3) - float(1e-2) || results(i, 1) >= float(3) - float(1e-2));
}


TEST(ResponseOptimizationNonSmooth, AbsGreaterEqualBranchesBySign)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(-5), float(5), float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("abs(x1 - x2)", ComparisonOperator::GreaterEqualTo, float(2), float(0));
    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_GE(abs(results(i, 0) - results(i, 1)), float(2) - float(1e-2));
}


TEST(ResponseOptimizationNonSmooth, NestedMaxAbsStaysInsideBox)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(-5), float(5), float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    // max(x1, |x2|) <= 1  <=>  x1 <= 1 AND |x2| <= 1
    opt.set_formula_constraint("max(x1, abs(x2))", ComparisonOperator::LessEqualTo, float(0), float(1));
    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_LE(results(i, 0), float(1) + float(1e-2));
        EXPECT_LE(abs(results(i, 1)), float(1) + float(1e-2));
    }
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
// Mixed-integer repair: masked affine projector (step 1)
// -----------------------------------------------------------------------------

TEST(MixedIntegerProjector, FixedBinariesHoldWhileContinuousReproject)
{
    // Portfolio-style budget + buy-in over 4 weights w0..w3 and 4 indicators z0..z3.
    // The indicators are pinned to a cardinality-2 pattern {z0=z1=1, z2=z3=0} and held
    // fixed; the projector must land the continuous weights on the affine manifold
    //   sum w = 1,  w_i <= z_i,  w_i >= 0.01 z_i
    // without ever moving an indicator.
    const vector<NamedColumn> inputs =
        make_named_columns({ "w0", "w1", "w2", "w3", "z0", "z1", "z2", "z3" });
    const vector<NamedColumn> outputs;   // input-only system, no network

    auto make_fc = [&](const string& expression, ComparisonOperator op, float low, float up)
    {
        MultivariateConstraint constraint;
        constraint.expression = expression;
        constraint.comparison_operator = op;
        constraint.low_bound = low;
        constraint.up_bound = up;
        constraint.compiled = compile_formula(expression, inputs, outputs);
        constraint.kind = classify(constraint);
        return constraint;
    };

    vector<MultivariateConstraint> constraints;
    constraints.push_back(make_fc("w0 + w1 + w2 + w3", ComparisonOperator::EqualTo, float(1), float(1)));
    for (int i = 0; i < 4; ++i)
    {
        const string w = "w" + to_string(i);
        const string z = "z" + to_string(i);
        constraints.push_back(make_fc(w + " - " + z,        ComparisonOperator::LessEqualTo,    float(0), float(0)));
        constraints.push_back(make_fc(w + " - 0.01 * " + z, ComparisonOperator::GreaterEqualTo, float(0), float(0)));
    }

    const Index n = 8;
    const VectorR inferior = VectorR::Zero(n);
    const VectorR superior = VectorR::Ones(n);

    const Index rows = 200;
    MatrixR points(rows, n);
    set_random_uniform(points, float(0), float(1));   // weights start anywhere in [0,1]
    for (Index r = 0; r < rows; ++r)
    {
        points(r, 4) = float(1); points(r, 5) = float(1);   // z0 = z1 = 1
        points(r, 6) = float(0); points(r, 7) = float(0);   // z2 = z3 = 0
    }

    vector<char> fixed(n, 0);
    fixed[4] = fixed[5] = fixed[6] = fixed[7] = 1;          // hold all indicators fixed

    repair_affine_inputs_with_fixed(points, inferior, superior, constraints, fixed);

    for (Index r = 0; r < rows; ++r)
    {
        // Indicators are untouched (exactly, not just to tolerance).
        EXPECT_EQ(points(r, 4), float(1)); EXPECT_EQ(points(r, 5), float(1));
        EXPECT_EQ(points(r, 6), float(0)); EXPECT_EQ(points(r, 7), float(0));

        const float budget = points(r, 0) + points(r, 1) + points(r, 2) + points(r, 3);
        EXPECT_NEAR(budget, float(1), float(1e-3)) << "budget violated at row " << r;

        for (int i = 0; i < 4; ++i)
        {
            EXPECT_LE(points(r, i), points(r, 4 + i) + float(1e-3))
                << "buy-in upper violated at row " << r << " col " << i;
            EXPECT_GE(points(r, i), float(0.01) * points(r, 4 + i) - float(1e-3))
                << "buy-in lower violated at row " << r << " col " << i;
        }

        // Unselected weights must collapse to ~0.
        EXPECT_NEAR(points(r, 2), float(0), float(1e-3)) << "w2 not zeroed at row " << r;
        EXPECT_NEAR(points(r, 3), float(0), float(1e-3)) << "w3 not zeroed at row " << r;
    }
}


// -----------------------------------------------------------------------------
// Mixed-integer repair: lattice clamp-and-carry (step 2)
// -----------------------------------------------------------------------------

namespace
{
    MultivariateConstraint make_integer_constraint(const string& expression,
                                              const vector<NamedColumn>& inputs,
                                              ComparisonOperator op, float low, float up)
    {
        MultivariateConstraint constraint;
        constraint.expression = expression;
        constraint.comparison_operator = op;
        constraint.low_bound = low;
        constraint.up_bound = up;
        constraint.compiled = compile_formula(expression, inputs, /*outputs*/ {});
        constraint.kind = classify(constraint);
        return constraint;
    }
}

TEST(MixedIntegerCarry, SingleBudgetStaysOnLatticeAndFeasible)
{
    // 3 integer vars n0..n2 in [0,5]; knapsack budget n0 + n1 + n2 <= 4.
    const vector<NamedColumn> inputs = make_named_columns({ "n0", "n1", "n2" });
    const MultivariateConstraint budget =
        make_integer_constraint("n0 + n1 + n2", inputs, ComparisonOperator::LessEqualTo, float(0), float(4));

    const Index n = 3;
    const VectorR inferior = VectorR::Zero(n);
    const VectorR superior = VectorR::Constant(n, float(5));

    const Index rows = 300;
    MatrixR points(rows, n);
    set_random_uniform(points, float(0), float(5));   // continuous; the carry lattices it

    repair_single_affine_integer(points, inferior, superior, budget);

    for (Index r = 0; r < rows; ++r)
    {
        for (Index j = 0; j < n; ++j)
        {
            EXPECT_NEAR(points(r, j), round(points(r, j)), float(1e-4)) << "not integral at " << r << "," << j;
            EXPECT_GE(points(r, j), float(0));
            EXPECT_LE(points(r, j), float(5));
        }
        EXPECT_LE(points(r, 0) + points(r, 1) + points(r, 2), float(4) + float(1e-3))
            << "budget violated at row " << r;
    }
}

TEST(MixedIntegerCarry, SingleEqualityLandsExactlyOnLattice)
{
    // n0 + n1 + n2 == 3, each in [0,5]; unit coefficients => the carry must hit it exactly.
    const vector<NamedColumn> inputs = make_named_columns({ "n0", "n1", "n2" });
    const MultivariateConstraint exact =
        make_integer_constraint("n0 + n1 + n2", inputs, ComparisonOperator::EqualTo, float(3), float(3));

    const Index n = 3;
    const VectorR inferior = VectorR::Zero(n);
    const VectorR superior = VectorR::Constant(n, float(5));

    const Index rows = 300;
    MatrixR points(rows, n);
    set_random_uniform(points, float(0), float(5));

    repair_single_affine_integer(points, inferior, superior, exact);

    for (Index r = 0; r < rows; ++r)
    {
        for (Index j = 0; j < n; ++j)
            EXPECT_NEAR(points(r, j), round(points(r, j)), float(1e-4)) << "not integral at " << r << "," << j;
        EXPECT_NEAR(points(r, 0) + points(r, 1) + points(r, 2), float(3), float(1e-3))
            << "equality not met at row " << r;
    }
}

TEST(MixedIntegerCarry, PureIntegerKnapsackWiredIntoSolve)
{
    // End-to-end: a pure-integer knapsack n0+n1+n2 <= 4 over three integer variables routes
    // through the mixed-integer pump, where the lattice clamp-and-carry fast path solves it.
    MinimalApproximation setup({ "n0", "n1", "n2" }, { "y" }, float(0), float(5), float(-1), float(1));
    vector<Variable> input_variables = setup.network->get_input_variables();
    for (int i = 0; i < 3; ++i) input_variables[i].type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("n0 + n1 + n2", ComparisonOperator::LessEqualTo, float(0), float(4));

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index r = 0; r < results.rows(); ++r)
    {
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(results(r, j), round(results(r, j)), float(1e-3)) << "n" << j << " not integral at row " << r;
        EXPECT_LE(results(r, 0) + results(r, 1) + results(r, 2), float(4) + float(1e-2)) << "knapsack violated at row " << r;
    }
}


// -----------------------------------------------------------------------------
// Mixed-integer repair: box-aware K-hot draw (step 3, Hole A)
// -----------------------------------------------------------------------------

TEST(MixedIntegerKHot, DrawsExactlyKHonoringPins)
{
    const Index count = 8, k = 3;
    vector<char> force_on(count, 0), force_off(count, 0);
    force_on[1] = 1;                       // index 1 must be selected
    force_off[5] = 1; force_off[6] = 1;    // indices 5,6 excluded

    for (int trial = 0; trial < 200; ++trial)
    {
        vector<float> out;
        ASSERT_TRUE(draw_k_hot(count, k, force_on, force_off, out));
        ASSERT_EQ(static_cast<Index>(out.size()), count);

        float sum = 0;
        for (const float v : out) { EXPECT_TRUE(v == float(0) || v == float(1)); sum += v; }
        EXPECT_EQ(sum, float(k));
        EXPECT_EQ(out[1], float(1));        // forced on
        EXPECT_EQ(out[5], float(0));        // forced off
        EXPECT_EQ(out[6], float(0));
    }
}

TEST(MixedIntegerKHot, ReportsInfeasiblePins)
{
    const Index count = 4;
    vector<float> out;

    // Too many forced on (3 > k=2).
    {
        vector<char> force_on(count, 0), force_off(count, 0);
        force_on[0] = force_on[1] = force_on[2] = 1;
        EXPECT_FALSE(draw_k_hot(count, 2, force_on, force_off, out));
    }
    // Too few free to reach k (only index 3 free, need 2).
    {
        vector<char> force_on(count, 0), force_off(count, 0);
        force_off[0] = force_off[1] = force_off[2] = 1;
        EXPECT_FALSE(draw_k_hot(count, 2, force_on, force_off, out));
    }
    // Contradictory pin (forced on and off).
    {
        vector<char> force_on(count, 0), force_off(count, 0);
        force_on[0] = 1; force_off[0] = 1;
        EXPECT_FALSE(draw_k_hot(count, 1, force_on, force_off, out));
    }
}


// -----------------------------------------------------------------------------
// Mixed-integer end-to-end: portfolio buy-in + budget + cardinality (steps 3-4)
// -----------------------------------------------------------------------------

TEST(MixedIntegerPortfolio, BuyInBudgetCardinalityYieldsFeasiblePoints)
{
    // OR-Library port1 MIQP shape, scaled down: A assets, choose exactly K, weights on a
    // simplex with a buy-in floor coupling each continuous weight to its binary indicator.
    //   sum_i w_i = 1,  0.01 z_i <= w_i <= z_i,  sum_i z_i = K,  w_i in [0,1], z_i in {0,1}.
    const int A = 8;
    const int K = 3;

    vector<string> indicator_names;
    vector<string> input_names;
    for (int i = 0; i < A; ++i) input_names.push_back("w" + to_string(i));
    for (int i = 0; i < A; ++i) { indicator_names.push_back("z" + to_string(i)); input_names.push_back("z" + to_string(i)); }

    MinimalApproximation setup(input_names, { "y" }, float(0), float(1), float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    for (int i = 0; i < A; ++i) input_variables[A + i].type = VariableType::Binary;   // indicators
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    string budget;
    for (int i = 0; i < A; ++i) budget += (i ? " + " : "") + ("w" + to_string(i));
    opt.set_formula_constraint(budget, ComparisonOperator::EqualTo, float(1), float(1));

    for (int i = 0; i < A; ++i)
    {
        const string w = "w" + to_string(i);
        const string z = "z" + to_string(i);
        opt.set_formula_constraint(w + " - " + z,        ComparisonOperator::LessEqualTo,    float(0), float(0));
        opt.set_formula_constraint(w + " - 0.01 * " + z, ComparisonOperator::GreaterEqualTo, float(0), float(0));
    }

    opt.set_cardinality_constraint(indicator_names, K);

    opt.set_iterations(3);
    opt.set_evaluations_number(1500);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0) << "no feasible point for buy-in + budget + cardinality";

    for (Index r = 0; r < results.rows(); ++r)
    {
        float weight_sum = 0, indicator_sum = 0;
        for (int i = 0; i < A; ++i)
        {
            const float w = results(r, i);
            const float z = results(r, A + i);
            weight_sum += w;
            indicator_sum += z;

            EXPECT_NEAR(z, round(z), float(1e-3)) << "indicator z" << i << " not binary at row " << r;
            EXPECT_LE(w, z + float(1e-2))         << "buy-in upper w" << i << " <= z" << i << " at row " << r;
            EXPECT_GE(w, float(0.01) * z - float(1e-2)) << "buy-in lower at row " << r << " asset " << i;
        }
        EXPECT_NEAR(weight_sum, float(1), float(2e-2))    << "budget violated at row " << r;
        EXPECT_NEAR(indicator_sum, float(K), float(1e-2)) << "cardinality violated at row " << r;
    }
}


TEST(MixedIntegerPortfolio, ExploreExploitRatioPreservesFeasibility)
{
    // A non-default exploration_ratio exercises both the explore (free K-hot) and exploit
    // (incumbent-preferred K-hot) branches across iterations; feasibility + cardinality must
    // still hold regardless of the split.
    const int A = 6;
    const int K = 2;

    vector<string> indicator_names, input_names;
    for (int i = 0; i < A; ++i) input_names.push_back("w" + to_string(i));
    for (int i = 0; i < A; ++i) { indicator_names.push_back("z" + to_string(i)); input_names.push_back("z" + to_string(i)); }

    MinimalApproximation setup(input_names, { "y" }, float(0), float(1), float(-1), float(1));
    vector<Variable> input_variables = setup.network->get_input_variables();
    for (int i = 0; i < A; ++i) input_variables[A + i].type = VariableType::Binary;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

    string budget;
    for (int i = 0; i < A; ++i) budget += (i ? " + " : "") + ("w" + to_string(i));
    opt.set_formula_constraint(budget, ComparisonOperator::EqualTo, float(1), float(1));
    for (int i = 0; i < A; ++i)
    {
        const string w = "w" + to_string(i), z = "z" + to_string(i);
        opt.set_formula_constraint(w + " - " + z,        ComparisonOperator::LessEqualTo,    float(0), float(0));
        opt.set_formula_constraint(w + " - 0.01 * " + z, ComparisonOperator::GreaterEqualTo, float(0), float(0));
    }
    opt.set_cardinality_constraint(indicator_names, K);

    opt.set_exploration_ratio(float(0.5));   // half explore, half exploit
    opt.set_iterations(4);
    opt.set_evaluations_number(1200);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0) << "explore/exploit split lost feasibility";
    for (Index r = 0; r < results.rows(); ++r)
    {
        float weight_sum = 0, indicator_sum = 0;
        for (int i = 0; i < A; ++i)
        {
            const float w = results(r, i), z = results(r, A + i);
            weight_sum += w; indicator_sum += z;
            EXPECT_LE(w, z + float(1e-2)) << "buy-in upper at row " << r << " asset " << i;
            EXPECT_GE(w, float(0.01) * z - float(1e-2)) << "buy-in lower at row " << r << " asset " << i;
        }
        EXPECT_NEAR(weight_sum, float(1), float(2e-2))    << "budget at row " << r;
        EXPECT_NEAR(indicator_sum, float(K), float(1e-2)) << "cardinality at row " << r;
    }
}


// -----------------------------------------------------------------------------
// Generalized cardinality over non-binary variables: choose exactly K, rest = 0
// -----------------------------------------------------------------------------

TEST(ContinuousCardinality, SelectsExactlyKNonzeroRestZero)
{
    // Cardinality over CONTINUOUS variables x0..x5 in [0,10]: exactly K active, the
    // remaining A-K forced to exactly 0. No other constraints, so the k-hot mask in the
    // sampler is the sole mechanism (the repair path early-returns with no input constraints).
    const int A = 6;
    const int K = 2;

    vector<string> input_names;
    for (int i = 0; i < A; ++i) input_names.push_back("x" + to_string(i));

    MinimalApproximation setup(input_names, { "y" }, float(0), float(10), float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_cardinality_constraint(input_names, K);

    opt.set_iterations(2);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();
    ASSERT_GT(results.rows(), 0) << "no feasible point for continuous cardinality";

    for (Index r = 0; r < results.rows(); ++r)
    {
        int nonzero = 0, zero = 0;
        for (int i = 0; i < A; ++i)
        {
            const float x = results(r, i);
            EXPECT_GE(x, float(-1e-4))          << "x" << i << " below box at row " << r;
            EXPECT_LE(x, float(10) + float(1e-3)) << "x" << i << " above box at row " << r;
            if (abs(x) > float(1e-6)) ++nonzero; else ++zero;
        }
        // Off-columns are forced to exactly 0; an active column may coincide with ~0 (option a),
        // so the robust invariants are: at most K active and at least A-K exact zeros.
        EXPECT_LE(nonzero, K)     << "more than K active variables at row " << r;
        EXPECT_GE(zero, A - K)    << "fewer than A-K zeroed variables at row " << r;
    }
}


TEST(IntegerCardinality, ExactlyKActiveIntegersRestZero)
{
    // Cardinality over INTEGER variables n0..n4 in [0,5]: exactly K active, rest 0. Active
    // integers are guaranteed nonzero (nudged off 0) and stay on the integer grid, and may
    // range over [1,5] rather than being pinned to 1.
    const int A = 5;
    const int K = 3;

    vector<string> input_names;
    for (int i = 0; i < A; ++i) input_names.push_back("n" + to_string(i));

    MinimalApproximation setup(input_names, { "y" }, float(0), float(5), float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    for (Variable& variable : input_variables) variable.type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_cardinality_constraint(input_names, K);

    opt.set_iterations(2);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();
    ASSERT_GT(results.rows(), 0) << "no feasible point for integer cardinality";

    for (Index r = 0; r < results.rows(); ++r)
    {
        int nonzero = 0;
        for (int i = 0; i < A; ++i)
        {
            const float n = results(r, i);
            EXPECT_NEAR(n, round(n), float(1e-3)) << "n" << i << " not integer at row " << r;
            EXPECT_GE(n, float(-1e-3))            << "n" << i << " below box at row " << r;
            EXPECT_LE(n, float(5) + float(1e-3))  << "n" << i << " above box at row " << r;
            if (abs(n) > float(0.5)) ++nonzero;
        }
        // Integers on -> guaranteed nonzero, so the active count is exactly K.
        EXPECT_EQ(nonzero, K) << "active integer count != K at row " << r;
    }
}


TEST(BinaryCardinality, FreeModeIsAtMostK)
{
    // force_nonzero = false: choose up to K binary slots that MAY be 1, the rest pinned to 0.
    // The result is a sparsity budget -> at most K ones (possibly fewer), never more.
    const int A = 6;
    const int K = 3;

    vector<string> input_names;
    for (int i = 0; i < A; ++i) input_names.push_back("b" + to_string(i));

    MinimalApproximation setup(input_names, { "y" }, float(0), float(1), float(-1), float(1));

    vector<Variable> input_variables = setup.network->get_input_variables();
    for (Variable& variable : input_variables) variable.type = VariableType::Binary;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_cardinality_constraint(input_names, K, /*force_nonzero=*/false);

    opt.set_iterations(2);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();
    ASSERT_GT(results.rows(), 0) << "no feasible point for free-mode binary cardinality";

    for (Index r = 0; r < results.rows(); ++r)
    {
        int ones = 0;
        for (int i = 0; i < A; ++i)
        {
            const float b = results(r, i);
            EXPECT_NEAR(b, round(b), float(1e-3)) << "b" << i << " not binary at row " << r;
            if (b > float(0.5)) ++ones;
        }
        EXPECT_LE(ones, K) << "more than K ones under free/at-most-K mode at row " << r;
    }
}


// -----------------------------------------------------------------------------
// Single-variable affine constraint -> domain box promotion (B)
// -----------------------------------------------------------------------------

TEST(SingleVariablePromotion, AffineConstraintBecomesBox)
{
    // "2*x1 - 6 <= 0"  ->  x1 <= 3, folded into the box; every result must respect it.
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(0), float(10), float(-1), float(1));
    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("2 * x1 - 6", ComparisonOperator::LessEqualTo, float(0), float(0));

    opt.set_iterations(3);
    opt.set_evaluations_number(500);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index r = 0; r < results.rows(); ++r)
        EXPECT_LE(results(r, 0), float(3) + float(1e-3)) << "x1 not boxed to <= 3 at row " << r;
}

TEST(SingleVariablePromotion, EmptyIntersectionThrows)
{
    // Existing box [5,10] intersected with the implied x1 <= 3 is empty -> must throw.
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(0), float(10), float(-1), float(1));
    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_constraint("x1", ComparisonOperator::Between, float(5), float(10));
    opt.set_formula_constraint("x1", ComparisonOperator::LessEqualTo, float(0), float(3));

    opt.set_iterations(2);
    opt.set_evaluations_number(200);

    EXPECT_ANY_THROW(opt.perform_response_optimization());
}

TEST(SingleVariablePromotion, IntegerPromotionRespectsLattice)
{
    // A promoted box on an integer variable must still yield integral, in-box results.
    MinimalApproximation setup({ "x1", "x2" }, { "y" }, float(0), float(10), float(-1), float(1));
    vector<Variable> input_variables = setup.network->get_input_variables();
    input_variables[0].type = VariableType::Integer;
    setup.network->set_input_variables(input_variables);

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_formula_constraint("x1", ComparisonOperator::LessEqualTo, float(0), float(5));

    opt.set_iterations(3);
    opt.set_evaluations_number(500);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index r = 0; r < results.rows(); ++r)
    {
        EXPECT_NEAR(results(r, 0), round(results(r, 0)), float(1e-3)) << "x1 not integral at row " << r;
        EXPECT_LE(results(r, 0), float(5) + float(1e-3)) << "x1 not boxed at row " << r;
    }
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
    opt.set_branch_mode(ResponseOptimization::BranchMode::Exhaustive);

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
