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

vector<pair<string, Index>> make_named_columns(const vector<string>& names)
{
    vector<pair<string, Index>> out;
    out.reserve(names.size());
    for (Index i = 0; i < static_cast<Index>(names.size()); ++i)
        out.emplace_back(names[i], i);
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

}



TEST(FormulaExpression, LinearSumIsAffine)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const vector<pair<string, Index>> outputs = make_named_columns({});

    const CompiledFormula f = compile_formula("x1 + 2*x2 - 3", inputs, outputs);

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_EQ(f.scope, FormulaScope::InputsOnly);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(1), float(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), float(2), float(1e-6));
    EXPECT_NEAR(f.affine_constant, float(-3), float(1e-6));
}


TEST(FormulaExpression, UnaryNegationFlipsCoefficients)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("-x1 + x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(-1), float(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), float(1), float(1e-6));
}


TEST(FormulaExpression, ConstantScalingDistributesOverSum)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("3*(x1 + x2)", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(3), float(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), float(3), float(1e-6));
}


TEST(FormulaExpression, DivisionByConstantIsAffine)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });
    const CompiledFormula f = compile_formula("x1 / 4", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), float(0.25), float(1e-6));
}


TEST(FormulaExpression, ProductOfVariablesIsNonlinear)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("x1 * x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, DivisionByVariableIsNonlinear)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("x1 / x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, SqrtFunctionIsNonlinear)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });
    const CompiledFormula f = compile_formula("sqrt(x1) + 1", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Nonlinear);
}


TEST(FormulaExpression, PowerWithNonUnitExponentIsNonlinear)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });
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



TEST(FormulaExpression, EvaluateAffineRespectsSignedCoefficients)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("-x1 + 2*x2 + 1", inputs, {});

    VectorR in(2); in << float(3), float(5);
    VectorR out(0);

    EXPECT_NEAR(f.evaluate(in, out), float(8), float(1e-5));
}


TEST(FormulaExpression, EvaluateNonlinearExpression)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("sqrt(x1) + x2^2", inputs, {});

    VectorR in(2); in << float(9), float(3);
    VectorR out(0);

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
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });

    const CompiledFormula a = compile_formula("2 * x1 + x2", inputs, {});
    const CompiledFormula b = compile_formula("2 * (x1 + x2)", inputs, {});

    VectorR in(2); in << float(3), float(5);
    VectorR out(0);

    EXPECT_NEAR(a.evaluate(in, out), float(11), float(1e-5));
    EXPECT_NEAR(b.evaluate(in, out), float(16), float(1e-5));
}


TEST(FormulaExpression, MinMaxFunctions)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula fmin = compile_formula("min(x1, x2)", inputs, {});
    const CompiledFormula fmax = compile_formula("max(x1, x2)", inputs, {});

    VectorR in(2); in << float(2), float(7);
    VectorR out(0);

    EXPECT_NEAR(fmin.evaluate(in, out), float(2), float(1e-5));
    EXPECT_NEAR(fmax.evaluate(in, out), float(7), float(1e-5));
    EXPECT_EQ(fmin.shape, FormulaShape::Nonlinear);
}



TEST(FormulaExpression, UnknownIdentifierThrows)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });

    EXPECT_THROW(compile_formula("x1 + z9", inputs, {}), runtime_error);
}


TEST(FormulaExpression, UnknownFunctionThrows)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });

    EXPECT_THROW(compile_formula("bogus(x1)", inputs, {}), runtime_error);
}


TEST(FormulaExpression, EmptyExpressionThrows)
{
    EXPECT_THROW(compile_formula("", {}, {}), runtime_error);
}


TEST(FormulaExpression, ExpressionWithoutVariablesThrows)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });
    EXPECT_THROW(compile_formula("1 + 2", inputs, {}), runtime_error);
}


TEST(FormulaExpression, WrongFunctionArityThrows)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });

    EXPECT_THROW(compile_formula("sqrt(x1, x2)", inputs, {}), runtime_error);
    EXPECT_THROW(compile_formula("min(x1)", inputs, {}), runtime_error);
}


TEST(FormulaExpression, MismatchedParenthesesThrow)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1" });

    EXPECT_THROW(compile_formula("(x1 + 1", inputs, {}), runtime_error);
}



TEST(ResponseOptimizationFormula, AffineInputConstraintFiltersResults)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);

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



TEST(RepairBlockDecomposition, DisjointAffineBlockStaysExactBesideNonlinearBlock)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x0", "x1", "x2", "x3" });

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

    opt.set_formula_constraint(
        [](const VectorR& in, const VectorR&) { return abs(in(0) - in(1)); },
        ComparisonOperator::LessEqualTo,
        float(0), float(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_LE(abs(results(i, 0) - results(i, 1)), float(1) + float(1e-3));
}



TEST(NonSmoothExpand, SmoothExpressionIsSingleBranch)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("x1 + x2", ComparisonOperator::LessEqualTo, float(0), float(3), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    ASSERT_EQ(branches[0].size(), size_t(1));
    EXPECT_EQ(branches[0][0].compiled.shape, FormulaShape::Affine);
}


TEST(NonSmoothExpand, MinGreaterEqualIsAndIntersection)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("min(x1, x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    EXPECT_EQ(branches[0].size(), size_t(2));
    for (const auto& c : branches[0])
        EXPECT_EQ(c.comparison_operator, ComparisonOperator::GreaterEqualTo);
}


TEST(NonSmoothExpand, MaxLessEqualIsAndIntersection)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("max(x1, x2)", ComparisonOperator::LessEqualTo, float(0), float(2), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    EXPECT_EQ(branches[0].size(), size_t(2));
}


TEST(NonSmoothExpand, AbsLessEqualIsInterval)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
    const auto branches = expand_constraint("abs(x1 - x2)", ComparisonOperator::LessEqualTo, float(0), float(1), inputs, {});

    ASSERT_EQ(branches.size(), size_t(1));
    ASSERT_EQ(branches[0].size(), size_t(1));
    EXPECT_EQ(branches[0][0].comparison_operator, ComparisonOperator::Between);
    EXPECT_NEAR(branches[0][0].low_bound, float(-1), float(1e-6));
    EXPECT_NEAR(branches[0][0].up_bound, float(1), float(1e-6));
}


TEST(NonSmoothExpand, OrCasesBranch)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });

    EXPECT_EQ(expand_constraint("min(x1, x2)", ComparisonOperator::LessEqualTo, float(0), float(1), inputs, {}).size(), size_t(2));
    EXPECT_EQ(expand_constraint("max(x1, x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0), inputs, {}).size(), size_t(2));
    EXPECT_EQ(expand_constraint("abs(x1 - x2)", ComparisonOperator::GreaterEqualTo, float(1), float(0), inputs, {}).size(), size_t(2));
}


TEST(NonSmoothExpand, NestedYieldsRegionProduct)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "x1", "x2" });
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

    opt.clear_objectives("x1");
    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.clear_objectives();
    EXPECT_EQ(opt.get_objectives_number(), 0);

    opt.clear_constraints();
    opt.clear_time_roles();
    SUCCEED();
}



namespace
{

float reachable_output_median(ResponseOptimization& opt, Index output_column, Index samples = 512)
{
    const MatrixR inputs = opt.calculate_random_inputs(opt.get_original_domain("Input"), samples);

    const MatrixR outputs = opt.calculate_outputs(inputs);

    vector<float> values(outputs.rows());
    for (Index i = 0; i < outputs.rows(); ++i)
        values[static_cast<size_t>(i)] = outputs(i, output_column);

    ranges::sort(values);
    return values[values.size() / 2];
}

}


TEST(ResponseOptimizationFixed, FixedInputIsConvertedToBox)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    opt.set_objective("x1", ResponseOptimization::Sense::Fixed, float(3));

    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.set_iterations(4);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_NEAR(results(i, 0), float(3), float(1e-2))
            << "row " << i << " x1=" << results(i, 0) << " (should be pinned to 3)";
}


TEST(ResponseOptimizationFixed, FixedOutputPureInverseSolve)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());

    const float target = reachable_output_median(opt, 0);

    opt.set_objective("y", ResponseOptimization::Sense::Fixed, target);

    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.set_relative_tolerance(float(1e-2));
    opt.set_iterations(6);
    opt.set_evaluations_number(1200);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_NEAR(results(i, 2), target, float(5e-2))
            << "row " << i << " y=" << results(i, 2) << " target=" << target;
}


TEST(ResponseOptimizationFixed, FixedMixedWithOptimizingStaysSingleObjective)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y1", "y2" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());

    const float target = reachable_output_median(opt, 1);

    opt.set_objective("y1", ResponseOptimization::Sense::Minimize);
    opt.set_objective("y2", ResponseOptimization::Sense::Fixed, target);

    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.set_relative_tolerance(float(1e-2));
    opt.set_iterations(6);
    opt.set_evaluations_number(1200);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_NEAR(results(i, 3), target, float(5e-2))
            << "row " << i << " y2=" << results(i, 3) << " target=" << target;
}


TEST(ResponseOptimizationFixed, MultipleFixedOutputsRemainSingleObjective)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y1", "y2" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());

    const float target1 = reachable_output_median(opt, 0);
    const float target2 = reachable_output_median(opt, 1);

    opt.set_objective("y1", ResponseOptimization::Sense::Fixed, target1);
    opt.set_objective("y2", ResponseOptimization::Sense::Fixed, target2);

    EXPECT_EQ(opt.get_objectives_number(), 2);

    opt.set_relative_tolerance(float(2e-2));
    opt.set_iterations(6);
    opt.set_evaluations_number(1500);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_NEAR(results(i, 2), target1, float(8e-2)) << "row " << i << " y1";
        EXPECT_NEAR(results(i, 3), target2, float(8e-2)) << "row " << i << " y2";
    }
}


TEST(ResponseOptimizationFixed, ClearObjectivesResetsFixedValues)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" });

    ResponseOptimization opt(setup.network.get());

    opt.set_objective("y", ResponseOptimization::Sense::Fixed, float(0.25));
    EXPECT_EQ(opt.get_objectives_number(), 1);

    opt.clear_objectives("y");
    EXPECT_EQ(opt.get_objectives_number(), 0);

    opt.set_objective("y", ResponseOptimization::Sense::Minimize);
    EXPECT_EQ(opt.get_objectives_number(), 1);
    SUCCEED();
}



TEST(ResponseOptimizationRobust, EmptyFrontReturnsMinusOne)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" });
    ResponseOptimization opt(setup.network.get());

    const auto [index, row] = opt.get_robust_point(MatrixR(), 0.5f);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(row.size(), 0);
}


TEST(ResponseOptimizationRobust, BalanceSelectsCentralOrRobustEndpoints)
{
    MinimalApproximation setup({ "x1", "x2", "x3" }, { "y" },
                                float(0), float(10),
                                float(-1), float(1));

    ResponseOptimization opt(setup.network.get());

    const float target = reachable_output_median(opt, 0);
    opt.set_objective("y", ResponseOptimization::Sense::Fixed, target);
    opt.set_relative_tolerance(float(2e-2));
    opt.set_iterations(6);
    opt.set_evaluations_number(1500);

    const MatrixR front = opt.perform_response_optimization();
    ASSERT_GT(front.rows(), 2);

    const Index inputs_number = setup.network->get_inputs_number();

    const auto [i_central, row_central] = opt.get_robust_point(front, float(0));
    const auto [i_mid,     row_mid]     = opt.get_robust_point(front, float(0.5));
    const auto [i_robust,  row_robust]  = opt.get_robust_point(front, float(1));

    for (const Index idx : { i_central, i_mid, i_robust })
    {
        ASSERT_GE(idx, Index(0));
        ASSERT_LT(idx, front.rows());
    }
    EXPECT_TRUE(row_robust.isApprox(front.row(i_robust).transpose()));

    const auto domain = opt.get_original_domain("Input");
    VectorR span(inputs_number);
    for (Index c = 0; c < inputs_number; ++c)
        span(c) = domain.superior_frontier(c) - domain.inferior_frontier(c);

    const auto sensitivity = [&](const VectorR& x) -> float
    {
        MatrixR probe(2 * inputs_number, inputs_number);
        for (Index c = 0; c < inputs_number; ++c)
        {
            const float h = max(1e-4f, 1e-3f * span(c));
            probe.row(2 * c)     = x.transpose(); probe(2 * c, c)     += h;
            probe.row(2 * c + 1) = x.transpose(); probe(2 * c + 1, c) -= h;
        }
        const MatrixR out = opt.calculate_outputs(probe);
        double s = 0.0;
        for (Index c = 0; c < inputs_number; ++c)
        {
            const float h = max(1e-4f, 1e-3f * span(c));
            const float d = (out(2 * c, 0) - out(2 * c + 1, 0)) / (2.0f * h);
            s += double(d * span(c)) * double(d * span(c));
        }
        return float(sqrt(s));
    };

    const auto min_margin = [&](const VectorR& x) -> float
    {
        float worst = 1.0f;
        for (Index c = 0; c < inputs_number; ++c)
        {
            const float half = 0.5f * span(c);
            if (half < 1e-6f) continue;
            worst = min(worst, min(x(c) - domain.inferior_frontier(c),
                                   domain.superior_frontier(c) - x(c)) / half);
        }
        return worst;
    };

    const VectorR x_central = row_central.head(inputs_number);
    const VectorR x_robust  = row_robust.head(inputs_number);

    const float span_max = max(sensitivity(x_central), sensitivity(x_robust));

    EXPECT_LE(sensitivity(x_robust), sensitivity(x_central) + 0.05f * span_max + 1e-3f);

    EXPECT_GE(min_margin(x_central), min_margin(x_robust) - 1e-3f);
}



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



TEST(MixedIntegerProjector, FixedBinariesHoldWhileContinuousReproject)
{
    const vector<pair<string, Index>> inputs =
        make_named_columns({ "w0", "w1", "w2", "w3", "z0", "z1", "z2", "z3" });
    const vector<pair<string, Index>> outputs;

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
    set_random_uniform(points, float(0), float(1));
    for (Index r = 0; r < rows; ++r)
    {
        points(r, 4) = float(1); points(r, 5) = float(1);
        points(r, 6) = float(0); points(r, 7) = float(0);
    }

    vector<char> fixed(n, 0);
    fixed[4] = fixed[5] = fixed[6] = fixed[7] = 1;

    repair_affine_inputs_with_fixed(points, inferior, superior, constraints, fixed);

    for (Index r = 0; r < rows; ++r)
    {
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

        EXPECT_NEAR(points(r, 2), float(0), float(1e-3)) << "w2 not zeroed at row " << r;
        EXPECT_NEAR(points(r, 3), float(0), float(1e-3)) << "w3 not zeroed at row " << r;
    }
}



namespace
{
    MultivariateConstraint make_integer_constraint(const string& expression,
                                              const vector<pair<string, Index>>& inputs,
                                              ComparisonOperator op, float low, float up)
    {
        MultivariateConstraint constraint;
        constraint.expression = expression;
        constraint.comparison_operator = op;
        constraint.low_bound = low;
        constraint.up_bound = up;
        constraint.compiled = compile_formula(expression, inputs,             {});
        constraint.kind = classify(constraint);
        return constraint;
    }
}

TEST(MixedIntegerCarry, SingleBudgetStaysOnLatticeAndFeasible)
{
    const vector<pair<string, Index>> inputs = make_named_columns({ "n0", "n1", "n2" });
    const MultivariateConstraint budget =
        make_integer_constraint("n0 + n1 + n2", inputs, ComparisonOperator::LessEqualTo, float(0), float(4));

    const Index n = 3;
    const VectorR inferior = VectorR::Zero(n);
    const VectorR superior = VectorR::Constant(n, float(5));

    const Index rows = 300;
    MatrixR points(rows, n);
    set_random_uniform(points, float(0), float(5));

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
    const vector<pair<string, Index>> inputs = make_named_columns({ "n0", "n1", "n2" });
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



TEST(MixedIntegerKHot, DrawsExactlyKHonoringPins)
{
    const Index count = 8, k = 3;
    vector<char> force_on(count, 0), force_off(count, 0);
    force_on[1] = 1;
    force_off[5] = 1; force_off[6] = 1;

    for (int trial = 0; trial < 200; ++trial)
    {
        vector<float> out;
        ASSERT_TRUE(draw_k_hot(count, k, force_on, force_off, out));
        ASSERT_EQ(static_cast<Index>(out.size()), count);

        float sum = 0;
        for (const float v : out) { EXPECT_TRUE(v == float(0) || v == float(1)); sum += v; }
        EXPECT_EQ(sum, float(k));
        EXPECT_EQ(out[1], float(1));
        EXPECT_EQ(out[5], float(0));
        EXPECT_EQ(out[6], float(0));
    }
}

TEST(MixedIntegerKHot, ReportsInfeasiblePins)
{
    const Index count = 4;
    vector<float> out;

    {
        vector<char> force_on(count, 0), force_off(count, 0);
        force_on[0] = force_on[1] = force_on[2] = 1;
        EXPECT_FALSE(draw_k_hot(count, 2, force_on, force_off, out));
    }
    {
        vector<char> force_on(count, 0), force_off(count, 0);
        force_off[0] = force_off[1] = force_off[2] = 1;
        EXPECT_FALSE(draw_k_hot(count, 2, force_on, force_off, out));
    }
    {
        vector<char> force_on(count, 0), force_off(count, 0);
        force_on[0] = 1; force_off[0] = 1;
        EXPECT_FALSE(draw_k_hot(count, 1, force_on, force_off, out));
    }
}



TEST(MixedIntegerPortfolio, BuyInBudgetCardinalityYieldsFeasiblePoints)
{
    const int A = 8;
    const int K = 3;

    vector<string> indicator_names;
    vector<string> input_names;
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


TEST(MixedIntegerPortfolio, Port1ScaleBuyInBudgetCardinalityIsFeasible)
{
    const int A = 31;
    const int K = 10;

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

    opt.set_iterations(3);
    opt.set_evaluations_number(2000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0) << "no feasible point for port1-scale buy-in + budget + cardinality";

    for (Index r = 0; r < results.rows(); ++r)
    {
        float weight_sum = 0, indicator_sum = 0;
        for (int i = 0; i < A; ++i)
        {
            const float w = results(r, i), z = results(r, A + i);
            weight_sum += w; indicator_sum += z;
            EXPECT_NEAR(z, round(z), float(1e-3))       << "indicator z" << i << " not binary at row " << r;
            EXPECT_LE(w, z + float(1e-2))               << "buy-in upper at row " << r << " asset " << i;
            EXPECT_GE(w, float(0.01) * z - float(1e-2)) << "buy-in lower at row " << r << " asset " << i;
        }
        EXPECT_NEAR(weight_sum, float(1), float(2e-2))    << "budget violated at row " << r;
        EXPECT_NEAR(indicator_sum, float(K), float(1e-2)) << "cardinality violated at row " << r;
    }
}


TEST(MixedIntegerPortfolio, ExploreExploitRatioPreservesFeasibility)
{
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

    opt.set_exploration_ratio(float(0.5));
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



TEST(ContinuousCardinality, SelectsExactlyKNonzeroRestZero)
{
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
        EXPECT_LE(nonzero, K)     << "more than K active variables at row " << r;
        EXPECT_GE(zero, A - K)    << "fewer than A-K zeroed variables at row " << r;
    }
}


TEST(IntegerCardinality, ExactlyKActiveIntegersRestZero)
{
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
        EXPECT_EQ(nonzero, K) << "active integer count != K at row " << r;
    }
}


TEST(BinaryCardinality, FreeModeIsAtMostK)
{
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
    opt.set_cardinality_constraint(input_names, K,                   false);

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



TEST(SingleVariablePromotion, AffineConstraintBecomesBox)
{
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



TEST(ResponseOptimizationAllowedSet, FreeInputIsDrawnFromTheSet)
{
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



TEST(ResponseOptimizationCategory, ExplorationSamplesEveryCategoryAndTracksFrequency)
{
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
    ApproximationNetwork network(Shape{ 4 }, Shape{ 4 }, Shape{ 1 });

    Variable x; x.name = "x"; x.set_role("Input"); x.type = VariableType::Numeric;
    Variable cat; cat.name = "cat"; cat.set_role("Input"); cat.type = VariableType::Categorical;
    cat.set_categories({ "A", "B", "C" });
    network.set_input_variables({ x, cat });

    Variable y; y.name = "y"; y.set_role("Target"); y.type = VariableType::Numeric;
    network.set_output_variables({ y });

    vector<Descriptives> in_desc(4);
    in_desc[0] = Descriptives(float(0), float(10), float(5), float(2.5));
    for (Index j = 1; j < 4; ++j)
        in_desc[j] = Descriptives(float(0), float(1), float(0.5), float(0.5));
    static_cast<Scaling*>(network.get_first("Scaling"))->set_descriptives(in_desc);

    static_cast<Unscaling*>(network.get_first("Unscaling"))
        ->set_descriptives({ Descriptives(float(-1), float(1), float(0), float(0.5)) });

    ResponseOptimization opt(&network);
    opt.set_objective("y", ResponseOptimization::Sense::Maximize);
    opt.set_constraint("cat", vector<float>{ float(0), float(2) });

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    ASSERT_EQ(results.cols(), 5);
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
