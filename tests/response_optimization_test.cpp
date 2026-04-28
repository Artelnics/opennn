//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   T E S T S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#include "../opennn/formula_expression.h"
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


type lookup_coeff(const vector<pair<Index, type>>& terms, Index column)
{
    for (const auto& [col, coeff] : terms)
        if (col == column) return coeff;
    return type(0);
}


struct MinimalApproximation
{
    unique_ptr<ApproximationNetwork> network;

    MinimalApproximation(const vector<string>& input_names,
                         const vector<string>& output_names,
                         type input_min = type(0),
                         type input_max = type(10),
                         type output_min = type(-1),
                         type output_max = type(1))
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
            input_vars[i].role = "Input";
            input_vars[i].type = VariableType::Numeric;
        }
        network->set_input_variables(input_vars);

        vector<Variable> output_vars(n_outputs);
        for (Index i = 0; i < n_outputs; ++i)
        {
            output_vars[i].name = output_names[i];
            output_vars[i].role = "Target";
            output_vars[i].type = VariableType::Numeric;
        }
        network->set_output_variables(output_vars);

        vector<Descriptives> in_desc(n_inputs);
        for (Descriptives& d : in_desc)
        {
            d.minimum = input_min;
            d.maximum = input_max;
            d.mean = (input_min + input_max) * type(0.5);
            d.standard_deviation = (input_max - input_min) * type(0.25);
        }
        auto* scaling = static_cast<Scaling<2>*>(network->get_first("Scaling2d"));
        scaling->set_descriptives(in_desc);

        vector<Descriptives> out_desc(n_outputs);
        for (Descriptives& d : out_desc)
        {
            d.minimum = output_min;
            d.maximum = output_max;
            d.mean = (output_min + output_max) * type(0.5);
            d.standard_deviation = (output_max - output_min) * type(0.25);
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
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), type(1), type(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), type(2), type(1e-6));
    EXPECT_NEAR(f.affine_constant, type(-3), type(1e-6));
}


TEST(FormulaExpression, UnaryNegationFlipsCoefficients)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("-x1 + x2", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), type(-1), type(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), type(1), type(1e-6));
}


TEST(FormulaExpression, ConstantScalingDistributesOverSum)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("3*(x1 + x2)", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), type(3), type(1e-6));
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 1), type(3), type(1e-6));
}


TEST(FormulaExpression, DivisionByConstantIsAffine)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1" });
    const CompiledFormula f = compile_formula("x1 / 4", inputs, {});

    EXPECT_EQ(f.shape, FormulaShape::Affine);
    EXPECT_NEAR(lookup_coeff(f.affine_input_terms, 0), type(0.25), type(1e-6));
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

    VectorR in(2); in << type(3), type(5);
    VectorR out(0);

    // Expected: -3 + 10 + 1 = 8
    EXPECT_NEAR(f.evaluate(in, out), type(8), type(1e-5));
}


TEST(FormulaExpression, EvaluateNonlinearExpression)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula f = compile_formula("sqrt(x1) + x2^2", inputs, {});

    VectorR in(2); in << type(9), type(3);
    VectorR out(0);

    // Expected: 3 + 9 = 12
    EXPECT_NEAR(f.evaluate(in, out), type(12), type(1e-5));
}


TEST(FormulaExpression, EvaluateUsesOutputsForMixedScope)
{
    const CompiledFormula f = compile_formula("x1 + 2*y1",
                                              make_named_columns({ "x1" }),
                                              make_named_columns({ "y1" }));

    VectorR in(1); in << type(1);
    VectorR out(1); out << type(4);

    EXPECT_NEAR(f.evaluate(in, out), type(9), type(1e-5));
}


TEST(FormulaExpression, ParenthesesOverridePrecedence)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });

    const CompiledFormula a = compile_formula("2 * x1 + x2", inputs, {});
    const CompiledFormula b = compile_formula("2 * (x1 + x2)", inputs, {});

    VectorR in(2); in << type(3), type(5);
    VectorR out(0);

    EXPECT_NEAR(a.evaluate(in, out), type(11), type(1e-5));
    EXPECT_NEAR(b.evaluate(in, out), type(16), type(1e-5));
}


TEST(FormulaExpression, MinMaxFunctions)
{
    const vector<NamedColumn> inputs = make_named_columns({ "x1", "x2" });
    const CompiledFormula fmin = compile_formula("min(x1, x2)", inputs, {});
    const CompiledFormula fmax = compile_formula("max(x1, x2)", inputs, {});

    VectorR in(2); in << type(2), type(7);
    VectorR out(0);

    EXPECT_NEAR(fmin.evaluate(in, out), type(2), type(1e-5));
    EXPECT_NEAR(fmax.evaluate(in, out), type(7), type(1e-5));
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
                                type(0), type(10),
                                type(-1), type(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_condition("y", ResponseOptimization::ConditionType::Minimize);

    // Require x1 + x2 <= 1.0 over a domain with each variable in [0, 10]
    opt.set_formula_constraint("x1 + x2",
                               ResponseOptimization::ConditionType::LessEqualTo,
                               type(0), type(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const type sum = results(i, 0) + results(i, 1);
        EXPECT_LE(sum, type(1) + type(1e-3))
            << "row " << i << " x1=" << results(i,0) << " x2=" << results(i,1);
    }
}


TEST(ResponseOptimizationFormula, AffineEqualityLandsOnHyperplane)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                type(0), type(10),
                                type(-1), type(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_condition("y", ResponseOptimization::ConditionType::Minimize);

    // Force x1 + x2 = 5
    opt.set_formula_constraint("x1 + x2",
                               ResponseOptimization::ConditionType::EqualTo,
                               type(5));

    opt.set_iterations(3);
    opt.set_evaluations_number(600);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const type sum = results(i, 0) + results(i, 1);
        EXPECT_NEAR(sum, type(5), type(5e-2))
            << "row " << i << " x1=" << results(i,0) << " x2=" << results(i,1);
    }
}


TEST(ResponseOptimizationFormula, NonlinearInputConstraintFilters)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                type(-5), type(5),
                                type(-1), type(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_condition("y", ResponseOptimization::ConditionType::Minimize);

    // Require x1^2 + x2^2 <= 4 (points inside circle of radius 2).
    opt.set_formula_constraint("x1^2 + x2^2",
                               ResponseOptimization::ConditionType::LessEqualTo,
                               type(0), type(4));

    opt.set_iterations(3);
    opt.set_evaluations_number(1000);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
    {
        const type r2 = results(i, 0) * results(i, 0) + results(i, 1) * results(i, 1);
        EXPECT_LE(r2, type(4) + type(1e-2));
    }
}


TEST(ResponseOptimizationFormula, CallbackConstraintFilters)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                type(0), type(10),
                                type(-1), type(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_condition("y", ResponseOptimization::ConditionType::Minimize);

    // Callback: |x1 - x2| <= 1
    opt.set_formula_constraint(
        [](const VectorR& in, const VectorR& /*out*/) { return abs(in(0) - in(1)); },
        ResponseOptimization::ConditionType::LessEqualTo,
        type(0), type(1));

    opt.set_iterations(3);
    opt.set_evaluations_number(800);

    const MatrixR results = opt.perform_response_optimization();

    ASSERT_GT(results.rows(), 0);
    for (Index i = 0; i < results.rows(); ++i)
        EXPECT_LE(abs(results(i, 0) - results(i, 1)), type(1) + type(1e-3));
}


TEST(ResponseOptimizationFormula, InfeasibleConstraintThrows)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                type(0), type(1),
                                type(-1), type(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_condition("y", ResponseOptimization::ConditionType::Minimize);

    // Unreachable: inputs live in [0,1] but we demand x1 + x2 ∈ [90, 100]
    opt.set_formula_constraint("x1 + x2",
                               ResponseOptimization::ConditionType::Between,
                               type(90), type(100));

    opt.set_iterations(2);
    opt.set_evaluations_number(200);
    opt.set_max_oversample_factor(2);

    EXPECT_THROW(opt.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationFormula, NoFormulaConstraintsPreservesBaseline)
{
    MinimalApproximation setup({ "x1", "x2" }, { "y" },
                                type(0), type(10),
                                type(-1), type(1));

    ResponseOptimization opt(setup.network.get());
    opt.set_condition("y", ResponseOptimization::ConditionType::Minimize);

    opt.set_iterations(2);
    opt.set_evaluations_number(300);

    const MatrixR results = opt.perform_response_optimization();
    ASSERT_GT(results.rows(), 0);

    // Each sampled input must live inside the declared domain.
    for (Index i = 0; i < results.rows(); ++i)
    {
        EXPECT_GE(results(i, 0), type(0) - type(1e-3));
        EXPECT_LE(results(i, 0), type(10) + type(1e-3));
        EXPECT_GE(results(i, 1), type(0) - type(1e-3));
        EXPECT_LE(results(i, 1), type(10) + type(1e-3));
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
