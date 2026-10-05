// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "tests/common/pch.h"

#include "tests/response_optimization/synthetic_fixture.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/response_optimization/feasibility_repair_system.h"

namespace
{

using Repair = FeasibilityRepairSystem;
using Constraint = ResponseOptimization::Constraint;

class FeasibilitySolver : public testing::TestWithParam<Repair::Solver>
{
protected:

    MinimalApproximation setup{{"x1", "x2"}, {"y"}};

    pair<VectorR, VectorR> domain() const
    {
        return {VectorR::Zero(2), VectorR::Constant(2, 10.0f)};
    }

    void SetUp() override
    {
        setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    }
};

}


TEST_P(FeasibilitySolver, RepairsMixedInputAndOutputEquations)
{
    const auto input_equation = compile_expression("x1^2", setup.network.get());
    const auto output_equation = compile_expression("x2 + y", setup.network.get());
    const vector<Constraint> constraints{{input_equation, Condition::Equal, {1.0f}},
                                         {output_equation, Condition::Equal, {2.0f}}};
    Repair repair(setup.network.get(), constraints, domain());
    repair.set_solver(GetParam());

    VectorR point(2); point << 4.0f, 5.0f;
    const auto [input, output] = repair.solve(point);

    ASSERT_EQ(input.size(), 2);
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(input(0), 1.0f, 1e-4f);
    EXPECT_NEAR(input(1) + output(0), 2.0f, 1e-4f);
    const VectorR expected = setup.network->calculate_outputs(input.transpose()).row(0).transpose();
    EXPECT_TRUE(output.isApprox(expected, 1e-6f));
}


TEST_P(FeasibilitySolver, RepairsBothSidesOfAnInterval)
{
    const auto equation = compile_expression("x1^2", setup.network.get());
    const vector<Constraint> constraints{{equation, Condition::Between, {1.0f, 4.0f}}};
    Repair repair(setup.network.get(), constraints, domain());
    repair.set_solver(GetParam());

    for (const float start : {0.5f, 3.0f})
    {
        SCOPED_TRACE(start);
        VectorR point(2); point << start, 2.0f;
        const auto [input, output] = repair.solve(point);
        ASSERT_EQ(input.size(), 2);
        const float value = equation.evaluate(input, output);
        EXPECT_GE(value, 1.0f - 1e-5f);
        EXPECT_LE(value, 4.0f + 1e-5f);
    }
}


TEST_P(FeasibilitySolver, ReusesUpdatedDomainsAndRefreshesOutputs)
{
    const auto equation = compile_expression("x1 + x2", setup.network.get());
    const vector<Constraint> constraints{{equation, Condition::Equal, {5.0f}}};
    Repair repair(setup.network.get(), constraints, domain());
    repair.set_solver(GetParam());

    for (const float fixed : {1.0f, 3.0f})
    {
        auto bounds = domain();
        bounds.first(0) = bounds.second(0) = fixed;
        repair.set_domain(move(bounds));
        const auto [input, output] = repair.solve(VectorR::Constant(2, 4.0f));
        ASSERT_EQ(input.size(), 2);
        EXPECT_FLOAT_EQ(input(0), fixed);
        EXPECT_NEAR(input.sum(), 5.0f, 1e-4f);

        setup.network->set_parameters(VectorR::Constant(setup.network->get_parameters_buffer_size(), 0.1f*fixed));
        const VectorR expected = setup.network->calculate_outputs(input.transpose()).row(0).transpose();
        ASSERT_FALSE(output.isApprox(expected, 1e-6f));
        const auto repeated = repair.solve(input);
        ASSERT_EQ(repeated.first.size(), 2);
        EXPECT_EQ((repeated.first - input).squaredNorm(), 0.0f);
        EXPECT_TRUE(repeated.second.isApprox(expected, 1e-6f));
    }
}


TEST_P(FeasibilitySolver, PreservesAnEqualityWhileRepairingAnInequality)
{
    const auto equality = compile_expression("x1 + x2", setup.network.get());
    const auto inequality = compile_expression("x1", setup.network.get());
    const vector<Constraint> constraints{{equality, Condition::Equal, {3.0f}},
                                         {inequality, Condition::GreaterEqual, {2.0f}}};
    Repair repair(setup.network.get(), constraints, domain());
    repair.set_solver(GetParam());
    const auto result = repair.solve(VectorR::Constant(2, 1.5f));
    ASSERT_EQ(result.first.size(), 2);
    EXPECT_NEAR(result.first.sum(), 3.0f, 1e-5f);
    EXPECT_GE(result.first(0), 2.0f - 2e-6f);
}


TEST_P(FeasibilitySolver, RejectsAnUnrepairedPointWhenTheEvaluationLimitIsReached)
{
    const auto equation = compile_expression("x1^2", setup.network.get());
    const vector<Constraint> constraints{{equation, Condition::Equal, {1.0f}}};
    Repair repair(setup.network.get(), constraints, domain(), 1, 1);
    repair.set_solver(GetParam());
    const auto result = repair.solve(VectorR::Constant(2, 4.0f));
    EXPECT_EQ(result.first.size(), 0);
    EXPECT_EQ(result.second.size(), 0);
}


TEST_P(FeasibilitySolver, RejectsImpossibleAndNonfiniteEquations)
{
    for (const string& expression : {"x1^2 + 1", "sqrt(-x1 - 1)"})
    {
        SCOPED_TRACE(expression);
        const auto equation = compile_expression(expression, setup.network.get());
        const vector<Constraint> constraints{{equation, Condition::Equal, {0.0f}}};
        Repair repair(setup.network.get(), constraints, domain());
        repair.set_solver(GetParam());
        const auto result = repair.solve(VectorR::Constant(2, 2.0f));
        EXPECT_EQ(result.first.size(), 0);
        EXPECT_EQ(result.second.size(), 0);
    }
}


TEST_P(FeasibilitySolver, RoundsDiscreteEquationsBeforeAcceptingPoints)
{
    const auto integer = compile_expression("x1", setup.network.get());
    const auto membership = compile_expression("x2", setup.network.get());
    const vector<Constraint> constraints{{integer, Condition::Integer, {}},
                                         {membership, Condition::AllowedSet, {9.0f, 1.0f, 5.0f}}};
    Repair repair(setup.network.get(), constraints, domain());
    repair.set_solver(GetParam());

    VectorR point(2); point << 2.7f, 3.4f;
    const auto [input, output] = repair.solve(point);
    ASSERT_EQ(input.size(), 2);
    EXPECT_FLOAT_EQ(input(0), 3.0f);
    EXPECT_FLOAT_EQ(input(1), 5.0f);
    EXPECT_EQ(output.size(), 1);
}


TEST_P(FeasibilitySolver, RejectsNonfiniteInputsAndNetworkOutputs)
{
    Repair repair(setup.network.get(), {}, domain());
    repair.set_solver(GetParam());
    const float invalid = numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(repair.solve(VectorR::Constant(2, invalid)).first.size(), 0);
    setup.network->set_parameters(VectorR::Constant(setup.network->get_parameters_buffer_size(), invalid));
    const auto result = repair.solve(VectorR::Ones(2));
    EXPECT_EQ(result.first.size(), 0);
    EXPECT_EQ(result.second.size(), 0);
}


TEST_P(FeasibilitySolver, RejectsInvalidBoundsAndKeepsThePreviousDomain)
{
    const auto equation = compile_expression("x1", setup.network.get());
    const float invalid = numeric_limits<float>::quiet_NaN();
    const vector<Constraint> malformed{{equation, Condition::Between, {invalid, 1.0f}},
                                       {equation, Condition::AllowedSet, {invalid}},
                                       {equation, Condition::Between, {1.0f}},
                                       {equation, Condition::Between, {2.0f, 1.0f}},
                                       {equation, Condition::Equal, {}},
                                       {equation, Condition::AllowedSet, {}},
                                       {compile_expression("x1 + x2", setup.network.get()), Condition::Cardinality, {1.0f}},
                                       {compile_expression("x1^2", setup.network.get()), Condition::Integer, {}},
                                       {equation, static_cast<Condition>(-1), {1.0f}}};
    for (const Constraint& constraint : malformed)
        EXPECT_THROW((Repair(setup.network.get(), span<const Constraint>(&constraint, 1), domain())), runtime_error);
    for (const float tolerance : {0.0f, -1.0f, invalid, numeric_limits<float>::infinity()})
        EXPECT_THROW((Repair(setup.network.get(), {}, domain(), 1, 50, 1e-6f, 0.1f, tolerance)), runtime_error);

    Repair repair(setup.network.get(), {}, domain());
    repair.set_solver(GetParam());
    auto bounds = domain();
    bounds.first(0) = 11.0f;
    EXPECT_THROW(repair.set_domain(move(bounds)), runtime_error);
    EXPECT_THROW(repair.set_domain({VectorR::Constant(2, -numeric_limits<float>::max()),
                                    VectorR::Constant(2, numeric_limits<float>::max())}), runtime_error);
    const auto result = repair.solve(VectorR::Constant(2, 3.0f));
    ASSERT_EQ(result.first.size(), 2);
    EXPECT_EQ((result.first - VectorR::Constant(2, 3.0f)).squaredNorm(), 0.0f);
}


TEST_P(FeasibilitySolver, DistinguishesStrictAndInclusiveBounds)
{
    const auto equation = compile_expression("x1", setup.network.get());
    for (const Condition condition : {Condition::Greater, Condition::Less,
                                       Condition::GreaterEqual, Condition::LessEqual})
    {
        SCOPED_TRACE(int(condition));
        const vector<Constraint> constraints{{equation, condition, {1.0f}}};
        Repair repair(setup.network.get(), constraints, domain(), 0);
        repair.set_solver(GetParam());
        const auto result = repair.solve(VectorR::Ones(2));
        const bool inclusive = condition == Condition::GreaterEqual || condition == Condition::LessEqual;
        EXPECT_EQ(result.first.size(), inclusive ? 2 : 0);
    }
}


TEST_P(FeasibilitySolver, UsesTheConfiguredDiscreteToleranceForCompoundExpressions)
{
    const auto equation = compile_expression("x1 + x2", setup.network.get());
    for (const Condition condition : {Condition::Integer, Condition::AllowedSet})
        for (const float tolerance : {1e-3f, 1e-2f})
        {
            SCOPED_TRACE(int(condition));
            SCOPED_TRACE(tolerance);
            const vector<Constraint> constraints{{equation, condition,
                condition == Condition::AllowedSet ? vector<float>{3.0f} : vector<float>{}}};
            Repair repair(setup.network.get(), constraints, domain(), 0, 50, 1e-6f, 0.1f, tolerance);
            repair.set_solver(GetParam());
            VectorR point(2); point << 1.0f, 2.005f;
            const auto result = repair.solve(point);
            EXPECT_EQ(result.first.size(), tolerance > 0.005f ? 2 : 0);
        }
}


TEST_P(FeasibilitySolver, AppliesCardinalityAndRejectsConflictingForcedMembers)
{
    for (const Index budget : {Index(0), Index(1)})
    {
        SCOPED_TRACE(budget);
        const auto equation = compile_elementary_symmetric({0, 1}, {10.0f, 10.0f}, budget + 1, 1e-3f);
        const vector<Constraint> constraints{{equation, Condition::Cardinality, {float(budget)}}};
        Repair repair(setup.network.get(), constraints, domain());
        repair.set_solver(GetParam());
        VectorR point(2); point << 2.0f, 5.0f;
        const auto result = repair.solve(point);
        ASSERT_EQ(result.first.size(), 2);
        EXPECT_EQ((result.first.array() != 0.0f).count(), budget);
        EXPECT_FLOAT_EQ(result.first(0), 0.0f);
        EXPECT_FLOAT_EQ(result.first(1), budget == 1 ? 5.0f : 0.0f);

        auto bounds = domain();
        bounds.first.setOnes();
        repair.set_domain(move(bounds));
        EXPECT_EQ(repair.solve(point).first.size(), 0);
    }
}


TEST_P(FeasibilitySolver, RepairsThroughScaledFrozenDenseLayersAndUpdatedWeights)
{
    MinimalApproximation model({"x1", "x2"}, {"y1", "y2"});
    Network& network = *model.network;
    auto* hidden = static_cast<opennn::Dense*>(network.get_layer(1).get());
    auto* output_layer = static_cast<opennn::Dense*>(network.get_layer(2).get());
    hidden->set_activation_function("Identity");
    hidden->set_use_bias(false);
    hidden->set_is_trainable(false);
    auto* scaling = static_cast<Scaling*>(network.get_first("Scaling"));
    scaling->set_scalers(ScalerMethod::MeanStandardDeviation);
    scaling->set_descriptives({Descriptives(-10, 10, 1, 2), Descriptives(-10, 10, -2, 4)});
    auto* unscaling = static_cast<Unscaling*>(network.get_first("Unscaling"));
    unscaling->set_scalers(ScalerMethod::MeanStandardDeviation);
    unscaling->set_descriptives({Descriptives(-100, 100, 10, 2), Descriptives(-100, 100, -3, 0.5f)});
    network.compile(Device::CPU);
    network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));
    hidden->get_parameter_views()[0].as_matrix() << 1, 2, 0, 0, -1, 1, 0, 0;
    output_layer->get_parameter_views()[0].as_vector() << 0.5f, -1.0f;
    output_layer->get_parameter_views()[1].as_matrix() << 2, -1, 1, 3, 0, 0, 0, 0;
    const vector<Constraint> constraints{{compile_expression("x1 + y1", &network), Condition::Equal, {14.0f}},
                                         {compile_expression("x2 - y2", &network), Condition::Equal, {3.25f}}};
    Repair repair(&network, constraints, domain());
    repair.set_solver(GetParam());
    VectorR point(2); point << 7.0f, 8.0f;
    for (Index i = 0; i < 2; i++)
    {
        SCOPED_TRACE(i);
        output_layer->get_parameter_views()[1].as_matrix()(0, 0) = 2.0f + float(i);
        network.mark_parameters_changed();
        const auto [input, output] = repair.solve(point);
        ASSERT_EQ(input.size(), 2);
        ASSERT_EQ(output.size(), 2);
        EXPECT_NEAR(input(0), i == 0 ? 2.0f : 18.0f/7.0f, 1e-4f);
        EXPECT_NEAR(input(1), i == 0 ? 4.0f : 38.0f/7.0f, 1e-4f);
        EXPECT_NEAR(input(0) + output(0), 14.0f, 2e-5f);
        EXPECT_NEAR(input(1) - output(1), 3.25f, 5e-6f);
        EXPECT_TRUE(output.isApprox(network.calculate_outputs(input.transpose()).row(0).transpose(), 1e-6f));
    }
}


TEST_P(FeasibilitySolver, RepairsThroughNonlinearDenseActivations)
{
    for (const string& activation : {"Tanh", "Sigmoid", "ReLU", "LeakyReLU"})
    {
        SCOPED_TRACE(activation);
        Network network;
        network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, activation));
        network.set_input_variables({Variable("x1", "Input")});
        network.set_output_variables({Variable("y", "Target")});
        network.compile(Device::CPU);
        network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));
        network.get_layer(0)->get_parameter_views()[1].as_vector()(0) = 1.0f;
        network.mark_parameters_changed();
        const float expected = activation == "ReLU" ? 0.5f : -0.25f;
        const float target = expected + network.calculate_outputs(MatrixR::Constant(1, 1, expected))(0, 0);
        const vector<Constraint> constraints{{compile_expression("x1 + y", &network), Condition::Equal, {target}}};
        Repair repair(&network, constraints, {VectorR::Constant(1, -1.0f), VectorR::Ones(1)});
        repair.set_solver(GetParam());
        const auto [input, output] = repair.solve(VectorR::Constant(1, activation == "ReLU" ? 0.8f : -0.5f));
        ASSERT_EQ(input.size(), 1);
        ASSERT_EQ(output.size(), 1);
        EXPECT_NEAR(input(0), expected, 1e-4f);
        EXPECT_NEAR(input(0) + output(0), target, 2e-6f);
    }
}


TEST_P(FeasibilitySolver, FallsBackForLogarithmicScaling)
{
    Network network;
    network.add_layer(make_unique<Scaling>(Shape{1}, ScalerMethod::Logarithm));
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    network.set_input_variables({Variable("x1", "Input")});
    network.set_output_variables({Variable("y", "Target")});
    network.compile(Device::CPU);
    network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));
    network.get_layer(1)->get_parameter_views()[1].as_vector()(0) = 1.0f;
    network.mark_parameters_changed();
    const vector<Constraint> constraints{{compile_expression("y", &network), Condition::Equal, {log(2.0f)}}};
    Repair repair(&network, constraints, {VectorR::Constant(1, 0.1f), VectorR::Constant(1, 10.0f)});
    repair.set_solver(GetParam());
    const auto [input, output] = repair.solve(VectorR::Constant(1, 4.0f));
    ASSERT_EQ(input.size(), 1);
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(input(0), 2.0f, 1e-4f);
    EXPECT_NEAR(output(0), log(2.0f), 2e-6f);
}


TEST_P(FeasibilitySolver, RepairsAtClampingBoundariesAndOnConstantSegments)
{
    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    network.add_layer(make_unique<Clamping>(Shape{1}));
    network.set_input_variables({Variable("x1", "Input")});
    network.set_output_variables({Variable("y", "Target")});
    network.compile(Device::CPU);
    network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));
    network.get_layer(0)->get_parameter_views()[1].as_vector()(0) = 1.0f;
    network.mark_parameters_changed();
    auto* clamping = static_cast<Clamping*>(network.get_layer(1).get());
    clamping->set_lower_bound(0, 0.0f);
    clamping->set_upper_bound(0, 1.0f);
    for (const bool plateau : {false, true})
    {
        SCOPED_TRACE(plateau);
        const vector<Constraint> constraints{{compile_expression(plateau ? "x1 + y" : "y", &network),
                                             Condition::Equal, {plateau ? 2.5f : 0.5f}}};
        Repair repair(&network, constraints, {VectorR::Zero(1), VectorR::Constant(1, 4.0f)});
        repair.set_solver(GetParam());
        const auto [input, output] = repair.solve(VectorR::Constant(1, plateau ? 2.0f : 0.0f));
        ASSERT_EQ(input.size(), 1);
        ASSERT_EQ(output.size(), 1);
        EXPECT_NEAR(input(0), plateau ? 1.5f : 0.5f, 1e-5f);
        EXPECT_NEAR(output(0), plateau ? 1.0f : 0.5f, 1e-5f);
    }
    clamping->set_lower_bound(0, 1.0f);
    const vector<Constraint> impossible{{compile_expression("y", &network), Condition::Equal, {2.0f}}};
    Repair repair(&network, impossible, {VectorR::Zero(1), VectorR::Constant(1, 4.0f)});
    repair.set_solver(GetParam());
    EXPECT_EQ(repair.solve(VectorR::Constant(1, 2.0f)).first.size(), 0);
}


TEST(FeasibilityGaussNewton, RepairsWhenFiniteDifferencesLoseOutputChanges)
{
    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    network.set_input_variables({Variable("x1", "Input")});
    network.set_output_variables({Variable("y", "Target")});
    network.compile(Device::CPU);
    network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));
    network.get_layer(0)->get_parameter_views()[0].as_vector()(0) = 1000000.0f;
    network.get_layer(0)->get_parameter_views()[1].as_vector()(0) = 1.0f;
    network.mark_parameters_changed();
    ASSERT_FLOAT_EQ(network.calculate_outputs(MatrixR::Constant(1, 1, 0.01f))(0, 0), 1000000.0f);
    const vector<Constraint> constraints{{compile_expression("y - 1000000", &network), Condition::Equal, {2.0f}}};
    Repair repair(&network, constraints, {VectorR::Zero(1), VectorR::Constant(1, 10.0f)}, 1, 1);
    repair.set_solver(Repair::Solver::GaussNewton);
    const auto [input, output] = repair.solve(VectorR::Zero(1));
    ASSERT_EQ(input.size(), 1);
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(input(0), 2.0f, 1e-5f);
    EXPECT_FLOAT_EQ(output(0), 1000002.0f);
}


TEST(FeasibilityGaussNewton, FindsTheMinimumNormStepWithUnequalSpansAndAFixedColumn)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"});
    const auto equation = compile_expression("x1 + x2/2 + x3/10", setup.network.get());
    const auto redundant = compile_expression("2*x1 + x2 + x3/5", setup.network.get());
    VectorR lower(3); lower << 2.0f, 0.0f, 0.0f;
    VectorR upper(3); upper << 2.0f, 2.0f, 10.0f;
    VectorR point(3); point << 9.0f, 0.0f, 0.0f;

    for (const bool duplicate : {false, true})
    {
        SCOPED_TRACE(duplicate);
        vector<Constraint> constraints{{equation, Condition::Equal, {3.0f}}};
        if (duplicate) constraints.push_back({redundant, Condition::Equal, {6.0f}});
        Repair repair(setup.network.get(), constraints, {lower, upper});
        repair.set_solver(Repair::Solver::GaussNewton);
        const auto [input, output] = repair.solve(point);
        ASSERT_EQ(input.size(), 3);
        ASSERT_EQ(output.size(), 1);
        EXPECT_TRUE(input.allFinite());
        EXPECT_FLOAT_EQ(input(0), 2.0f);
        EXPECT_NEAR(input(1), 1.0f, 1e-5f);
        EXPECT_NEAR(input(2), 5.0f, 1e-5f);
    }
}


TEST(FeasibilityGaussNewton, RejectsAStationaryInfeasiblePoint)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    const vector<Constraint> constraints{{compile_expression("x1^2", setup.network.get()),
                                         Condition::Equal, {1.0f}}};
    Repair repair(setup.network.get(), constraints, {VectorR::Zero(2), VectorR::Constant(2, 10.0f)});
    repair.set_solver(Repair::Solver::GaussNewton);
    pair<VectorR, VectorR> result;
    EXPECT_NO_THROW(result = repair.solve(VectorR::Zero(2)));
    EXPECT_EQ(result.first.size(), 0);
    EXPECT_EQ(result.second.size(), 0);
}


TEST(FeasibilityGaussNewton, ProjectsAnOvershootingStepAndContinuesRepair)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    const vector<Constraint> constraints{{compile_expression("x1^2", setup.network.get()),
                                         Condition::Equal, {1.0f}}};
    VectorR lower(2); lower << 0.0f, 3.0f;
    VectorR upper(2); upper << 2.0f, 3.0f;
    Repair repair(setup.network.get(), constraints, {lower, upper});
    repair.set_solver(Repair::Solver::GaussNewton);
    VectorR point(2); point << 0.1f, 3.0f;
    const auto [input, output] = repair.solve(point);
    ASSERT_EQ(input.size(), 2);
    EXPECT_TRUE(input.allFinite());
    EXPECT_TRUE((input.array() >= lower.array()).all());
    EXPECT_TRUE((input.array() <= upper.array()).all());
    EXPECT_NEAR(input(0), 1.0f, 1e-5f);
    EXPECT_FLOAT_EQ(input(1), 3.0f);
    EXPECT_EQ(output.size(), 1);
}


TEST(FeasibilityGaussNewton, AppliesASmallCorrectionBeforeStopping)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    const vector<Constraint> constraints{{compile_expression("x1", setup.network.get()),
                                         Condition::Equal, {1.0f}}};
    Repair repair(setup.network.get(), constraints, {VectorR::Zero(2), VectorR::Constant(2, 10.0f)}, 1, 1);
    repair.set_solver(Repair::Solver::GaussNewton);
    const auto result = repair.solve(VectorR::Constant(2, 1.0f + 2e-6f));
    ASSERT_EQ(result.first.size(), 2);
    EXPECT_FLOAT_EQ(result.first(0), 1.0f);
    EXPECT_FLOAT_EQ(result.first(1), 1.0f + 2e-6f);
}


TEST(FeasibilityGaussNewton, ContinuesSmallProjectedStepsUntilFeasible)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    const vector<Constraint> constraints{{compile_expression("x1 + x2", setup.network.get()),
                                         Condition::Equal, {5.0f}}};
    Repair repair(setup.network.get(), constraints, {VectorR::Zero(2), VectorR::Constant(2, 10.0f)});
    repair.set_solver(Repair::Solver::GaussNewton);
    VectorR point(2); point << 9.0f, 0.0f;
    const auto result = repair.solve(point);
    ASSERT_EQ(result.first.size(), 2);
    EXPECT_TRUE(result.first.allFinite());
    EXPECT_TRUE((result.first.array() >= 0.0f).all());
    EXPECT_TRUE((result.first.array() <= 10.0f).all());
    EXPECT_NEAR(result.first.sum(), 5.0f, 5e-6f);
    EXPECT_FLOAT_EQ(result.first(1), 0.0f);
}


TEST(FeasibilityGaussNewton, AccountsForProportionalAndCappedMarginsInOneStep)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    const auto equation = compile_expression("x1", setup.network.get());
    for (const Condition condition : {Condition::GreaterEqual, Condition::Between})
    {
        SCOPED_TRACE(int(condition));
        const vector<Constraint> constraints{{equation, condition,
            condition == Condition::Between ? vector<float>{1.0f, 1.01f} : vector<float>{1.0f}}};
        Repair repair(setup.network.get(), constraints, {VectorR::Zero(2), VectorR::Constant(2, 10.0f)}, 1, 1);
        repair.set_solver(Repair::Solver::GaussNewton);
        const auto result = repair.solve(VectorR::Zero(2));
        ASSERT_EQ(result.first.size(), 2);
        EXPECT_NEAR(result.first(0), condition == Condition::Between ? 1.005f : 1.0f, 1e-6f);
        EXPECT_FLOAT_EQ(result.first(1), 0.0f);
    }
}


INSTANTIATE_TEST_SUITE_P(RepairBackends, FeasibilitySolver,
                        testing::Values(Repair::Solver::LevenbergMarquardt, Repair::Solver::GaussNewton),
                        [](const testing::TestParamInfo<Repair::Solver>& info)
                        { return info.param == Repair::Solver::GaussNewton ? "GaussNewton" : "LevenbergMarquardt"; });
