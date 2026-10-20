// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

// Every condition and objective sense a response optimization understands, stated against
// a small untrained network and run through both solvers.
//
// The networks carry no training, so no case asks what response comes back. Each one asks
// only where the result may sit: inside the box, on a hyperplane, on a lattice, one hot
// across a categorical block. That holds whatever the weights are.
//
// Cases that are refused before any search starts are grouped first, under
// ResponseOptimizationSetup. The ones that run are parameterised over both solvers, so a
// condition that only one of them honours fails here.
//
// The expression language the conditions are written in is tested in
// expression_test.cpp.

#include "tests/common/pch.h"

#include "tests/response_optimization/synthetic_fixture.h"

#include "opennn/core/profiler.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/feasibility_repair_system.h"
#include "opennn/response_optimization/genetic_response.h"

namespace
{

enum class Driver { Contraction, Genetic };

unique_ptr<ResponseOptimization> make_driver(const Driver driver, Network* network)
{
    if (driver == Driver::Genetic) return make_unique<GeneticResponse>(network);

    return make_unique<DomainContraction>(network);
}


string driver_name(const testing::TestParamInfo<Driver>& info)
{
    return info.param == Driver::Genetic ? "Genetic" : "Contraction";
}


class ResponseDriver : public testing::TestWithParam<Driver>
{
protected:

    void SetUp() override { set_seed(1234); }
};


class RepairProbe : public ResponseOptimization
{
public:

    explicit RepairProbe(Network* network) : ResponseOptimization(network) {}

    using ResponseOptimization::calculate_domain;
    using ResponseOptimization::calculate_pareto_front;
    using ResponseOptimization::calculate_random_input;
    using ResponseOptimization::constraints;
    using ResponseOptimization::input_bounds;
    using ResponseOptimization::get_feasible_points;
    using ResponseOptimization::get_sampling_failure;
    using ResponseOptimization::make_feasibility_system;

private:

    MatrixR single_optimization() override { return {}; }
    MatrixR multi_optimization() override { return {}; }
};

}


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


TEST(ResponseOptimizationSetup, CardinalityNeedsAListOfAtLeastTwoInputs)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; ", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1 * x2", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; y", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_NO_THROW(optimization.add_constraint("x1; x2; x3", Condition::Cardinality, {2.0f}));
}


TEST(ResponseOptimizationSetup, CardinalityRejectsArithmeticInItsList)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1 + x2 + x3", Condition::Cardinality, {2.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("2*x1; x2", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1 - x2; x3", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; x2 + 5", Condition::Cardinality, {1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; x1; x2", Condition::Cardinality, {1.0f}), runtime_error);
}


TEST(ResponseOptimizationSetup, CardinalityNeedsAWholeBudgetInsideItsGroup)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1; x2; x3", Condition::Cardinality, {}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; x2; x3", Condition::Cardinality, {-1.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; x2; x3", Condition::Cardinality, {4.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1; x2; x3", Condition::Cardinality, {1.5f}), runtime_error);
}


TEST(ResponseOptimizationSetup, CardinalityOverAVariableThatCannotReachZeroThrows)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"}, 2.0f, 10.0f);

    DomainContraction optimization(setup.network.get());

    optimization.add_objective("y", Sense::Minimize);
    optimization.add_constraint("x1; x2; x3", Condition::Cardinality, {2.0f});

    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationSetup, DiscreteConditionsTakeAnInputOrAnExpressionOfSeveral)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_NO_THROW(optimization.add_constraint("x1", Condition::Integer));
    EXPECT_NO_THROW(optimization.add_constraint("x1 + x2", Condition::Integer));
    EXPECT_NO_THROW(optimization.add_constraint("x1 / x2", Condition::AllowedSet, {1.0f, 2.0f}));

    EXPECT_THROW(optimization.add_constraint("2*x1", Condition::Integer), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1 + y", Condition::AllowedSet, {1.0f, 2.0f}), runtime_error);
}


TEST_P(ResponseDriver, IntegerOnAnExpressionKeepsItWhole)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Maximize);
    optimization->add_constraint("x1 + x2", Condition::Integer);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        const float sum = results(i, 0) + results(i, 1);

        EXPECT_LT(abs(sum - round(sum)), 2e-3f) << "row " << i << " x1 + x2 = " << sum;
    }
}


TEST(ResponseOptimizationSetup, AnIntegerWithNoWholeNumberInItsRangeThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    DomainContraction optimization(setup.network.get());

    optimization.add_objective("y", Sense::Minimize);
    optimization.add_constraint("x1", Condition::Integer);
    optimization.add_constraint("x1", Condition::Between, {1.2f, 1.8f});

    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationSetup, AnAllowedSetWithNoValueInItsRangeThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    DomainContraction optimization(setup.network.get());

    optimization.add_objective("y", Sense::Minimize);
    optimization.add_constraint("x1", Condition::AllowedSet, {1.0f, 9.0f});
    optimization.add_constraint("x1", Condition::Between, {4.0f, 6.0f});

    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationSetup, InvalidConstraintValuesThrow)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::Between, {5.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1", Condition::LessEqual, {}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1", Condition::AllowedSet, {}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1", Condition::Between, {5.0f, 2.0f}), runtime_error);
}


TEST(ResponseOptimizationSetup, ConstraintsThatEmptyAColumnThrowWhenTheDomainIsBuilt)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    DomainContraction optimization(setup.network.get());

    optimization.add_objective("y", Sense::Minimize);

    optimization.add_constraint("x1", Condition::GreaterEqual, {8.0f});
    optimization.add_constraint("2 * x1", Condition::LessEqual, {4.0f});

    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


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

    optimization->add_constraint("2 * x1 - 6", Condition::LessEqual, {0.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_LE(results(i, 0), 3.0f + 1e-3f) << "row " << i;
}


TEST_P(ResponseDriver, OutputConstraintHolds)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

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

        EXPECT_LT(min(min(abs(x1 - 1.0f), abs(x1 - 5.0f)), abs(x1 - 9.0f)), 1e-5f)
            << "row " << i << " x1=" << x1 << " is not in {1, 5, 9}";
    }
}


TEST_P(ResponseDriver, IntegerKeepsResultsOnWholeNumbers)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);
    optimization->add_constraint("x1", Condition::Integer);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        const float x1 = results(i, 0);

        EXPECT_LT(abs(x1 - round(x1)), 1e-5f) << "row " << i << " x1=" << x1 << " is not a whole number";

        EXPECT_GE(x1, 0.0f) << "row " << i;
        EXPECT_LE(x1, 10.0f) << "row " << i;
    }
}


TEST_P(ResponseDriver, AnAllowedSetOfTwoValuesActsAsABinaryVariable)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);
    optimization->add_constraint("x1", Condition::AllowedSet, {0.0f, 1.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        const float x1 = results(i, 0);

        EXPECT_LT(min(abs(x1), abs(x1 - 1.0f)), 1e-5f)
            << "row " << i << " x1=" << x1 << " is neither 0 nor 1";
    }
}


TEST_P(ResponseDriver, TwoLatticeConditionsOnOneVariableAgreeInEitherOrder)
{
    for (const bool integer_first : {true, false})
    {
        MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

        const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

        optimization->add_objective("y", Sense::Minimize);

        if (integer_first) optimization->add_constraint("x1", Condition::Integer);

        optimization->add_constraint("x1", Condition::AllowedSet, {2.0f, 4.0f, 6.0f});

        if (!integer_first) optimization->add_constraint("x1", Condition::Integer);

        const MatrixR results = optimization->perform_response_optimization();

        ASSERT_GT(results.rows(), 0);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float x1 = results(i, 0);

            EXPECT_LT(min(min(abs(x1 - 2.0f), abs(x1 - 4.0f)), abs(x1 - 6.0f)), 1e-5f)
                << "integer first " << integer_first << ", row " << i << " x1=" << x1;
        }
    }
}


TEST_P(ResponseDriver, CardinalityLeavesAtMostTheBudgetInPlay)
{
    MinimalApproximation setup({"x1", "x2", "x3", "x4"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("x1 + x2 + x3 + x4", Sense::Maximize);
    optimization->add_constraint("x1; x2; x3; x4", Condition::Cardinality, {2.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        Index in_play = 0;

        for (Index j = 0; j < 4; j++)
            if (results(i, j) != 0.0f) in_play++;

        EXPECT_LE(in_play, 2) << "row " << i << " keeps " << in_play << " of the 4 variables in play";
    }
}


TEST_P(ResponseDriver, SamplingBudgetsLimitFailedRepairs)
{
    MinimalApproximation setup({"x1"}, {"y"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));

    const auto optimization = make_driver(GetParam(), setup.network.get());
    optimization->add_objective("x1", Sense::Minimize);
    optimization->add_constraint("x1", Condition::Equal, {0.0f});
    optimization->add_constraint("y", Condition::GreaterEqual, {1.0f});
    optimization->set_points_number(3);
    optimization->set_iterations_number(2);

    const auto expect_attempts = [&](const long expected)
    {
        SCOPED_TRACE(expected);
        const bool profiling_enabled = profiler::is_enabled();
        profiler::set_enabled(true);
        const long previous_calls = profiler::stats().call_count("fp:set");

        EXPECT_THROW(optimization->perform_response_optimization(), runtime_error);

        const long calls = profiler::stats().call_count("fp:set") - previous_calls;
        profiler::set_enabled(profiling_enabled);
        EXPECT_EQ(calls, expected);
    };

    expect_attempts(6);
    optimization->set_sampling_budget_multiplier(1);
    expect_attempts(3);
    optimization->set_sampling_budget_multiplier(4);
    expect_attempts(12);
    optimization->set_iterations_number(5);
    expect_attempts(12);
    optimization->set_maximum_consecutive_failures(2);
    expect_attempts(2);
    optimization->set_maximum_consecutive_failures(20);
    expect_attempts(12);
    optimization->set_maximum_consecutive_failures(0);
    expect_attempts(12);
    optimization->set_sampling_budget_multiplier(0);
    expect_attempts(15);
}


TEST_P(ResponseDriver, FailureLimitPreservesPartialResultsAndResetsAfterSuccess)
{
    CategoricalApproximation setup({}, "material", {"first", "best", "rejected"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));

    RepairProbe probe(setup.network.get());
    const auto domain = probe.calculate_domain();

    for (const bool interrupted : {false, true})
    {
        SCOPED_TRACE(interrupted);
        const vector<Index> sequence = interrupted ? vector<Index>{0, 2, 2, 1} : vector<Index>{2, 0, 2, 1};

        unsigned seed = 0;
        for (; seed < 1024; seed++)
        {
            set_seed(seed);
            bool matches = true;
            for (const Index category : sequence)
                matches = (probe.calculate_random_input(domain)(category) == 1.0f) && matches;
            if (matches) break;
        }
        ASSERT_LT(seed, 1024u);

        const auto optimization = make_driver(GetParam(), setup.network.get());
        optimization->add_objective("best", Sense::Maximize);
        optimization->add_constraint("first + best + y1", Condition::Equal, {1.0f});
        optimization->set_points_number(2);
        optimization->set_iterations_number(1);
        optimization->set_sampling_budget_multiplier(4);
        optimization->set_maximum_consecutive_failures(2);

        set_seed(seed);
        MatrixR results;
        ASSERT_NO_THROW(results = optimization->perform_response_optimization());
        ASSERT_EQ(results.rows(), 1);
        const Index category = read_category(results, 0, 0, 3);
        if (interrupted && GetParam() == Driver::Genetic)
            EXPECT_TRUE(category == 0 || category == 1);
        else
            EXPECT_EQ(category, interrupted ? 0 : 1);
    }
}


TEST_P(ResponseDriver, SelectsEitherFeasibilitySolver)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    const auto optimization = make_driver(GetParam(), setup.network.get());
    optimization->add_constraint("x1^2 + x2^2", Condition::Equal, {25.0f});
    optimization->add_objective("x1", Sense::Maximize);
    optimization->set_points_number(4);
    optimization->set_iterations_number(1);
    for (const auto solver : {ResponseOptimization::FeasibilitySolver::LevenbergMarquardt,
                              ResponseOptimization::FeasibilitySolver::GaussNewton})
    {
        SCOPED_TRACE(int(solver));
        optimization->set_feasibility_solver(solver);
        set_seed(1234);
        const MatrixR result = optimization->perform_response_optimization();
        ASSERT_EQ(result.rows(), 1);
        EXPECT_TRUE(result.allFinite());
        EXPECT_NEAR(result.leftCols(2).squaredNorm(), 25.0f, 1e-4f);
    }
}


TEST(Feasibility, SamplingAndBatchRepairUseTheirExplicitDomain)
{
    CategoricalApproximation setup({"x"}, "material", {"first", "second"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    RepairProbe probe(setup.network.get());
    auto domain = probe.calculate_domain();
    probe.input_bounds.second(2) = 0.0f;
    domain.first(0) = domain.second(0) = 3.0f;
    domain.second(1) = 0.0f;
    auto feasibility_system = probe.make_feasibility_system(domain);
    for (Index i = 0; i < 4; i++)
    {
        const VectorR sampled = probe.calculate_random_input(domain);
        EXPECT_FLOAT_EQ(sampled(0), 3.0f);
        EXPECT_FLOAT_EQ(sampled(1), 0.0f);
        EXPECT_FLOAT_EQ(sampled(2), 1.0f);
        const auto [input, output] = probe.get_feasible_points(sampled, feasibility_system);
        ASSERT_EQ(input.size(), 3);
        EXPECT_EQ((input - sampled).squaredNorm(), 0.0f);
        EXPECT_TRUE(output.allFinite());
    }
    EXPECT_FLOAT_EQ(probe.input_bounds.second(2), 0.0f);
}


TEST(Feasibility, ConfiguredSolverIsUsedByBothRepairEntryPoints)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1 + x2", Condition::Equal, {5.0f});
    probe.calculate_domain();
    probe.input_bounds.second(1) = 2.0f;
    probe.set_feasibility_solver(ResponseOptimization::FeasibilitySolver::GaussNewton);
    auto feasibility_system = probe.make_feasibility_system(probe.input_bounds);
    const auto direct = probe.get_feasible_points(VectorR::Zero(2));
    const auto reused = probe.get_feasible_points(VectorR::Zero(2), feasibility_system);
    ASSERT_EQ(direct.first.size(), 2);
    ASSERT_EQ(reused.first.size(), 2);
    EXPECT_NEAR(direct.first(0), 125.0f/26.0f, 1e-5f);
    EXPECT_NEAR(direct.first(1), 5.0f/26.0f, 1e-5f);
    EXPECT_EQ((direct.first - reused.first).squaredNorm(), 0.0f);
    EXPECT_THROW(probe.set_feasibility_solver(static_cast<ResponseOptimization::FeasibilitySolver>(1)), runtime_error);
}


TEST_P(ResponseDriver, SamplingFailureNamesTheMissedCatalogueConstraint)
{
    MinimalApproximation setup({"x"}, {"y"});
    const auto optimization = make_driver(GetParam(), setup.network.get());
    optimization->add_objective("x", Sense::Minimize);
    optimization->add_constraint("x", Condition::AllowedSet, {1.0f});
    optimization->add_constraint("x*x", Condition::Equal, {2.0f});
    optimization->set_points_number(1);
    optimization->set_iterations_number(1);
    try
    {
        optimization->perform_response_optimization();
        FAIL() << "The catalogue cannot meet this equality.";
    }
    catch (const runtime_error& error)
    {
        const string message = error.what();
        EXPECT_NE(message.find("constraint 'x*x' (Equal): value 1, required value 2"), string::npos);
        EXPECT_NE(message.find("absolute miss 1 against tolerance"), string::npos);
        EXPECT_NE(message.find("consider Between"), string::npos);
    }
}


TEST(Feasibility, DiagnosticTracksRoundedCandidatesAndResetsBetweenRuns)
{
    MinimalApproximation setup({"x"}, {"y"});
    for (const auto solver : {ResponseOptimization::FeasibilitySolver::LevenbergMarquardt,
                              ResponseOptimization::FeasibilitySolver::GaussNewton})
    {
        RepairProbe probe(setup.network.get());
        probe.add_objective("x", Sense::Minimize);
        probe.add_constraint("x", Condition::AllowedSet, {1.0f, 2.0f});
        probe.add_constraint("x*x", Condition::Equal, {2.0f});
        probe.set_feasibility_solver(solver);
        probe.set_feasibility_rounds(2);
        auto repair = probe.make_feasibility_system(probe.calculate_domain());
        EXPECT_EQ(probe.get_feasible_points(VectorR::Ones(1), repair).first.size(), 0);
        const string message = probe.get_sampling_failure();
        EXPECT_NE(message.find("value 1, required value 2, absolute miss 1"), string::npos);
        EXPECT_GT(repair.get_closest_failure().first, 4e5);
        EXPECT_LT(repair.get_closest_failure().first, 6e5);
        EXPECT_EQ(probe.get_feasible_points(VectorR::Constant(1, 2.0f), repair).first.size(), 0);
        EXPECT_EQ(probe.get_sampling_failure(), message);
        probe.perform_response_optimization();
        EXPECT_EQ(probe.get_sampling_failure().find("constraint '"), string::npos);
        probe.get_feasible_points(VectorR::Ones(1));
        EXPECT_NE(probe.get_sampling_failure().find("constraint '"), string::npos);
        probe.set(setup.network.get());
        EXPECT_EQ(probe.get_sampling_failure().find("constraint '"), string::npos);
    }
}


TEST(Feasibility, DiagnosticRanksTheWorstMissOfEachCandidate)
{
    MinimalApproximation setup({"x"}, {"y"});
    RepairProbe probe(setup.network.get());
    probe.add_constraint("x*x", Condition::Equal, {1.0f});
    probe.add_constraint("(x+1)^2", Condition::Equal, {16.0f});
    FeasibilityRepairSystem repair(setup.network.get(), probe.constraints, probe.calculate_domain(), 0);
    for (const float value : {3.0f, 1.0f, 2.0f})
        EXPECT_EQ(probe.get_feasible_points(VectorR::Constant(1, value), repair).first.size(), 0);
    EXPECT_NE(probe.get_sampling_failure().find("constraint '(x+1)^2' (Equal): value 4"), string::npos);
    EXPECT_GT(repair.get_closest_failure().first, 7e5);
    EXPECT_LT(repair.get_closest_failure().first, 8e5);
}


TEST(Feasibility, DiagnosticUsesDiscreteToleranceAndSkipsNonfiniteValues)
{
    MinimalApproximation setup({"x", "z"}, {"y"});
    for (const auto condition : {Condition::Integer, Condition::AllowedSet})
    {
        RepairProbe probe(setup.network.get());
        probe.add_constraint("x+z", condition, condition == Condition::Integer ? vector<float>{} : vector<float>{0.0f, 1.0f});
        FeasibilityRepairSystem repair(setup.network.get(), probe.constraints, probe.calculate_domain(), 0);
        EXPECT_EQ(probe.get_feasible_points(VectorR::Constant(2, 0.125f), repair).first.size(), 0);
        EXPECT_NE(probe.get_sampling_failure().find("absolute miss 0.25 against tolerance 0.001"), string::npos);
        EXPECT_GT(repair.get_closest_failure().first, 249.0);
        EXPECT_LT(repair.get_closest_failure().first, 250.0);
    }
    RepairProbe probe(setup.network.get());
    probe.add_constraint("sqrt(-x-1)", Condition::Equal, {1.0f});
    probe.calculate_domain();
    EXPECT_EQ(probe.get_feasible_points(VectorR::Ones(2)).first.size(), 0);
    EXPECT_EQ(probe.get_sampling_failure().find("constraint '"), string::npos);
}


TEST(Feasibility, DiagnosticRetainsMissesSmallerThanTheValuePrecision)
{
    MinimalApproximation setup({"x"}, {"y"});
    const vector<ResponseOptimization::Constraint> constraints{
        {compile_expression("x", setup.network.get()), Condition::Equal, {1e20f}}};
    FeasibilityRepairSystem repair(setup.network.get(), constraints,
        {VectorR::Zero(1), VectorR::Constant(1, 2e20f)}, 0, 1, 1e-30f);
    EXPECT_EQ(repair.solve(VectorR::Constant(1, nextafter(1e20f, MAX))).first.size(), 0);
    EXPECT_TRUE(isfinite(repair.get_closest_failure().first));
    EXPECT_GT(repair.get_closest_failure().first, 1.0);
    EXPECT_FALSE(repair.get_closest_failure().second.empty());
}


TEST(Feasibility, RepairBudgetsControlNonlinearRepair)
{
    MinimalApproximation setup({"x1"}, {"y"});
    const VectorR point = VectorR::Constant(1, 4.0f);

    for (const float scale : {1.0f, 1000.0f})
    {
        SCOPED_TRACE(scale);
        RepairProbe probe(setup.network.get());
        probe.add_constraint(to_string(scale) + " * x1^2", Condition::Equal, {scale});
        probe.calculate_domain();
        probe.set_feasibility_evaluations(1);
        probe.set_feasibility_rounds(1);

        EXPECT_EQ(probe.get_feasible_points(point).first.size(), 0);

        probe.set_feasibility_evaluations(50);
        const VectorR repaired = probe.get_feasible_points(point).first;
        ASSERT_EQ(repaired.size(), 1);
        EXPECT_NEAR(repaired(0), 1.0f, 1e-4f);

        probe.set_feasibility_evaluations(1);
        probe.set_feasibility_rounds(10);
        for (const float start : {4.0f, 9.0f})
        {
            const VectorR retried = probe.get_feasible_points(VectorR::Constant(1, start)).first;
            ASSERT_EQ(retried.size(), 1);
            EXPECT_NEAR(retried(0), 1.0f, 1e-4f);
        }
    }
}


TEST(Feasibility, FeasiblePointsPassThroughWithMinimalBudgets)
{
    MinimalApproximation setup({"x1"}, {"y"});
    RepairProbe probe(setup.network.get());
    probe.set_feasibility_evaluations(1);
    probe.set_feasibility_rounds(1);

    const VectorR point = VectorR::Constant(1, 2.0f);
    const VectorR expected_output = setup.network->calculate_outputs(point.transpose()).row(0).transpose();

    for (const bool constrained : {false, true})
    {
        if (constrained) probe.add_constraint("x1^2", Condition::Between, {1.0f, 9.0f});
        probe.calculate_domain();

        const auto [input, output] = probe.get_feasible_points(point);
        ASSERT_EQ(input.size(), point.size());
        EXPECT_EQ((input - point).squaredNorm(), 0.0f);
        ASSERT_EQ(output.size(), expected_output.size());
        EXPECT_TRUE(output.isApprox(expected_output, 1e-6f));
    }
}


TEST(Feasibility, InputOnlyRepairAvoidsIntermediateNetworkCalls)
{
    MinimalApproximation setup({"x1"}, {"y"});
    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1^2", Condition::Equal, {1.0f});
    probe.calculate_domain();
    probe.set_feasibility_rounds(1);

    for (const bool succeeds : {false, true})
    {
        SCOPED_TRACE(succeeds);
        probe.set_feasibility_evaluations(succeeds ? 50 : 1);

        const bool profiling_enabled = profiler::is_enabled();
        profiler::set_enabled(true);
        const long previous_calls = profiler::stats().call_count("fp:set");

        pair<VectorR, VectorR> result;
        EXPECT_NO_THROW(result = probe.get_feasible_points(VectorR::Constant(1, 4.0f)));

        const long calls = profiler::stats().call_count("fp:set") - previous_calls;
        profiler::set_enabled(profiling_enabled);

        EXPECT_EQ(calls, succeeds ? 1 : 0);
        ASSERT_EQ(result.first.size(), succeeds ? 1 : 0);
        ASSERT_EQ(result.second.size(), succeeds ? 1 : 0);

        if (succeeds)
        {
            EXPECT_NEAR(result.first(0), 1.0f, 1e-4f);
            const VectorR expected = setup.network->calculate_outputs(result.first.transpose()).row(0).transpose();
            EXPECT_TRUE(result.second.isApprox(expected, 1e-6f));
        }
    }
}


TEST(Feasibility, OutputCoupledRepairReturnsOutputAtTheRepairedPoint)
{
    MinimalApproximation setup({"x1"}, {"y"});
    setup.network->set_parameters(VectorR::Constant(setup.network->get_parameters_buffer_size(), 0.1f));

    const float target = setup.network->calculate_outputs(MatrixR::Constant(1, 1, 1.0f))(0, 0);

    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1^2", Condition::Equal, {1.0f});
    probe.calculate_domain();

    ASSERT_EQ(probe.get_feasible_points(VectorR::Constant(1, 4.0f)).first.size(), 1);
    probe.add_constraint("x1^2 + y", Condition::Equal, {1.0f + target});
    probe.calculate_domain();

    for (const float start : {4.0f, 10.0f})
    {
        SCOPED_TRACE(start);
        const auto [input, output] = probe.get_feasible_points(VectorR::Constant(1, start));

        ASSERT_EQ(input.size(), 1);
        ASSERT_EQ(output.size(), 1);
        EXPECT_NEAR(input(0), 1.0f, 1e-4f);

        const VectorR expected = setup.network->calculate_outputs(input.transpose()).row(0).transpose();
        EXPECT_TRUE(output.isApprox(expected, 1e-6f));
        EXPECT_NEAR(input(0)*input(0) + output(0), 1.0f + target, 1e-4f);
    }
}


TEST(Feasibility, CopiesKeepTheirOwnConstraints)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    RepairProbe source(setup.network.get());
    source.add_constraint("x1 + x2", Condition::Equal, {5.0f});
    source.calculate_domain();

    RepairProbe copied(source);
    RepairProbe assigned(nullptr);
    assigned = source;

    source.add_constraint("x1 + x2", Condition::Equal, {12.0f});

    VectorR point(2); point << 2.0f, 3.0f;

    for (const RepairProbe* copy : {&copied, &assigned})
    {
        SCOPED_TRACE(copy == &copied ? "copy construction" : "copy assignment");

        const auto [input, output] = copy->get_feasible_points(point);
        ASSERT_EQ(input.size(), point.size());
        EXPECT_EQ((input - point).squaredNorm(), 0.0f);
        EXPECT_EQ(output.size(), 1);

        const VectorR repaired = copy->get_feasible_points(VectorR::Constant(2, 4.0f)).first;
        ASSERT_EQ(repaired.size(), point.size());
        EXPECT_NEAR(repaired.sum(), 5.0f, 1e-4f);
    }
}


TEST(Feasibility, RepeatedRoundedCycleStopsBeforeTheRoundBudget)
{
    MinimalApproximation setup({"x1"}, {"y"}, 0.0f, 1.0f);
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));

    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1", Condition::AllowedSet, {0.0f, 1.0f});
    probe.add_constraint("(x1 - 0.5)^2 + y", Condition::Equal, {-0.35f});
    probe.calculate_domain();
    probe.set_feasibility_evaluations(1);

    vector<long> calls;
    for (const Index rounds : {Index(1), Index(2), Index(100)})
    {
        SCOPED_TRACE(rounds);
        probe.set_feasibility_rounds(rounds);
        const bool profiling_enabled = profiler::is_enabled();
        profiler::set_enabled(true);
        const long previous_calls = profiler::stats().call_count("fp:set");

        pair<VectorR, VectorR> result;
        EXPECT_NO_THROW(result = probe.get_feasible_points(VectorR::Zero(1)));

        calls.push_back(profiler::stats().call_count("fp:set") - previous_calls);
        profiler::set_enabled(profiling_enabled);
        EXPECT_EQ(result.first.size(), 0);
        EXPECT_EQ(result.second.size(), 0);
    }

    EXPECT_GT(calls[1], calls[0]);
    EXPECT_EQ(calls[2], calls[1]);
}


TEST(CardinalityRepair, ReturnsOnlyKSparsePoints)
{
    set_seed(1234);

    for (const Index variables_number : {Index(4), Index(8)})
        for (const Index kept : {Index(1), variables_number/2, variables_number - 1})
            for (const bool budget : {false, true})
            {
                vector<string> names;

                for (Index i = 1; i <= variables_number; i++)
                    names.push_back("x" + to_string(i));

                MinimalApproximation setup(names, {"y"}, 0.0f, 10.0f);

                RepairProbe probe(setup.network.get());

                string list;
                string sum;

                for (const string& name : names)
                {
                    list += (list.empty() ? "" : "; ") + name;
                    sum += (sum.empty() ? "" : " + ") + name;
                }

                probe.add_constraint(list, Condition::Cardinality, {float(kept)});

                const float total = 2.5f*float(kept);

                if (budget) probe.add_constraint(sum, Condition::Equal, {total});

                const pair<VectorR, VectorR> domain = probe.calculate_domain();

                Index repaired = 0;

                for (Index draw = 0; draw < 50; draw++)
                {
                    const auto [input, output] = probe.get_feasible_points(probe.calculate_random_input(domain));

                    if (input.size() == 0) continue;

                    repaired++;

                    const Index in_play = Index((input.array() != 0.0f).count());

                    EXPECT_LE(in_play, kept) << "n=" << variables_number << " k=" << kept
                                             << " budget=" << budget << " draw=" << draw;

                    if (budget) EXPECT_NEAR(input.sum(), total, 1e-2f*total);
                }

                EXPECT_GT(repaired, 0) << "n=" << variables_number << " k=" << kept << " budget=" << budget;
            }
}


TEST(CardinalityRepair, CutRanksMembersByTheirShareOfTheirRange)
{
    MinimalApproximation setup({"x1", "x2", "x3", "x4"}, {"y"});

    vector<Descriptives> ranges = make_descriptives(4, 0.0f, 100.0f);
    ranges[2] = ranges[3] = Descriptives(0.0f, 1.0f, 0.5f, 0.25f);
    static_cast<Scaling*>(setup.network->get_first("Scaling"))->set_descriptives(ranges);

    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1; x2; x3; x4", Condition::Cardinality, {2.0f});
    probe.calculate_domain();

    VectorR point(4);
    point << 10.0f, 10.0f, 0.9f, 0.9f;

    const auto [input, output] = probe.get_feasible_points(point);

    ASSERT_EQ(input.size(), 4);
    EXPECT_EQ(input(0), 0.0f) << "x1 holds 10% of its range and should have been cut";
    EXPECT_EQ(input(1), 0.0f) << "x2 holds 10% of its range and should have been cut";
    EXPECT_GT(input(2), 0.0f) << "x3 holds 90% of its range and should have been kept";
    EXPECT_GT(input(3), 0.0f) << "x4 holds 90% of its range and should have been kept";
}


TEST(CardinalityRepair, CutKeepsMembersTheBoxCannotSwitchOff)
{
    vector<string> names;
    for (Index i = 1; i <= 18; i++) names.push_back("x" + to_string(i));

    MinimalApproximation setup(names, {"y"});

    string list;
    for (const string& name : names) list += (list.empty() ? "" : "; ") + name;

    RepairProbe probe(setup.network.get());
    probe.add_constraint(list, Condition::Cardinality, {4.0f});
    probe.calculate_domain();

    probe.input_bounds.first(17) = 1.0f;

    VectorR point = VectorR::Zero(18);
    point.head(4).setConstant(5.0f);
    point(17) = 1.0f;

    const auto [input, output] = probe.get_feasible_points(point);

    ASSERT_EQ(input.size(), 18) << "keeping x18 and three others is feasible, so a point is expected";
    EXPECT_LE((input.array() != 0.0f).count(), 4) << "more than the budget in play: " << input.transpose();
    EXPECT_GE(input(17), 1.0f) << "x18 cannot leave [1, 10], so it has to be one of the four kept";
}


TEST(CardinalityRepair, TooManyMembersTheBoxCannotSwitchOffIsInfeasible)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"});

    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1; x2; x3", Condition::Cardinality, {1.0f});
    probe.calculate_domain();

    probe.input_bounds.first(1) = 1.0f;
    probe.input_bounds.first(2) = 1.0f;

    VectorR point(3);
    point << 5.0f, 4.0f, 2.0f;

    EXPECT_EQ(probe.get_feasible_points(point).first.size(), 0) << "two members cannot be off and the budget is one";
}


TEST_P(ResponseDriver, CardinalityWithARoomyBudgetRestrictsNothing)
{
    MinimalApproximation setup({"x1", "x2", "x3"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("x1 + x2 + x3", Sense::Maximize);
    optimization->add_constraint("x1; x2; x3", Condition::Cardinality, {3.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        for (Index j = 0; j < 3; j++)
            EXPECT_GT(results(i, j), 1e-2f) << "row " << i << " switched off x" << j + 1 << " for nothing";
}


TEST_P(ResponseDriver, IntegerHoldsAlongsideABandOnTheSameVariable)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 0.0f, 10.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y", Sense::Minimize);
    optimization->add_constraint("x1", Condition::Integer);
    optimization->add_constraint("x1", Condition::Between, {2.5f, 7.5f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
    {
        const float x1 = results(i, 0);

        EXPECT_LT(abs(x1 - round(x1)), 1e-5f) << "row " << i << " x1=" << x1 << " is not a whole number";

        EXPECT_GE(x1, 3.0f - 1e-5f) << "row " << i << " x1=" << x1;
        EXPECT_LE(x1, 7.0f + 1e-5f) << "row " << i << " x1=" << x1;
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


TEST_P(ResponseDriver, CategoricalConstraintSelectsRequestedCategory)
{
    for (const string& name : {"copper", "material.copper"})
    {
        SCOPED_TRACE(name);
        set_seed(1234);
        CategoricalApproximation setup({"x1", "x2"}, "material", {"steel", "copper", "brass"});

        const auto optimization = make_driver(GetParam(), setup.network.get());

        optimization->add_objective("y1", Sense::Minimize);
        optimization->add_constraint(name, Condition::Equal, {1.0f});

        const MatrixR results = optimization->perform_response_optimization();

        ASSERT_GT(results.rows(), 0);

        for (Index i = 0; i < results.rows(); i++)
            EXPECT_EQ(read_category(results, i, 2, 3), 1) << "row " << i;
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


TEST_P(ResponseDriver, RejectsUndefinedAndOverflowedConstraints)
{
    for (const bool overflow : {false, true})
    {
        MinimalApproximation setup({"x1", "x2"}, {"y"},
                                   overflow ? 100.0f : -2.0f,
                                   overflow ? 101.0f : -1.0f);

        const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

        optimization->set_points_number(4);
        optimization->set_iterations_number(2);

        optimization->add_objective("x1", Sense::Minimize);
        optimization->add_constraint(overflow ? "exp(x1)" : "sqrt(x1)", Condition::GreaterEqual, {1.0f});

        EXPECT_THROW(optimization->perform_response_optimization(), runtime_error);
    }
}


TEST_P(ResponseDriver, FiniteNonlinearConstraintsStillProduceFeasiblePoints)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"}, 1.0f, 4.0f);

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("x1", Sense::Minimize);
    optimization->add_constraint("sqrt(x1)", Condition::GreaterEqual, {1.0f});

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_EQ(results.rows(), 1);

    EXPECT_TRUE(results.allFinite());

    EXPECT_GE(sqrt(results(0, 0)), 1.0f - EPSILON);
}


TEST(ResponseOptimizationSetup, RejectsNonFiniteSettingsAndInvalidInputBounds)
{
    MinimalApproximation setup({"x1"}, {"y"});
    DomainContraction optimization(setup.network.get());

    for (const float value : {numeric_limits<float>::quiet_NaN(),
                              numeric_limits<float>::infinity(),
                              -numeric_limits<float>::infinity()})
    {
        SCOPED_TRACE(value);
        EXPECT_THROW(optimization.add_constraint("x1", Condition::LessEqual, {value}), runtime_error);
        EXPECT_THROW(optimization.add_objective("x1", Sense::Fixed, value), runtime_error);
        EXPECT_THROW(optimization.set_feasibility_margin_factor(value), runtime_error);
        EXPECT_THROW(optimization.set_contraction_factor(value), runtime_error);
    }

    RepairProbe probe(setup.network.get());
    for (const auto& [lower, upper] : vector<pair<float, float>>{
             {numeric_limits<float>::quiet_NaN(), 1.0f},
             {0.0f, numeric_limits<float>::infinity()}, {2.0f, 1.0f}})
    {
        static_cast<Scaling*>(setup.network->get_first("Scaling"))
            ->set_descriptives(make_descriptives(1, lower, upper));
        EXPECT_THROW(probe.calculate_domain(), runtime_error);
    }
}


TEST(ResponseOptimizationSetup, RebindingClearsCompiledObjectivesAndConstraints)
{
    MinimalApproximation original({"x1", "x2"}, {"y"});
    MinimalApproximation replacement({"x2"}, {"y"});
    DomainContraction optimization(original.network.get());
    optimization.set_points_number(2);
    optimization.set_iterations_number(1);
    optimization.add_objective("x2", Sense::Maximize);
    optimization.add_constraint("x2", Condition::Equal, {7.0f});

    for (const float target : {3.0f, 2.0f})
    {
        optimization.set(replacement.network.get());
        EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
        optimization.add_objective("x2", Sense::Minimize);
        optimization.add_constraint("x2", Condition::Equal, {target});
        const MatrixR results = optimization.perform_response_optimization();
        ASSERT_EQ(results.rows(), 1);
        EXPECT_FLOAT_EQ(results(0, 0), target);
    }

    optimization.set();
    EXPECT_THROW(optimization.perform_response_optimization(), runtime_error);
}


TEST(ResponseOptimizationSetup, RejectsCategoricalBoundsWithoutAOneHotAssignment)
{
    CategoricalApproximation setup({}, "material", {"first", "second", "third"});

    for (const Index scenario : {Index(0), Index(1), Index(2), Index(3)})
    {
        SCOPED_TRACE(scenario);
        RepairProbe probe(setup.network.get());
        if (scenario == 0)
            for (const string& category : {"first", "second", "third"})
                probe.add_constraint(category, Condition::LessEqual, {0.0f});
        else if (scenario == 1)
            for (const string& category : {"first", "second"})
                probe.add_constraint(category, Condition::GreaterEqual, {0.5f});
        else if (scenario == 2)
            probe.add_constraint("first", Condition::Between, {0.25f, 0.75f});
        else
            probe.add_constraint("first", Condition::AllowedSet, {0.2f, 0.8f});

        EXPECT_THROW(probe.calculate_domain(), runtime_error);
    }

    RepairProbe probe(setup.network.get());
    probe.add_constraint("first", Condition::AllowedSet, {0.0f, 0.2f, 1.0f});
    probe.calculate_domain();
    for (const Index category : {Index(0), Index(1)})
    {
        VectorR point = VectorR::Zero(3);
        point(category) = 0.8f;
        const VectorR repaired = probe.get_feasible_points(point).first;
        ASSERT_EQ(repaired.size(), 3);
        EXPECT_FLOAT_EQ(repaired(category), 1.0f);
        EXPECT_FLOAT_EQ(repaired.sum(), 1.0f);
        EXPECT_TRUE(((repaired.array() == 0.0f) || (repaired.array() == 1.0f)).all());
    }
}


TEST_P(ResponseDriver, RejectsOverflowingSamplingBudgetBeforeAllocation)
{
    MinimalApproximation setup({"x1"}, {"y"});
    const auto optimization = make_driver(GetParam(), setup.network.get());
    optimization->add_objective("x1", Sense::Minimize);
    optimization->set_points_number(numeric_limits<Index>::max());
    optimization->set_sampling_budget_multiplier(2);
    EXPECT_THROW(optimization->perform_response_optimization(), runtime_error);
}


TEST_P(ResponseDriver, RejectsObjectivesThatAreNeverFinite)
{
    MinimalApproximation setup({"x1"}, {"y"});
    for (const string& expression : {"sqrt(-x1 - 1)", "exp(100*x1 + 100)"})
        for (const bool multiple : {false, true})
        {
            SCOPED_TRACE(expression);
            SCOPED_TRACE(multiple);
            const auto optimization = make_driver(GetParam(), setup.network.get());
            optimization->add_objective(expression, Sense::Maximize);
            if (multiple) optimization->add_objective("x1", Sense::Minimize);
            optimization->set_points_number(2);
            optimization->set_iterations_number(1);
            EXPECT_THROW(optimization->perform_response_optimization(), runtime_error);
        }
}


TEST_P(ResponseDriver, ResamplesCandidatesWithNonFiniteObjectives)
{
    MinimalApproximation setup({"x1"}, {"y"});
    for (const bool multiple : {false, true})
    {
        SCOPED_TRACE(multiple);
        const auto optimization = make_driver(GetParam(), setup.network.get());
        optimization->add_objective("sqrt(x1 - 5)", Sense::Maximize);
        if (multiple) optimization->add_objective("x1", Sense::Minimize);
        optimization->set_points_number(4);
        optimization->set_iterations_number(2);
        optimization->set_sampling_budget_multiplier(20);
        const MatrixR results = optimization->perform_response_optimization();
        ASSERT_GT(results.rows(), 0);
        EXPECT_TRUE(results.allFinite());
        EXPECT_TRUE((results.col(0).array() >= 5.0f).all());
    }
}


TEST(Feasibility, RejectsNonFiniteNetworkOutputsWithoutOutputConstraints)
{
    MinimalApproximation setup({"x1"}, {"y"});
    setup.network->set_parameters(VectorR::Constant(setup.network->get_parameters_buffer_size(),
                                                   numeric_limits<float>::quiet_NaN()));
    RepairProbe probe(setup.network.get());
    probe.add_objective("x1", Sense::Maximize);
    probe.calculate_domain();
    EXPECT_EQ(probe.get_feasible_points(VectorR::Constant(1, 5.0f)).first.size(), 0);
}


TEST(Feasibility, AllowedExpressionsUseDistanceToTheNearestValue)
{
    for (const bool fixed : {false, true})
        for (const bool coupled : {false, true})
        {
            SCOPED_TRACE(fixed);
            SCOPED_TRACE(coupled);
            MinimalApproximation setup({"x1", "x2"}, {"y"}, fixed ? 0.25f : 0.0f, fixed ? 0.25f : 10.0f);
            setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
            RepairProbe probe(setup.network.get());
            probe.add_constraint(coupled ? "x1 + x2 + y" : "x1 + x2",
                                 Condition::AllowedSet, {0.0f, 1.0f, 10000000.0f});
            probe.calculate_domain();
            const auto [input, output] = probe.get_feasible_points(VectorR::Constant(2, 0.25f));
            if (fixed)
                EXPECT_EQ(input.size(), 0);
            else
            {
                ASSERT_EQ(input.size(), 2);
                const float value = input.sum() + (coupled ? output(0) : 0.0f);
                EXPECT_LE(min(abs(value), abs(value - 1.0f)), 1.01e-3f);
            }
        }
}


TEST(Feasibility, LargeIntegersRemainFeasibleAfterRounding)
{
    MinimalApproximation setup({"x1"}, {"y"}, 1000000.0f, 1000010.0f);
    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1", Condition::Integer);
    probe.calculate_domain();
    const VectorR repaired = probe.get_feasible_points(VectorR::Constant(1, 1000003.25f)).first;
    ASSERT_EQ(repaired.size(), 1);
    EXPECT_FLOAT_EQ(repaired(0), 1000003.0f);
}


TEST(Feasibility, AnalyticGradientRepairsANarrowInputRange)
{
    MinimalApproximation setup({"x1"}, {"y"}, 0.0f, 1e-4f);
    RepairProbe probe(setup.network.get());
    probe.add_constraint("(10000*x1)^2", Condition::Equal, {0.25f});
    probe.calculate_domain();
    const VectorR repaired = probe.get_feasible_points(VectorR::Constant(1, 9e-5f)).first;
    ASSERT_EQ(repaired.size(), 1);
    EXPECT_NEAR(repaired(0), 5e-5f, 1e-8f);
}


TEST(Feasibility, NumericalGradientFitsInsideAnOffsetInputRange)
{
    MinimalApproximation setup({"x1"}, {"y"}, 1000.0f, 1001.0f);
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    RepairProbe probe(setup.network.get());
    probe.add_constraint("x1 + y", Condition::Equal, {1000.25f});
    probe.calculate_domain();
    const VectorR repaired = probe.get_feasible_points(VectorR::Constant(1, 1000.5f)).first;
    ASSERT_EQ(repaired.size(), 1);
    EXPECT_NEAR(repaired(0), 1000.25f, 2e-3f);
}


TEST(Feasibility, NumericalProbesHandleExpressionsUndefinedInOneDirection)
{
    MinimalApproximation setup({"x1"}, {"y"}, -1.0f, 1.0f);
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    for (const bool coupled : {false, true})
    {
        SCOPED_TRACE(coupled);
        RepairProbe probe(setup.network.get());
        if (coupled)
            probe.add_constraint("sqrt(0.0005 - x1) + y", Condition::Equal, {0.5f});
        else
        {
            probe.add_constraint("x1 + y", Condition::Equal, {-0.25f});
            probe.add_constraint("sqrt(0.0005 - x1)", Condition::Between, {0.0f, 2.0f});
        }
        probe.calculate_domain();
        const VectorR repaired = probe.get_feasible_points(VectorR::Zero(1)).first;
        ASSERT_EQ(repaired.size(), 1);
        EXPECT_NEAR(repaired(0), coupled ? -0.2495f : -0.25f, 1e-4f);
    }
}


TEST(DomainContraction, KeepsPreviousResultsWhenALaterBatchHasNoFeasiblePoint)
{
    CategoricalApproximation setup({}, "material", {"accepted", "rejected"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    RepairProbe probe(setup.network.get());
    const auto domain = probe.calculate_domain();
    unsigned seed = 0;
    for (; seed < 1024; seed++)
    {
        set_seed(seed);
        const bool first_accepted = probe.calculate_random_input(domain)(0) == 1.0f;
        if (probe.calculate_random_input(domain)(1) == 1.0f && first_accepted) break;
    }
    ASSERT_LT(seed, 1024u);

    for (const bool multiple : {false, true})
    {
        SCOPED_TRACE(multiple);
        DomainContraction optimization(setup.network.get());
        optimization.add_objective("accepted", Sense::Maximize);
        if (multiple) optimization.add_objective("accepted", Sense::Minimize);
        optimization.add_constraint("accepted + y1", Condition::Equal, {1.0f});
        optimization.set_points_number(1);
        optimization.set_iterations_number(2);
        optimization.set_sampling_budget_multiplier(4);
        optimization.set_maximum_consecutive_failures(1);
        set_seed(seed);
        const MatrixR results = optimization.perform_response_optimization();
        ASSERT_EQ(results.rows(), 1);
        EXPECT_EQ(read_category(results, 0, 0, 2), 0);
    }
}


TEST(ResponseOptimizationAdvisedPoint, EmptyResultsOrMissingObjectivesReturnNoPoint)
{
    MinimalApproximation setup({"x1"}, {"y"});
    DomainContraction optimization(setup.network.get());
    const MatrixR results = MatrixR::Zero(1, 2);

    const auto [missing_index, missing_point] = optimization.get_advised_point(results);
    EXPECT_EQ(missing_index, -1);
    EXPECT_EQ(missing_point.size(), 0);

    optimization.add_objective("x1", Sense::Minimize);
    const auto [empty_index, empty_point] = optimization.get_advised_point(MatrixR(0, 2));
    EXPECT_EQ(empty_index, -1);
    EXPECT_EQ(empty_point.size(), 0);
}


TEST(ResponseOptimizationAdvisedPoint, NormalizesMixedObjectivesUsingSuppliedOutputs)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    DomainContraction optimization(setup.network.get());
    optimization.add_objective("x1", Sense::Minimize);
    optimization.add_objective("2*y + x2", Sense::Maximize);
    optimization.add_objective("x2", Sense::Minimize);

    MatrixR results(3, 3);
    results << 0.0f, 7.0f, 0.0f,
               4.0f, 7.0f, 60.0f,
               10.0f, 7.0f, 100.0f;

    const auto [index, point] = optimization.get_advised_point(results);
    EXPECT_EQ(index, 1);
    ASSERT_EQ(point.size(), results.cols());
    EXPECT_TRUE(point.isApprox(results.row(1).transpose()));
}


TEST(ResponseOptimizationAdvisedPoint, KeepsTheFirstTiedRowAndReturnsASingleCandidate)
{
    MinimalApproximation setup({"x1"}, {"y"});
    DomainContraction optimization(setup.network.get());
    optimization.add_objective("x1", Sense::Minimize);
    optimization.add_objective("y", Sense::Maximize);

    MatrixR results(2, 2);
    results << 0.0f, 3.0f,
               10.0f, 13.0f;

    const auto [tied_index, tied_point] = optimization.get_advised_point(results);
    EXPECT_EQ(tied_index, 0);
    ASSERT_EQ(tied_point.size(), results.cols());
    EXPECT_TRUE(tied_point.isApprox(results.row(0).transpose()));

    const auto [single_index, single_point] = optimization.get_advised_point(results.bottomRows(1));
    EXPECT_EQ(single_index, 0);
    ASSERT_EQ(single_point.size(), results.cols());
    EXPECT_TRUE(single_point.isApprox(results.row(1).transpose()));
}


TEST(ResponseOptimizationAdvisedPoint, FixedObjectiveSelectsTheExactOrNearestTarget)
{
    MinimalApproximation setup({"x1"}, {"y"});
    DomainContraction optimization(setup.network.get());
    optimization.add_objective("x1", Sense::Fixed, 5.0f);

    MatrixR results(3, 2);
    results << 0.0f, 10.0f,
               4.0f, 20.0f,
               8.0f, 30.0f;

    for (const float nearest : {4.0f, 5.0f})
    {
        SCOPED_TRACE(nearest);
        results(1, 0) = nearest;
        const auto [index, point] = optimization.get_advised_point(results);
        EXPECT_EQ(index, 1);
        ASSERT_EQ(point.size(), results.cols());
        EXPECT_TRUE(point.isApprox(results.row(1).transpose()));
    }
}


INSTANTIATE_TEST_SUITE_P(Drivers,
                         ResponseDriver,
                         testing::Values(Driver::Contraction, Driver::Genetic),
                         driver_name);


TEST(GeneticResponse, KeepsUsefulPartialMutationBatch)
{
    CategoricalApproximation setup({}, "material", {"first", "best", "rejected"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    RepairProbe probe(setup.network.get());
    const auto domain = probe.calculate_domain();
    GeneticResponse optimization(setup.network.get());
    optimization.add_objective("best", Sense::Maximize);
    optimization.add_constraint("first + best + y1", Condition::Equal, {1.0f});
    optimization.set_points_number(2);
    optimization.set_iterations_number(1);
    optimization.set_sampling_budget_multiplier(1);
    optimization.set_maximum_consecutive_failures(1);

    bool retained_best = false;
    for (unsigned seed = 0; seed < 1024 && !retained_best; seed++)
    {
        SCOPED_TRACE(seed);
        set_seed(seed);
        if (probe.calculate_random_input(domain)(0) != 1.0f
         || probe.calculate_random_input(domain)(0) != 1.0f) continue;

        string warnings;
        const auto previous_level = logging::level();
        logging::set_level(logging::Level::Warning);
        logging::set_sink([&](logging::Level, string_view message) { warnings += message; });
        set_seed(seed);
        MatrixR results;
        EXPECT_NO_THROW(results = optimization.perform_response_optimization());
        logging::set_sink({});
        logging::set_level(previous_level);

        ASSERT_EQ(results.rows(), 1);
        ASSERT_EQ(results.cols(), 4);
        EXPECT_TRUE(results.allFinite());
        EXPECT_EQ(results(0, 0) + results(0, 1), 1.0f);
        retained_best = warnings.find("only 1 of 2 children survived mutation") != string::npos
                     && results(0, 1) == 1.0f;
    }
    EXPECT_TRUE(retained_best);
}


TEST(GeneticResponse, ContinuesFromPartialRecombinationBatch)
{
    MinimalApproximation setup({"x1"}, {"y"}, -1.0f, 1.0f);
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    RepairProbe probe(setup.network.get());
    const auto domain = probe.calculate_domain();
    const auto objective = compile_expression("sqrt(sin(100*x1))", setup.network.get(), "Objective");
    GeneticResponse optimization(setup.network.get());
    optimization.add_objective("sqrt(sin(100*x1))", Sense::Maximize);
    optimization.set_points_number(2);
    optimization.set_iterations_number(1);
    optimization.set_sampling_budget_multiplier(1);
    optimization.set_maximum_consecutive_failures(1);

    bool continued = false;
    for (unsigned seed = 0; seed < 1024 && !continued; seed++)
    {
        SCOPED_TRACE(seed);
        set_seed(seed);
        if (!isfinite(objective.evaluate(probe.calculate_random_input(domain), VectorR{}))
         || !isfinite(objective.evaluate(probe.calculate_random_input(domain), VectorR{}))) continue;

        string warnings;
        const auto previous_level = logging::level();
        logging::set_level(logging::Level::Warning);
        logging::set_sink([&](logging::Level, string_view message) { warnings += message; });
        set_seed(seed);
        MatrixR results;
        EXPECT_NO_THROW(results = optimization.perform_response_optimization());
        logging::set_sink({});
        logging::set_level(previous_level);

        ASSERT_EQ(results.rows(), 1);
        ASSERT_EQ(results.cols(), 2);
        EXPECT_TRUE(results.allFinite());
        EXPECT_GE(results(0, 0), -1.0f);
        EXPECT_LE(results(0, 0), 1.0f);
        EXPECT_TRUE(isfinite(objective.evaluate(results.row(0).head(1).transpose(), VectorR{})));
        const size_t partial = warnings.find("only 1 of 2 children could be recombined");
        continued = partial != string::npos
                 && warnings.find("children survived mutation", partial) != string::npos;
    }
    EXPECT_TRUE(continued);
}


TEST(ResponseOptimizationPareto, DominatedOutliersDoNotChangeTradeoffs)
{
    RepairProbe probe(nullptr);
    MatrixR scores(4, 2);
    scores << 0.0f, 1.0f,
              1.0f, 0.0f,
              -2e6f, -2e6f,
              0.0f, 1.0f;
    EXPECT_EQ(probe.calculate_pareto_front(MatrixR(scores.topRows(2))), (vector<Index>{0, 1}));
    EXPECT_EQ(probe.calculate_pareto_front(scores), (vector<Index>{0, 1}));
    scores(3, 0) = 1e-8f;
    EXPECT_EQ(probe.calculate_pareto_front(scores), (vector<Index>{1, 3}));
}


TEST(ResponseOptimizationPareto, HandlesExtremeFiniteScores)
{
    RepairProbe probe(nullptr);
    MatrixR scores(3, 2);
    scores << -2e38f, -2e38f,
              2e38f, 2e38f,
              2e38f, 2e38f;
    EXPECT_EQ(probe.calculate_pareto_front(scores), (vector<Index>{1}));
    scores << -2e38f, 2e38f,
              0.0f, 0.0f,
              2e38f, -2e38f;
    EXPECT_EQ(probe.calculate_pareto_front(scores), (vector<Index>{0, 1, 2}));
}


TEST(ResponseOptimizationAdvisedPoint, HandlesWideAndTinyFiniteScoreRanges)
{
    MinimalApproximation setup({"x"}, {"first", "second", "constant"});
    RepairProbe probe(setup.network.get());
    for (const string name : {"first", "second", "constant"})
        probe.add_objective(name, Sense::Maximize);
    for (const float scale : {2e38f, 1e-40f})
    {
        SCOPED_TRACE(scale);
        MatrixR results(4, 4);
        results << 0.0f, -scale, scale, 7.0f,
                   1.0f, 0.5f*scale, -0.5f*scale, 7.0f,
                   2.0f, 0.0f, 0.0f, 7.0f,
                   3.0f, scale, -scale, 7.0f;
        const auto [index, point] = probe.get_advised_point(results);
        EXPECT_EQ(index, 2);
        EXPECT_TRUE(point.isApprox(results.row(2).transpose()));
    }
}


TEST(GeneticResponse, FixedVariablesDoNotSuppressMutationOfFreeInput)
{
    vector<string> names{"free_input"};
    for (Index i = 1; i < 32; i++) names.push_back("fixed_" + to_string(i));
    CategoricalApproximation setup(names, "material", {"kept", "closed"});
    setup.network->set_parameters(VectorR::Zero(setup.network->get_parameters_buffer_size()));
    RepairProbe probe(setup.network.get());
    pair<VectorR, VectorR> domain{VectorR::Zero(34), VectorR::Zero(34)};
    domain.second(0) = 10.0f;
    domain.first(32) = domain.second(32) = 1.0f;
    unsigned seed = 0;
    for (; seed < 1024; seed++)
    {
        set_seed(seed);
        if (probe.calculate_random_input(domain)(0) < 1.0f) break;
    }
    ASSERT_LT(seed, 1024u);

    for (const bool all_fixed : {false, true})
    {
        SCOPED_TRACE(all_fixed);
        GeneticResponse optimization(setup.network.get());
        optimization.add_objective("free_input", Sense::Maximize);
        for (Index i = all_fixed ? 0 : 1; i < 32; i++)
            optimization.add_constraint(names[size_t(i)], Condition::Equal, {0.0f});
        optimization.add_constraint("kept", Condition::Equal, {1.0f});
        optimization.set_points_number(1);
        optimization.set_iterations_number(all_fixed ? 1 : 64);
        optimization.set_sampling_budget_multiplier(1);
        set_seed(seed);
        const MatrixR results = optimization.perform_response_optimization();
        ASSERT_EQ(results.rows(), 1);
        ASSERT_EQ(results.cols(), 35);
        EXPECT_TRUE(results.allFinite());
        EXPECT_TRUE(results.row(0).segment(1, 31).isZero());
        EXPECT_EQ(read_category(results, 0, 32, 2), 0);
        if (all_fixed) EXPECT_FLOAT_EQ(results(0, 0), 0.0f);
        else EXPECT_GT(results(0, 0), 8.0f);
    }
}
