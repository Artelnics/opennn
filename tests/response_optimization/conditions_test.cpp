//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N D I T I O N   T E S T S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

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
// The same conditions against the trained concrete network are in
// concrete_scenarios_test.cpp; the expression language they are written in is in
// expression_test.cpp.

#include "tests/pch.h"

#include "tests/response_optimization/synthetic_fixture.h"

#include "opennn/core/profiler.h"
#include "opennn/response_optimization/domain_contraction.h"
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


// Reaches the repair without a solver around it.

class RepairProbe : public ResponseOptimization
{
public:

    explicit RepairProbe(Network* network) : ResponseOptimization(network) {}

    using ResponseOptimization::calculate_domain;
    using ResponseOptimization::calculate_random_input;
    using ResponseOptimization::solve;

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


// The list used to be read as a sum, which quietly dropped coefficients and repeated members.

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
        // With the only input fixed, each unsuccessful repair needs one inference.
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

        // Find the desired successes/failures without assuming a platform's RNG sequence.
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
        if (interrupted && GetParam() == Driver::Genetic)
        {
            EXPECT_THROW(optimization->perform_response_optimization(), runtime_error);
            continue;
        }

        MatrixR results;
        ASSERT_NO_THROW(results = optimization->perform_response_optimization());
        ASSERT_EQ(results.rows(), 1);
        EXPECT_EQ(read_category(results, 0, 0, 3), interrupted ? 0 : 1);
    }
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

        EXPECT_EQ(probe.solve(point).first.size(), 0);

        probe.set_feasibility_evaluations(50);
        const VectorR repaired = probe.solve(point).first;
        ASSERT_EQ(repaired.size(), 1);
        EXPECT_NEAR(repaired(0), 1.0f, 1e-4f);

        probe.set_feasibility_evaluations(1);
        probe.set_feasibility_rounds(10);
        for (const float start : {4.0f, 9.0f})
        {
            const VectorR retried = probe.solve(VectorR::Constant(1, start)).first;
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

        const auto [input, output] = probe.solve(point);
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
        EXPECT_NO_THROW(result = probe.solve(VectorR::Constant(1, 4.0f)));

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

    // Adding the output row between solves must update which expressions need inference.
    ASSERT_EQ(probe.solve(VectorR::Constant(1, 4.0f)).first.size(), 1);
    probe.add_constraint("x1^2 + y", Condition::Equal, {1.0f + target});
    probe.calculate_domain();

    for (const float start : {4.0f, 10.0f})
    {
        SCOPED_TRACE(start);
        const auto [input, output] = probe.solve(VectorR::Constant(1, start));

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

        const auto [input, output] = copy->solve(point);
        ASSERT_EQ(input.size(), point.size());
        EXPECT_EQ((input - point).squaredNorm(), 0.0f);
        EXPECT_EQ(output.size(), 1);

        const VectorR repaired = copy->solve(VectorR::Constant(2, 4.0f)).first;
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
        EXPECT_NO_THROW(result = probe.solve(VectorR::Zero(1)));

        calls.push_back(profiler::stats().call_count("fp:set") - previous_calls);
        profiler::set_enabled(profiling_enabled);
        EXPECT_EQ(result.first.size(), 0);
        EXPECT_EQ(result.second.size(), 0);
    }

    // Each short LM step rounds to the other endpoint: 0 -> 1 -> 0.
    EXPECT_GT(calls[1], calls[0]);
    EXPECT_EQ(calls[2], calls[1]);
}


// The repair on its own, from random starts: whatever it returns has at most k counted inputs
// that are not exactly zero, alone and beside a budget row that couples every counted input.

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
                    const auto [input, output] = probe.solve(probe.calculate_random_input(domain));

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


// An expression that is undefined over the whole box, or that overflows on it, leaves no
// feasible point to repair towards, and the run has to say so rather than return a number.

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
        const VectorR repaired = probe.solve(point).first;
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
    EXPECT_EQ(probe.solve(VectorR::Constant(1, 5.0f)).first.size(), 0);
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
            const auto [input, output] = probe.solve(VectorR::Constant(2, 0.25f));
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
    const VectorR repaired = probe.solve(VectorR::Constant(1, 1000003.25f)).first;
    ASSERT_EQ(repaired.size(), 1);
    EXPECT_FLOAT_EQ(repaired(0), 1000003.0f);
}


TEST(Feasibility, AnalyticGradientRepairsANarrowInputRange)
{
    MinimalApproximation setup({"x1"}, {"y"}, 0.0f, 1e-4f);
    RepairProbe probe(setup.network.get());
    probe.add_constraint("(10000*x1)^2", Condition::Equal, {0.25f});
    probe.calculate_domain();
    const VectorR repaired = probe.solve(VectorR::Constant(1, 9e-5f)).first;
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
    const VectorR repaired = probe.solve(VectorR::Constant(1, 1000.5f)).first;
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
        const VectorR repaired = probe.solve(VectorR::Zero(1)).first;
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


INSTANTIATE_TEST_SUITE_P(Drivers,
                         ResponseDriver,
                         testing::Values(Driver::Contraction, Driver::Genetic),
                         driver_name);

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
