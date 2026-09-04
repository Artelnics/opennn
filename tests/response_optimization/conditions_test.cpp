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

#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"

namespace
{

enum class Driver { Contraction, Genetic };

unique_ptr<ResponseOptimization> make_driver(const Driver driver, NeuralNetwork* network)
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


TEST(ResponseOptimizationSetup, NonFiniteConstraintValueThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::LessEqual,
                                             {numeric_limits<float>::infinity()}),
                 runtime_error);
}


TEST(ResponseOptimizationSetup, AllowedSetNeedsAtLeastOneValue)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::AllowedSet, {}), runtime_error);
}


TEST(ResponseOptimizationSetup, IntegerConditionOnlyAppliesToASingleVariable)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1 + x2", Condition::Integer), runtime_error);
    EXPECT_NO_THROW(optimization.add_constraint("x1", Condition::Integer));
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


TEST(ResponseOptimizationSetup, MissingConditionValuesThrow)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

    EXPECT_THROW(optimization.add_constraint("x1", Condition::Between, {5.0f}), runtime_error);
    EXPECT_THROW(optimization.add_constraint("x1", Condition::LessEqual, {}), runtime_error);
}


TEST(ResponseOptimizationSetup, EmptyBetweenIntervalThrows)
{
    MinimalApproximation setup({"x1", "x2"}, {"y"});

    DomainContraction optimization(setup.network.get());

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
            if (abs(results(i, j)) > 1e-2f) in_play++;

        EXPECT_LE(in_play, 2) << "row " << i << " keeps " << in_play << " of the 4 variables in play";
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


TEST_P(ResponseDriver, CategoricalResultsAreOneHot)
{
    CategoricalApproximation setup({"x1", "x2"}, "material", {"steel", "copper", "brass"});

    const unique_ptr<ResponseOptimization> optimization = make_driver(GetParam(), setup.network.get());

    optimization->add_objective("y1", Sense::Minimize);

    const MatrixR results = optimization->perform_response_optimization();

    ASSERT_GT(results.rows(), 0);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_GE(read_category(results, i, 2, 3), 0)
            << "row " << i << " holds "
            << results(i, 2) << ", " << results(i, 3) << ", " << results(i, 4);
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


INSTANTIATE_TEST_SUITE_P(Drivers,
                         ResponseDriver,
                         testing::Values(Driver::Contraction, Driver::Genetic),
                         driver_name);

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
