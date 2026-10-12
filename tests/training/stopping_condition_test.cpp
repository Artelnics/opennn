// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "tests/common/pch.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training/loss.h"
#include "opennn/training/adam.h"
#include "opennn/training/quasi_newton.h"
#include "opennn/training/training_result.h"
#include "gtest/gtest.h"

using namespace opennn;

// Both training loops share the stopping rules: Adam runs the mini-batch loop,
// quasi-Newton the full-batch one.

namespace
{
    struct Problem
    {
        TabularDataset dataset;
        ApproximationNetwork network;
        Loss loss;

        explicit Problem(const unsigned seed)
            : dataset(16, {2}, {1}),
              network({2}, {4}, {1}),
              loss(&network, &dataset)
        {
            set_seed(seed);
            dataset.set_data_random();
            dataset.set_sample_roles("Training");
            loss.set_error(Loss::Error::MeanSquaredError);
        }
    };

    void expect_stopped(const TrainingResult& result, const StoppingCondition condition, const Index epochs)
    {
        ASSERT_TRUE(result.stopping_condition.has_value());
        EXPECT_EQ(*result.stopping_condition, condition);
        EXPECT_EQ(result.get_epochs_number(), epochs);
        EXPECT_TRUE(isfinite(result.get_training_error()));
    }
}

TEST(StoppingConditionTest, ZeroEpochLimitStillRunsOneEpoch)
{
    Problem adam_problem(21);
    Adam adam(&adam_problem.loss);
    adam.set_display(false);
    adam.set_maximum_epochs(0);
    expect_stopped(adam.train(), StoppingCondition::MaximumEpochsNumber, 1);

    Problem newton_problem(21);
    QuasiNewton quasi_newton(&newton_problem.loss);
    quasi_newton.set_display(false);
    quasi_newton.set_minimum_loss_decrease(-numeric_limits<float>::max());
    quasi_newton.set_maximum_epochs(0);
    expect_stopped(quasi_newton.train(), StoppingCondition::MaximumEpochsNumber, 1);
}

TEST(StoppingConditionTest, EpochBudgetKeepsOneHistoryEntryPerEpoch)
{
    Problem adam_problem(22);
    Adam adam(&adam_problem.loss);
    adam.set_display(false);
    adam.set_maximum_epochs(4);
    const TrainingResult adam_result = adam.train();
    expect_stopped(adam_result, StoppingCondition::MaximumEpochsNumber, 4);
    EXPECT_EQ(adam_result.validation_error_history.size(), 0);
    EXPECT_TRUE(adam_result.training_error_history.allFinite());

    Problem newton_problem(22);
    QuasiNewton quasi_newton(&newton_problem.loss);
    quasi_newton.set_display(false);
    quasi_newton.set_minimum_loss_decrease(-numeric_limits<float>::max());
    quasi_newton.set_maximum_epochs(4);
    const TrainingResult newton_result = quasi_newton.train();
    expect_stopped(newton_result, StoppingCondition::MaximumEpochsNumber, 4);
    EXPECT_TRUE(newton_result.training_error_history.allFinite());
}

TEST(StoppingConditionTest, LossGoalTakesPriorityOverTheBudgets)
{
    Problem adam_problem(23);
    Adam adam(&adam_problem.loss);
    adam.set_display(false);
    adam.set_loss_goal(numeric_limits<float>::max());
    adam.set_maximum_time(0.0f);
    adam.set_maximum_epochs(1);
    expect_stopped(adam.train(), StoppingCondition::LossGoal, 1);

    Problem newton_problem(23);
    QuasiNewton quasi_newton(&newton_problem.loss);
    quasi_newton.set_display(false);
    quasi_newton.set_loss_goal(numeric_limits<float>::max());
    quasi_newton.set_maximum_time(0.0f);
    quasi_newton.set_maximum_epochs(1);
    expect_stopped(quasi_newton.train(), StoppingCondition::LossGoal, 1);
}

TEST(StoppingConditionTest, TimeLimitIsReportedBeforeTheEpochLimit)
{
    Problem adam_problem(24);
    Adam adam(&adam_problem.loss);
    adam.set_display(false);
    adam.set_maximum_time(0.0f);
    adam.set_maximum_epochs(1);
    expect_stopped(adam.train(), StoppingCondition::MaximumTime, 1);

    Problem newton_problem(24);
    QuasiNewton quasi_newton(&newton_problem.loss);
    quasi_newton.set_display(false);
    quasi_newton.set_minimum_loss_decrease(-numeric_limits<float>::max());
    quasi_newton.set_maximum_time(0.0f);
    quasi_newton.set_maximum_epochs(1000);
    expect_stopped(quasi_newton.train(), StoppingCondition::MaximumTime, 1);
}

TEST(StoppingConditionTest, MinimumLossDecreaseNeedsAPreviousEpoch)
{
    Problem problem(25);
    QuasiNewton quasi_newton(&problem.loss);
    quasi_newton.set_display(false);
    // Any decrease is too small, but the first epoch has nothing to compare with.
    quasi_newton.set_minimum_loss_decrease(numeric_limits<float>::max());
    quasi_newton.set_maximum_epochs(1000);
    expect_stopped(quasi_newton.train(), StoppingCondition::MinimumLossDecrease, 2);
}

TEST(StoppingConditionTest, LabelsStayStableForReports)
{
    TrainingResult result;
    EXPECT_EQ(result.write_stopping_condition(), "None");

    const pair<StoppingCondition, string> labels[] =
    {
        {StoppingCondition::LossGoal, "Loss goal"},
        {StoppingCondition::MinimumLossDecrease, "Minimum loss decrease"},
        {StoppingCondition::MaximumValidationErrorIncreases, "Maximum validation error increases"},
        {StoppingCondition::MaximumTime, "Maximum training time"},
        {StoppingCondition::MaximumEpochsNumber, "Maximum epochs number"}
    };

    for (const auto& [condition, label] : labels)
    {
        result.stopping_condition = condition;
        EXPECT_EQ(result.write_stopping_condition(), label);
    }
}
