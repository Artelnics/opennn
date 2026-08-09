#include "tests/pch.h"

#include "opennn/dataset/dataset.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/model_selection/model_selection.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/model_selection/growing_neurons.h"

using namespace opennn;

TEST(ModelSelectionTest, DefaultConstructor)
{
    ModelSelection model_selection;
}

TEST(ModelSelectionTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;

    ModelSelection model_selection(&training_strategy);
}

