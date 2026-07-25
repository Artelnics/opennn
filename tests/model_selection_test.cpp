#include "pch.h"

#include "opennn/dataset.h"
#include "opennn/training_strategy.h"
#include "opennn/model_selection.h"
#include "opennn/standard_networks.h"
#include "opennn/growing_neurons.h"

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

