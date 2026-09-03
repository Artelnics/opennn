#include "tests/pch.h"

#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/models/models.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

#include <fstream>

using namespace opennn;

namespace
{

class TrainingArtifactProbe final : public Optimizer
{
public:

    using Optimizer::Optimizer;

    void prepare() { prepare_training_artifacts(); }
};

}

TEST(TrainingStrategy, DefaultConstructor)
{
    TrainingStrategy training_strategy;

    EXPECT_EQ(training_strategy.get_neural_network(), nullptr);
    EXPECT_EQ(training_strategy.get_dataset(), nullptr);
}

TEST(TrainingStrategy, GeneralConstructor)
{
    TabularDataset dataset(10, {2}, {1});
    dataset.set_data_random();

    ApproximationNetwork neural_network({2}, {3}, {1});

    TrainingStrategy training_strategy_1(&neural_network, &dataset);

    EXPECT_EQ(training_strategy_1.get_neural_network(), &neural_network);
    EXPECT_EQ(training_strategy_1.get_dataset(), &dataset);
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Approximation);
    EXPECT_EQ(training_strategy_1.get_loss()->get_name(), "MeanSquaredError");
    EXPECT_EQ(training_strategy_1.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");
}

TEST(TrainingStrategy, UsesExplicitNetworkTask)
{
    TabularDataset dataset(10, {2}, {1});
    NeuralNetwork neural_network;
    neural_network.set_task(NetworkTask::LanguageModeling);

    TrainingStrategy training_strategy(&neural_network, &dataset);

    EXPECT_EQ(training_strategy.get_loss()->get_name(), "CrossEntropyError3d");
    EXPECT_EQ(training_strategy.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");

    const auto* adam = dynamic_cast<const AdaptiveMomentEstimation*>(
        training_strategy.get_optimization_algorithm());
    ASSERT_NE(adam, nullptr);
    EXPECT_FLOAT_EQ(adam->get_learning_rate(), 0.0001f);
}

TEST(TrainingStrategy, ClassificationFamilyUsesOptimizerTaskDefaults)
{
    TabularDataset dataset(10, {2}, {2});
    NeuralNetwork neural_network;
    neural_network.set_task(NetworkTask::ImageClassification);

    TrainingStrategy training_strategy(&neural_network, &dataset);

    EXPECT_EQ(training_strategy.get_loss()->get_name(), "CrossEntropy");
    EXPECT_EQ(training_strategy.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");
    EXPECT_EQ(training_strategy.get_optimization_algorithm()->get_maximum_epochs(), 100);
}

TEST(TrainingStrategy, DoesNotInferTaskFromTopology)
{
    TabularDataset dataset(10, {2}, {2});
    NeuralNetwork neural_network;
    neural_network.add_layer(
        make_unique<opennn::Dense>(Shape{2}, Shape{2}, "Softmax"));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    EXPECT_EQ(neural_network.get_task(), NetworkTask::Generic);
    EXPECT_EQ(training_strategy.get_loss()->get_name(), "MeanSquaredError");
    EXPECT_EQ(training_strategy.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");
}

TEST(TrainingStrategy, ClassificationDefaultsUseDeclaredTask)
{
    TabularDataset binary_dataset(10, {2}, {1});
    ClassificationNetwork binary_network({2}, {3}, {1});
    TrainingStrategy binary_strategy(&binary_network, &binary_dataset);

    EXPECT_EQ(binary_network.get_task(), NetworkTask::Classification);
    EXPECT_EQ(binary_strategy.get_loss()->get_name(), "WeightedSquaredError");
    EXPECT_EQ(binary_strategy.get_optimization_algorithm()->get_name(),
              "QuasiNewtonMethod");

    TabularDataset multiclass_dataset(10, {2}, {3});
    ClassificationNetwork multiclass_network({2}, {3}, {3});
    TrainingStrategy multiclass_strategy(&multiclass_network, &multiclass_dataset);

    EXPECT_EQ(multiclass_strategy.get_loss()->get_name(), "CrossEntropy");
    EXPECT_EQ(multiclass_strategy.get_optimization_algorithm()->get_name(),
              "QuasiNewtonMethod");
}

TEST(TrainingStrategy, RebindsLossDependencies)
{
    TabularDataset first_dataset(10, {2}, {1});
    TabularDataset second_dataset(10, {2}, {1});
    ApproximationNetwork first_network({2}, {3}, {1});
    ApproximationNetwork second_network({2}, {3}, {1});

    TrainingStrategy training_strategy(&first_network, &first_dataset);
    training_strategy.set_neural_network(&second_network);
    training_strategy.set_dataset(&second_dataset);

    ASSERT_NE(training_strategy.get_loss(), nullptr);
    EXPECT_EQ(training_strategy.get_loss()->get_neural_network(), &second_network);
    EXPECT_EQ(training_strategy.get_loss()->get_dataset(), &second_dataset);

    training_strategy.set();
    EXPECT_EQ(training_strategy.get_loss(), nullptr);
    EXPECT_EQ(training_strategy.get_optimization_algorithm(), nullptr);
}

TEST(TrainingStrategy, InitializesWhenNetworkIsSetLater)
{
    TabularDataset dataset(10, {2}, {1});
    ApproximationNetwork neural_network({2}, {3}, {1});

    TrainingStrategy training_strategy;
    training_strategy.set_dataset(&dataset);
    training_strategy.set_neural_network(&neural_network);

    ASSERT_NE(training_strategy.get_loss(), nullptr);
    ASSERT_NE(training_strategy.get_optimization_algorithm(), nullptr);
    EXPECT_EQ(training_strategy.get_loss()->get_neural_network(), &neural_network);
    EXPECT_EQ(training_strategy.get_loss()->get_dataset(), &dataset);
}

TEST(TrainingStrategy, TransfersTranslationVocabularies)
{
    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_translation_vocabulary_test.txt";
    {
        ofstream file(path);
        file << "hello world\thola mundo\n"
             << "good night\tbuenas noches\n";
    }

    LanguageDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_display(false);
    dataset.set_data_path(path);
    dataset.read_txt();

    Transformer transformer(
        dataset.get_shape("Input")[0],
        dataset.get_shape("Decoder")[0],
        dataset.get_input_vocabulary_size(),
        dataset.get_target_vocabulary_size(),
        8, 2, 16, 1);

    Loss loss(&transformer, &dataset);
    TrainingArtifactProbe optimizer(&loss);
    optimizer.prepare();

    EXPECT_EQ(transformer.get_input_vocabulary(), dataset.get_input_vocabulary());
    EXPECT_EQ(transformer.get_target_vocabulary(), dataset.get_target_vocabulary());

    error_code error;
    filesystem::remove(path, error);
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
