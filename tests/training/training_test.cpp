#include "tests/pch.h"

#include "opennn/network/layers/dense_layer.h"
#include "opennn/models/models.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training/adaptive_moment_estimation.h"
#include "opennn/training/levenberg_marquardt.h"
#include "opennn/training/quasi_newton.h"
#include "opennn/training/training.h"

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

TEST(Training, DefaultConstructor)
{
    Training training;

    EXPECT_EQ(training.get_network(), nullptr);
    EXPECT_EQ(training.get_dataset(), nullptr);
}

TEST(Training, SerializesTrainingConfiguration)
{
    TabularDataset dataset(4, {2}, {1});
    ApproximationNetwork network({2}, {3}, {1});
    Training training(&network, &dataset);
    training.set_optimization_algorithm("AdaptiveMomentEstimation");
    auto* optimizer = dynamic_cast<AdaptiveMomentEstimation*>(training.get_optimization_algorithm());
    ASSERT_NE(optimizer, nullptr);
    optimizer->set_learning_rate(0.0125f);
    optimizer->set_maximum_epochs(3);
    optimizer->set_display(false);

    JsonWriter writer;
    training.to_JSON(writer);
    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));
    ASSERT_TRUE(document.get_root().has("Training"));
    EXPECT_FALSE(document.get_root().has("TrainingStrategy"));

    const filesystem::path path = filesystem::temp_directory_path() / "opennn_training_name_test.json";
    training.save(path);
    for (const bool from_file : {false, true})
    {
        Training restored(&network, &dataset);
        if (from_file) restored.load(path);
        else restored.from_JSON(document);
        EXPECT_EQ(restored.get_network(), &network);
        EXPECT_EQ(restored.get_dataset(), &dataset);
        EXPECT_EQ(restored.get_loss()->get_name(), training.get_loss()->get_name());
        const auto* restored_optimizer =
            dynamic_cast<const AdaptiveMomentEstimation*>(restored.get_optimization_algorithm());
        ASSERT_NE(restored_optimizer, nullptr);
        EXPECT_FLOAT_EQ(restored_optimizer->get_learning_rate(), 0.0125f);
        EXPECT_EQ(restored_optimizer->get_maximum_epochs(), 3);
        EXPECT_FALSE(restored_optimizer->get_display());
    }
    filesystem::remove(path);

    const JsonDocument previous =
        JsonDocument::wrap("TrainingStrategy", document.get_root().at("Training"));
    EXPECT_THROW(training.from_JSON(previous), runtime_error);
}

TEST(Training, SerializesRenamedOptimizers)
{
    TabularDataset dataset(4, {2}, {1});
    ApproximationNetwork network({2}, {3}, {1});
    for (const string name : {"LevenbergMarquardt", "QuasiNewton"})
    {
        SCOPED_TRACE(name);
        Training training(&network, &dataset);
        training.set_optimization_algorithm(name);
        training.get_optimization_algorithm()->set_maximum_epochs(7);
        JsonWriter writer;
        training.to_JSON(writer);
        JsonDocument document;
        document.set_root(Json::parse(writer.c_str()));
        const Json& optimizer_json = document.get_root().at("Training").at("Optimizer");
        EXPECT_EQ(optimizer_json.at("OptimizationMethod").as_string(), name);
        EXPECT_TRUE(optimizer_json.has(name));

        Training restored(&network, &dataset);
        restored.from_JSON(document);
        const Optimizer* optimizer = restored.get_optimization_algorithm();
        EXPECT_EQ(optimizer->get_name(), name);
        EXPECT_EQ(optimizer->get_maximum_epochs(), 7);
        if (name == "QuasiNewton")
            EXPECT_NE(dynamic_cast<const QuasiNewton*>(optimizer), nullptr);
        else
            EXPECT_NE(dynamic_cast<const LevenbergMarquardt*>(optimizer), nullptr);
    }

    Training training(&network, &dataset);
    EXPECT_THROW(training.set_optimization_algorithm("QuasiNewtonMethod"), runtime_error);
}

TEST(Training, GeneralConstructor)
{
    TabularDataset dataset(10, {2}, {1});
    dataset.set_data_random();

    ApproximationNetwork network({2}, {3}, {1});

    Training training_1(&network, &dataset);

    EXPECT_EQ(training_1.get_network(), &network);
    EXPECT_EQ(training_1.get_dataset(), &dataset);
    EXPECT_EQ(network.get_task(), NetworkTask::Approximation);
    EXPECT_EQ(training_1.get_loss()->get_name(), "MeanSquaredError");
    EXPECT_EQ(training_1.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");
}

TEST(Training, UsesExplicitNetworkTask)
{
    TabularDataset dataset(10, {2}, {1});
    Network network;
    network.set_task(NetworkTask::LanguageModeling);

    Training training(&network, &dataset);

    EXPECT_EQ(training.get_loss()->get_name(), "CrossEntropyError3d");
    EXPECT_EQ(training.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");

    const auto* adam = dynamic_cast<const AdaptiveMomentEstimation*>(
        training.get_optimization_algorithm());
    ASSERT_NE(adam, nullptr);
    EXPECT_FLOAT_EQ(adam->get_learning_rate(), 0.0001f);
}

TEST(Training, ClassificationFamilyUsesOptimizerTaskDefaults)
{
    TabularDataset dataset(10, {2}, {2});
    Network network;
    network.set_task(NetworkTask::ImageClassification);

    Training training(&network, &dataset);

    EXPECT_EQ(training.get_loss()->get_name(), "CrossEntropy");
    EXPECT_EQ(training.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");
    EXPECT_EQ(training.get_optimization_algorithm()->get_maximum_epochs(), 100);
}

TEST(Training, DoesNotInferTaskFromTopology)
{
    TabularDataset dataset(10, {2}, {2});
    Network network;
    network.add_layer(
        make_unique<opennn::Dense>(Shape{2}, Shape{2}, "Softmax"));

    Training training(&network, &dataset);

    EXPECT_EQ(network.get_task(), NetworkTask::Generic);
    EXPECT_EQ(training.get_loss()->get_name(), "MeanSquaredError");
    EXPECT_EQ(training.get_optimization_algorithm()->get_name(),
              "AdaptiveMomentEstimation");
}

TEST(Training, ClassificationDefaultsUseDeclaredTask)
{
    TabularDataset binary_dataset(10, {2}, {1});
    ClassificationNetwork binary_network({2}, {3}, {1});
    Training binary_strategy(&binary_network, &binary_dataset);

    EXPECT_EQ(binary_network.get_task(), NetworkTask::Classification);
    EXPECT_EQ(binary_strategy.get_loss()->get_name(), "WeightedSquaredError");
    EXPECT_EQ(binary_strategy.get_optimization_algorithm()->get_name(),
              "QuasiNewton");

    TabularDataset multiclass_dataset(10, {2}, {3});
    ClassificationNetwork multiclass_network({2}, {3}, {3});
    Training multiclass_strategy(&multiclass_network, &multiclass_dataset);

    EXPECT_EQ(multiclass_strategy.get_loss()->get_name(), "CrossEntropy");
    EXPECT_EQ(multiclass_strategy.get_optimization_algorithm()->get_name(),
              "QuasiNewton");
}

TEST(Training, RebindsLossDependencies)
{
    TabularDataset first_dataset(10, {2}, {1});
    TabularDataset second_dataset(10, {2}, {1});
    ApproximationNetwork first_network({2}, {3}, {1});
    ApproximationNetwork second_network({2}, {3}, {1});

    Training training(&first_network, &first_dataset);
    training.set_network(&second_network);
    training.set_dataset(&second_dataset);

    ASSERT_NE(training.get_loss(), nullptr);
    EXPECT_EQ(training.get_loss()->get_network(), &second_network);
    EXPECT_EQ(training.get_loss()->get_dataset(), &second_dataset);

    training.set();
    EXPECT_EQ(training.get_loss(), nullptr);
    EXPECT_EQ(training.get_optimization_algorithm(), nullptr);
}

TEST(Training, InitializesWhenNetworkIsSetLater)
{
    TabularDataset dataset(10, {2}, {1});
    ApproximationNetwork network({2}, {3}, {1});

    Training training;
    training.set_dataset(&dataset);
    training.set_network(&network);

    ASSERT_NE(training.get_loss(), nullptr);
    ASSERT_NE(training.get_optimization_algorithm(), nullptr);
    EXPECT_EQ(training.get_loss()->get_network(), &network);
    EXPECT_EQ(training.get_loss()->get_dataset(), &dataset);
}

TEST(Training, TransfersTranslationVocabularies)
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
