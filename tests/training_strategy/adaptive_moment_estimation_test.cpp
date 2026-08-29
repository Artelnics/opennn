#include "tests/pch.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/core/device_backend.h"

#include "tests/test_helpers.h"
#include "gtest/gtest.h"

using namespace opennn;
using namespace opennn_test;

namespace
{
    filesystem::path write_image_classification_dataset()
    {
        const filesystem::path root = filesystem::temp_directory_path() / "opennn_adam_image_classification";
        filesystem::remove_all(root);

        const filesystem::path class_a = root / "red";
        const filesystem::path class_b = root / "blue";
        filesystem::create_directories(class_a);
        filesystem::create_directories(class_b);

        for (int i = 0; i < 4; ++i)
        {
            write_bmp_24(class_a / ("a_" + to_string(i) + ".bmp"), 8, 8, 220, 20, 20);
            write_bmp_24(class_b / ("b_" + to_string(i) + ".bmp"), 8, 8, 20, 20, 220);
        }

        return root;
    }

    void remove_image_classification_dataset(const filesystem::path& root)
    {
        error_code error;
        filesystem::remove_all(root, error);
    }

    MatrixR separable_classification_data(Index samples_number, Index inputs_number)
    {
        MatrixR data(samples_number, inputs_number + 1);
        for (Index i = 0; i < samples_number; ++i)
        {
            float sum = 0.0f;
            for (Index j = 0; j < inputs_number; ++j)
            {
                const float value = ((i * 7 + j * 13) % 100) / 50.0f - 1.0f;
                data(i, j) = value;
                sum += value;
            }
            data(i, inputs_number) = sum > 0.0f ? 1.0f : 0.0f;
        }
        return data;
    }

    string write_text_classification_file()
    {
        const string content =
            "great phone excellent product fantastic value\tGood\n"
            "terrible battery awful screen broken charger\tBad\n"
            "amazing camera wonderful design superb quality\tGood\n"
            "useless device horrible support waste money\tBad\n"
            "love this product highly recommend brilliant\tGood\n"
            "worst purchase ever defective unit disappointed\tBad\n"
            "excellent service great value happy customer\tGood\n"
            "poor quality cheap material total garbage\tBad\n";

        const string file_path = (filesystem::temp_directory_path() / "opennn_adam_text_classification.txt").string();

        ofstream outfile(file_path);
        outfile << content;
        outfile.close();

        return file_path;
    }

    void remove_text_classification_file(const string& file_path)
    {
        error_code error;
        filesystem::remove(file_path, error);
        filesystem::remove_all(file_path + ".cache", error);
    }
}

class AdaptiveMomentEstimationTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        set_threads_number(0);
    }
};

TEST_F(AdaptiveMomentEstimationTest, DefaultConstructor)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    EXPECT_EQ(adaptive_moment_estimation.get_loss() == nullptr, true);
}

TEST_F(AdaptiveMomentEstimationTest, GeneralConstructor)
{
    Loss loss;
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    EXPECT_TRUE(adaptive_moment_estimation.get_loss() != nullptr);
}

TEST_F(AdaptiveMomentEstimationTest, TrainApproximationCPU)
{
    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_maximum_epochs(2);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(1);
    TabularDataset dataset_long(16, {2}, {1});
    dataset_long.set_data_random();
    dataset_long.set_sample_roles("Training");
    ApproximationNetwork network_long({2}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_maximum_epochs(200);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(AdaptiveMomentEstimationTest, TrainsRemainderBatchCPU)
{
    TabularDataset dataset(5, { 2 }, { 1 });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({ 2 }, { 3 }, { 1 });
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(2);
    adam.set_maximum_epochs(0);
    adam.set_display(false);

    Index batches_processed = 0;
    adam.post_batch_callback = [&](NeuralNetwork*) { ++batches_processed; };

    EXPECT_TRUE(isfinite(adam.train().get_training_error()));
    EXPECT_EQ(batches_processed, 3);
}

TEST_F(AdaptiveMomentEstimationTest, TrainingScalingCleanupSurvivesCallbackException)
{
    TabularDataset dataset(4, {1}, {1});
    MatrixR raw(4, 2);
    raw << 10.0f, 100.0f,
           20.0f, 200.0f,
           30.0f, 300.0f,
           40.0f, 400.0f;
    dataset.set_data(raw);
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_variable_scalers("MinimumMaximum");

    ApproximationNetwork network({1}, {2}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(2);
    adam.set_maximum_epochs(0);
    adam.set_display(false);
    adam.post_batch_callback = [](NeuralNetwork*)
    {
        throw runtime_error("intentional callback failure");
    };

    EXPECT_THROW(adam.train(), runtime_error);
    EXPECT_TRUE(dataset.get_data().isApprox(raw, 0.0f));

    std::array<float, 4> inputs{};
    dataset.fill_inputs({0, 1, 2, 3}, {0}, inputs.data(), FillMode::Training);
    for (Index i = 0; i < 4; ++i)
        EXPECT_NEAR(inputs[size_t(i)], raw(i, 0), EPSILON);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(AdaptiveMomentEstimationTest, TrainApproximationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_maximum_epochs(2);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(1);
    TabularDataset dataset_long(16, {2}, {1});
    dataset_long.set_data_random();
    dataset_long.set_sample_roles("Training");
    ApproximationNetwork network_long({2}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_maximum_epochs(200);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(AdaptiveMomentEstimationTest, TrainsRemainderBatchGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    TabularDataset dataset(5, { 2 }, { 1 });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({ 2 }, { 3 }, { 1 });
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(2);
    adam.set_cuda_graph(true);
    adam.set_joint_gradient_arena(true);
    adam.set_maximum_epochs(0);
    adam.set_display(false);

    Index batches_processed = 0;
    adam.post_batch_callback = [&](NeuralNetwork*) { ++batches_processed; };

    EXPECT_TRUE(isfinite(adam.train().get_training_error()));
    EXPECT_EQ(batches_processed, 3);
}

TEST_F(AdaptiveMomentEstimationTest, JointGradientArenaMatchesContiguousAdamGPU)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    set_seed(17);
    TabularDataset dataset(8, {3}, {2});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork joint_network({3}, {7}, {2});
    ApproximationNetwork contiguous_network({3}, {7}, {2});
    const VectorR initial_parameters = joint_network.get_parameters_map();
    contiguous_network.set_parameters(initial_parameters);

    const auto train_once = [&](ApproximationNetwork& network,
                                const bool joint_gradient_arena)
    {
        Loss loss(&network, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);

        AdaptiveMomentEstimation adam(&loss);
        adam.set_batch_size(8);
        adam.set_maximum_epochs(0);
        adam.set_display(false);
        adam.set_shuffle(false);
        adam.set_cuda_graph(false);
        adam.set_joint_gradient_arena(joint_gradient_arena);
        EXPECT_TRUE(isfinite(adam.train().get_training_error()));

        return VectorR(network.get_parameters_map());
    };

    const VectorR joint_parameters = train_once(joint_network, true);
    const VectorR contiguous_parameters = train_once(contiguous_network, false);

    ASSERT_EQ(joint_parameters.size(), contiguous_parameters.size());
    EXPECT_TRUE(logical_parameters_are_approx(
        joint_network.get_parameter_specs(),
        joint_parameters, contiguous_parameters, 1.0e-7f));
}

TEST_F(AdaptiveMomentEstimationTest, CudaGraphGroupedHostStagingReplay)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    set_seed(1);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(1);
    adam.set_cuda_graph(true);
    adam.set_maximum_epochs(1);
    adam.set_display(false);

    EXPECT_TRUE(isfinite(adam.train().get_training_error()));
}

// The encoder-decoder Transformer's training step must capture into a CUDA
// graph: an exported valid-length record that took a host round trip
// (compute_token_valid_lengths D2H + stream sync) once broke every capture
// and left the whole run eager. Both attention paths are covered, fused
// (SDPA) and unfused, in the precision the benchmarks train in.
static void expect_transformer_step_captures(Type dtype, Index sdpa_min_sequence_length)
{
    Configuration::instance().set(Device::CUDA, dtype);

    string content;
    for (Index sample = 0; sample < 16; ++sample)
    {
        for (Index token = 0; token < 14; ++token)
            content += "w" + to_string((sample * 7 + token * 3) % 23) + (token + 1 < 14 ? " " : "");
        content += "\t";
        for (Index token = 0; token < 15; ++token)
            content += "t" + to_string((sample * 5 + token * 2) % 19) + (token + 1 < 15 ? " " : "");
        content += "\n";
    }
    const string file_path = (filesystem::temp_directory_path()
                              / ("opennn_adam_transformer_graph_" + to_string(int(dtype)) + ".txt")).string();
    { ofstream out(file_path); out << content; }

    set_seed(3);
    LanguageDataset dataset(file_path);
    dataset.set_display(false);
    dataset.split_samples(1.0f, 0.0f, 0.0f);

    Transformer transformer(dataset.get_shape("Input")[0], dataset.get_shape("Decoder")[0],
                            dataset.get_input_vocabulary_size(), dataset.get_target_vocabulary_size(),
                            32, 4, 64, 1);
    transformer.set_attention_sdpa_min_sequence_length(sdpa_min_sequence_length);

    Loss loss(&transformer, &dataset);
    loss.set_error(Loss::Error::CrossEntropy3d);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(2);           // 8 whole batches: one grouped graph
    adam.set_cuda_graph(true);
    adam.set_joint_gradient_arena(true);
    adam.set_maximum_epochs(1);
    adam.set_display(false);

    EXPECT_TRUE(isfinite(adam.train().get_training_error()));
    EXPECT_FALSE(adam.get_cuda_graph_capture_failed());

    filesystem::remove(file_path);
}

TEST_F(AdaptiveMomentEstimationTest, CudaGraphCapturesTransformerStepSdpaBf16)
{
    expect_transformer_step_captures(Type::BF16, 1);
}

TEST_F(AdaptiveMomentEstimationTest, CudaGraphCapturesTransformerStepUnfusedFp32)
{
    expect_transformer_step_captures(Type::FP32, 1 << 20);
}

TEST_F(AdaptiveMomentEstimationTest, CudaGraphGroupedResidentBf16Replay)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    set_seed(1);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
    const MatrixR raw = dataset.get_data();

    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(1);
    adam.set_cuda_graph(true);
    adam.set_maximum_epochs(1);
    adam.set_display(false);

    EXPECT_TRUE(isfinite(adam.train().get_training_error()));
    EXPECT_TRUE(dataset.get_data().isApprox(raw, 0.0f));
    EXPECT_TRUE(dataset.requests_device_residency());
    EXPECT_FALSE(dataset.is_device_resident());
}

TEST_F(AdaptiveMomentEstimationTest, ResidentDatasetCleanupSurvivesCallbackException)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    TabularDataset dataset(4, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

    ApproximationNetwork network({2}, {4}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(2);
    adam.set_maximum_epochs(0);
    adam.set_display(false);
    adam.post_batch_callback = [](NeuralNetwork*)
    {
        throw runtime_error("intentional callback failure");
    };

    EXPECT_THROW(adam.train(), runtime_error);
    EXPECT_TRUE(dataset.requests_device_residency());
    EXPECT_FALSE(dataset.is_device_resident());
}
#endif

TEST_F(AdaptiveMomentEstimationTest, TrainClassificationCPU)
{
    const MatrixR classification_data = separable_classification_data(16, 3);

    set_seed(2);
    TabularDataset dataset_short(16, {3}, {1});
    dataset_short.set_data(classification_data);
    dataset_short.set_sample_roles("Training");
    ClassificationNetwork network_short({3}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_learning_rate(0.01f);
    adam_short.set_maximum_epochs(2);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(2);
    TabularDataset dataset_long(16, {3}, {1});
    dataset_long.set_data(classification_data);
    dataset_long.set_sample_roles("Training");
    ClassificationNetwork network_long({3}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_learning_rate(0.01f);
    adam_long.set_maximum_epochs(300);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(AdaptiveMomentEstimationTest, TrainClassificationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    const MatrixR classification_data = separable_classification_data(16, 3);

    set_seed(2);
    TabularDataset dataset_short(16, {3}, {1});
    dataset_short.set_data(classification_data);
    dataset_short.set_sample_roles("Training");
    ClassificationNetwork network_short({3}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_learning_rate(0.01f);
    adam_short.set_maximum_epochs(2);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(2);
    TabularDataset dataset_long(16, {3}, {1});
    dataset_long.set_data(classification_data);
    dataset_long.set_sample_roles("Training");
    ClassificationNetwork network_long({3}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_learning_rate(0.01f);
    adam_long.set_maximum_epochs(300);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}
#endif

static void expect_forecasting_training_reduces_error()
{
    set_seed(3);
    TimeSeriesDataset dataset(24, {1}, {1});
    dataset.set_data_random();
    dataset.set_past_time_steps(3);
    dataset.set_future_time_steps(1);
    dataset.set_sample_roles("Training");
    ForecastingNetwork network(dataset.get_input_shape(), {4}, dataset.get_target_shape());

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_display(false);

    adam.set_maximum_epochs(0);
    const type error_before = adam.train().get_training_error();

    adam.set_maximum_epochs(200);
    const type error_after = adam.train().get_training_error();

    EXPECT_LT(error_after, error_before);
}

TEST_F(AdaptiveMomentEstimationTest, TrainForecastingCPU)
{
    expect_forecasting_training_reduces_error();
}

#ifdef OPENNN_HAS_CUDA
TEST_F(AdaptiveMomentEstimationTest, TrainForecastingGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    expect_forecasting_training_reduces_error();
}
#endif

TEST_F(AdaptiveMomentEstimationTest, TrainImageClassificationCPU)
{
    const filesystem::path root = write_image_classification_dataset();

    set_seed(4);
    ImageDataset dataset_short(root);
    dataset_short.set_sample_roles("Training");
    ImageClassificationNetwork network_short(dataset_short.get_input_shape(), {4}, dataset_short.get_target_shape());
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_learning_rate(0.01f);
    adam_short.set_maximum_epochs(1);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(4);
    ImageDataset dataset_long(root);
    dataset_long.set_sample_roles("Training");
    ImageClassificationNetwork network_long(dataset_long.get_input_shape(), {4}, dataset_long.get_target_shape());
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_learning_rate(0.01f);
    adam_long.set_maximum_epochs(60);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    remove_image_classification_dataset(root);

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(AdaptiveMomentEstimationTest, TrainImageClassificationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    const filesystem::path root = write_image_classification_dataset();

    set_seed(4);
    ImageDataset dataset_short(root);
    dataset_short.set_sample_roles("Training");
    ImageClassificationNetwork network_short(dataset_short.get_input_shape(), {4}, dataset_short.get_target_shape());
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_learning_rate(0.01f);
    adam_short.set_maximum_epochs(1);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(4);
    ImageDataset dataset_long(root);
    dataset_long.set_sample_roles("Training");
    ImageClassificationNetwork network_long(dataset_long.get_input_shape(), {4}, dataset_long.get_target_shape());
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_learning_rate(0.01f);
    adam_long.set_maximum_epochs(60);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    remove_image_classification_dataset(root);

    EXPECT_LT(error_long, error_short);
}
#endif

TEST_F(AdaptiveMomentEstimationTest, TrainTextClassificationCPU)
{
    const string file_path = write_text_classification_file();

    set_seed(5);
    LanguageDataset dataset_short;
    dataset_short.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset_short.set_separator(Dataset::Separator::Tab);
    dataset_short.set_has_header(false);
    dataset_short.set_display(false);
    dataset_short.set_data_path(file_path);
    dataset_short.read_txt();
    dataset_short.set_sample_roles("Training");
    TextClassificationNetwork network_short(
        {dataset_short.get_input_vocabulary_size(), dataset_short.get_maximum_input_sequence_length(), 16},
        {2},
        {dataset_short.get_maximum_target_sequence_length()});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_maximum_epochs(2);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(5);
    LanguageDataset dataset_long;
    dataset_long.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset_long.set_separator(Dataset::Separator::Tab);
    dataset_long.set_has_header(false);
    dataset_long.set_display(false);
    dataset_long.set_data_path(file_path);
    dataset_long.read_txt();
    dataset_long.set_sample_roles("Training");
    TextClassificationNetwork network_long(
        {dataset_long.get_input_vocabulary_size(), dataset_long.get_maximum_input_sequence_length(), 16},
        {2},
        {dataset_long.get_maximum_target_sequence_length()});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_maximum_epochs(150);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    remove_text_classification_file(file_path);

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(AdaptiveMomentEstimationTest, TrainTextClassificationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    const string file_path = write_text_classification_file();

    set_seed(5);
    LanguageDataset dataset_short;
    dataset_short.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset_short.set_separator(Dataset::Separator::Tab);
    dataset_short.set_has_header(false);
    dataset_short.set_display(false);
    dataset_short.set_data_path(file_path);
    dataset_short.read_txt();
    dataset_short.set_sample_roles("Training");
    TextClassificationNetwork network_short(
        {dataset_short.get_input_vocabulary_size(), dataset_short.get_maximum_input_sequence_length(), 16},
        {2},
        {dataset_short.get_maximum_target_sequence_length()});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_short(&loss_short);
    adam_short.set_maximum_epochs(2);
    adam_short.set_display(false);
    const type error_short = adam_short.train().get_training_error();

    set_seed(5);
    LanguageDataset dataset_long;
    dataset_long.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset_long.set_separator(Dataset::Separator::Tab);
    dataset_long.set_has_header(false);
    dataset_long.set_display(false);
    dataset_long.set_data_path(file_path);
    dataset_long.read_txt();
    dataset_long.set_sample_roles("Training");
    TextClassificationNetwork network_long(
        {dataset_long.get_input_vocabulary_size(), dataset_long.get_maximum_input_sequence_length(), 16},
        {2},
        {dataset_long.get_maximum_target_sequence_length()});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    AdaptiveMomentEstimation adam_long(&loss_long);
    adam_long.set_maximum_epochs(150);
    adam_long.set_display(false);
    const type error_long = adam_long.train().get_training_error();

    remove_text_classification_file(file_path);

    EXPECT_LT(error_long, error_short);
}
#endif

TEST_F(AdaptiveMomentEstimationTest, LearningRateEffect)
{
    set_seed(6);
    TabularDataset dataset_trained(16, {2}, {1});
    dataset_trained.set_data_random();
    dataset_trained.set_sample_roles("Training");
    ApproximationNetwork network_trained({2}, {6}, {1});
    Loss loss_trained(&network_trained, &dataset_trained);
    loss_trained.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_trained(&loss_trained);
    adam_trained.set_learning_rate(0.01f);
    adam_trained.set_maximum_epochs(2);
    adam_trained.set_display(false);
    const type error_before = adam_trained.train().get_training_error();
    adam_trained.set_maximum_epochs(200);
    const type error_after = adam_trained.train().get_training_error();
    EXPECT_LT(error_after, error_before);

    set_seed(6);
    TabularDataset dataset_frozen(16, {2}, {1});
    dataset_frozen.set_data_random();
    dataset_frozen.set_sample_roles("Training");
    ApproximationNetwork network_frozen({2}, {6}, {1});
    Loss loss_frozen(&network_frozen, &dataset_frozen);
    loss_frozen.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_frozen(&loss_frozen);
    adam_frozen.set_learning_rate(0.0f);
    adam_frozen.set_maximum_epochs(2);
    adam_frozen.set_display(false);
    const type frozen_before = adam_frozen.train().get_training_error();
    adam_frozen.set_maximum_epochs(200);
    const type frozen_after = adam_frozen.train().get_training_error();
    EXPECT_FLOAT_EQ(frozen_after, frozen_before);
}

TEST_F(AdaptiveMomentEstimationTest, BetaSettersConverge)
{
    set_seed(7);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);
    adaptive_moment_estimation.set_beta_1(0.9f);
    adaptive_moment_estimation.set_beta_2(0.999f);
    adaptive_moment_estimation.set_learning_rate(0.01f);
    adaptive_moment_estimation.set_display(false);

    adaptive_moment_estimation.set_maximum_epochs(2);
    const type error_short = adaptive_moment_estimation.train().get_training_error();
    adaptive_moment_estimation.set_maximum_epochs(200);
    const type error_long = adaptive_moment_estimation.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(AdaptiveMomentEstimationTest, StoppingMaximumEpochs)
{
    set_seed(8);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);
    adaptive_moment_estimation.set_maximum_epochs(5);
    adaptive_moment_estimation.set_display(false);

    const TrainingResult training_results = adaptive_moment_estimation.train();

    EXPECT_EQ(training_results.get_epochs_number(), 5);
    EXPECT_EQ(training_results.get_epochs_number(), training_results.training_error_history.size());
}

TEST_F(AdaptiveMomentEstimationTest, StoppingLossGoal)
{
    set_seed(9);
    TabularDataset dataset(4, {1}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({1}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    const type training_loss_goal = type(0.1);
    adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    adaptive_moment_estimation.set_learning_rate(0.05f);
    adaptive_moment_estimation.set_maximum_epochs(10000);
    adaptive_moment_estimation.set_maximum_time(1000.0);
    adaptive_moment_estimation.set_display(false);

    const TrainingResult training_results = adaptive_moment_estimation.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}

TEST_F(AdaptiveMomentEstimationTest, StoppingMaximumTime)
{
    set_seed(10);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);
    adaptive_moment_estimation.set_maximum_epochs(1000000);
    adaptive_moment_estimation.set_maximum_time(0.5);
    adaptive_moment_estimation.set_display(false);

    const time_t start = time(nullptr);
    const TrainingResult training_results = adaptive_moment_estimation.train();
    const double elapsed = difftime(time(nullptr), start);

    EXPECT_LT(training_results.get_epochs_number(), 1000000);
    EXPECT_LT(elapsed, 30.0);
}

TEST_F(AdaptiveMomentEstimationTest, Determinism)
{
    set_threads_number(1);

    set_seed(11);
    TabularDataset dataset_first(16, {2}, {1});
    dataset_first.set_data_random();
    dataset_first.set_sample_roles("Training");
    ApproximationNetwork network_first({2}, {6}, {1});
    Loss loss_first(&network_first, &dataset_first);
    loss_first.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_first(&loss_first);
    adam_first.set_batch_size(16);
    adam_first.set_workers_number(1);
    adam_first.set_maximum_epochs(50);
    adam_first.set_display(false);
    const type error_first = adam_first.train().get_training_error();

    set_seed(11);
    TabularDataset dataset_second(16, {2}, {1});
    dataset_second.set_data_random();
    dataset_second.set_sample_roles("Training");
    ApproximationNetwork network_second({2}, {6}, {1});
    Loss loss_second(&network_second, &dataset_second);
    loss_second.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adam_second(&loss_second);
    adam_second.set_batch_size(16);
    adam_second.set_workers_number(1);
    adam_second.set_maximum_epochs(50);
    adam_second.set_display(false);
    const type error_second = adam_second.train().get_training_error();

    EXPECT_FLOAT_EQ(error_first, error_second);
}

TEST_F(AdaptiveMomentEstimationTest, RepeatedTrainingResetsState)
{
    set_threads_number(1);
    set_seed(17);

    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({2}, {6}, {1});
    const VectorR initial_parameters = network.get_parameters_map();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adam(&loss);
    adam.set_batch_size(16);
    adam.set_workers_number(1);
    adam.set_maximum_epochs(20);
    adam.set_display(false);

    set_seed(19);
    const float first_error = adam.train().get_training_error();

    network.set_parameters(initial_parameters);
    set_seed(19);
    const float second_error = adam.train().get_training_error();

    EXPECT_FLOAT_EQ(first_error, second_error);
}
