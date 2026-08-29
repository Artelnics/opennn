#include "tests/pch.h"
#include "opennn/core/configuration.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/core/random_utilities.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/stochastic_gradient_descent.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/device_backend.h"

#include "tests/test_helpers.h"
#include "gtest/gtest.h"

using namespace opennn;
using namespace opennn_test;

namespace
{
    filesystem::path write_sgd_image_classification_dataset()
    {
        const filesystem::path root = filesystem::temp_directory_path() / "opennn_sgd_image_classification";
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

    void remove_sgd_image_classification_dataset(const filesystem::path& root)
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

    string write_sgd_text_classification_file()
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

        const string file_path = (filesystem::temp_directory_path() / "opennn_sgd_text_classification.txt").string();

        ofstream outfile(file_path);
        outfile << content;
        outfile.close();

        return file_path;
    }

    void remove_sgd_text_classification_file(const string& file_path)
    {
        error_code error;
        filesystem::remove(file_path, error);
        filesystem::remove_all(file_path + ".cache", error);
    }

    class GradientClipProbe final : public Optimizer
    {
    public:
        static void clip(BackPropagation& back_propagation, float max_norm)
        {
            clip_gradient_norm(back_propagation, max_norm);
        }
    };
}

class StochasticGradientDescentTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        set_threads_number(0);
    }
};

TEST_F(StochasticGradientDescentTest, GpuClipWorkspaceIsBackwardOwned)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);

    NeuralNetwork neural_network;
    neural_network.add_layer(
        make_unique<opennn::Dense>(Shape{2}, Shape{1}, "Identity"));
    neural_network.compile(Device::CUDA);
    neural_network.set_parameters_random();

    Loss loss(&neural_network);
    BackPropagation first(1, loss);
    BackPropagation second(1, loss);

    const Index gradient_size = first.gradient.size_in_floats();
    vector<float> gradient(size_t(gradient_size), 2.0f);
    const Index gradient_bytes = gradient_size * Index(sizeof(float));
    device::copy_async(first.gradient.data(), gradient.data(), gradient_bytes,
                       device::CopyKind::HostToDevice);
    device::copy_async(second.gradient.data(), gradient.data(), gradient_bytes,
                       device::CopyKind::HostToDevice);

    GradientClipProbe::clip(first, 1.0f);
    GradientClipProbe::clip(second, 1.0f);

    vector<float> clipped(static_cast<size_t>(gradient_size), 0.0f);
    device::copy_async(clipped.data(), first.gradient.data(), gradient_bytes,
                       device::CopyKind::DeviceToHost);
    device::synchronize(device::get_compute_stream());

    ASSERT_FALSE(first.execution_workspace.empty());
    ASSERT_FALSE(second.execution_workspace.empty());
    EXPECT_NE(first.execution_workspace.data(), second.execution_workspace.data());

    const float clipped_norm = sqrt(inner_product(
        clipped.begin(), clipped.end(), clipped.begin(), 0.0f));
    EXPECT_NEAR(clipped_norm, 1.0f, 1.0e-5f);
}

TEST_F(StochasticGradientDescentTest, GpuClipSupportsTailAndCudaGraph)
{
    if (!device::has_cuda_device())
        GTEST_SKIP() << "No CUDA device.";

    Configuration::instance().set(Device::CUDA, Type::FP32);

    TabularDataset dataset(10, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({2}, {4}, {1});
    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    StochasticGradientDescent optimizer(&loss);
    optimizer.set_initial_learning_rate(0.01f);
    optimizer.set_batch_size(4);
    optimizer.set_gradient_clip_norm(0.5f);
    optimizer.set_cuda_graph(true);
    optimizer.set_maximum_epochs(2);
    optimizer.set_display(false);

    EXPECT_TRUE(isfinite(optimizer.train().get_training_error()));
}

TEST_F(StochasticGradientDescentTest, DefaultConstructor)
{
    StochasticGradientDescent stochastic_gradient_descent;

    EXPECT_TRUE(stochastic_gradient_descent.get_loss() == nullptr);
}

TEST_F(StochasticGradientDescentTest, GeneralConstructor)
{
    Loss loss;
    StochasticGradientDescent stochastic_gradient_descent(&loss);

    EXPECT_TRUE(stochastic_gradient_descent.get_loss() != nullptr);
}

TEST_F(StochasticGradientDescentTest, TrainApproximationCPU)
{
    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(1);
    TabularDataset dataset_long(16, {2}, {1});
    dataset_long.set_data_random();
    dataset_long.set_sample_roles("Training");
    ApproximationNetwork network_long({2}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(300);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(StochasticGradientDescentTest, TrainApproximationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    set_seed(1);
    TabularDataset dataset_short(16, {2}, {1});
    dataset_short.set_data_random();
    dataset_short.set_sample_roles("Training");
    ApproximationNetwork network_short({2}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(1);
    TabularDataset dataset_long(16, {2}, {1});
    dataset_long.set_data_random();
    dataset_long.set_sample_roles("Training");
    ApproximationNetwork network_long({2}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(300);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(StochasticGradientDescentTest, JointGradientArenaMatchesContiguousSgdGPU)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    set_seed(19);
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

        StochasticGradientDescent sgd(&loss);
        sgd.set_initial_learning_rate(0.01f);
        sgd.set_momentum(0.9f);
        sgd.set_nesterov(true);
        sgd.set_batch_size(8);
        sgd.set_maximum_epochs(0);
        sgd.set_display(false);
        sgd.set_shuffle(false);
        sgd.set_cuda_graph(false);
        sgd.set_joint_gradient_arena(joint_gradient_arena);
        EXPECT_TRUE(isfinite(sgd.train().get_training_error()));

        return VectorR(network.get_parameters_map());
    };

    const VectorR joint_parameters = train_once(joint_network, true);
    const VectorR contiguous_parameters = train_once(contiguous_network, false);

    ASSERT_EQ(joint_parameters.size(), contiguous_parameters.size());
    EXPECT_TRUE(logical_parameters_are_approx(
        joint_network.get_parameter_specs(),
        joint_parameters, contiguous_parameters, 1.0e-7f));
}

TEST_F(StochasticGradientDescentTest, JointGradientArenaSupportsCudaGraphRemainderGPU)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);

    set_seed(23);
    TabularDataset dataset(10, {3}, {2});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({3}, {7}, {2});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    StochasticGradientDescent sgd(&loss);
    sgd.set_initial_learning_rate(0.01f);
    sgd.set_momentum(0.9f);
    sgd.set_nesterov(true);
    sgd.set_batch_size(4);
    sgd.set_maximum_epochs(0);
    sgd.set_display(false);
    sgd.set_cuda_graph(true);
    sgd.set_joint_gradient_arena(true);

    Index batches_processed = 0;
    sgd.post_batch_callback = [&](NeuralNetwork*) { ++batches_processed; };

    EXPECT_TRUE(isfinite(sgd.train().get_training_error()));
    EXPECT_EQ(batches_processed, 3);
    EXPECT_FALSE(sgd.get_cuda_graph_capture_failed());
}
#endif

TEST_F(StochasticGradientDescentTest, TrainClassificationCPU)
{
    const MatrixR classification_data = separable_classification_data(16, 3);

    set_seed(2);
    TabularDataset dataset_short(16, {3}, {1});
    dataset_short.set_data(classification_data);
    dataset_short.set_sample_roles("Training");
    ClassificationNetwork network_short({3}, {6}, {1});
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.1f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(2);
    TabularDataset dataset_long(16, {3}, {1});
    dataset_long.set_data(classification_data);
    dataset_long.set_sample_roles("Training");
    ClassificationNetwork network_long({3}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.1f);
    sgd_long.set_maximum_epochs(400);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(StochasticGradientDescentTest, TrainClassificationGPU)
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
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.1f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(2);
    TabularDataset dataset_long(16, {3}, {1});
    dataset_long.set_data(classification_data);
    dataset_long.set_sample_roles("Training");
    ClassificationNetwork network_long({3}, {6}, {1});
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.1f);
    sgd_long.set_maximum_epochs(400);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}
#endif

TEST_F(StochasticGradientDescentTest, TrainForecastingCPU)
{
    set_seed(3);
    TimeSeriesDataset dataset_short(24, {1}, {1});
    dataset_short.set_data_random();
    dataset_short.set_past_time_steps(3);
    dataset_short.set_future_time_steps(1);
    dataset_short.set_sample_roles("Training");
    ForecastingNetwork network_short(dataset_short.get_input_shape(), {4}, dataset_short.get_target_shape());
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(3);
    TimeSeriesDataset dataset_long(24, {1}, {1});
    dataset_long.set_data_random();
    dataset_long.set_past_time_steps(3);
    dataset_long.set_future_time_steps(1);
    dataset_long.set_sample_roles("Training");
    ForecastingNetwork network_long(dataset_long.get_input_shape(), {4}, dataset_long.get_target_shape());
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(300);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(StochasticGradientDescentTest, TrainForecastingGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    set_seed(3);
    TimeSeriesDataset dataset_short(24, {1}, {1});
    dataset_short.set_data_random();
    dataset_short.set_past_time_steps(3);
    dataset_short.set_future_time_steps(1);
    dataset_short.set_sample_roles("Training");
    ForecastingNetwork network_short(dataset_short.get_input_shape(), {4}, dataset_short.get_target_shape());
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(3);
    TimeSeriesDataset dataset_long(24, {1}, {1});
    dataset_long.set_data_random();
    dataset_long.set_past_time_steps(3);
    dataset_long.set_future_time_steps(1);
    dataset_long.set_sample_roles("Training");
    ForecastingNetwork network_long(dataset_long.get_input_shape(), {4}, dataset_long.get_target_shape());
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(300);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}
#endif

TEST_F(StochasticGradientDescentTest, TrainImageClassificationCPU)
{
    const filesystem::path root = write_sgd_image_classification_dataset();

    set_seed(4);
    ImageDataset dataset_short(root);
    dataset_short.set_sample_roles("Training");
    ImageClassificationNetwork network_short(dataset_short.get_input_shape(), {4}, dataset_short.get_target_shape());
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(1);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(4);
    ImageDataset dataset_long(root);
    dataset_long.set_sample_roles("Training");
    ImageClassificationNetwork network_long(dataset_long.get_input_shape(), {4}, dataset_long.get_target_shape());
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(80);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    remove_sgd_image_classification_dataset(root);

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(StochasticGradientDescentTest, TrainImageClassificationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    const filesystem::path root = write_sgd_image_classification_dataset();

    set_seed(4);
    ImageDataset dataset_short(root);
    dataset_short.set_sample_roles("Training");
    ImageClassificationNetwork network_short(dataset_short.get_input_shape(), {4}, dataset_short.get_target_shape());
    Loss loss_short(&network_short, &dataset_short);
    loss_short.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(1);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

    set_seed(4);
    ImageDataset dataset_long(root);
    dataset_long.set_sample_roles("Training");
    ImageClassificationNetwork network_long(dataset_long.get_input_shape(), {4}, dataset_long.get_target_shape());
    Loss loss_long(&network_long, &dataset_long);
    loss_long.set_error(Loss::Error::CrossEntropy);
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(80);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    remove_sgd_image_classification_dataset(root);

    EXPECT_LT(error_long, error_short);
}
#endif

TEST_F(StochasticGradientDescentTest, TrainTextClassificationCPU)
{
    const string file_path = write_sgd_text_classification_file();

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
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

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
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(200);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    remove_sgd_text_classification_file(file_path);

    EXPECT_LT(error_long, error_short);
}

#ifdef OPENNN_HAS_CUDA
TEST_F(StochasticGradientDescentTest, TrainTextClassificationGPU)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);

    const string file_path = write_sgd_text_classification_file();

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
    StochasticGradientDescent sgd_short(&loss_short);
    sgd_short.set_initial_learning_rate(0.05f);
    sgd_short.set_maximum_epochs(2);
    sgd_short.set_display(false);
    const type error_short = sgd_short.train().get_training_error();

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
    StochasticGradientDescent sgd_long(&loss_long);
    sgd_long.set_initial_learning_rate(0.05f);
    sgd_long.set_maximum_epochs(200);
    sgd_long.set_display(false);
    const type error_long = sgd_long.train().get_training_error();

    remove_sgd_text_classification_file(file_path);

    EXPECT_LT(error_long, error_short);
}
#endif

TEST_F(StochasticGradientDescentTest, MomentumConverges)
{
    set_seed(6);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_initial_learning_rate(0.05f);
    stochastic_gradient_descent.set_momentum(0.9f);
    stochastic_gradient_descent.set_display(false);

    stochastic_gradient_descent.set_maximum_epochs(2);
    const type error_short = stochastic_gradient_descent.train().get_training_error();
    stochastic_gradient_descent.set_maximum_epochs(300);
    const type error_long = stochastic_gradient_descent.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(StochasticGradientDescentTest, NesterovConverges)
{
    set_seed(7);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_initial_learning_rate(0.05f);
    stochastic_gradient_descent.set_momentum(0.9f);
    stochastic_gradient_descent.set_nesterov(true);
    stochastic_gradient_descent.set_display(false);

    stochastic_gradient_descent.set_maximum_epochs(2);
    const type error_short = stochastic_gradient_descent.train().get_training_error();
    stochastic_gradient_descent.set_maximum_epochs(300);
    const type error_long = stochastic_gradient_descent.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(StochasticGradientDescentTest, InitialDecayConverges)
{
    set_seed(8);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_initial_learning_rate(0.05f);
    stochastic_gradient_descent.set_initial_decay(0.01f);
    stochastic_gradient_descent.set_display(false);

    stochastic_gradient_descent.set_maximum_epochs(2);
    const type error_short = stochastic_gradient_descent.train().get_training_error();
    stochastic_gradient_descent.set_maximum_epochs(300);
    const type error_long = stochastic_gradient_descent.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(StochasticGradientDescentTest, BatchSizeConverges)
{
    set_seed(9);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_initial_learning_rate(0.05f);
    stochastic_gradient_descent.set_batch_size(4);
    stochastic_gradient_descent.set_display(false);

    stochastic_gradient_descent.set_maximum_epochs(2);
    const type error_short = stochastic_gradient_descent.train().get_training_error();
    stochastic_gradient_descent.set_maximum_epochs(300);
    const type error_long = stochastic_gradient_descent.train().get_training_error();

    EXPECT_LT(error_long, error_short);
}

TEST_F(StochasticGradientDescentTest, StoppingMaximumEpochs)
{
    set_seed(10);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_maximum_epochs(5);
    stochastic_gradient_descent.set_display(false);

    const TrainingResult training_results = stochastic_gradient_descent.train();

    EXPECT_EQ(training_results.get_epochs_number(), 5);
    EXPECT_EQ(training_results.get_epochs_number(), training_results.training_error_history.size());
}

TEST_F(StochasticGradientDescentTest, StoppingLossGoal)
{
    set_seed(11);
    TabularDataset dataset(4, {1}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({1}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);

    const type training_loss_goal = type(0.1);
    stochastic_gradient_descent.set_loss_goal(training_loss_goal);
    stochastic_gradient_descent.set_initial_learning_rate(0.1f);
    stochastic_gradient_descent.set_maximum_epochs(10000);
    stochastic_gradient_descent.set_maximum_time(1000.0);
    stochastic_gradient_descent.set_display(false);

    const TrainingResult training_results = stochastic_gradient_descent.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}

TEST_F(StochasticGradientDescentTest, StoppingMaximumTime)
{
    set_seed(12);
    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");
    ApproximationNetwork network({2}, {6}, {1});
    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_maximum_epochs(1000000);
    stochastic_gradient_descent.set_maximum_time(0.5);
    stochastic_gradient_descent.set_display(false);

    const time_t start = time(nullptr);
    const TrainingResult training_results = stochastic_gradient_descent.train();
    const double elapsed = difftime(time(nullptr), start);

    EXPECT_LT(training_results.get_epochs_number(), 1000000);
    EXPECT_LT(elapsed, 30.0);
}

TEST_F(StochasticGradientDescentTest, Determinism)
{
    set_threads_number(1);

    set_seed(13);
    TabularDataset dataset_first(16, {2}, {1});
    dataset_first.set_data_random();
    dataset_first.set_sample_roles("Training");
    ApproximationNetwork network_first({2}, {6}, {1});
    Loss loss_first(&network_first, &dataset_first);
    loss_first.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_first(&loss_first);
    sgd_first.set_initial_learning_rate(0.05f);
    sgd_first.set_batch_size(16);
    sgd_first.set_workers_number(1);
    sgd_first.set_maximum_epochs(50);
    sgd_first.set_display(false);
    const type error_first = sgd_first.train().get_training_error();

    set_seed(13);
    TabularDataset dataset_second(16, {2}, {1});
    dataset_second.set_data_random();
    dataset_second.set_sample_roles("Training");
    ApproximationNetwork network_second({2}, {6}, {1});
    Loss loss_second(&network_second, &dataset_second);
    loss_second.set_error(Loss::Error::MeanSquaredError);
    StochasticGradientDescent sgd_second(&loss_second);
    sgd_second.set_initial_learning_rate(0.05f);
    sgd_second.set_batch_size(16);
    sgd_second.set_workers_number(1);
    sgd_second.set_maximum_epochs(50);
    sgd_second.set_display(false);
    const type error_second = sgd_second.train().get_training_error();

    EXPECT_FLOAT_EQ(error_first, error_second);
}

TEST_F(StochasticGradientDescentTest, RepeatedTrainingResetsState)
{
    set_threads_number(1);
    set_seed(23);

    TabularDataset dataset(16, {2}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork network({2}, {6}, {1});
    const VectorR initial_parameters = network.get_parameters_map();

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    StochasticGradientDescent sgd(&loss);
    sgd.set_initial_learning_rate(0.05f);
    sgd.set_initial_decay(0.01f);
    sgd.set_momentum(0.9f);
    sgd.set_batch_size(16);
    sgd.set_workers_number(1);
    sgd.set_maximum_epochs(20);
    sgd.set_display(false);

    set_seed(29);
    const float first_error = sgd.train().get_training_error();

    network.set_parameters(initial_parameters);
    set_seed(29);
    const float second_error = sgd.train().get_training_error();

    EXPECT_FLOAT_EQ(first_error, second_error);
}


// initial_learning_rate and initial_decay used to be declared without an
// initialiser and filled in by set_default(). They now carry their values in
// the header, so this pins them: a default-constructed SGD must still report
// what set_default() used to assign.
TEST_F(StochasticGradientDescentTest, HeaderDefaultsMatchWhatSetDefaultUsedToAssign)
{
    const StochasticGradientDescent stochastic_gradient_descent;

    EXPECT_FLOAT_EQ(stochastic_gradient_descent.get_initial_learning_rate(), 0.001f);
    EXPECT_EQ(stochastic_gradient_descent.get_batch_size(), 0);
    EXPECT_EQ(stochastic_gradient_descent.get_maximum_epochs(), 1000);
}
