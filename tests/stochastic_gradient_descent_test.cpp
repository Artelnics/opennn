#include "pch.h"
#include "../opennn/configuration.h"
#include "../opennn/random_utilities.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/time_series_dataset.h"
#include "../opennn/language_dataset.h"
#include "../opennn/image_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/loss.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/device_backend.h"
#include "gtest/gtest.h"

using namespace opennn;

namespace
{
    void write_u16(vector<uint8_t>& bytes, uint16_t value)
    {
        bytes.push_back(uint8_t(value & 0xFF));
        bytes.push_back(uint8_t((value >> 8) & 0xFF));
    }

    void write_u32(vector<uint8_t>& bytes, uint32_t value)
    {
        bytes.push_back(uint8_t(value & 0xFF));
        bytes.push_back(uint8_t((value >> 8) & 0xFF));
        bytes.push_back(uint8_t((value >> 16) & 0xFF));
        bytes.push_back(uint8_t((value >> 24) & 0xFF));
    }

    void write_bmp_24(const filesystem::path& path, int width, int height, uint8_t red, uint8_t green, uint8_t blue)
    {
        const int bytes_per_pixel = 3;
        const int row_stride = ((width * bytes_per_pixel + 3) / 4) * 4;
        const uint32_t pixel_array_size = uint32_t(row_stride * height);
        const uint32_t header_size = 54;
        const uint32_t file_size = header_size + pixel_array_size;

        vector<uint8_t> bytes;
        bytes.push_back('B');
        bytes.push_back('M');
        write_u32(bytes, file_size);
        write_u16(bytes, 0);
        write_u16(bytes, 0);
        write_u32(bytes, header_size);
        write_u32(bytes, 40);
        write_u32(bytes, uint32_t(width));
        write_u32(bytes, uint32_t(height));
        write_u16(bytes, 1);
        write_u16(bytes, 24);
        write_u32(bytes, 0);
        write_u32(bytes, pixel_array_size);
        write_u32(bytes, 2835);
        write_u32(bytes, 2835);
        write_u32(bytes, 0);
        write_u32(bytes, 0);

        for (int y = 0; y < height; ++y)
        {
            int written = 0;
            for (int x = 0; x < width; ++x)
            {
                bytes.push_back(blue);
                bytes.push_back(green);
                bytes.push_back(red);
                written += bytes_per_pixel;
            }
            while (written < row_stride)
            {
                bytes.push_back(0);
                ++written;
            }
        }

        ofstream out(path, ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), streamsize(bytes.size()));
        out.close();
    }

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
}

class StochasticGradientDescentTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);
        Backend::instance().set_threads_number(0);
    }
};

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
#endif


TEST_F(StochasticGradientDescentTest, TrainClassificationCPU)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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

    EXPECT_LE(training_results.get_epochs_number(), 5);
}


TEST_F(StochasticGradientDescentTest, StoppingLossGoal)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);

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
    Configuration::instance().set(Device::CPU, Type::FP32);
    Backend::instance().set_threads_number(1);

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
