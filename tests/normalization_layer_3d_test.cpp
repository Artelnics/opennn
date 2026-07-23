#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/random_utilities.h"
#include "opennn/normalization_layer_3d.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;


TEST(Normalization3dTest, DefaultConstructor)
{
    Normalization3d normalization_3d;

    EXPECT_EQ(normalization_3d.get_input_shape().rank, 2);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], 0);
    EXPECT_EQ(normalization_3d.get_output_shape().rank, 2);
    EXPECT_EQ(normalization_3d.get_output_shape()[0], 0);
}


TEST(Normalization3dTest, GeneralConstructor)
{
    const Index sequence_length = 15;
    const Index embedding_dimension = 32;

    Normalization3d normalization_3d({sequence_length, embedding_dimension}, "encoder_norm");

    EXPECT_EQ(normalization_3d.get_name(), "Normalization3d");
    EXPECT_EQ(normalization_3d.get_label(), "encoder_norm");
    EXPECT_EQ(normalization_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(normalization_3d.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], sequence_length);
    EXPECT_EQ(normalization_3d.get_input_shape()[1], embedding_dimension);
    EXPECT_EQ(normalization_3d.get_output_shape()[0], sequence_length);
    EXPECT_EQ(normalization_3d.get_output_shape()[1], embedding_dimension);
}


TEST(Normalization3dTest, ForwardMatchesHandComputedLayerNorm)
{
    const Index batch_size = 1;
    const Index seq = 1;
    const Index dim = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Normalization3d>(Shape{seq, dim}, "norm"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 inputs(batch_size, seq, dim);
    inputs.data()[0] = type(1);
    inputs.data()[1] = type(2);
    inputs.data()[2] = type(3);
    inputs.data()[3] = type(4);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs.data(), {batch_size, seq, dim}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    const float* output_data = output_view.as<type>();

    const type inv_std = type(1) / std::sqrt(type(1.25));

    EXPECT_NEAR(output_data[0], type(-1.5) * inv_std, 1.0e-4f);
    EXPECT_NEAR(output_data[1], type(-0.5) * inv_std, 1.0e-4f);
    EXPECT_NEAR(output_data[2], type(0.5)  * inv_std, 1.0e-4f);
    EXPECT_NEAR(output_data[3], type(1.5)  * inv_std, 1.0e-4f);
}


TEST(Normalization3dTest, BackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Normalization3d>(input_shape, "norm"), {-1});
    const Index norm_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(norm_index)->get_output_shape()),
                             {norm_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}


TEST(Normalization3dTest, FusedResidualAddForward)
{
    const Index batch_size = 1;
    const Index seq = 1;
    const Index dim = 4;

    auto norm = make_unique<Normalization3d>(Shape{seq, dim}, "fused_norm");
    norm->set_fuse_add(true);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(norm), {-1, -2});
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 main_input(batch_size, seq, dim);
    Tensor3 residual_input(batch_size, seq, dim);

    main_input.data()[0]     = type(1);
    main_input.data()[1]     = type(2);
    main_input.data()[2]     = type(3);
    main_input.data()[3]     = type(4);

    residual_input.data()[0] = type(1);
    residual_input.data()[1] = type(0);
    residual_input.data()[2] = type(1);
    residual_input.data()[3] = type(0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = {
        TensorView(main_input.data(), {batch_size, seq, dim}),
        TensorView(residual_input.data(), {batch_size, seq, dim})
    };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    const float* output_data = output_view.as<type>();

    EXPECT_NEAR(output_data[0], type(-1), 1.0e-4f);
    EXPECT_NEAR(output_data[1], type(-1), 1.0e-4f);
    EXPECT_NEAR(output_data[2], type(1),  1.0e-4f);
    EXPECT_NEAR(output_data[3], type(1),  1.0e-4f);
}


TEST(Normalization3dTest, RMSGeneralConstructor)
{
    const Index sequence_length = 15;
    const Index embedding_dimension = 32;

    Normalization3d rms_normalization_3d({sequence_length, embedding_dimension}, "decoder_norm");
    rms_normalization_3d.set_method(NormalizationMethod::RMS);

    EXPECT_EQ(rms_normalization_3d.get_name(), "RMSNormalization3d");
    EXPECT_EQ(rms_normalization_3d.get_label(), "decoder_norm");
    EXPECT_EQ(rms_normalization_3d.get_method(), NormalizationMethod::RMS);
    EXPECT_EQ(rms_normalization_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(rms_normalization_3d.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(rms_normalization_3d.get_parameters_number(), embedding_dimension);   // weight only, no beta
}


TEST(Normalization3dTest, RMSFuseAddRejected)
{
    Normalization3d rms_normalization_3d({2, 4}, "norm");
    rms_normalization_3d.set_method(NormalizationMethod::RMS);

    EXPECT_ANY_THROW(rms_normalization_3d.set_fuse_add(true));
}


// For x = [1,2,3,4]: mean(x^2) = 7.5, so y = x / sqrt(7.5) (no mean subtraction).
TEST(Normalization3dTest, RMSForwardMatchesHandComputedRMSNorm)
{
    const Index batch_size = 1;
    const Index seq = 1;
    const Index dim = 4;

    auto norm = make_unique<Normalization3d>(Shape{seq, dim}, "norm");
    norm->set_method(NormalizationMethod::RMS);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(norm));
    neural_network.compile();
    neural_network.set_parameters_random();   // RMSNorm weight initializes to ones

    Tensor3 inputs(batch_size, seq, dim);
    inputs.data()[0] = type(1);
    inputs.data()[1] = type(2);
    inputs.data()[2] = type(3);
    inputs.data()[3] = type(4);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs.data(), {batch_size, seq, dim}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    const float* output_data = output_view.as<type>();

    const type inv_rms = type(1) / std::sqrt(type(7.5));

    EXPECT_NEAR(output_data[0], type(1) * inv_rms, 1.0e-4f);
    EXPECT_NEAR(output_data[1], type(2) * inv_rms, 1.0e-4f);
    EXPECT_NEAR(output_data[2], type(3) * inv_rms, 1.0e-4f);
    EXPECT_NEAR(output_data[3], type(4) * inv_rms, 1.0e-4f);
}


TEST(Normalization3dTest, RMSBackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    auto norm = make_unique<Normalization3d>(input_shape, "norm");
    norm->set_method(NormalizationMethod::RMS);
    neural_network.add_layer(move(norm), {-1});
    const Index norm_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(norm_index)->get_output_shape()),
                             {norm_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}


TEST(Normalization3dTest, RMSSaveLoadRoundTrip)
{
    const Index batch_size = 2;
    const Index seq = 3;
    const Index dim = 4;
    const float epsilon = 1.0e-5f;

    auto norm = make_unique<Normalization3d>(Shape{seq, dim}, "norm");
    norm->set_method(NormalizationMethod::RMS);
    norm->set_epsilon(epsilon);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(norm));
    neural_network.compile();
    neural_network.set_parameters_random();

    vector<float> input_values(size_t(batch_size * seq * dim));
    for (size_t i = 0; i < input_values.size(); ++i)
        input_values[i] = 0.25f * float(i) - 1.0f;

    auto make_inputs = [&]() {
        return vector<TensorView>{ TensorView(input_values.data(), {batch_size, seq, dim}) };
    };

    ForwardPropagation forward_before(batch_size, &neural_network);
    vector<TensorView> inputs_before = make_inputs();
    neural_network.forward_propagate(inputs_before, forward_before, false);
    const TensorView out_before = forward_before.get_outputs();
    const vector<float> expected(out_before.as<float>(), out_before.as<float>() + out_before.size());

    const string path = (filesystem::temp_directory_path() / "opennn_rms_norm_roundtrip.json").string();
    neural_network.save(path);

    NeuralNetwork loaded;
    loaded.load(path);

    const auto* loaded_norm = dynamic_cast<const Normalization3d*>(loaded.get_layer(Index(0)).get());
    ASSERT_NE(loaded_norm, nullptr);
    EXPECT_EQ(loaded_norm->get_method(), NormalizationMethod::RMS);
    EXPECT_NEAR(loaded_norm->get_epsilon(), epsilon, 1.0e-9f);

    ForwardPropagation forward_after(batch_size, &loaded);
    vector<TensorView> inputs_after = make_inputs();
    loaded.forward_propagate(inputs_after, forward_after, false);
    const TensorView out_after = forward_after.get_outputs();

    for (Index i = 0; i < out_after.size(); ++i)
        EXPECT_NEAR(out_after.as<float>()[i], expected[size_t(i)], 1.0e-5f);

    error_code file_error;
    filesystem::remove(path, file_error);
    filesystem::remove(filesystem::path(path).replace_extension(".bin"), file_error);
}


TEST(Normalization3dTest, FusedResidualAddGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<opennn::Dense>(input_shape, Shape{embedding_dimension}, "Identity"),
                             {-1});
    const Index dense_index = neural_network.get_layers_number() - 1;

    auto norm = make_unique<Normalization3d>(input_shape, "fused_norm");
    norm->set_fuse_add(true);
    neural_network.add_layer(move(norm), {dense_index, -1});
    const Index norm_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(norm_index)->get_output_shape()),
                             {norm_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}
