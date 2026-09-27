#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/network/layers/detection_layer.h"
#include "opennn/network/layers/convolutional_layer.h"
#include "opennn/network/network.h"
#include "opennn/training/loss.h"
#include "opennn/dataset/batch.h"
#include "opennn/models/models.h"

#include "tests/test_helpers.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace opennn;
using namespace opennn_test;

TEST(YoloFPN, SingleHeadLargeGridNoObjectGradientMatchesNumerical)
{
    // Fixture: two object-free 8x8 images of clearly different, unsaturated
    // colours, each paired with an empty label file, and a single class.
    TempDir dir("opennn_yolo_fpn_single_head_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "single.names", {"target"});

    struct UniformImage
    {
        const char* stem;
        std::uint8_t red, green, blue;
    };

    const UniformImage images[] = {{"warm", 192, 64, 96}, {"cool", 64, 160, 200}};

    for (const UniformImage& image : images)
    {
        write_bmp_24(images_dir / (string(image.stem) + ".bmp"), 8, 8, image.red, image.green, image.blue);
        ofstream empty_label(labels_dir / (string(image.stem) + ".txt"));
    }

    constexpr Index grid_size = 8;
    constexpr Index boxes_per_cell = 3;
    constexpr Index classes_number = 1;
    constexpr Index head_channels = boxes_per_cell * (5 + classes_number);

    const Shape input_shape{8, 8, 3};
    const vector<std::array<float, 2>> anchors{{0.08f, 0.08f}, {0.14f, 0.14f}, {0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, input_shape, grid_size, boxes_per_cell, anchors);

    // Every error and gradient evaluation below must see the same data.
    {
        YoloDataset::AugmentationPolicy no_augmentation;
        no_augmentation.enabled = false;
        dataset.set_augmentation_policy(no_augmentation);
    }

    // Layer order matters: the packed attribution below walks layers by index.
    Network network;
    network.add_layer(make_unique<Convolutional>(
        input_shape, Shape{3, 3, 3, 4},
        "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "features"));
    network.add_layer(make_unique<Convolutional>(
        Shape{grid_size, grid_size, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "head_logits"));
    network.add_layer(make_unique<Detection>(
        Shape{grid_size, grid_size, head_channels}, anchors, "head_detection"));

    // compile() zeroes the parameters, and calculate_gradient rejects an
    // all-zero vector, so the whole buffer (padding included) is set here.
    network.compile();
    network.get_parameters_map().setConstant(0.05f);

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    // Smoke probe of the parameter-override forward pass: central differences
    // on one first-layer weight at several step sizes. Prints only.
    {
        const Index samples_number = dataset.get_samples_number("Training");

        Batch batch(samples_number, &dataset, network.get_config());
        batch.fill(dataset.get_sample_indices("Training"), dataset.get_feature_selection());

#ifdef OPENNN_HAS_CUDA
        if (network.is_gpu())
        {
            batch.upload_to_device_batch_async(batch, device::get_transfer_stream());
            batch.wait_h2d_complete();
        }
#endif

        ForwardPropagation forward_propagation(samples_number, &network);

        network.forward_propagate(batch.get_inputs(), forward_propagation, ForwardPropagationMode::Training);
        const float current_error = loss.calculate_error(batch, forward_propagation).error;

        const Network& const_network = network;
        const VectorR current_parameters = const_network.get_parameters_map();

        constexpr Index probed_index = 13;
        VectorR shifted_parameters = current_parameters;

        const auto error_with_shift = [&](const float shift) -> float
        {
            shifted_parameters(probed_index) = current_parameters(probed_index) + shift;
            network.forward_propagate(batch.get_inputs(), shifted_parameters, forward_propagation);
            return loss.calculate_error(batch, forward_propagation).error;
        };

        cout << "[override probe] error at current parameters: " << current_error << "\n"
             << "[override probe] parameter " << probed_index
             << " = " << current_parameters(probed_index) << "\n";

        const float default_step = calculate_h(current_parameters(probed_index));

        for (const float step : {default_step, 1e-4f, 1e-3f, 1e-2f})
        {
            const float estimate = (error_with_shift(step) - error_with_shift(-step)) / (2.0f * step);
            cout << "[override probe]   step " << step << ": dE/dp ~ " << estimate << "\n";
        }

        const bool restored = (const_network.get_parameters_map().array() == current_parameters.array()).all();
        cout << "[override probe] parameters restored: " << (restored ? "yes" : "no") << "\n";
    }

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    // Each layer owns a packed range [start, start + P_i); the checked prefix
    // is the concatenation of those ranges, not the aligned buffer layout.
    Index packed_total = 0;
    Index worst_index = 0;
    float worst_difference = 0.0f;

    for (Index layer_index = 0; layer_index < network.get_layers_number(); ++layer_index)
    {
        const unique_ptr<Layer>& layer = network.get_layer(layer_index);
        const Index layer_parameters = layer->get_parameters_number();

        cout << "[gradient check] layer " << layer_index << " '" << layer->get_label() << "': ";

        if (layer_parameters == 0)
        {
            cout << "no parameters\n";
            continue;
        }

        float layer_worst = 0.0f;

        for (Index k = packed_total; k < packed_total + layer_parameters; ++k)
        {
            const float difference = abs(gradient(k) - numerical_gradient(k));

            layer_worst = max(layer_worst, difference);

            if (difference > worst_difference)
            {
                worst_difference = difference;
                worst_index = k;
            }
        }

        cout << layer_parameters << " parameters, max |analytic - numerical| = " << layer_worst << "\n";

        packed_total += layer_parameters;
    }

    cout << "[gradient check] worst index " << worst_index
         << ": analytic " << gradient(worst_index)
         << ", numerical " << numerical_gradient(worst_index) << "\n"
         << "[gradient check] gradient size " << gradient.size()
         << ", packed total " << packed_total
         << ", parameter buffer size " << network.get_parameters_buffer_size() << "\n";

    EXPECT_LT(worst_difference, 0.05f);
}

TEST(YoloFPN, MultiHeadNoObjectGradientMatchesNumerical)
{

    TempDir dir("opennn_yolo_fpn_test_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "classes.names", {"only"});

    write_bmp_24(images_dir / "a.bmp", 8, 8, 200, 100, 50);
    write_bmp_24(images_dir / "b.bmp", 8, 8,  50, 200, 100);
    { ofstream empty_a(labels_dir / "a.txt"); }
    { ofstream empty_b(labels_dir / "b.txt"); }

    constexpr Index input_H = 8;
    constexpr Index input_W = 8;
    constexpr Index classes_number = 1;
    constexpr Index boxes_per_head = 3;
    constexpr Index head_channels = boxes_per_head * (5 + classes_number);

    const vector<Index> head_grids{2, 4, 8};

    const vector<std::array<float, 2>> anchors_large {{0.6f, 0.6f}, {0.7f, 0.7f}, {0.8f, 0.8f}};
    const vector<std::array<float, 2>> anchors_medium{{0.3f, 0.3f}, {0.4f, 0.4f}, {0.5f, 0.5f}};
    const vector<std::array<float, 2>> anchors_small {{0.1f, 0.1f}, {0.15f, 0.15f}, {0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{input_H, input_W, 3},
                              2,                    3, anchors_large);
    dataset.set_multi_scale_heads(head_grids, {anchors_large, anchors_medium, anchors_small});
    {
        YoloDataset::AugmentationPolicy no_augmentation;
        no_augmentation.enabled = false;
        dataset.set_augmentation_policy(no_augmentation);
    }

    Network network;

    const Index s1 = network.add_layer(make_unique<Convolutional>(
                         Shape{input_H, input_W, 3}, Shape{3, 3, 3, 4},
                         "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "stage1"));

    const Index s2 = network.add_layer(make_unique<Convolutional>(
                         Shape{input_H, input_W, 4}, Shape{3, 3, 4, 4},
                         "Identity", Shape{2, 2}, "Same", BatchNormalization::No, "stage2"));

    const Index s3 = network.add_layer(make_unique<Convolutional>(
                         Shape{4, 4, 4}, Shape{3, 3, 4, 4},
                         "Identity", Shape{2, 2}, "Same", BatchNormalization::No, "stage3"));

    network.add_layer(make_unique<Convolutional>(
        Shape{2, 2, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits_large"), {s3});
    network.add_layer(make_unique<Detection>(
        Shape{2, 2, head_channels}, anchors_large, "detection_large"));

    network.add_layer(make_unique<Convolutional>(
        Shape{4, 4, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits_medium"), {s2});
    network.add_layer(make_unique<Detection>(
        Shape{4, 4, head_channels}, anchors_medium, "detection_medium"));

    network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits_small"), {s1});
    network.add_layer(make_unique<Detection>(
        Shape{input_H, input_W, head_channels}, anchors_small, "detection_small"));

    network.compile();
    network.get_parameters_map().setConstant(0.05f);

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    Index sum_per_layer = 0;
    for (Index i = 0; i < network.get_layers_number(); ++i)
        sum_per_layer += network.get_layer(i)->get_parameters_number();

    Index worst_idx = 0;
    float worst = 0.0f;
    for (Index i = 0; i < sum_per_layer; ++i)
    {
        const float d = abs(gradient(i) - numerical_gradient(i));
        if (d > worst) { worst = d; worst_idx = i; }
    }

    EXPECT_LT(worst, 0.01f)
        << "gradient mismatch at parameter " << worst_idx
        << ": analytic " << gradient(worst_idx)
        << ", numerical " << numerical_gradient(worst_idx)
        << " (difference " << worst << ")";
}

TEST(YoloFPN, C3k2ScalingParameterCountsMonotonic)
{
    const Shape input_shape{320, 320, 3};
    const int classes = 4;
    const vector<std::array<float, 2>> anchors(9, {0.1f, 0.1f});
    const int grid_size = 10;

    using B  = Yolo::Backbone;
    using CA = Yolo::ClassActivation;
    using HS = Yolo::HeadStyle;
    using BA = Yolo::BodyActivation;
    using MS = Yolo::ModelSize;

    Index prev_params = 0;
    for (MS ms : {MS::n, MS::s, MS::m, MS::l, MS::x})
    {
        Yolo network(input_shape, classes, anchors, grid_size,
                       B::CSPDarknet53v11, CA::Sigmoid, HS::FPNv8,
                       BA::LeakyReLU, false, 16, ms);
        const Index params = network.get_parameters_number();
        EXPECT_GT(params, prev_params)
            << "Parameter count did not increase for model_size=" << static_cast<int>(ms);
        prev_params = params;
    }
}


// reg_max, use_sppf and model_size are read by some backbone/head branches and
// ignored by the rest. A combination that ignores one used to build a network
// that quietly disagreed with what the caller asked for, so each is rejected.
TEST(YoloFPN, RejectsParametersTheChosenBackboneAndHeadIgnore)
{
    const Shape input_shape{320, 320, 3};
    const Index classes = 4;
    const vector<std::array<float, 2>> anchors(9, {0.1f, 0.1f});
    const Index grid_size = 10;

    using B  = Yolo::Backbone;
    using CA = Yolo::ClassActivation;
    using HS = Yolo::HeadStyle;
    using BA = Yolo::BodyActivation;
    using MS = Yolo::ModelSize;

    const auto build = [&](B backbone, HS head, bool use_sppf, Index reg_max, MS model_size)
    {
        Yolo network(input_shape, classes, anchors, grid_size,
                            backbone, CA::Sigmoid, head, BA::LeakyReLU,
                            use_sppf, reg_max, model_size);
    };

    // reg_max is a distribution-focal parameter: only the v8 head reads it.
    EXPECT_THROW(build(B::Darknet53, HS::PANet, false, 16, MS::l), runtime_error);

    // SPPF is inserted by the Darknet53 FPN/PANet neck and nowhere else.
    EXPECT_THROW(build(B::Darknet53, HS::FPNv8, true, 16, MS::l), runtime_error);

    // model_size scales the v11 backbone's width and depth multipliers.
    EXPECT_THROW(build(B::Darknet53, HS::PANet, false, 1, MS::n), runtime_error);

    // v11 builds a v8 head only, and used to throw after building the backbone.
    EXPECT_THROW(build(B::CSPDarknet53v11, HS::PANet, false, 16, MS::l), runtime_error);

    // The combinations each parameter does apply to still build.
    EXPECT_NO_THROW(build(B::Darknet53, HS::PANet, true, 1, MS::l));
    EXPECT_NO_THROW(build(B::CSPDarknet53v11, HS::FPNv8, false, 16, MS::n));
}
