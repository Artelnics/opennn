#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/yolo_dataset.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/concatenation_layer.h"
#include "opennn/neural_network/layers/non_max_suppression_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/loss.h"

#include "tests/test_helpers.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace opennn;
using namespace opennn_test;

namespace {

struct YoloLossFixture
{
    TempDir dir{"opennn_yolo_loss_test_"};
    filesystem::path images_dir;
    filesystem::path labels_dir;

    static constexpr Index W = 2;
    static constexpr Index H = 2;
    static constexpr Index grid = 2;
    static constexpr Index B = 2;
    static constexpr Index C = 1;
    static constexpr Index channels = B * (5 + C);

    const vector<std::array<float, 2>> anchors{{0.2f, 0.2f}, {0.5f, 0.5f}};

    YoloLossFixture()
    {
        images_dir = dir.path / "images";
        labels_dir = dir.path / "labels";
        filesystem::create_directories(images_dir);
        filesystem::create_directories(labels_dir);
        write_classes(labels_dir / "classes.names", {"only"});
    }
};

void build_yolo_network(NeuralNetwork& net, const YoloLossFixture& f)
{
    net.add_layer(make_unique<Convolutional>(Shape{f.H, f.W, 3},
                                             Shape{1, 1, 3, f.channels},
                                             "Identity",
                                             Shape{1, 1},
                                             "Same",
                                             BatchNormalization::No,
                                             "yolo_logits"));
    net.add_layer(make_unique<Detection>(Shape{f.grid, f.grid, f.channels}, f.anchors, "detection"));
    net.compile();
    net.get_parameters_map().setConstant(0.1f);
}

}

TEST(YoloLoss, OutputDeltaLayersFollowSelectedLoss)
{
    for (const bool v8 : {false, true})
    {
        SCOPED_TRACE(v8 ? "DetectionV8" : "Detection");

        constexpr Index height = 2;
        constexpr Index width = 2;
        constexpr Index features = 4;
        const Index head_channels = v8 ? 5 : 6;
        const vector<std::array<float, 2>> anchors{{0.5f, 0.5f}};

        NeuralNetwork network;
        const Index stem = network.add_layer(make_unique<Convolutional>(
                               Shape{height, width, 3}, Shape{1, 1, 3, features},
                               "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "stem"));

        const auto add_head = [&](const string& suffix)
        {
            const Index logits = network.add_layer(make_unique<Convolutional>(
                                     Shape{height, width, features}, Shape{1, 1, features, head_channels},
                                     "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits_" + suffix), {stem});

            if (v8)
                network.add_layer(make_unique<DetectionV8>(
                    Shape{height, width, head_channels}, "detection_" + suffix), {logits});
            else
                network.add_layer(make_unique<Detection>(
                    Shape{height, width, head_channels}, anchors, "detection_" + suffix), {logits});

            return network.get_layers_number() - 1;
        };

        const Index first_head = add_head("first");
        const Index last_head = add_head("last");
        network.compile();

        Loss standard_loss(&network);
        EXPECT_EQ(standard_loss.get_output_delta_layer_indices(), vector<Index>{last_head});

        BackPropagation standard_back_propagation(2, standard_loss);
        EXPECT_TRUE(standard_back_propagation.output_deltas[size_t(first_head)].empty());
        EXPECT_FALSE(standard_back_propagation.output_deltas[size_t(last_head)].empty());

        Loss yolo_loss(&network);
        yolo_loss.set_error(Loss::Error::Yolo);
        EXPECT_EQ(yolo_loss.get_output_delta_layer_indices(),
                  (vector<Index>{first_head, last_head}));

        InferenceShapePolicy policy;
        policy.retained_output_layers = yolo_loss.get_output_delta_layer_indices();
        ForwardPropagation validation_propagation(
            2, &network, ForwardPropagationMode::Inference, policy);

        const TensorView& first_output =
            validation_propagation.slots[size_t(first_head)].back();
        const TensorView& last_output =
            validation_propagation.slots[size_t(last_head)].back();
        ASSERT_FALSE(first_output.empty());
        ASSERT_FALSE(last_output.empty());

        const auto first_begin = reinterpret_cast<uintptr_t>(first_output.get_data());
        const auto first_end = first_begin + uintptr_t(first_output.byte_size());
        const auto last_begin = reinterpret_cast<uintptr_t>(last_output.get_data());
        const auto last_end = last_begin + uintptr_t(last_output.byte_size());
        EXPECT_TRUE(first_end <= last_begin || last_end <= first_begin);

        BackPropagation yolo_back_propagation(2, yolo_loss);
        EXPECT_FALSE(yolo_back_propagation.output_deltas[size_t(first_head)].empty());
        EXPECT_FALSE(yolo_back_propagation.output_deltas[size_t(last_head)].empty());
    }
}

TEST(YoloLoss, InferencePolicyRetainsConsumedHead)
{
    constexpr Index height = 2;
    constexpr Index width = 2;
    constexpr Index features = 4;
    constexpr Index head_channels = 6;
    const vector<std::array<float, 2>> anchors{{0.5f, 0.5f}};

    NeuralNetwork network;
    const Index stem = network.add_layer(make_unique<Convolutional>(
                           Shape{height, width, 3}, Shape{1, 1, 3, features},
                           "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "stem"));

    const Index logits = network.add_layer(make_unique<Convolutional>(
                             Shape{height, width, features}, Shape{1, 1, features, head_channels},
                             "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits"), {stem});

    const Index detection = network.add_layer(make_unique<Detection>(
                                Shape{height, width, head_channels}, anchors, "detection"), {logits});

    network.add_layer(make_unique<NonMaxSuppression>(
        Shape{height, width, head_channels}, 1, 0.5f, 0.4f, "nms"), {detection});

    const Index tail = network.add_layer(make_unique<Convolutional>(
                           Shape{height, width, features}, Shape{1, 1, features, features},
                           "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "tail"), {stem});
    network.compile();

    Loss yolo_loss(&network);
    yolo_loss.set_error(Loss::Error::Yolo);
    ASSERT_EQ(yolo_loss.get_output_delta_layer_indices(), vector<Index>{detection});

    InferenceShapePolicy policy;
    policy.retained_output_layers = yolo_loss.get_output_delta_layer_indices();
    ForwardPropagation validation_propagation(
        2, &network, ForwardPropagationMode::Inference, policy);

    const TensorView& detection_output =
        validation_propagation.slots[size_t(detection)].back();
    const TensorView& tail_output = validation_propagation.slots[size_t(tail)].back();
    ASSERT_FALSE(detection_output.empty());
    ASSERT_FALSE(tail_output.empty());

    const auto detection_begin = reinterpret_cast<uintptr_t>(detection_output.get_data());
    const auto detection_end = detection_begin + uintptr_t(detection_output.byte_size());
    const auto tail_begin = reinterpret_cast<uintptr_t>(tail_output.get_data());
    const auto tail_end = tail_begin + uintptr_t(tail_output.byte_size());
    EXPECT_TRUE(detection_end <= tail_begin || tail_end <= detection_begin);
}

TEST(YoloLoss, NoObjectGradientMatchesNumericalGradient)
{

    YoloLossFixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    { ofstream empty_a(f.labels_dir / "a.txt"); }
    { ofstream empty_b(f.labels_dir / "b.txt"); }

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, f.B, f.anchors);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    NeuralNetwork neural_network;
    build_yolo_network(neural_network, f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

TEST(YoloLoss, WithObjectGradientMatchesV1Approximation)
{

    YoloLossFixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    write_label(f.labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_label(f.labels_dir / "b.txt", 0, 0.25f, 0.75f, 0.2f, 0.2f);

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, f.B, f.anchors);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    NeuralNetwork neural_network;
    build_yolo_network(neural_network, f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

namespace {

struct YoloLossV8Fixture
{
    TempDir dir{"opennn_yolo_loss_test_"};
    filesystem::path images_dir;
    filesystem::path labels_dir;

    static constexpr Index W = 2;
    static constexpr Index H = 2;
    static constexpr Index grid = 2;
    static constexpr Index C = 1;
    static constexpr Index ch = 4 + C;

    YoloLossV8Fixture()
    {
        images_dir = dir.path / "images";
        labels_dir = dir.path / "labels";
        filesystem::create_directories(images_dir);
        filesystem::create_directories(labels_dir);
        write_classes(labels_dir / "classes.names", {"only"});
    }
};

void build_yolo_v8_network(NeuralNetwork& net, const YoloLossV8Fixture& f)
{
    net.add_layer(make_unique<Convolutional>(Shape{f.H, f.W, 3},
                                             Shape{1, 1, 3, f.ch},
                                             "Identity",
                                             Shape{1, 1},
                                             "Same",
                                             BatchNormalization::No,
                                             "v8_logits"));
    net.add_layer(make_unique<DetectionV8>(Shape{f.grid, f.grid, f.ch}, "detection_v8"));
    net.compile();
    net.get_parameters_map().setConstant(0.1f);
}

}

TEST(YoloLoss, V8UsesDetectionContractWithGenericDataset)
{
    YoloLossV8Fixture f;
    constexpr Index samples_number = 2;
    constexpr Index max_gt_boxes = 3;
    TabularDataset dataset(samples_number,
                           Shape{f.H, f.W, 3},
                           Shape{max_gt_boxes * 5});
    dataset.set_data_constant(0.0f);
    dataset.set_sample_roles(SampleRole::Training);

    NeuralNetwork neural_network;
    build_yolo_v8_network(neural_network, f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient          = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

TEST(YoloLoss, V8WithObjectGradientMatchesNumericalGradient)
{

    YoloLossV8Fixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    write_label(f.labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_label(f.labels_dir / "b.txt", 0, 0.25f, 0.75f, 0.2f, 0.2f);

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, 0, {});
    dataset.set_v8_mode(true);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    NeuralNetwork neural_network;
    build_yolo_v8_network(neural_network, f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient          = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

TEST(YoloLoss, V8DecoupledHeadGradientMatchesNumericalGradient)
{

    YoloLossV8Fixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    write_label(f.labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.4f, 0.4f);
    write_label(f.labels_dir / "b.txt", 0, 0.25f, 0.75f, 0.2f, 0.2f);

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, 0, {});
    dataset.set_v8_mode(true);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    NeuralNetwork net;
    constexpr Index head_ch = 4;
    const Shape feat{f.H, f.W, head_ch};

    const Index stem = net.add_layer(make_unique<Convolutional>(Shape{f.H, f.W, 3},
                                                                Shape{1, 1, 3, head_ch},
                                                                "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "stem"));

    const Index box_out = net.add_layer(make_unique<Convolutional>(feat, Shape{1, 1, head_ch, 4},
                                                                   "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "box_out"), {stem});

    const Index cls_out = net.add_layer(make_unique<Convolutional>(feat, Shape{1, 1, head_ch, f.C},
                                                                   "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "cls_out"), {stem});

    const Shape box_shape{f.grid, f.grid, 4};
    const Index cat = net.add_layer(make_unique<Concatenation>(box_shape, vector<Index>{4, f.C}, "cat"),
                                    {box_out, cls_out});
    net.add_layer(make_unique<DetectionV8>(Shape{f.grid, f.grid, f.ch}, "det_v8"), {cat});

    net.compile();
    net.get_parameters_map().setConstant(0.1f);

    Loss loss(&net, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient          = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

TEST(YoloLoss, V8TALVFLGradientMatchesNumerical)
{
    YoloLossV8Fixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    write_label(f.labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.6f, 0.6f);
    write_label(f.labels_dir / "b.txt", 0, 0.5f, 0.5f, 0.6f, 0.6f);

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, f.grid, 0, {});
    dataset.set_v8_mode(true);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    NeuralNetwork neural_network;
    build_yolo_v8_network(neural_network, f);

    Loss loss_fn(&neural_network, &dataset);
    loss_fn.set_error(Loss::Error::Yolo);
    loss_fn.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient           = calculate_gradient(loss_fn);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss_fn);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}

TEST(YoloLoss, V8DFLGradientMatchesNumerical)
{
    YoloLossV8Fixture f;
    write_bmp_24(f.images_dir / "a.bmp", f.W, f.H, 200, 100, 50);
    write_bmp_24(f.images_dir / "b.bmp", f.W, f.H,  50, 200, 100);
    write_label(f.labels_dir / "a.txt", 0, 0.5f, 0.5f, 0.6f, 0.6f);
    write_label(f.labels_dir / "b.txt", 0, 0.5f, 0.5f, 0.6f, 0.6f);

    constexpr Index grid = 2;
    constexpr Index C    = 1;
    constexpr Index rm   = 2;
    constexpr Index dfl_ch = 4 * rm + C;

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(f.images_dir, f.labels_dir, Shape{f.H, f.W, 3}, grid, 0, {});
    dataset.set_v8_mode(true);

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;
    dataset.set_augmentation_policy(no_aug);

    NeuralNetwork net;
    net.add_layer(make_unique<Convolutional>(Shape{f.H, f.W, 3},
                                             Shape{1, 1, 3, dfl_ch},
                                             "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "v8_logits"));
    net.add_layer(make_unique<DetectionV8>(Shape{grid, grid, dfl_ch}, rm, "detection_v8"));
    net.compile();
    net.get_parameters_map().setConstant(0.1f);

    Loss loss_fn(&net, &dataset);
    loss_fn.set_error(Loss::Error::Yolo);
    loss_fn.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient           = calculate_gradient(loss_fn);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss_fn);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 0.5f);
}
