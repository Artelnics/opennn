#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/dataset/batch.h"
#include "opennn/neural_network/standard_networks.h"

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

    constexpr Index input_H = 8, input_W = 8, classes_number = 1, boxes_per_cell = 3;
    constexpr Index head_channels = boxes_per_cell * (5 + classes_number);
    const vector<std::array<float, 2>> anchors{{0.1f, 0.1f}, {0.15f, 0.15f}, {0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{input_H, input_W, 3},
                              8, boxes_per_cell, anchors);
    {
        YoloDataset::AugmentationConfig no_aug; no_aug.enabled = false;
        dataset.set_augmentation(no_aug);
    }

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 3}, Shape{3, 3, 3, 4},
        "Identity", Shape{1, 1}, "Same", false, "conv1"));
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits"));
    neural_network.add_layer(make_unique<Detection>(
        Shape{input_H, input_W, head_channels}, anchors, "detection"));
    neural_network.compile();
    neural_network.get_parameters_map().setConstant(0.05f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    {
        const Index sn = dataset.get_samples_number("Training");
        const vector<Index> ti = dataset.get_sample_indices("Training");
        Batch batch_diag(sn, &dataset, neural_network.get_config());
        batch_diag.fill(ti, dataset.get_feature_selection());
        ForwardPropagation fp_diag(sn, &neural_network);
        neural_network.forward_propagate(batch_diag.get_inputs(), fp_diag, true);
        BackPropagation bp_diag(sn, loss);
        const float L0 = loss.calculate_error(batch_diag, fp_diag).error;

        VectorMap params = neural_network.get_parameters_map();
        VectorR perturbed = params;
        const float h = calculate_h(params(0));
        perturbed(0) += h;
        neural_network.forward_propagate(batch_diag.get_inputs(), perturbed, fp_diag);
        const float Lp = loss.calculate_error(batch_diag, fp_diag).error;
        perturbed(0) -= 2.0f*h;
        neural_network.forward_propagate(batch_diag.get_inputs(), perturbed, fp_diag);
        const float Lm = loss.calculate_error(batch_diag, fp_diag).error;
        neural_network.forward_propagate(batch_diag.get_inputs(), params, fp_diag);

        cout << "Manual check param[0]: L0=" << L0 << " L+=" << Lp << " L-=" << Lm
                  << " h=" << h << "\n"
                  << "  manual_num=" << (Lp-Lm)/(2.0f*h)
                  << "  c4 effect: 3boxes*0.05w*sum_cells_c4\n";

        for (float hh : {1e-4f, 1e-3f, 1e-2f}) {
            VectorR p_ph = VectorR(params); p_ph(0) += hh;
            VectorR p_mh = VectorR(params); p_mh(0) -= hh;
            neural_network.forward_propagate(batch_diag.get_inputs(), p_ph, fp_diag);
            const float Lph = loss.calculate_error(batch_diag, fp_diag).error;
            neural_network.forward_propagate(batch_diag.get_inputs(), p_mh, fp_diag);
            const float Lmh = loss.calculate_error(batch_diag, fp_diag).error;
            cout << "  h=" << hh << " num_grad=" << (Lph-Lmh)/(2.0f*hh) << "\n";
        }
        {VectorR r = VectorR(params); neural_network.forward_propagate(batch_diag.get_inputs(), r, fp_diag);}
    }
    neural_network.get_parameters_map().setConstant(0.05f);
    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    Index offset = 0;
    float worst = 0.0f;
    Index worst_idx = 0;
    for (Index li = 0; li < neural_network.get_layers_number(); ++li)
    {
        const Index np = neural_network.get_layer(li)->get_parameters_number();
        if (np == 0) { cout << "  layer " << li << " (" << neural_network.get_layer(li)->get_label() << "): no params\n"; continue; }
        float lworst = 0.0f; Index lwi = offset;
        for (Index k = offset; k < offset + np; ++k)
        {
            const float d = abs(gradient(k) - numerical_gradient(k));
            if (d > lworst) { lworst = d; lwi = k; }
            if (d > worst)  { worst  = d; worst_idx = k; }
        }
        cout << "  layer " << li << " (" << neural_network.get_layer(li)->get_label()
                  << "): worst=" << lworst << " at local[" << (lwi-offset) << "]"
                  << " grad=" << gradient(lwi) << " num=" << numerical_gradient(lwi) << "\n";
        cout << "    first 4 grad: ";
        for (Index k = offset; k < min(offset+4, offset+np); ++k) cout << gradient(k) << " ";
        cout << "\n    first 4 num:  ";
        for (Index k = offset; k < min(offset+4, offset+np); ++k) cout << numerical_gradient(k) << " ";
        cout << "\n";
        offset += np;
    }
    cout << "\n[SingleHead8x8] Overall worst at idx " << worst_idx
              << ": grad=" << gradient(worst_idx) << " num=" << numerical_gradient(worst_idx)
              << " ratio=" << gradient(worst_idx)/numerical_gradient(worst_idx) << "\n";

    cout << "gradient.size()=" << gradient.size()
              << " sum_per_layer=" << offset
              << " get_parameters_buffer_size()=" << neural_network.get_parameters_buffer_size() << "\n";
    const Index conv1_np = neural_network.get_layer(0)->get_parameters_number();
    cout << "conv1 params=" << conv1_np << " logits params=" << neural_network.get_layer(1)->get_parameters_number() << "\n";
    Index logits_true_start = conv1_np;
    for (Index k = conv1_np; k < gradient.size(); ++k)
        if (gradient(k) != 0.0f || numerical_gradient(k) != 0.0f)
            { logits_true_start = k; break; }
    cout << "logits_true_start=" << logits_true_start << "\n";
    cout << "logits grad[0..17]: ";
    for (Index k = logits_true_start; k < logits_true_start+18 && k < gradient.size(); ++k)
        cout << gradient(k) << " ";
    cout << "\nlogits num[0..17]: ";
    for (Index k = logits_true_start; k < logits_true_start+18 && k < numerical_gradient.size(); ++k)
        cout << numerical_gradient(k) << " ";
    cout << "\n";

    EXPECT_LT(worst, 0.05f);
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
        YoloDataset::AugmentationConfig no_aug; no_aug.enabled = false;
        dataset.set_augmentation(no_aug);
    }

    NeuralNetwork neural_network;

    const Index s1 = neural_network.add_layer(make_unique<Convolutional>(
                         Shape{input_H, input_W, 3}, Shape{3, 3, 3, 4},
                         "Identity", Shape{1, 1}, "Same", false, "stage1"));

    const Index s2 = neural_network.add_layer(make_unique<Convolutional>(
                         Shape{input_H, input_W, 4}, Shape{3, 3, 4, 4},
                         "Identity", Shape{2, 2}, "Same", false, "stage2"));

    const Index s3 = neural_network.add_layer(make_unique<Convolutional>(
                         Shape{4, 4, 4}, Shape{3, 3, 4, 4},
                         "Identity", Shape{2, 2}, "Same", false, "stage3"));

    neural_network.add_layer(make_unique<Convolutional>(
        Shape{2, 2, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits_large"), {s3});
    neural_network.add_layer(make_unique<Detection>(
        Shape{2, 2, head_channels}, anchors_large, "detection_large"));

    neural_network.add_layer(make_unique<Convolutional>(
        Shape{4, 4, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits_medium"), {s2});
    neural_network.add_layer(make_unique<Detection>(
        Shape{4, 4, head_channels}, anchors_medium, "detection_medium"));

    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits_small"), {s1});
    neural_network.add_layer(make_unique<Detection>(
        Shape{input_H, input_W, head_channels}, anchors_small, "detection_small"));

    neural_network.compile();
    neural_network.get_parameters_map().setConstant(0.05f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    Index sum_per_layer = 0;
    for (Index i = 0; i < neural_network.get_layers_number(); ++i)
        sum_per_layer += neural_network.get_layer(i)->get_parameters_number();

    Index worst_idx = 0;
    float worst = 0.0f;
    for (Index i = 0; i < sum_per_layer; ++i)
    {
        const float d = abs(gradient(i) - numerical_gradient(i));
        if (d > worst) { worst = d; worst_idx = i; }
    }

    EXPECT_LT(worst, 0.01f)
        << "Worst at idx " << worst_idx
        << ": grad=" << gradient(worst_idx)
        << " num=" << numerical_gradient(worst_idx);
}

TEST(YoloFPN, C3k2ScalingParameterCountsMonotonic)
{
    const Shape input_shape{320, 320, 3};
    const int classes = 4;
    const vector<std::array<float, 2>> anchors(9, {0.1f, 0.1f});
    const int grid_size = 10;

    using B  = YoloNetwork::Backbone;
    using CA = YoloNetwork::ClassActivation;
    using HS = YoloNetwork::HeadStyle;
    using BA = YoloNetwork::BodyActivation;
    using MS = YoloNetwork::ModelSize;

    Index prev_params = 0;
    for (MS ms : {MS::n, MS::s, MS::m, MS::l, MS::x})
    {
        YoloNetwork nn(input_shape, classes, anchors, grid_size,
                       B::CSPDarknet53v11, CA::Sigmoid, HS::FPNv8,
                       BA::LeakyReLU, false, 16, ms);
        const Index params = nn.get_parameters_number();
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

    using B  = YoloNetwork::Backbone;
    using CA = YoloNetwork::ClassActivation;
    using HS = YoloNetwork::HeadStyle;
    using BA = YoloNetwork::BodyActivation;
    using MS = YoloNetwork::ModelSize;

    const auto build = [&](B backbone, HS head, bool use_sppf, Index reg_max, MS model_size)
    {
        YoloNetwork network(input_shape, classes, anchors, grid_size,
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
