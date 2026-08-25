#include "tests/pch.h"

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/concatenation_layer.h"
#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/layers/activation_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"

#include "tests/test_helpers.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace opennn;
using namespace opennn_test;

TEST(YoloOverfit, SingleImageSingleClassLossDecreases)
{

    TempDir dir("opennn_yolo_overfit_");
    const filesystem::path images_dir = dir.path / "images";
    const filesystem::path labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200, 100,  50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 100, 200);

    {
        ofstream lf(labels_dir / "a.txt");
        lf << "0 0.5 0.5 0.4 0.4\n";
    }
    {
        ofstream lf(labels_dir / "b.txt");
        lf << "0 0.25 0.75 0.4 0.4\n";
    }
    {
        ofstream nf(labels_dir / "classes.names");
        nf << "object\n";
    }

    constexpr Index H = 32, W = 32, grid = 4, B = 1, C = 1;
    constexpr Index channels = B * (5 + C);
    const vector<std::array<float, 2>> anchors{{0.4f, 0.4f}};

    YoloDataset::AugmentationPolicy no_aug;
    no_aug.enabled = false;

    auto build_net = [&]() {
        auto net = make_unique<NeuralNetwork>();
        net->add_layer(make_unique<Convolutional>(
            Shape{H, W, 3}, Shape{3, 3, 3, 16}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::No, "conv1"));
        net->add_layer(make_unique<Convolutional>(
            Shape{H, W, 16}, Shape{1, 1, 16, channels}, "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits"));
        net->add_layer(make_unique<Detection>(Shape{grid, grid, channels}, anchors, "detection"));
        net->compile();
        net->get_parameters_map().setConstant(0.05f);
        return net;
    };

    float error_short;
    {
        YoloDataset ds;
        ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation_policy(no_aug);
        auto net = build_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(2);
        adam.set_display(false);
        error_short = adam.train().get_training_error();
    }

    float error_long;
    {
        YoloDataset ds;
        ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation_policy(no_aug);
        auto net = build_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(200);
        adam.set_display(false);
        error_long = adam.train().get_training_error();
    }

    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease: short=" << error_short << " long=" << error_long
        << " — systematic bug in YOLO loss/gradient/optimizer chain.";

    EXPECT_LT(error_long, error_short * 0.95f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") after 200 epochs — optimizer or gradient likely broken.";
}

TEST(YoloOverfit, SPPFGradientFlowsAndLossDecreases)
{

    TempDir dir("opennn_yolo_overfit_");
    const auto images_dir = dir.path / "images";
    const auto labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200, 50, 50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 200, 50);
    write_bmp_24(images_dir / "c.bmp", 32, 32,  50, 50, 200);
    write_bmp_24(images_dir / "d.bmp", 32, 32, 200, 200, 50);
    { ofstream f(labels_dir / "a.txt"); f << "0 0.5 0.5 0.4 0.4\n"; }
    { ofstream f(labels_dir / "b.txt"); f << "0 0.25 0.5 0.3 0.3\n"; }
    { ofstream f(labels_dir / "c.txt"); f << "0 0.75 0.5 0.3 0.3\n"; }
    { ofstream f(labels_dir / "d.txt"); f << "0 0.5 0.25 0.4 0.3\n"; }

    constexpr Index H = 32, W = 32, grid = 4, B = 1, C = 1;
    const vector<std::array<float, 2>> anchors{{0.4f, 0.4f}};
    constexpr Index ch = 16;
    constexpr Index logit_ch = B * (5 + C);

    YoloDataset::AugmentationPolicy no_aug; no_aug.enabled = false;

    auto build_sppf_net = [&]() {
        auto net = make_unique<NeuralNetwork>();

        net->add_layer(make_unique<Convolutional>(
            Shape{H, W, 3}, Shape{3, 3, 3, ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "conv_stem"));

        const Shape feat{H, W, ch};
        const Index stem_idx = net->get_layers_number() - 1;

        const Index p1_idx = net->add_layer(make_unique<Pooling>(feat, Shape{5,5}, Shape{1,1}, Shape{2,2},
                                                                       "MaxPooling", "sppf_p1"), {stem_idx});
        const Index p2_idx = net->add_layer(make_unique<Pooling>(feat, Shape{5,5}, Shape{1,1}, Shape{2,2},
                                                                       "MaxPooling", "sppf_p2"), {p1_idx});
        const Index p3_idx = net->add_layer(make_unique<Pooling>(feat, Shape{5,5}, Shape{1,1}, Shape{2,2},
                                                                       "MaxPooling", "sppf_p3"), {p2_idx});

        const Index cat_idx = net->add_layer(make_unique<Concatenation>(feat,
                                                                              vector<Index>{ch, ch, ch, ch}, "sppf_cat"),
                                             {stem_idx, p1_idx, p2_idx, p3_idx});

        net->add_layer(make_unique<Convolutional>(
            Shape{H, W, 4*ch}, Shape{1,1, 4*ch, ch}, "LeakyReLU", Shape{1,1}, "Same", BatchNormalization::Yes, "sppf_out"), {cat_idx});
        net->add_layer(make_unique<Convolutional>(
            Shape{H, W, ch}, Shape{1,1, ch, logit_ch}, "Identity", Shape{1,1}, "Same", BatchNormalization::No, "logits"));
        net->add_layer(make_unique<Detection>(Shape{grid, grid, logit_ch}, anchors, "detection"));

        net->compile();
        net->get_parameters_map().setConstant(0.05f);
        return net;
    };

    auto run = [&](Index epochs) -> float {
        YoloDataset ds; ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation_policy(no_aug);
        auto net = build_sppf_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(epochs);
        adam.set_display(false);
        return adam.train().get_training_error();
    };

    const float error_short = run(2);
    const float error_long  = run(150);

    EXPECT_FALSE(isnan(error_short)) << "NaN after 2 epochs — forward/backward through SPPF broken.";
    EXPECT_FALSE(isnan(error_long))  << "NaN after 150 epochs — SPPF gradient instability.";
    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease through SPPF layers: short=" << error_short << " long=" << error_long;
    EXPECT_LT(error_long, error_short * 0.90f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") — backprop through pooling+concat may be broken.";
}

TEST(YoloOverfit, CSPGradientFlowsAndLossDecreases)
{

    TempDir dir("opennn_yolo_overfit_");
    const auto images_dir = dir.path / "images";
    const auto labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200,  50,  50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 200,  50);
    write_bmp_24(images_dir / "c.bmp", 32, 32,  50,  50, 200);
    write_bmp_24(images_dir / "d.bmp", 32, 32, 200, 200,  50);
    { ofstream f(labels_dir / "a.txt"); f << "0 0.5 0.5 0.4 0.4\n"; }
    { ofstream f(labels_dir / "b.txt"); f << "0 0.25 0.5 0.3 0.3\n"; }
    { ofstream f(labels_dir / "c.txt"); f << "0 0.75 0.5 0.3 0.3\n"; }
    { ofstream f(labels_dir / "d.txt"); f << "0 0.5 0.25 0.4 0.3\n"; }
    { ofstream f(labels_dir / "classes.names"); f << "object\n"; }

    constexpr Index H = 32, W = 32, grid = 4, B = 1, C = 1;
    const vector<std::array<float, 2>> anchors{{0.4f, 0.4f}};
    constexpr Index ch = 8;
    constexpr Index half = ch / 2;
    constexpr Index logit_ch = B * (5 + C);

    YoloDataset::AugmentationPolicy no_aug; no_aug.enabled = false;

    auto build_csp_net = [&]() {
        auto net = make_unique<NeuralNetwork>();

        const Shape input{H, W, 3};
        const Index stem = net->add_layer(make_unique<Convolutional>(
                               input, Shape{3, 3, 3, ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "stem"));
        const Shape feat{H, W, ch};

        const Index branch1 = net->add_layer(make_unique<Convolutional>(
                                  feat, Shape{1, 1, ch, half}, "Identity", Shape{1, 1}, "Same", BatchNormalization::Yes, "csp_s1"), {stem});

        const Index b2_start = net->add_layer(make_unique<Convolutional>(
                                   feat, Shape{1, 1, ch, half}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "csp_s2"), {stem});

        const Shape hfeat{H, W, half};
        const Index b1c1 = net->add_layer(make_unique<Convolutional>(
                               hfeat, Shape{1, 1, half, half}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "csp_b1_c1"), {b2_start});
        const Index b1c2 = net->add_layer(make_unique<Convolutional>(
                               hfeat, Shape{3, 3, half, half}, "Identity", Shape{1, 1}, "Same", BatchNormalization::Yes, "csp_b1_c2"), {b1c1});
        const Index add = net->add_layer(make_unique<Addition>(hfeat, "csp_b1_add"), {b1c2, b2_start});
        const Index branch2 = net->add_layer(make_unique<Activation>(hfeat, "LeakyReLU", "csp_b1_act"), {add});

        const Index cat = net->add_layer(make_unique<Concatenation>(hfeat, vector<Index>{half, half}, "csp_cat"),
                                         {branch1, branch2});
        const Index merge = net->add_layer(make_unique<Convolutional>(
                                feat, Shape{1, 1, ch, ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "csp_merge"), {cat});

        net->add_layer(make_unique<Convolutional>(
            feat, Shape{1, 1, ch, logit_ch}, "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "logits"), {merge});
        net->add_layer(make_unique<Detection>(Shape{grid, grid, logit_ch}, anchors, "detection"));

        net->compile();
        net->get_parameters_map().setConstant(0.05f);
        return net;
    };

    auto run = [&](Index epochs) -> float {
        YoloDataset ds; ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, B, anchors);
        ds.set_augmentation_policy(no_aug);
        auto net = build_csp_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(epochs);
        adam.set_display(false);
        return adam.train().get_training_error();
    };

    const float error_short = run(2);
    const float error_long  = run(150);

    EXPECT_FALSE(isnan(error_short)) << "NaN after 2 epochs — forward/backward through CSP broken.";
    EXPECT_FALSE(isnan(error_long))  << "NaN after 150 epochs — CSP gradient instability.";
    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease through CSP layers: short=" << error_short << " long=" << error_long;
    EXPECT_LT(error_long, error_short * 0.90f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") — backprop through CSP split/concat/residual may be broken.";
}

TEST(YoloOverfit, V8AnchorFreeGradientFlowsAndLossDecreases)
{

    TempDir dir("opennn_yolo_overfit_");
    const auto images_dir = dir.path / "images";
    const auto labels_dir = dir.path / "labels";
    filesystem::create_directories(images_dir);
    filesystem::create_directories(labels_dir);

    write_bmp_24(images_dir / "a.bmp", 32, 32, 200,  50,  50);
    write_bmp_24(images_dir / "b.bmp", 32, 32,  50, 200,  50);
    write_bmp_24(images_dir / "c.bmp", 32, 32,  50,  50, 200);
    write_bmp_24(images_dir / "d.bmp", 32, 32, 200, 200,  50);

    { ofstream f(labels_dir / "a.txt"); f << "0 0.375 0.375 0.18 0.18\n"; }
    { ofstream f(labels_dir / "b.txt"); f << "0 0.125 0.125 0.15 0.15\n"; }
    { ofstream f(labels_dir / "c.txt"); f << "0 0.875 0.625 0.15 0.15\n"; }
    { ofstream f(labels_dir / "d.txt"); f << "0 0.125 0.875 0.15 0.15\n"; }
    { ofstream f(labels_dir / "classes.names"); f << "object\n"; }

    constexpr Index H = 32, W = 32, grid = 4, C = 1;
    constexpr Index head_ch = 8;
    constexpr Index det_ch  = 4 + C;

    YoloDataset::AugmentationPolicy no_aug; no_aug.enabled = false;

    auto build_v8_net = [&]() {
        auto net = make_unique<NeuralNetwork>();

        const Shape input{H, W, 3};
        const Index stem = net->add_layer(make_unique<Convolutional>(
                               input, Shape{3, 3, 3, head_ch}, "LeakyReLU", Shape{8, 8}, "Same", BatchNormalization::Yes, "stem"));
        const Shape feat{grid, grid, head_ch};

        const Index bc1 = net->add_layer(make_unique<Convolutional>(
                              feat, Shape{3, 3, head_ch, head_ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "box_c1"), {stem});
        const Index bc2 = net->add_layer(make_unique<Convolutional>(
                              feat, Shape{3, 3, head_ch, head_ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "box_c2"), {bc1});
        const Index box_out = net->add_layer(make_unique<Convolutional>(
                                  feat, Shape{1, 1, head_ch, 4}, "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "box_out"), {bc2});

        const Index cc1 = net->add_layer(make_unique<Convolutional>(
                              feat, Shape{3, 3, head_ch, head_ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "cls_c1"), {stem});
        const Index cc2 = net->add_layer(make_unique<Convolutional>(
                              feat, Shape{3, 3, head_ch, head_ch}, "LeakyReLU", Shape{1, 1}, "Same", BatchNormalization::Yes, "cls_c2"), {cc1});
        const Index cls_out = net->add_layer(make_unique<Convolutional>(
                                  feat, Shape{1, 1, head_ch, C}, "Identity", Shape{1, 1}, "Same", BatchNormalization::No, "cls_out"), {cc2});

        const Shape box_shape{grid, grid, 4};
        const Index cat = net->add_layer(make_unique<Concatenation>(box_shape, vector<Index>{4, C}, "cat"),
                                         {box_out, cls_out});
        net->add_layer(make_unique<DetectionV8>(Shape{grid, grid, det_ch}, "det"));

        net->compile();
        net->get_parameters_map().setConstant(0.05f);
        return net;
    };

    auto run = [&](Index epochs) -> float {
        YoloDataset ds; ds.set_display(false);
        ds.set(images_dir, labels_dir, Shape{H, W, 3}, grid, 0, {});
        ds.set_v8_mode(true);
        ds.set_augmentation_policy(no_aug);
        auto net = build_v8_net();
        Loss loss(net.get(), &ds);
        loss.set_error(Loss::Error::Yolo);
        loss.set_regularization(Loss::Regularization::NoRegularization);
        loss.set_yolo_focal_gamma(2.0f);
        loss.set_yolo_lambda_class(0.01f);
        AdaptiveMomentEstimation adam(&loss);
        adam.set_maximum_epochs(epochs);
        adam.set_learning_rate(1e-3f);
        adam.set_display(false);
        return adam.train().get_training_error();
    };

    const float error_short = run(2);
    const float error_long  = run(500);

    EXPECT_FALSE(isnan(error_short)) << "NaN after 2 epochs — v8 forward/backward broken.";
    EXPECT_FALSE(isnan(error_long))  << "NaN after 500 epochs — v8 gradient instability.";
    EXPECT_LT(error_long, error_short)
        << "Loss did not decrease: short=" << error_short << " long=" << error_long;
    EXPECT_LT(error_long, error_short * 0.90f)
        << "Loss barely decreased (" << error_short << " → " << error_long
        << ") — TAL/VFL backprop or v8 loss may be broken.";
}
