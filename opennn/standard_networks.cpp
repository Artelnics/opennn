//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "standard_networks.h"
#include "activation_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "dense_layer.h"
#include "bounding_layer.h"
#include "recurrent_layer.h"
#include "long_short_term_memory_layer.h"
#include "embedding_layer.h"
#include "convolutional_layer.h"
#include "detection_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "non_max_suppression_layer.h"
#include "flatten_layer.h"
#include "addition_layer.h"
#include "upsample_layer.h"
#include "concatenation_layer.h"
#include "normalization_layer_3d.h"
#include "multihead_attention_layer.h"
#include "string_utilities.h"

namespace opennn
{

static bool same_specs(const vector<vector<TensorSpec>>& a, const vector<vector<TensorSpec>>& b)
{
    return ranges::equal(a, b, [](const auto& x, const auto& y)
    {
        return ranges::equal(x, y, [](const TensorSpec& s, const TensorSpec& t)
        {
            return s.shape == t.shape && s.dtype == t.dtype;
        });
    });
}

static void recompile_if_specs_changed(NeuralNetwork& network,
                                       const vector<vector<TensorSpec>>& forward_before,
                                       const vector<vector<TensorSpec>>& backward_before)
{
    if (same_specs(forward_before, network.get_forward_specs(1))
        && same_specs(backward_before, network.get_backward_specs(1)))
        return;

    VectorR parameters_snapshot;
    if (network.get_parameters_size() > 0)
    {
        network.copy_parameters_host();
        parameters_snapshot = Eigen::Map<const VectorR>(network.get_parameters_data(),
                                                        network.get_parameters_size());
    }

    network.compile();

    if (parameters_snapshot.size() > 0)
        network.set_parameters(parameters_snapshot);
}

ApproximationNetwork::ApproximationNetwork(const Shape& input_shape,
                                           const Shape& complexity_dimensions,
                                           const Shape& output_shape,
                                           const string& hidden_activation) : NeuralNetwork()
{
    const Index complexity_size = complexity_dimensions.rank;

    add_layer(make_unique<Scaling>(input_shape));

    for (Index i = 0; i < complexity_size; ++i)
        add_layer(make_unique<Dense>(get_output_shape(),
                                       Shape{ complexity_dimensions[i] },
                                       hidden_activation,
                                       false,
                                       format("dense2d_layer_{}", i + 1)));

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   "Identity",
                                   false,
                                   "approximation_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    add_layer(make_unique<Bounding>(output_shape));

    compile();
    set_parameters_glorot();
}

ClassificationNetwork::ClassificationNetwork(const Shape& input_shape,
                                             const Shape& complexity_dimensions,
                                             const Shape& output_shape) : NeuralNetwork()
{
    const Index complexity_size = complexity_dimensions.rank;

    add_layer(make_unique<Scaling>(input_shape));

    for (Index i = 0; i < complexity_size; ++i)
        add_layer(make_unique<Dense>(get_output_shape(),
                                       Shape{complexity_dimensions[i]},
                                       "Tanh",
                                       false,
                                       format("dense2d_layer_{}", i + 1)));

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    compile();
    set_parameters_glorot();
}

ForecastingNetwork::ForecastingNetwork(const Shape& input_shape,
                                       const Shape& complexity_dimensions,
                                       const Shape& output_shape) : NeuralNetwork()
{
    add_layer(make_unique<Scaling>(input_shape));

    const Index recurrent_count = complexity_dimensions.rank;
    for (Index i = 0; i < recurrent_count; ++i)
    {
        const bool last = (i == recurrent_count - 1);
        auto recurrent = make_unique<Recurrent>(get_output_shape(),
                                                Shape{complexity_dimensions[i]},
                                                "Tanh",
                                                last ? "recurrent_layer"
                                                     : format("recurrent_layer_{}", i + 1));
        if (!last) recurrent->set_return_sequences(true);
        add_layer(move(recurrent));
    }

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 "Identity",
                                 false,
                                 "forecasting_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    auto bounding = make_unique<Bounding>(output_shape);
    bounding->set_bounding_method("NoBounding");
    add_layer(move(bounding));

    compile();
    set_parameters_glorot();
}

ForecastingLstmNetwork::ForecastingLstmNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape) : NeuralNetwork()
{
    add_layer(make_unique<Scaling>(input_shape));

    const Index lstm_count = complexity_dimensions.rank;
    for (Index i = 0; i < lstm_count; ++i)
    {
        const bool last = (i == lstm_count - 1);
        auto lstm = make_unique<LongShortTermMemory>(get_output_shape(),
                                                     Shape{complexity_dimensions[i]},
                                                     "Tanh",
                                                     "Sigmoid",
                                                     last ? "long_short_term_memory_layer"
                                                          : format("long_short_term_memory_layer_{}", i + 1));
        if (!last) lstm->set_return_sequences(true);
        add_layer(move(lstm));
    }

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 "Identity",
                                 false,
                                 "forecasting_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    auto bounding = make_unique<Bounding>(output_shape);
    bounding->set_bounding_method("NoBounding");
    add_layer(move(bounding));

    compile();
    set_parameters_glorot();
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape) : NeuralNetwork()
{
    add_layer(make_unique<Scaling>(input_shape));

    const Shape mapping_shape{ 10 };
    const Shape bottleneck_shape{ complexity_dimensions[0] };

    add_layer(make_unique<Dense>(input_shape,
                                 mapping_shape,
                                 "Tanh",
                                 false,
                                 "mapping_layer"));

    add_layer(make_unique<Dense>(mapping_shape,
                                 bottleneck_shape,
                                 "Identity",
                                 false,
                                 "bottleneck_layer"));

    add_layer(make_unique<Dense>(bottleneck_shape,
                                 mapping_shape,
                                 "Tanh",
                                 false,
                                 "demapping_layer"));

    add_layer(make_unique<Dense>(mapping_shape,
                                 Shape{ output_shape },
                                 "Identity",
                                 false,
                                 "output_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    compile();
    set_parameters_glorot();
}

#ifndef OPENNN_NO_VISION

ImageClassificationNetwork::ImageClassificationNetwork(const Shape& input_shape,
                                                       const Shape& complexity_dimensions,
                                                       const Shape& output_shape) : NeuralNetwork()
{
    throw_if(input_shape.rank != 3, "Input shape size is not 3.");

    auto scaling_layer = make_unique<Scaling>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    add_layer(move(scaling_layer));

    const Index complexity_size = complexity_dimensions.rank;

    for (Index i = 0; i < complexity_size; ++i)
    {
        const Shape kernel_shape = { 3, 3, get_output_shape()[2], complexity_dimensions[i] };
        const Shape stride_shape = { 1, 1 };

        add_layer(make_unique<Convolutional>(get_output_shape(),
                                             kernel_shape,
                                             "ReLU",
                                             stride_shape,
                                             "Same",
                                             false,
                                             format("convolutional_layer_{}", i + 1)));

        const Shape pool_dimensions = { 2, 2 };
        const Shape pooling_stride_shape = { 2, 2 };
        const Shape padding_dimensions = { 0, 0 };

        add_layer(make_unique<Pooling>(get_output_shape(),
                                       pool_dimensions,
                                       pooling_stride_shape,
                                       padding_dimensions,
                                       "MaxPooling",
                                       format("pooling_layer_{}", i + 1)));
    }

    add_layer(make_unique<Flatten>(get_output_shape()));

    const Index flatten_size = get_output_shape()[0];
    const Shape hidden_shape = { min(flatten_size, Index(128)) };

    add_layer(make_unique<Dense>(get_output_shape(),
                                   hidden_shape,
                                   "ReLU",
                                   false,
                                   "dense_2d_layer_1"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    compile();
    set_parameters_random();
}

ResNet::ResNet(const Shape& input_shape,
               const vector<Index>& blocks_per_stage,
               const Shape& initial_filters,
               const Shape& output_shape,
               bool use_bottleneck) : NeuralNetwork()
{
    throw_if(input_shape.rank != 3, "ResNet: input shape must be rank 3 (H, W, C).");
    throw_if(Index(blocks_per_stage.size()) != Index(initial_filters.rank),
             "ResNet: blocks_per_stage and initial_filters must have the same size.");
    throw_if(blocks_per_stage.empty(), "ResNet: at least one stage is required.");

    constexpr Index bottleneck_expansion = 4;

    auto add_conv = [&](Index input_index,
                        const Shape& kernel_shape, const char* activation,
                        const Shape& stride, const string& name) -> Index {
        add_layer(make_unique<Convolutional>(
                      get_layer(input_index)->get_output_shape(),
                      kernel_shape, activation, stride, "Same",
                      /*batch_normalization=*/true, name),
                  {input_index});
        return get_layers_number() - 1;
    };

    auto add_skip = [&](Index input_index, Index in_channels, Index out_channels,
                        Index stride, const string& prefix) -> Index {
        if (stride == 1 && in_channels == out_channels)
            return input_index;
        return add_conv(input_index,
                        Shape{1, 1, in_channels, out_channels}, "Identity",
                        Shape{stride, stride}, prefix + "_skip");
    };

    // The block-end convolution takes the skip branch as a second input; its
    // batch norm fuses the residual add and the final ReLU.
    auto add_residual_conv = [&](Index input_index, Index skip_index,
                                 const Shape& kernel_shape, const string& name) -> Index {
        auto conv = make_unique<Convolutional>(
            get_layer(input_index)->get_output_shape(),
            kernel_shape, "ReLU", Shape{1, 1}, "Same",
            /*batch_normalization=*/true, name);
        conv->set_residual(true);
        add_layer(move(conv), {input_index, skip_index});
        return get_layers_number() - 1;
    };

    auto add_basic_block = [&](Index input_index, size_t stage, Index block,
                               Index filters) -> Index {
        const Shape input_shape  = get_layer(input_index)->get_output_shape();
        const Index input_channels   = input_shape[2];
        const Index stride    = (stage > 0 && block == 0) ? 2 : 1;
        const string prefix   = format("s{}b{}", stage, block);

        const Index main_index = add_conv(input_index,
            Shape{3, 3, input_channels, filters}, "ReLU",
            Shape{stride, stride}, prefix + "_conv1");

        const Index skip_index = add_skip(input_index, input_channels, filters,
                                          stride, prefix);

        return add_residual_conv(main_index, skip_index,
            Shape{3, 3, filters, filters}, prefix + "_conv2");
    };

    auto add_bottleneck_block = [&](Index input_index, size_t stage, Index block,
                                    Index filters) -> Index {
        const Shape input_shape  = get_layer(input_index)->get_output_shape();
        const Index input_channels   = input_shape[2];
        const Index output_channels  = filters * bottleneck_expansion;
        const Index stride    = (stage > 0 && block == 0) ? 2 : 1;
        const string prefix   = format("s{}b{}", stage, block);

        Index main_index = add_conv(input_index,
            Shape{1, 1, input_channels, filters}, "ReLU",
            Shape{1, 1}, prefix + "_conv1");
        main_index = add_conv(main_index,
            Shape{3, 3, filters, filters}, "ReLU",
            Shape{stride, stride}, prefix + "_conv2");

        const Index skip_index = add_skip(input_index, input_channels, output_channels,
                                          stride, prefix);

        return add_residual_conv(main_index, skip_index,
            Shape{1, 1, filters, output_channels}, prefix + "_conv3");
    };


    auto scaling_layer = make_unique<Scaling>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    add_layer(move(scaling_layer));

    Index last_index = add_conv(0,
        Shape{7, 7, input_shape[2], initial_filters[0]}, "ReLU",
        Shape{2, 2}, "stem_conv");

    add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                   Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                                   "MaxPooling", "stem_pool"),
              {last_index});
    last_index = get_layers_number() - 1;

    for (size_t i = 0; i < blocks_per_stage.size(); ++i)
        for (Index j = 0; j < blocks_per_stage[i]; ++j)
            last_index = use_bottleneck
                ? add_bottleneck_block(last_index, i, j, initial_filters[i])
                : add_basic_block(last_index, i, j, initial_filters[i]);

    const Shape pre_pool = get_layer(last_index)->get_output_shape();
    add_layer(make_unique<Pooling>(pre_pool,
                                   Shape{pre_pool[0], pre_pool[1]},
                                   Shape{1, 1}, Shape{0, 0},
                                   "AveragePooling", "global_avg_pool"),
              {last_index});
    last_index = get_layers_number() - 1;

    add_layer(make_unique<Flatten>(get_layer(last_index)->get_output_shape()),
              {last_index});
    last_index = get_layers_number() - 1;

    add_layer(make_unique<Dense>(get_layer(last_index)->get_output_shape(),
                                 output_shape, "Softmax", false, "classifier"),
              {last_index});

    compile();
    set_parameters_random();
}

YoloNetwork::YoloNetwork(const Shape& input_shape,
                         Index classes_number,
                         const vector<array<float, 2>>& anchors,
                         Index grid_size,
                         Backbone backbone,
                         ClassActivation class_activation,
                         HeadStyle head_style,
                         BodyActivation body_activation) : NeuralNetwork()
{
    throw_if(input_shape.rank != 3, "YoloNetwork: input shape must be rank 3 (H, W, C).");
    throw_if(classes_number <= 0 || anchors.empty(),
             "YoloNetwork: classes_number and anchors must be valid.");
    throw_if(input_shape[0] != grid_size * 32 || input_shape[1] != grid_size * 32,
             "YoloNetwork: this minimal builder expects input H/W == grid_size * 32.");
    if (head_style == HeadStyle::FPN)
    {
        throw_if(backbone != Backbone::DarknetTiny && backbone != Backbone::DarknetTinyV3
                 && backbone != Backbone::Darknet53,
                 "YoloNetwork: HeadStyle::FPN requires DarknetTiny, DarknetTinyV3, or Darknet53.");
        throw_if(ssize(anchors) != 9 && ssize(anchors) != 6,
                 "YoloNetwork: HeadStyle::FPN expects 6 anchors (2-head) or 9 anchors (3-head).");
    }
    if (head_style == HeadStyle::PANet)
    {
        throw_if(backbone != Backbone::Darknet53,
                 "YoloNetwork: HeadStyle::PANet requires Backbone::Darknet53.");
        throw_if(ssize(anchors) != 9,
                 "YoloNetwork: HeadStyle::PANet requires exactly 9 anchors.");
    }

    // Single source of truth for every conv-layer activation string in this
    // network. Defaults to "ReLU" so call sites + saved Phase 1/2 weights
    // behave unchanged.
    const char* act = (body_activation == BodyActivation::LeakyReLU) ? "LeakyReLU" : "ReLU";

    const Shape stride{1, 1};
    const Shape stride_2{2, 2};
    const Shape pool{2, 2};
    const Shape pool_stride{2, 2};
    const Shape no_padding{0, 0};

    auto add_conv = [&](Index input_index, const Shape& kernel_shape,
                        const char* activation, const Shape& kernel_stride,
                        bool batch_norm, const string& name) -> Index {
        add_layer(make_unique<Convolutional>(
                      get_layer(input_index)->get_output_shape(),
                      kernel_shape, activation, kernel_stride, "Same",
                      batch_norm, name),
                  {input_index});
        return get_layers_number() - 1;
    };

    if (backbone == Backbone::Vgg)
    {
        const vector<Index> filters = {32, 64, 128, 256, 512};

        for (Index i = 0; i < ssize(filters); ++i)
        {
            const Shape conv_input_shape = (i == 0) ? input_shape : get_output_shape();

            add_layer(make_unique<Convolutional>(conv_input_shape,
                                                 Shape{3, 3, conv_input_shape[2], filters[size_t(i)]},
                                                 act, stride, "Same", true,
                                                 format("yolo_conv_{}", i + 1)));

            add_layer(make_unique<Pooling>(get_output_shape(), pool, pool_stride,
                                           no_padding, "MaxPooling",
                                           format("yolo_pool_{}", i + 1)));
        }

        add_layer(make_unique<Convolutional>(get_output_shape(),
                                             Shape{3, 3, get_output_shape()[2], 1024},
                                             act, stride, "Same", true,
                                             "yolo_conv_6"));
    }
    else if (backbone == Backbone::DarknetTinyV3)
    {
        // Standard YOLOv3-tiny backbone: 8 conv layers (all BN + act),
        // 5 conv+pool pairs + 3 additional convs.  No residuals.
        // Matches yolov3-tiny.weights exactly (8 backbone conv layers).

        const vector<Index> channels_seq = {16, 32, 64, 128, 256, 512, 1024, 256};
        const vector<bool>  has_pool     = {true, true, true, true, true, false, false, false};
        const vector<bool>  use_1x1      = {false, false, false, false, false, false, false, true};

        Index c3_index = -1;  // output of conv[4] (256ch, 26x26 feature map)
        Index last_index = -1;

        for (size_t i = 0; i < channels_seq.size(); ++i)
        {
            const Shape in_shape  = (i == 0) ? input_shape : get_layer(last_index)->get_output_shape();
            const Index  in_ch    = in_shape[2];
            const Index  out_ch   = channels_seq[i];
            const Index  ksize    = use_1x1[i] ? 1 : 3;

            add_layer(make_unique<Convolutional>(in_shape,
                                                 Shape{ksize, ksize, in_ch, out_ch},
                                                 act, stride, "Same", true,
                                                 format("dntv3_conv_{}", i + 1)));
            last_index = get_layers_number() - 1;

            if (has_pool[i])
            {
                add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                               pool, pool_stride, no_padding,
                                               "MaxPooling",
                                               format("dntv3_pool_{}", i + 1)));
                last_index = get_layers_number() - 1;
            }

            // conv[4] (i==4, 256ch) is the branch point for the small (26×26) head
            if (i == 4) c3_index = get_layers_number() - 1 - (has_pool[i] ? 1 : 0);
            // After the pool at i==4, c3_index points to the conv output before pool
        }

        // At this point:
        //   c3_index  → conv[4] output (256ch, 26x26, before maxpool)
        //   last_index → conv[7] output = p5_top (256ch, 13x13)

        if (head_style == HeadStyle::FPN)
        {
            // 2-head YOLOv3-tiny FPN: sort anchors by area, split into
            // small (3 anchors, 26×26) and large (3 anchors, 13×13).
            vector<array<float, 2>> anchors_sorted = anchors;
            ranges::sort(anchors_sorted, {},
                         [](const array<float, 2>& a) { return a[0] * a[1]; });

            const vector<array<float, 2>> anchors_small(anchors_sorted.begin(),     anchors_sorted.begin() + 3);
            const vector<array<float, 2>> anchors_large(anchors_sorted.begin() + 3, anchors_sorted.end());

            const Index detection_channels_per_head = 3 * (5 + classes_number);

            auto add_detection_head = [&](Index feature_index,
                                          const vector<array<float, 2>>& head_anchors,
                                          const string& name) -> Index {
                const Index logits_index = add_conv(feature_index,
                    Shape{1, 1, get_layer(feature_index)->get_output_shape()[2],
                          detection_channels_per_head},
                    "Identity", stride, false, "yolo_logits_" + name);

                add_layer(make_unique<Detection>(
                              get_layer(logits_index)->get_output_shape(),
                              head_anchors, "detection_" + name),
                          {logits_index});
                static_cast<Detection&>(*get_layers().back()).set_class_activation(
                    class_activation == ClassActivation::Sigmoid
                    ? Detection::ClassActivation::Sigmoid
                    : Detection::ClassActivation::Softmax);
                return get_layers_number() - 1;
            };

            // Large head (13×13): p5_top → Conv1×1 512ch → logits + detection
            const Index p5_conv = add_conv(last_index,
                Shape{3, 3, get_layer(last_index)->get_output_shape()[2], 512},
                act, stride, true, "fpn_p5_conv");
            add_detection_head(p5_conv, anchors_large, "large");

            // Small head (26×26): p5_top → Conv1×1 128ch → Upsample(2×)
            //   → Concat(p5_up 128ch + c3 256ch = 384ch) → Conv3×3 256ch → logits
            const Index p5_lateral = add_conv(last_index,
                Shape{1, 1, get_layer(last_index)->get_output_shape()[2], 128},
                act, stride, true, "fpn_p5_lateral");

            add_layer(make_unique<Upsample>(
                          get_layer(p5_lateral)->get_output_shape(),
                          /*scale_factor=*/2, "fpn_p5_upsample"),
                      {p5_lateral});
            const Index p5_up = get_layers_number() - 1;

            const Index c3_channels   = get_layer(c3_index)->get_output_shape()[2];
            const Index p5_up_channels = get_layer(p5_up)->get_output_shape()[2];
            add_layer(make_unique<Concatenation>(
                          get_layer(c3_index)->get_output_shape(),
                          vector<Index>{p5_up_channels, c3_channels},
                          "fpn_p4_concatenation"),
                      {p5_up, c3_index});
            const Index p4_concat = get_layers_number() - 1;

            const Index p4_conv = add_conv(p4_concat,
                Shape{3, 3, get_layer(p4_concat)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p4_conv");
            add_detection_head(p4_conv, anchors_small, "small");

            compile();
            set_parameters_random();
            return;
        }
    }
    else if (backbone == Backbone::Darknet53)
    {
        // Darknet53: 52 conv layers arranged as initial conv + 5 stages.
        // Each stage: stride-2 downsample conv + N residual blocks [Conv1×1 → Conv3×3 + skip].
        // Stages: {64,1},{128,2},{256,8},{512,8},{1024,4} → 32× total stride.
        // FPN taps C3=52×52×256, C4=26×26×512, C5=13×13×1024.

        auto add_residual_block = [&](Index input_index, Index channels, const string& prefix) -> Index {
            const Index half = channels / 2;
            Index x = add_conv(input_index, Shape{1, 1, channels, half},      act,        stride, true, prefix+"_c1");
            x       = add_conv(x,           Shape{3, 3, half,     channels},  "Identity", stride, true, prefix+"_c2");
            add_layer(make_unique<Addition>(get_layer(x)->get_output_shape(), prefix+"_add"), {x, input_index});
            const Index add_idx = get_layers_number() - 1;
            add_layer(make_unique<Activation>(get_layer(add_idx)->get_output_shape(), act, prefix+"_act"), {add_idx});
            return get_layers_number() - 1;
        };

        const vector<pair<Index,Index>> stages = {{64,1},{128,2},{256,8},{512,8},{1024,4}};

        // Initial conv: stride 1 (unlike DarknetTiny which uses stride 2 for the stem)
        add_layer(make_unique<Convolutional>(input_shape, Shape{3, 3, input_shape[2], 32},
                                             act, stride, "Same", true, "dn53_stem"));
        Index last_index = get_layers_number() - 1;

        Index c3_index = -1, c4_index = -1, c5_index = -1;

        for (size_t i = 0; i < stages.size(); ++i)
        {
            const Index ch     = stages[i].first;
            const Index nblocks = stages[i].second;
            const Index in_ch  = get_layer(last_index)->get_output_shape()[2];

            last_index = add_conv(last_index, Shape{3, 3, in_ch, ch}, act, stride_2, true,
                                  format("dn53_down_{}", i+1));
            for (Index j = 0; j < nblocks; ++j)
                last_index = add_residual_block(last_index, ch, format("dn53_s{}_b{}", i+1, j+1));

            if (i == 2) c3_index = last_index;
            if (i == 3) c4_index = last_index;
            if (i == 4) c5_index = last_index;
        }

        if (head_style == HeadStyle::FPN || head_style == HeadStyle::PANet)
        {
            throw_if(ssize(anchors) != 9, "YoloNetwork: Darknet53 FPN/PANet requires exactly 9 anchors.");

            vector<array<float,2>> anchors_sorted = anchors;
            ranges::sort(anchors_sorted, {}, [](const array<float,2>& a){ return a[0]*a[1]; });
            const vector<array<float,2>> anchors_small (anchors_sorted.begin(),     anchors_sorted.begin()+3);
            const vector<array<float,2>> anchors_medium(anchors_sorted.begin()+3,   anchors_sorted.begin()+6);
            const vector<array<float,2>> anchors_large (anchors_sorted.begin()+6,   anchors_sorted.end());

            const Index det_ch = 3 * (5 + classes_number);

            auto add_det_head = [&](Index feat_idx, const vector<array<float,2>>& head_anchors, const string& name) {
                const Index logits = add_conv(feat_idx,
                    Shape{1, 1, get_layer(feat_idx)->get_output_shape()[2], det_ch},
                    "Identity", stride, false, "yolo_logits_"+name);
                add_layer(make_unique<Detection>(get_layer(logits)->get_output_shape(),
                                                 head_anchors, "detection_"+name), {logits});
                static_cast<Detection&>(*get_layers().back()).set_class_activation(
                    class_activation == ClassActivation::Sigmoid
                    ? Detection::ClassActivation::Sigmoid : Detection::ClassActivation::Softmax);
            };

            // 5-conv neck block (YOLOv3 DBL×5)
            auto add_yolo_neck = [&](Index idx, Index in_ch,
                                     Index ch_small, Index ch_large, const string& pfx) -> Index {
                Index x = add_conv(idx, Shape{1, 1, in_ch,     ch_small}, act, stride, true, pfx+"_c1");
                x       = add_conv(x,   Shape{3, 3, ch_small, ch_large},  act, stride, true, pfx+"_c2");
                x       = add_conv(x,   Shape{1, 1, ch_large, ch_small},  act, stride, true, pfx+"_c3");
                x       = add_conv(x,   Shape{3, 3, ch_small, ch_large},  act, stride, true, pfx+"_c4");
                x       = add_conv(x,   Shape{1, 1, ch_large, ch_small},  act, stride, true, pfx+"_c5");
                return x;
            };

            // ── Top-down FPN path (shared by FPN and PANet) ──────────────────
            const Index p5n = add_yolo_neck(c5_index, 1024, 512, 1024, "neck_p5");

            const Index p5l = add_conv(p5n, Shape{1, 1, 512, 256}, act, stride, true, "neck_p5_lat");
            add_layer(make_unique<Upsample>(get_layer(p5l)->get_output_shape(), 2, "fpn_p5_up"), {p5l});
            const Index p5u = get_layers_number() - 1;

            add_layer(make_unique<Concatenation>(get_layer(c4_index)->get_output_shape(),
                                                 vector<Index>{256, 512}, "fpn_p4_cat"),
                      {p5u, c4_index});
            const Index p4c = get_layers_number() - 1;
            const Index p4n = add_yolo_neck(p4c, 768, 256, 512, "neck_p4");

            const Index p4l = add_conv(p4n, Shape{1, 1, 256, 128}, act, stride, true, "neck_p4_lat");
            add_layer(make_unique<Upsample>(get_layer(p4l)->get_output_shape(), 2, "fpn_p4_up"), {p4l});
            const Index p4u = get_layers_number() - 1;

            add_layer(make_unique<Concatenation>(get_layer(c3_index)->get_output_shape(),
                                                 vector<Index>{128, 256}, "fpn_p3_cat"),
                      {p4u, c3_index});
            const Index p3c = get_layers_number() - 1;
            const Index p3n = add_yolo_neck(p3c, 384, 128, 256, "neck_p3");

            if (head_style == HeadStyle::FPN)
            {
                // FPN: direct detection at each scale
                const Index p5d = add_conv(p5n, Shape{3, 3, 512, 1024}, act, stride, true, "neck_p5_pre");
                add_det_head(p5d, anchors_large, "large");
                const Index p4d = add_conv(p4n, Shape{3, 3, 256, 512}, act, stride, true, "neck_p4_pre");
                add_det_head(p4d, anchors_medium, "medium");
                const Index p3d = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "neck_p3_pre");
                add_det_head(p3d, anchors_small, "small");
            }
            else // PANet: small head at P3, then bottom-up path for medium and large
            {
                // Small head (52×52) — same as FPN
                const Index p3d = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "neck_p3_pre");
                add_det_head(p3d, anchors_small, "small");

                // 3-conv PAN block: reduce → expand → reduce
                auto add_pan_block = [&](Index idx, Index in_ch, Index ch_s, Index ch_l, const string& pfx) -> Index {
                    Index x = add_conv(idx, Shape{1, 1, in_ch, ch_s}, act, stride, true, pfx+"_c1");
                    x       = add_conv(x,   Shape{3, 3, ch_s,  ch_l}, act, stride, true, pfx+"_c2");
                    x       = add_conv(x,   Shape{1, 1, ch_l,  ch_s}, act, stride, true, pfx+"_c3");
                    return x;
                };

                // ── Bottom-up: P3 → N4 (medium head, 26×26) ─────────────────
                const Index n3_down = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride_2, true, "pan_n3_down");
                add_layer(make_unique<Concatenation>(get_layer(p4n)->get_output_shape(),
                                                     vector<Index>{256, 256}, "pan_n4_cat"),
                          {n3_down, p4n});
                const Index n4c = get_layers_number() - 1;
                const Index n4n = add_pan_block(n4c, 512, 256, 512, "pan_n4");
                const Index n4d = add_conv(n4n, Shape{3, 3, 256, 512}, act, stride, true, "pan_n4_pre");
                add_det_head(n4d, anchors_medium, "medium");

                // ── Bottom-up: N4 → N5 (large head, 13×13) ──────────────────
                const Index n4_down = add_conv(n4n, Shape{3, 3, 256, 512}, act, stride_2, true, "pan_n4_down");
                add_layer(make_unique<Concatenation>(get_layer(p5n)->get_output_shape(),
                                                     vector<Index>{512, 512}, "pan_n5_cat"),
                          {n4_down, p5n});
                const Index n5c = get_layers_number() - 1;
                const Index n5n = add_pan_block(n5c, 1024, 512, 1024, "pan_n5");
                const Index n5d = add_conv(n5n, Shape{3, 3, 512, 1024}, act, stride, true, "pan_n5_pre");
                add_det_head(n5d, anchors_large, "large");
            }

            compile();
            set_parameters_random();
            return;
        }
    }
    else
    {
        auto add_residual_block = [&](Index input_index, Index channels,
                                      const string& prefix) -> Index {
            const Index reduced = max<Index>(channels / 2, 1);

            Index main_index = add_conv(input_index,
                Shape{1, 1, channels, reduced}, act,
                stride, true, prefix + "_conv1");
            main_index = add_conv(main_index,
                Shape{3, 3, reduced, channels}, "Identity",
                stride, true, prefix + "_conv2");

            add_layer(make_unique<Addition>(get_layer(main_index)->get_output_shape(),
                                            prefix + "_add"),
                      {main_index, input_index});
            const Index add_index = get_layers_number() - 1;

            add_layer(make_unique<Activation>(get_layer(add_index)->get_output_shape(),
                                              act, prefix + "_relu"),
                      {add_index});
            return get_layers_number() - 1;
        };

        const vector<pair<Index, Index>> stages = {
            { 64, 1},
            {128, 1},
            {256, 1},
            {512, 1},
        };

        add_layer(make_unique<Convolutional>(input_shape,
                                             Shape{3, 3, input_shape[2], 32},
                                             act, stride_2, "Same", true,
                                             "darknet_stem"));
        Index last_index = get_layers_number() - 1;

        Index c3_index = -1;
        Index c4_index = -1;
        Index c5_index = -1;

        for (size_t i = 0; i < stages.size(); ++i)
        {
            const Index channels = stages[i].first;
            const Index blocks_number  = stages[i].second;
            const Index input_channels = get_layer(last_index)->get_output_shape()[2];

            last_index = add_conv(last_index,
                Shape{3, 3, input_channels, channels}, act,
                stride_2, true, format("darknet_down_{}", i + 1));

            for (Index j = 0; j < blocks_number; ++j)
                last_index = add_residual_block(last_index, channels,
                    format("darknet_s{}_b{}", i + 1, j));

            if (i == 1) c3_index = last_index;
            if (i == 2) c4_index = last_index;
            if (i == 3) c5_index = last_index;
        }

        if (head_style == HeadStyle::FPN)
        {
            throw_if(ssize(anchors) != 9,
                     "YoloNetwork: DarknetTiny FPN (3-head) requires exactly 9 anchors.");
            vector<array<float, 2>> anchors_sorted = anchors;
            ranges::sort(anchors_sorted, {},
                         [](const array<float, 2>& a) { return a[0] * a[1]; });

            const vector<array<float, 2>> anchors_small (anchors_sorted.begin(),     anchors_sorted.begin() + 3);
            const vector<array<float, 2>> anchors_medium(anchors_sorted.begin() + 3, anchors_sorted.begin() + 6);
            const vector<array<float, 2>> anchors_large (anchors_sorted.begin() + 6, anchors_sorted.end());

            const Index detection_channels_per_head = 3 * (5 + classes_number);

            auto add_detection_head = [&](Index feature_index,
                                          const vector<array<float, 2>>& head_anchors,
                                          const string& name) -> Index {
                const Index logits_index = add_conv(feature_index,
                    Shape{1, 1, get_layer(feature_index)->get_output_shape()[2],
                          detection_channels_per_head},
                    "Identity", stride, false, "yolo_logits_" + name);

                add_layer(make_unique<Detection>(
                              get_layer(logits_index)->get_output_shape(),
                              head_anchors, "detection_" + name),
                          {logits_index});
                static_cast<Detection&>(*get_layers().back()).set_class_activation(
                    class_activation == ClassActivation::Sigmoid
                    ? Detection::ClassActivation::Sigmoid
                    : Detection::ClassActivation::Softmax);
                return get_layers_number() - 1;
            };

            const Index p5_lateral = add_conv(c5_index,
                Shape{1, 1, get_layer(c5_index)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p5_lateral");
            add_detection_head(p5_lateral, anchors_large, "large");

            add_layer(make_unique<Upsample>(
                          get_layer(p5_lateral)->get_output_shape(),
                          /*scale_factor=*/2, "fpn_p5_upsample"),
                      {p5_lateral});
            const Index p5_up = get_layers_number() - 1;

            const Index c4_channels = get_layer(c4_index)->get_output_shape()[2];
            const Index p5_up_channels = get_layer(p5_up)->get_output_shape()[2];
            add_layer(make_unique<Concatenation>(
                          get_layer(c4_index)->get_output_shape(),
                          vector<Index>{p5_up_channels, c4_channels},
                          "fpn_p4_concatenation"),
                      {p5_up, c4_index});
            const Index p4_concatenation = get_layers_number() - 1;

            const Index p4_lateral = add_conv(p4_concatenation,
                Shape{1, 1, get_layer(p4_concatenation)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p4_lateral");
            add_detection_head(p4_lateral, anchors_medium, "medium");

            add_layer(make_unique<Upsample>(
                          get_layer(p4_lateral)->get_output_shape(),
                          /*scale_factor=*/2, "fpn_p4_upsample"),
                      {p4_lateral});
            const Index p4_up = get_layers_number() - 1;

            const Index c3_channels = get_layer(c3_index)->get_output_shape()[2];
            const Index p4_up_channels = get_layer(p4_up)->get_output_shape()[2];
            add_layer(make_unique<Concatenation>(
                          get_layer(c3_index)->get_output_shape(),
                          vector<Index>{p4_up_channels, c3_channels},
                          "fpn_p3_concatenation"),
                      {p4_up, c3_index});
            const Index p3_concatenation = get_layers_number() - 1;

            const Index p3_lateral = add_conv(p3_concatenation,
                Shape{1, 1, get_layer(p3_concatenation)->get_output_shape()[2], 128},
                act, stride, true, "fpn_p3_lateral");
            add_detection_head(p3_lateral, anchors_small, "small");

            compile();
            set_parameters_random();
            return;
        }
    }

    const Index detection_channels = ssize(anchors) * (5 + classes_number);

    add_layer(make_unique<Convolutional>(get_output_shape(),
                                         Shape{1, 1, get_output_shape()[2], detection_channels},
                                         "Identity", stride, "Same", false,
                                         "yolo_logits"));

    add_layer(make_unique<Detection>(get_output_shape(), anchors, "detection_layer"));
    static_cast<Detection&>(*get_layers().back()).set_class_activation(
        class_activation == ClassActivation::Sigmoid
        ? Detection::ClassActivation::Sigmoid
        : Detection::ClassActivation::Softmax);

    add_layer(make_unique<NonMaxSuppression>(get_output_shape(),
                                             ssize(anchors),
                                             0.5f,
                                             0.4f,
                                             "non_max_suppression_layer"));

    compile();
    set_parameters_random();
}

TextClassificationNetwork::TextClassificationNetwork(const Shape& input_shape,
                                                     const Shape& complexity_dimensions,
                                                     const Shape& output_shape,
                                                     PoolingMethod pooling_method) : NeuralNetwork()
{
    const Index vocabulary_size = input_shape[0];
    const Index sequence_length = input_shape[1];
    const Index embedding_dimension = input_shape[2];
    const Index heads_number = complexity_dimensions[0];
    const Index hidden_neurons = complexity_dimensions.rank > 1 ? complexity_dimensions[1] : 64;

    auto embedding_layer = make_unique<Embedding>(Shape({vocabulary_size, sequence_length}),
                                                  embedding_dimension,
                                                  "embedding_layer");
    embedding_layer->set_scale_embedding(true);
    embedding_layer->set_add_positional_encoding(true);
    add_layer(move(embedding_layer));

    add_layer(make_unique<MultiHeadAttention>(
        Shape({sequence_length, embedding_dimension}),
        heads_number,
        "multihead_attention_layer"));

    add_layer(make_unique<Pooling3d>(get_output_shape(), pooling_method));

    add_layer(make_unique<Dense>(get_output_shape(), Shape({hidden_neurons}), "ReLU", false, "hidden_layer"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                 false,
                                 "classification_layer"));

    compile();
    set_parameters_glorot();
}

static Index add_residual_and_norm(NeuralNetwork& network,
                                   const Shape& shape,
                                   const string& norm_label,
                                   Index left_index, Index right_index)
{
    auto norm = make_unique<Normalization3d>(shape, norm_label);
    norm->set_fuse_add(true);
    network.add_layer(move(norm), {left_index, right_index});
    return network.get_layers_number() - 1;
}

static Index add_feed_forward(NeuralNetwork& network,
                              const Shape& input_shape, Index ff_dim,
                              const string& internal_label,
                              const string& external_label)
{
    const Index seq_len = input_shape[0];
    const Index emb_dim = input_shape[1];
    network.add_layer(make_unique<Dense>(input_shape, Shape{ff_dim},
                                         "ReLU", false, internal_label));
    network.add_layer(make_unique<Dense>(Shape{seq_len, ff_dim}, Shape{emb_dim},
                                         "Identity", false, external_label));
    return network.get_layers_number() - 1;
}

Transformer::Transformer(Index input_sequence_length,
                         Index decoder_sequence_length,
                         Index input_vocabulary_size,
                         Index output_vocabulary_size,
                         Index embedding_dimension,
                         Index heads_number,
                         Index feed_forward_dimension,
                         Index layers_number)
    : NeuralNetwork()
{
    throw_if(input_sequence_length == 0 ||
             decoder_sequence_length == 0 ||
             input_vocabulary_size == 0 ||
             output_vocabulary_size == 0 ||
             embedding_dimension == 0 ||
             heads_number == 0 ||
             feed_forward_dimension == 0 ||
             layers_number == 0,
             "Transformer: all dimensions must be > 0.");

    throw_if(embedding_dimension % heads_number != 0,
             "Transformer: embedding_dimension must be divisible by heads_number.");

    auto decoder_embedding = make_unique<Embedding>(
        Shape{output_vocabulary_size, decoder_sequence_length},
        embedding_dimension, "decoder_embedding");
    decoder_embedding->set_scale_embedding(true);
    decoder_embedding->set_add_positional_encoding(true);
    add_layer(move(decoder_embedding), {-1});
    Index current_decoder_index = get_layers_number() - 1;

    auto encoder_embedding = make_unique<Embedding>(
        Shape{input_vocabulary_size, input_sequence_length},
        embedding_dimension, "encoder_embedding");
    encoder_embedding->set_scale_embedding(true);
    encoder_embedding->set_add_positional_encoding(true);
    add_layer(move(encoder_embedding), {-2});
    Index current_encoder_index = get_layers_number() - 1;

    const Shape encoder_shape{input_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        add_layer(make_unique<MultiHeadAttention>(encoder_shape, heads_number,
                                                  "encoder_self_attention" + suffix),
                  {current_encoder_index});
        const Index attn_index = get_layers_number() - 1;

        const Index norm1_index = add_residual_and_norm(*this, encoder_shape,
            "encoder_self_attention_normalization" + suffix,
            current_encoder_index, attn_index);

        const Index ff_index = add_feed_forward(*this, encoder_shape, feed_forward_dimension,
            "encoder_internal_dense" + suffix,
            "encoder_external_dense" + suffix);

        current_encoder_index = add_residual_and_norm(*this, encoder_shape,
            "encoder_dense_normalization" + suffix,
            norm1_index, ff_index);
    }

    const Index encoder_final_output_index = current_encoder_index;

    const Shape decoder_shape{decoder_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        auto decoder_self_attention = make_unique<MultiHeadAttention>(
            decoder_shape, heads_number, "decoder_self_attention" + suffix);
        decoder_self_attention->set(decoder_sequence_length, decoder_sequence_length,
                                    embedding_dimension, heads_number,
                                    true,
                                    "decoder_self_attention" + suffix);
        add_layer(move(decoder_self_attention), {current_decoder_index});
        const Index self_attn_index = get_layers_number() - 1;

        const Index norm1_index = add_residual_and_norm(*this, decoder_shape,
            "decoder_self_attention_normalization" + suffix,
            current_decoder_index, self_attn_index);

        add_layer(make_unique<MultiHeadAttention>(decoder_shape, encoder_shape,
                                                  heads_number,
                                                  "cross_attention" + suffix),
                  {norm1_index, encoder_final_output_index});
        const Index cross_attn_index = get_layers_number() - 1;

        const Index norm2_index = add_residual_and_norm(*this, decoder_shape,
            "cross_attention_normalization" + suffix,
            norm1_index, cross_attn_index);

        const Index ff_index = add_feed_forward(*this, decoder_shape, feed_forward_dimension,
            "decoder_internal_dense" + suffix,
            "decoder_external_dense" + suffix);

        current_decoder_index = add_residual_and_norm(*this, decoder_shape,
            "decoder_dense_normalization" + suffix,
            norm2_index, ff_index);
    }


    add_layer(make_unique<Dense>(decoder_shape, Shape{output_vocabulary_size},
                                 "Softmax", false, "output_projection"));

    compile();
    set_parameters_glorot();
}

void Transformer::set_dropout_rate(const float new_dropout_rate)
{
    const auto forward_before = get_forward_specs(1);
    const auto backward_before = get_backward_specs(1);

    for (auto& layer : get_layers())
    {
        if (!layer) continue;

        const string& label = layer->get_label();
        const bool is_ffn_dense = starts_with_any(label,
                                                  {"encoder_internal_dense",
                                                   "encoder_external_dense",
                                                   "decoder_internal_dense",
                                                   "decoder_external_dense"});

        if (is_ffn_dense)
        {
            if (auto* dense = dynamic_cast<Dense*>(layer.get()))
                dense->set_dropout_rate(new_dropout_rate);
        }
        else if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
        {
            mha->set_dropout_rate(new_dropout_rate);
        }
    }

    recompile_if_specs_changed(*this, forward_before, backward_before);
}

void Transformer::set_attention_sdpa_auto(bool new_sdpa_auto)
{
    const auto forward_before = get_forward_specs(1);
    const auto backward_before = get_backward_specs(1);

    for (auto& layer : get_layers())
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
            mha->set_sdpa_auto(new_sdpa_auto);

    recompile_if_specs_changed(*this, forward_before, backward_before);
}

void Transformer::set_attention_sdpa_min_sequence_length(Index new_threshold)
{
    const auto forward_before = get_forward_specs(1);
    const auto backward_before = get_backward_specs(1);

    for (auto& layer : get_layers())
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
            mha->set_sdpa_min_sequence_length(new_threshold);

    recompile_if_specs_changed(*this, forward_before, backward_before);
}

Index Transformer::get_input_sequence_length() const
{
    return get_layer("encoder_embedding")->get_input_shape()[0];
}

Index Transformer::get_decoder_sequence_length() const
{
    return get_layer("decoder_embedding")->get_input_shape()[0];
}

Index Transformer::get_embedding_dimension() const
{
    return get_layer(0)->get_output_shape().back();
}

Index Transformer::get_heads_number() const
{
    if (auto* mha = dynamic_cast<const MultiHeadAttention*>(get_first(LayerType::MultiHeadAttention)))
        return mha->get_heads_number();

    return 0;
}

TextGenerationNetwork::TextGenerationNetwork(Index sequence_length,
                                             Index vocabulary_size,
                                             Index embedding_dimension,
                                             Index heads_number,
                                             Index feed_forward_dimension,
                                             Index layers_number,
                                             bool pre_normalization)
    : NeuralNetwork()
{
    throw_if(sequence_length == 0 ||
             vocabulary_size == 0 ||
             embedding_dimension == 0 ||
             heads_number == 0 ||
             feed_forward_dimension == 0 ||
             layers_number == 0,
             "TextGenerationNetwork: all dimensions must be > 0.");

    throw_if(embedding_dimension % heads_number != 0,
             "TextGenerationNetwork: embedding_dimension must be divisible by heads_number.");

    auto embedding = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length},
        embedding_dimension, "embedding");
    embedding->set_scale_embedding(true);
    embedding->set_add_positional_encoding(true);
    add_layer(move(embedding), {-1});
    Index current_index = get_layers_number() - 1;

    const Shape block_shape{sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        Index attention_input_index = current_index;

        if (pre_normalization)
        {
            add_layer(make_unique<Normalization3d>(block_shape,
                                                   "attention_normalization" + suffix),
                      {current_index});
            attention_input_index = get_layers_number() - 1;
        }

        auto self_attention = make_unique<MultiHeadAttention>(
            block_shape, heads_number, "self_attention" + suffix);
        self_attention->set(sequence_length, sequence_length,
                            embedding_dimension, heads_number,
                            true,
                            "self_attention" + suffix);
        add_layer(move(self_attention), {attention_input_index});
        const Index attn_index = get_layers_number() - 1;

        if (pre_normalization)
        {
            add_layer(make_unique<Addition>(block_shape, "attention_addition" + suffix),
                      {current_index, attn_index});
            const Index residual_index = get_layers_number() - 1;

            add_layer(make_unique<Normalization3d>(block_shape,
                                                   "dense_normalization" + suffix),
                      {residual_index});

            const Index ff_index = add_feed_forward(*this, block_shape, feed_forward_dimension,
                "internal_dense" + suffix,
                "external_dense" + suffix);

            add_layer(make_unique<Addition>(block_shape, "dense_addition" + suffix),
                      {residual_index, ff_index});
            current_index = get_layers_number() - 1;
        }
        else
        {
            const Index norm1_index = add_residual_and_norm(*this, block_shape,
                "self_attention_normalization" + suffix,
                current_index, attn_index);

            const Index ff_index = add_feed_forward(*this, block_shape, feed_forward_dimension,
                "internal_dense" + suffix,
                "external_dense" + suffix);

            current_index = add_residual_and_norm(*this, block_shape,
                "dense_normalization" + suffix,
                norm1_index, ff_index);
        }
    }

    if (pre_normalization)
        add_layer(make_unique<Normalization3d>(block_shape, "final_normalization"),
                  {current_index});

    add_layer(make_unique<Dense>(block_shape, Shape{vocabulary_size},
                                 "Softmax", false, "output_projection"));

    compile();
    set_parameters_glorot();
}

void TextGenerationNetwork::set_dropout_rate(const float new_dropout_rate)
{
    const auto forward_before = get_forward_specs(1);
    const auto backward_before = get_backward_specs(1);

    for (auto& layer : get_layers())
    {
        if (!layer) continue;

        const string& label = layer->get_label();
        const bool is_ffn_dense = starts_with_any(label,
                                                  {"internal_dense",
                                                   "external_dense"});

        if (is_ffn_dense)
        {
            if (auto* dense = dynamic_cast<Dense*>(layer.get()))
                dense->set_dropout_rate(new_dropout_rate);
        }
        else if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
        {
            mha->set_dropout_rate(new_dropout_rate);
        }
    }

    recompile_if_specs_changed(*this, forward_before, backward_before);
}

void TextGenerationNetwork::set_attention_sdpa_auto(bool new_sdpa_auto)
{
    const auto forward_before = get_forward_specs(1);
    const auto backward_before = get_backward_specs(1);

    for (auto& layer : get_layers())
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
            mha->set_sdpa_auto(new_sdpa_auto);

    recompile_if_specs_changed(*this, forward_before, backward_before);
}

Index TextGenerationNetwork::get_sequence_length() const
{
    return get_layer("embedding")->get_input_shape()[0];
}

Index TextGenerationNetwork::get_embedding_dimension() const
{
    return get_layer(0)->get_output_shape().back();
}

Index TextGenerationNetwork::get_heads_number() const
{
    if (auto* mha = dynamic_cast<const MultiHeadAttention*>(get_first(LayerType::MultiHeadAttention)))
        return mha->get_heads_number();

    return 0;
}

#endif // OPENNN_NO_VISION

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
