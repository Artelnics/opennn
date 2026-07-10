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
#include "tokenizer_layer.h"
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
#include "random_utilities.h"
#include "device_backend.h"

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
                                       format("dense_layer_{}", i + 1)));

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
                                       format("dense_layer_{}", i + 1)));

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

    add_layer(make_unique<Dense>(get_output_shape(), Shape({hidden_neurons}), "ReLU", false, "dense_layer_1"));

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
                              const string& external_label,
                              const string& internal_activation = "ReLU")
{
    const Index seq_len = input_shape[0];
    const Index emb_dim = input_shape[1];
    network.add_layer(make_unique<Dense>(input_shape, Shape{ff_dim},
                                         internal_activation, false, internal_label));
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

    add_layer(make_unique<Tokenizer>(Shape{decoder_sequence_length}, "decoder_tokenizer"), {-1});
    const Index decoder_tokenizer_index = get_layers_number() - 1;

    auto decoder_embedding = make_unique<Embedding>(
        Shape{output_vocabulary_size, decoder_sequence_length},
        embedding_dimension, "decoder_embedding");
    decoder_embedding->set_scale_embedding(true);
    decoder_embedding->set_add_positional_encoding(true);
    add_layer(move(decoder_embedding), {decoder_tokenizer_index});
    Index current_decoder_index = get_layers_number() - 1;

    add_layer(make_unique<Tokenizer>(Shape{input_sequence_length}, "encoder_tokenizer"), {-2});
    const Index encoder_tokenizer_index = get_layers_number() - 1;

    auto encoder_embedding = make_unique<Embedding>(
        Shape{input_vocabulary_size, input_sequence_length},
        embedding_dimension, "encoder_embedding");
    encoder_embedding->set_scale_embedding(true);
    encoder_embedding->set_add_positional_encoding(true);
    add_layer(move(encoder_embedding), {encoder_tokenizer_index});
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
    return get_layer("decoder_embedding")->get_output_shape().back();
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
                                             bool pre_normalization,
                                             bool scale_embedding,
                                             bool learned_positional,
                                             const string& feed_forward_activation)
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

    add_layer(make_unique<Tokenizer>(Shape{sequence_length}, "tokenizer"), {-1});
    const Index tokenizer_index = get_layers_number() - 1;

    auto embedding = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length},
        embedding_dimension, "embedding");
    embedding->set_scale_embedding(scale_embedding);
    if (learned_positional)
        embedding->set_learned_positional(true);
    else
        embedding->set_add_positional_encoding(true);
    add_layer(move(embedding), {tokenizer_index});
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
                "external_dense" + suffix,
                feed_forward_activation);

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
                "external_dense" + suffix,
                feed_forward_activation);

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

static Index add_bert_encoder(NeuralNetwork& net,
                              Index sequence_length, Index vocabulary_size, Index hidden_size,
                              Index heads_number, Index intermediate_size, Index layers_number,
                              Index type_vocabulary_size)
{
    throw_if(sequence_length == 0 || vocabulary_size == 0 || hidden_size == 0 ||
             heads_number == 0 || intermediate_size == 0 || layers_number == 0 ||
             type_vocabulary_size == 0,
             "BERT: all dimensions must be > 0.");

    throw_if(hidden_size % heads_number != 0,
             "BERT: hidden_size must be divisible by heads_number.");

    const Shape seq_hidden{sequence_length, hidden_size};

    auto add_residual_norm = [&](const string& label, Index left, Index right) -> Index {
        auto norm = make_unique<Normalization3d>(seq_hidden, label);
        norm->set_fuse_add(true);
        net.add_layer(move(norm), {left, right});
        return net.get_layers_number() - 1;
    };

    auto word_embeddings = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length}, hidden_size, "word_embeddings");
    word_embeddings->set_learned_positional(true);
    word_embeddings->set_export_valid_lengths(true);
    net.add_layer(move(word_embeddings), {-1});
    const Index word_index = net.get_layers_number() - 1;

    net.add_layer(make_unique<Embedding>(
                      Shape{type_vocabulary_size + 1, sequence_length}, hidden_size, "token_type_embeddings"),
                  {-2});
    const Index type_index = net.get_layers_number() - 1;

    Index current = add_residual_norm("embeddings_layer_norm", word_index, type_index);

    for (Index i = 0; i < layers_number; ++i)
    {
        const string sfx = format("_{}", i + 1);

        net.add_layer(make_unique<MultiHeadAttention>(seq_hidden, heads_number, "attention" + sfx),
                      {current});
        const Index attention_index = net.get_layers_number() - 1;

        const Index attention_norm_index =
            add_residual_norm("attention_layer_norm" + sfx, current, attention_index);

        net.add_layer(make_unique<Dense>(seq_hidden, Shape{intermediate_size},
                                         "GELU", false, "intermediate" + sfx),
                      {attention_norm_index});
        net.add_layer(make_unique<Dense>(Shape{sequence_length, intermediate_size}, Shape{hidden_size},
                                         "Identity", false, "feed_forward_output" + sfx));
        const Index feed_forward_index = net.get_layers_number() - 1;

        current = add_residual_norm("output_layer_norm" + sfx, attention_norm_index, feed_forward_index);
    }

    return current;
}

Bert::Bert(Index sequence_length,
           Index vocabulary_size,
           Index hidden_size,
           Index heads_number,
           Index intermediate_size,
           Index layers_number,
           Index type_vocabulary_size)
    : NeuralNetwork()
{
    add_bert_encoder(*this, sequence_length, vocabulary_size, hidden_size, heads_number,
                     intermediate_size, layers_number, type_vocabulary_size);
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
    return get_layer("embedding")->get_output_shape().back();
}

// --- Text inference (decode / generate / chat) ---
//
// The tokenizer layers carried by Transformer and TextGenerationNetwork hold
// the vocabulary, so a model loaded from disk decodes without any dataset.
// All per-prompt state lives in a GenerationSession created at execution time,
// like the optimizers and calculate_outputs do.

TokenCallback stream_token_callback(ostream& out, bool& first_token, bool raw)
{
    return [&out, &first_token, raw](const string& token)
    {
        if (raw) { out << token << flush; return; }

        const bool is_punctuation = token.size() == 1
            && string_view(",.!?;:").find(token[0]) != string_view::npos;

        if (!first_token && !is_punctuation) out << ' ';
        out << token << flush;
        first_token = false;
    };
}

Index sample_token(VectorR& probabilities,
                   const SamplingConfig& sampling_config,
                   const vector<Index>& history)
{
    const Index vocabulary_size = probabilities.size();

    SamplingConfig config = sampling_config;
    config.temperature = max(config.temperature, 0.0f);
    if (config.repetition_penalty <= 0.0f) config.repetition_penalty = 1.0f;
    config.top_k = max(config.top_k, Index(0));
    config.top_p = clamp(config.top_p, 0.0f, 1.0f);

    if (config.temperature == 0.0f)
    {
        Index best;
        probabilities.maxCoeff(&best);
        return best;
    }

    const VectorR original = probabilities;

    if (config.repetition_penalty != 1.0f)
        for (Index token_id : history)
            if (token_id >= 0 && token_id < vocabulary_size)
                probabilities(token_id) /= config.repetition_penalty;

    if (config.temperature != 1.0f)
    {
        const float inverse_temperature = 1.0f / config.temperature;
        for (Index i = 0; i < vocabulary_size; ++i)
            probabilities(i) = pow(max(probabilities(i), 0.0f), inverse_temperature);
    }

    if (config.top_k > 0 && config.top_k < vocabulary_size)
    {
        vector<pair<float, Index>> indexed(vocabulary_size);
        for (Index i = 0; i < vocabulary_size; ++i) indexed[i] = {probabilities(i), i};
        nth_element(indexed.begin(),
                    indexed.begin() + config.top_k,
                    indexed.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
        vector<char> keep(vocabulary_size, 0);
        for (Index i = 0; i < config.top_k; ++i) keep[indexed[i].second] = 1;
        for (Index i = 0; i < vocabulary_size; ++i) if (!keep[i]) probabilities(i) = 0.0f;
    }

    if (config.top_p < 1.0f && config.top_p > 0.0f)
    {
        vector<pair<float, Index>> sorted_probabilities(vocabulary_size);
        float total = 0.0f;
        for (Index i = 0; i < vocabulary_size; ++i)
        {
            sorted_probabilities[i] = {probabilities(i), i};
            total += probabilities(i);
        }
        if (total > 0.0f)
        {
            ranges::sort(sorted_probabilities,
                         [](const auto& a, const auto& b) { return a.first > b.first; });
            float cumulative_probability = 0.0f;
            vector<char> keep(vocabulary_size, 0);
            for (const auto& [probability, token_id] : sorted_probabilities)
            {
                cumulative_probability += probability / total;
                keep[token_id] = 1;
                if (cumulative_probability >= config.top_p) break;
            }
            for (Index i = 0; i < vocabulary_size; ++i) if (!keep[i]) probabilities(i) = 0.0f;
        }
    }

    const float sum = probabilities.sum();
    if (sum <= 0.0f)
    {
        Index best;
        original.maxCoeff(&best);
        return best;
    }

    const float sample_threshold = random_uniform(0.0f, sum);
    float cumulative_probability = 0.0f;
    for (Index i = 0; i < vocabulary_size; ++i)
    {
        cumulative_probability += probabilities(i);
        if (cumulative_probability >= sample_threshold) return i;
    }
    return vocabulary_size - 1;
}

namespace
{

constexpr Index pad_token_id     = 0;
constexpr Index unknown_token_id = 1;
constexpr Index start_token_id   = 2;
constexpr Index end_token_id     = 3;

bool is_printable_token(Index token_id)
{
    return token_id != pad_token_id
        && token_id != start_token_id
        && token_id != end_token_id;
}

SamplingConfig greedy_config()
{
    return {.temperature = 0.0f};
}

Tokenizer& get_tokenizer_layer(const NeuralNetwork& network, const string& label, const char* method)
{
    Tokenizer* tokenizer_layer = nullptr;

    try
    {
        tokenizer_layer = dynamic_cast<Tokenizer*>(network.get_layer(label).get());
    }
    catch (const exception&)
    {
    }

    throw_if(!tokenizer_layer,
             format("{}: network has no '{}' layer. Rebuild the network or re-save the model with a tokenizer.",
                    method, label));

    return *tokenizer_layer;
}

struct GenerationSession
{
    Buffer arena{Device::CUDA};
    TensorView source_ids_device;
    TensorView target_ids_device;
    unique_ptr<ForwardPropagation> forward_propagation;
    vector<TensorView> inputs;

    Tensor2 source_ids;
    Tensor2 target_ids;
    vector<Index> history;
    VectorR distribution;
    vector<uint16_t> bf16_staging;

    Index input_sequence_length = 0;
    Index sequence_length = 0;

    Index decoder_embedding_index = -1;
    Index encoder_embedding_index = -1;
    Index encoder_last_index      = -1;
    Index decoder_first_index     = -1;
    Index output_projection_index = -1;
};

void identify_seq2seq_ranges(const NeuralNetwork& network, GenerationSession& session)
{
    const auto& layers = network.get_layers();
    const Index layers_number = ssize(layers);

    session.decoder_embedding_index = network.get_layer_index("decoder_embedding");
    session.encoder_embedding_index = network.get_layer_index("encoder_embedding");

    Index first_cross_attention_index = -1;
    for (Index i = 0; i < layers_number; ++i)
    {
        if (layers[i]->get_label().starts_with("cross_attention_"))
        {
            first_cross_attention_index = i;
            break;
        }
    }
    throw_if(first_cross_attention_index < 0,
             "Transformer::decode: no 'cross_attention_*' layer found.");

    const vector<Index>& cross_sources = network.get_source_layers()[first_cross_attention_index];
    throw_if(cross_sources.size() < 2 || cross_sources[1] < 0,
             "Transformer::decode: first cross_attention layer must have 2 valid inputs (decoder, encoder).");

    session.encoder_last_index = cross_sources[1];

    session.decoder_first_index = session.encoder_last_index + 1;
    throw_if(session.decoder_first_index >= layers_number,
             "Transformer::decode: decoder stack first index out of range.");
    throw_if(layers[session.decoder_first_index]->get_label() != "decoder_self_attention_1",
             format("Transformer::decode: layer after encoder expected to be 'decoder_self_attention_1', found '{}'.",
                    layers[session.decoder_first_index]->get_label()));

    session.output_projection_index = layers_number - 1;
    throw_if(layers[session.output_projection_index]->get_label() != "output_projection",
             format("Transformer::decode: last layer expected to be 'output_projection', found '{}'.",
                    layers[session.output_projection_index]->get_label()));
}

void prepare_device(NeuralNetwork& network)
{
    network.copy_parameters_device();
    network.link_parameters();
    network.copy_states_device();
    network.link_states();
}

void size_session_distribution(const NeuralNetwork& network, GenerationSession& session)
{
    const Index vocabulary_size = network.get_layers().back()->get_output_shape().back();
    session.distribution = VectorR::Zero(vocabulary_size);
    session.bf16_staging.assign(static_cast<size_t>(vocabulary_size), 0);
}

GenerationSession make_seq2seq_session(Transformer& network)
{
    throw_if(!network.is_gpu() || !device::is_cuda_build(),
             "Transformer::decode requires GPU configuration.");

    GenerationSession session;

    session.input_sequence_length = network.get_input_sequence_length();
    session.sequence_length       = network.get_decoder_sequence_length();

    throw_if(network.get_input_vocabulary().empty(),
             "Transformer::decode: input vocabulary is empty. Set it with set_input_vocabulary/set_input_tokenizer or load a model saved with one.");
    throw_if(network.get_target_vocabulary().empty(),
             "Transformer::decode: target vocabulary is empty. Set it with set_target_vocabulary/set_target_tokenizer or load a model saved with one.");

    prepare_device(network);

    identify_seq2seq_ranges(network, session);

    constexpr Index batch_size = 1;

    const Index source_bytes = get_aligned_bytes(batch_size * session.input_sequence_length, Type::FP32);
    const Index target_bytes = get_aligned_bytes(batch_size * session.sequence_length, Type::FP32);

    session.arena.resize_bytes(source_bytes + target_bytes, Device::CUDA);

    char* const base = session.arena.as<char>();
    session.source_ids_device = TensorView(base,
                                           {batch_size, session.input_sequence_length},
                                           Type::FP32,
                                           Device::CUDA);
    session.target_ids_device = TensorView(base + source_bytes,
                                           {batch_size, session.sequence_length},
                                           Type::FP32,
                                           Device::CUDA);

    session.forward_propagation = make_unique<ForwardPropagation>(batch_size, &network);

    session.source_ids = Tensor2(batch_size, session.input_sequence_length);
    session.target_ids = Tensor2(batch_size, session.sequence_length);
    session.history.reserve(session.sequence_length);

    session.inputs = {session.target_ids_device, session.source_ids_device};

    size_session_distribution(network, session);

    return session;
}

GenerationSession make_decoder_only_session(TextGenerationNetwork& network)
{
    throw_if(!network.is_gpu() || !device::is_cuda_build(),
             "TextGenerationNetwork::generate requires GPU configuration.");

    GenerationSession session;

    session.sequence_length = network.get_sequence_length();

    throw_if(network.get_vocabulary().empty(),
             "TextGenerationNetwork::generate: vocabulary is empty. Set it with set_vocabulary/set_tokenizer or load a model saved with one.");

    prepare_device(network);

    network.get_layer_index("embedding");
    throw_if(network.get_layers().back()->get_label() != "output_projection",
             format("TextGenerationNetwork::generate: last layer expected to be 'output_projection', found '{}'.",
                    network.get_layers().back()->get_label()));

    constexpr Index batch_size = 1;

    session.arena.resize_bytes(get_aligned_bytes(batch_size * session.sequence_length, Type::FP32), Device::CUDA);

    session.target_ids_device = TensorView(session.arena.as<char>(),
                                           {batch_size, session.sequence_length},
                                           Type::FP32,
                                           Device::CUDA);

    session.forward_propagation = make_unique<ForwardPropagation>(batch_size, &network);

    session.target_ids = Tensor2(batch_size, session.sequence_length);
    session.history.reserve(session.sequence_length);

    session.inputs = {session.target_ids_device};

    size_session_distribution(network, session);

    return session;
}

void reset_per_prompt_state(GenerationSession& session)
{
    session.target_ids.setConstant(pad_token_id);
    session.target_ids(0, 0) = start_token_id;
    session.history.clear();

    cudaStream_t stream = Backend::get_compute_stream();
    device::copy_async(session.target_ids_device.data,
                       session.target_ids.data(),
                       session.target_ids_device.byte_size(),
                       device::CopyKind::HostToDevice,
                       stream);
}

void encode_source(Transformer& network, GenerationSession& session, const string& source)
{
    const TokenizerOperator* input_tokenizer = network.get_input_tokenizer();
    const auto& input_vocabulary_map = input_tokenizer->get_vocabulary_map();

    session.source_ids.setConstant(pad_token_id);
    session.source_ids(0, 0) = start_token_id;

    const vector<string> source_tokens = input_tokenizer->tokenize(source);
    Index write_index = 1;
    for (const string& token : source_tokens)
    {
        if (write_index >= session.input_sequence_length) break;

        const auto it = input_vocabulary_map.find(token);
        session.source_ids(0, write_index) = (it != input_vocabulary_map.end())
                                                 ? static_cast<float>(it->second)
                                                 : unknown_token_id;
        ++write_index;
    }
    if (write_index < session.input_sequence_length)
        session.source_ids(0, write_index) = end_token_id;

    cudaStream_t stream = Backend::get_compute_stream();
    device::copy_async(session.source_ids_device.data,
                       session.source_ids.data(),
                       session.source_ids_device.byte_size(),
                       device::CopyKind::HostToDevice,
                       stream);
    network.forward_propagate(session.inputs, *session.forward_propagation, false,
                              session.encoder_embedding_index,
                              session.encoder_last_index);
}

void read_distribution(GenerationSession& session, Index position)
{
    cudaStream_t stream = Backend::get_compute_stream();

    const TensorView output_view = session.forward_propagation->get_outputs();
    const Index vocabulary_size = output_view.shape[2];
    const Index slice_offset = position * vocabulary_size;
    if (output_view.is_bf16())
    {
        device::copy_async(session.bf16_staging.data(),
                           output_view.as<bfloat16>() + slice_offset,
                           vocabulary_size * Index(sizeof(uint16_t)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
        for (Index i = 0; i < vocabulary_size; ++i)
        {
            session.distribution(i) = bit_cast<float>(static_cast<uint32_t>(session.bf16_staging[size_t(i)]) << 16);
        }
    }
    else if (output_view.is_fp32())
    {
        device::copy_async(session.distribution.data(),
                           output_view.as<float>() + slice_offset,
                           vocabulary_size * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
    }
    else
    {
        throw runtime_error("Text generation: unsupported output dtype.");
    }
}

Index decode_step(Transformer& network, GenerationSession& session,
                  Index step_index, const SamplingConfig& config)
{
    network.forward_propagate(session.inputs, *session.forward_propagation, false,
                              session.decoder_embedding_index,
                              session.decoder_embedding_index);

    network.forward_propagate(session.inputs, *session.forward_propagation, false,
                              session.decoder_first_index,
                              session.output_projection_index);

    read_distribution(session, step_index - 1);

    return sample_token(session.distribution, config, session.history);
}

Index generate_step(TextGenerationNetwork& network, GenerationSession& session,
                    const vector<Index>& context, const SamplingConfig& config)
{
    // Sliding window over the tail of the context: causal masking makes the
    // trailing PAD positions irrelevant to the predictions at earlier positions.
    const Index window_length = min(ssize(context), session.sequence_length);
    const Index window_start = ssize(context) - window_length;

    session.target_ids.setConstant(pad_token_id);
    for (Index j = 0; j < window_length; ++j)
        session.target_ids(0, j) = static_cast<float>(context[size_t(window_start + j)]);

    cudaStream_t stream = Backend::get_compute_stream();
    device::copy_async(session.target_ids_device.data,
                       session.target_ids.data(),
                       session.target_ids_device.byte_size(),
                       device::CopyKind::HostToDevice,
                       stream);

    network.forward_propagate(session.inputs, *session.forward_propagation, false);

    read_distribution(session, window_length - 1);

    return sample_token(session.distribution, config, session.history);
}

string assemble_output_string(const GenerationSession& session, const vector<string>& output_vocabulary)
{
    string result;
    for (Index i = 1; i < session.sequence_length; ++i)
    {
        const Index token_id = static_cast<Index>(session.target_ids(0, i));
        if (token_id == end_token_id || token_id == pad_token_id) break;

        if (token_id < 0 || token_id >= ssize(output_vocabulary)) continue;

        if (!result.empty()) result += " ";
        result += output_vocabulary[size_t(token_id)];
    }
    return result;
}

string decode_with_session(Transformer& network, GenerationSession& session,
                           const string& source, const SamplingConfig& config,
                           const TokenCallback& on_token)
{
    reset_per_prompt_state(session);
    encode_source(network, session, source);

    const Index generation_limit = (config.maximum_tokens > 0)
        ? min(config.maximum_tokens + Index(1), session.sequence_length)
        : session.sequence_length;

    const vector<string>& output_vocabulary = network.get_target_vocabulary();

    for (Index i = 1; i < generation_limit; ++i)
    {
        const Index next_token_id = decode_step(network, session, i, config);

        session.target_ids(0, i) = static_cast<float>(next_token_id);
        session.history.push_back(next_token_id);

        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(session.target_ids_device.as<float>() + i,
                           &session.target_ids(0, i),
                           Index(sizeof(float)),
                           device::CopyKind::HostToDevice,
                           stream);

        if (next_token_id == end_token_id)
            break;

        if (on_token && is_printable_token(next_token_id)
            && next_token_id >= 0 && next_token_id < ssize(output_vocabulary))
            on_token(output_vocabulary[size_t(next_token_id)]);
    }

    return assemble_output_string(session, output_vocabulary);
}

string generate_with_session(TextGenerationNetwork& network, GenerationSession& session,
                             const string& prompt, const SamplingConfig& config,
                             const TokenCallback& on_token)
{
    const TokenizerOperator* tokenizer = network.get_tokenizer();
    const vector<string>& vocabulary = tokenizer->get_vocabulary();

    // A subword tokenizer decodes text directly; word-level maps ids through
    // the vocabulary and joins them with spaces.
    const bool word_level = tokenizer->get_kind() == "WordLevel";

    vector<Index> context = tokenizer->encode(prompt);

    throw_if(context.empty(),
             "TextGenerationNetwork::generate: prompt produced no tokens.");

    session.history = context;

    const Index maximum_new_tokens = (config.maximum_tokens > 0)
        ? config.maximum_tokens
        : session.sequence_length;
    const Index prompt_size = ssize(context);

    string result;

    for (Index step = 0; step < maximum_new_tokens; ++step)
    {
        const Index next_token_id = generate_step(network, session, context, config);

        context.push_back(next_token_id);
        session.history.push_back(next_token_id);

        if (next_token_id == pad_token_id
            || next_token_id < 0 || next_token_id >= ssize(vocabulary))
            continue;

        if (!word_level)
        {
            // Re-decode the generated tail so subwords join into text correctly,
            // and stream only the newly produced suffix.
            const vector<Index> generated(context.begin() + prompt_size, context.end());
            const string text = tokenizer->decode(generated);
            if (on_token && text.size() > result.size())
                on_token(text.substr(result.size()));
            result = text;
        }
        else
        {
            const string& token = vocabulary[size_t(next_token_id)];
            if (!result.empty()) result += " ";
            result += token;
            if (on_token) on_token(token);
        }
    }

    return result;
}

void chat_loop(const function<void(const string&)>& answer_prompt)
{
    cout << "Enter prompts. Empty line or Ctrl+D to exit.\n";

    string prompt_line;
    while (true)
    {
        cout << "\n> " << flush;
        if (!getline(cin, prompt_line) || prompt_line.empty()) break;

        answer_prompt(prompt_line);
        cout << "\n";
    }
    cout << "Bye!\n";
}

}

Transformer::Transformer(const filesystem::path& path)
    : NeuralNetwork(path)
{
}

void Transformer::set_input_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::set_input_tokenizer")
        .set_tokenizer(move(new_tokenizer));
}

void Transformer::set_target_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::set_target_tokenizer")
        .set_tokenizer(move(new_tokenizer));
}

void Transformer::set_input_vocabulary(const vector<string>& new_vocabulary)
{
    get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::set_input_vocabulary")
        .set_vocabulary(new_vocabulary);
}

void Transformer::set_target_vocabulary(const vector<string>& new_vocabulary)
{
    get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::set_target_vocabulary")
        .set_vocabulary(new_vocabulary);
}

const TokenizerOperator* Transformer::get_input_tokenizer() const
{
    return get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::get_input_tokenizer").get_tokenizer();
}

const TokenizerOperator* Transformer::get_target_tokenizer() const
{
    return get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::get_target_tokenizer").get_tokenizer();
}

const vector<string>& Transformer::get_input_vocabulary() const
{
    return get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::get_input_vocabulary").get_vocabulary();
}

const vector<string>& Transformer::get_target_vocabulary() const
{
    return get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::get_target_vocabulary").get_vocabulary();
}

string Transformer::decode(const string& source)
{
    return decode(source, greedy_config(), TokenCallback{});
}

string Transformer::decode(const string& source, const SamplingConfig& config)
{
    return decode(source, config, TokenCallback{});
}

string Transformer::decode(const string& source, const TokenCallback& on_token)
{
    return decode(source, greedy_config(), on_token);
}

string Transformer::decode(const string& source,
                           const SamplingConfig& config,
                           const TokenCallback& on_token)
{
    GenerationSession session = make_seq2seq_session(*this);

    return decode_with_session(*this, session, source, config, on_token);
}

string Transformer::decode_to_stream(const string& source, ostream& out)
{
    return decode_to_stream(source, greedy_config(), out);
}

string Transformer::decode_to_stream(const string& source,
                                     const SamplingConfig& config,
                                     ostream& out)
{
    bool first_token = true;

    return decode(source, config, stream_token_callback(out, first_token, false));
}

void Transformer::chat()
{
    chat(greedy_config());
}

void Transformer::chat(const SamplingConfig& config)
{
    GenerationSession session = make_seq2seq_session(*this);

    chat_loop([&](const string& prompt_line)
    {
        bool first_token = true;
        decode_with_session(*this, session, prompt_line, config,
                            stream_token_callback(cout, first_token, false));
    });
}

TextGenerationNetwork::TextGenerationNetwork(const filesystem::path& path)
    : NeuralNetwork(path)
{
}

void TextGenerationNetwork::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::set_tokenizer")
        .set_tokenizer(move(new_tokenizer));
}

void TextGenerationNetwork::set_vocabulary(const vector<string>& new_vocabulary)
{
    get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::set_vocabulary")
        .set_vocabulary(new_vocabulary);
}

const TokenizerOperator* TextGenerationNetwork::get_tokenizer() const
{
    return get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::get_tokenizer").get_tokenizer();
}

const vector<string>& TextGenerationNetwork::get_vocabulary() const
{
    return get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::get_vocabulary").get_vocabulary();
}

string TextGenerationNetwork::generate(const string& prompt)
{
    return generate(prompt, greedy_config(), TokenCallback{});
}

string TextGenerationNetwork::generate(const string& prompt, const SamplingConfig& config)
{
    return generate(prompt, config, TokenCallback{});
}

string TextGenerationNetwork::generate(const string& prompt, const TokenCallback& on_token)
{
    return generate(prompt, greedy_config(), on_token);
}

string TextGenerationNetwork::generate(const string& prompt,
                                       const SamplingConfig& config,
                                       const TokenCallback& on_token)
{
    GenerationSession session = make_decoder_only_session(*this);

    return generate_with_session(*this, session, prompt, config, on_token);
}

string TextGenerationNetwork::generate_to_stream(const string& prompt, ostream& out)
{
    return generate_to_stream(prompt, greedy_config(), out);
}

string TextGenerationNetwork::generate_to_stream(const string& prompt,
                                                 const SamplingConfig& config,
                                                 ostream& out)
{
    bool first_token = true;
    const bool raw = get_tokenizer() && get_tokenizer()->get_kind() != "WordLevel";

    return generate(prompt, config, stream_token_callback(out, first_token, raw));
}

void TextGenerationNetwork::chat()
{
    chat(greedy_config());
}

void TextGenerationNetwork::chat(const SamplingConfig& config)
{
    GenerationSession session = make_decoder_only_session(*this);

    const bool raw = get_tokenizer() && get_tokenizer()->get_kind() != "WordLevel";

    chat_loop([&](const string& prompt_line)
    {
        bool first_token = true;
        generate_with_session(*this, session, prompt_line, config,
                              stream_token_callback(cout, first_token, raw));
    });
}

Index Bert::get_sequence_length() const
{
    return get_layer("word_embeddings")->get_input_shape()[0];
}

Index Bert::get_hidden_size() const
{
    return get_layer(0)->get_output_shape().back();
}

Index TextGenerationNetwork::get_heads_number() const
{
    if (auto* mha = dynamic_cast<const MultiHeadAttention*>(get_first(LayerType::MultiHeadAttention)))
        return mha->get_heads_number();

    return 0;
}

Index Bert::get_heads_number() const
{
    if (auto* mha = dynamic_cast<const MultiHeadAttention*>(get_first(LayerType::MultiHeadAttention)))
        return mha->get_heads_number();

    return 0;
}

BertForSequenceClassification::BertForSequenceClassification(Index sequence_length,
                                                             Index vocabulary_size,
                                                             Index hidden_size,
                                                             Index heads_number,
                                                             Index intermediate_size,
                                                             Index layers_number,
                                                             Index labels_number,
                                                             Index type_vocabulary_size)
    : NeuralNetwork()
{
    throw_if(labels_number == 0, "BertForSequenceClassification: labels_number must be > 0.");

    const Index encoder_index = add_bert_encoder(*this, sequence_length, vocabulary_size, hidden_size,
                                                 heads_number, intermediate_size, layers_number,
                                                 type_vocabulary_size);

    add_layer(make_unique<Pooling3d>(Shape{sequence_length, hidden_size},
                                     PoolingMethod::FirstToken, "cls_pooling"),
              {encoder_index});

    add_layer(make_unique<Dense>(Shape{hidden_size}, Shape{hidden_size}, "Tanh", false, "pooler"));

    add_layer(make_unique<Dense>(Shape{hidden_size}, Shape{labels_number},
                                 labels_number == 1 ? "Sigmoid" : "Softmax", false, "classifier"));

    compile();
    set_parameters_glorot();
}

void BertForSequenceClassification::set_dropout_rate(const float new_dropout_rate)
{
    const auto forward_before = get_forward_specs(1);
    const auto backward_before = get_backward_specs(1);

    for (auto& layer : get_layers())
    {
        if (!layer) continue;

        if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
        {
            mha->set_dropout_rate(new_dropout_rate);
            continue;
        }

        if (starts_with_any(layer->get_label(), {"feed_forward_output", "pooler"}))
            if (auto* dense = dynamic_cast<Dense*>(layer.get()))
                dense->set_dropout_rate(new_dropout_rate);
    }

    recompile_if_specs_changed(*this, forward_before, backward_before);
}

#endif // OPENNN_NO_VISION

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
