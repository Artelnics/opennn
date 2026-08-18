//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/standard_networks.h"

#include <utility>

#include "opennn/core/string_utilities.h"
#include "opennn/neural_network/layers/activation_layer.h"
#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/layers/bounding_layer.h"
#include "opennn/neural_network/layers/c2psa_layer.h"
#include "opennn/neural_network/layers/concatenation_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/grouped_query_attention_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include "opennn/neural_network/layers/non_max_suppression_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/pooling_layer_3d.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/tokenizer_layer.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/neural_network/layers/upsample_layer.h"

namespace opennn
{

static void recompile_if_specs_changed(NeuralNetwork& network,
                                       const vector<vector<TensorSpec>>& forward_before,
                                       const vector<vector<TensorSpec>>& backward_before)
{
    if (forward_before == network.get_forward_specs(1)
        && backward_before == network.get_backward_specs(1))
        return;

    VectorR parameters_snapshot;
    if (network.get_parameters_buffer_size() > 0)
    {
        network.copy_parameters_host();
        parameters_snapshot = network.get_parameters_map();
    }

    network.compile();

    if (parameters_snapshot.size() > 0)
        network.set_parameters(parameters_snapshot);
}

static void finalize_build(NeuralNetwork& network)
{
    network.compile();
    network.set_parameters_glorot();
}

static void add_dense_stack(NeuralNetwork& network,
                            const Shape& complexity_dimensions,
                            const string& hidden_activation)
{
    for (size_t i = 0; i < complexity_dimensions.get_rank(); ++i)
        network.add_layer(make_unique<Dense>(network.get_output_shape(),
                                             Shape{ complexity_dimensions[i] },
                                             hidden_activation,
                                             false,
                                             format("dense_layer_{}", i + 1)));
}

template<typename MakeLayer>
static void add_recurrent_stack(NeuralNetwork& network,
                                const Shape& complexity_dimensions,
                                const string& base_label,
                                MakeLayer make_layer)
{
    const Index layer_count = complexity_dimensions.get_rank();

    for (Index i = 0; i < layer_count; ++i)
    {
        const bool last = (i == layer_count - 1);
        auto layer = make_layer(network.get_output_shape(),
                                Shape{complexity_dimensions[i]},
                                last ? base_label : format("{}_{}", base_label, i + 1));
        if (!last) layer->set_return_sequences(true);
        network.add_layer(std::move(layer));
    }
}

static void add_regression_output(NeuralNetwork& network,
                                  const Shape& output_shape,
                                  const string& output_label,
                                  const char* bounding_method)
{
    network.add_layer(make_unique<Dense>(network.get_output_shape(),
                                         output_shape,
                                         "Identity",
                                         false,
                                         output_label));

    network.add_layer(make_unique<Unscaling>(output_shape));

    auto bounding = make_unique<Bounding>(output_shape);
    if (bounding_method) bounding->set_bounding_method(bounding_method);
    network.add_layer(std::move(bounding));
}

ApproximationNetwork::ApproximationNetwork(const Shape& input_shape,
                                           const Shape& complexity_dimensions,
                                           const Shape& output_shape,
                                           const string& hidden_activation)
    : NeuralNetwork(NetworkTask::Approximation)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, hidden_activation);

    add_regression_output(*this, output_shape, "approximation_layer", nullptr);

    finalize_build(*this);
}

ClassificationNetwork::ClassificationNetwork(const Shape& input_shape,
                                             const Shape& complexity_dimensions,
                                             const Shape& output_shape)
    : NeuralNetwork(NetworkTask::Classification)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, "Tanh");

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    finalize_build(*this);
}

ForecastingNetwork::ForecastingNetwork(const Shape& input_shape,
                                       const Shape& complexity_dimensions,
                                       const Shape& output_shape)
    : NeuralNetwork(NetworkTask::Forecasting)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_recurrent_stack(*this, complexity_dimensions, "recurrent_layer",
                        [](const Shape& in, const Shape& out, const string& label)
                        { return make_unique<Recurrent>(in, out, "Tanh", label); });

    add_regression_output(*this, output_shape, "forecasting_layer", "NoBounding");

    finalize_build(*this);
}

ForecastingLstmNetwork::ForecastingLstmNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape)
    : NeuralNetwork(NetworkTask::Forecasting)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_recurrent_stack(*this, complexity_dimensions, "long_short_term_memory_layer",
                        [](const Shape& in, const Shape& out, const string& label)
                        { return make_unique<LongShortTermMemory>(in, out, "Tanh", "Sigmoid", label); });

    add_regression_output(*this, output_shape, "forecasting_layer", "NoBounding");

    finalize_build(*this);
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape)
    : NeuralNetwork(NetworkTask::AutoAssociation)
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

    finalize_build(*this);
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& encoder_dimensions,
                                               const string& hidden_activation,
                                               const string& output_activation)
    : NeuralNetwork(NetworkTask::AutoAssociation)
{
    throw_if(input_shape.empty(),
             "AutoAssociationNetwork: input shape cannot be empty.");
    throw_if(encoder_dimensions.empty(),
             "AutoAssociationNetwork: encoder dimensions cannot be empty.");

    add_layer(make_unique<Scaling>(input_shape));

    for (size_t i = 0; i < encoder_dimensions.get_rank(); ++i)
    {
        throw_if(encoder_dimensions[i] <= 0,
                 "AutoAssociationNetwork: encoder dimensions must be positive.");

        const bool bottleneck = i == encoder_dimensions.get_rank() - 1;
        add_layer(make_unique<Dense>(get_output_shape(),
                                     Shape{encoder_dimensions[i]},
                                     hidden_activation,
                                     false,
                                     bottleneck ? "bottleneck_layer"
                                                : format("encoder_layer_{}", i + 1)));
    }

    Index decoder = 1;
    for (Index i = Index(encoder_dimensions.get_rank()) - 2; i >= 0; --i, ++decoder)
        add_layer(make_unique<Dense>(get_output_shape(),
                                     Shape{encoder_dimensions[i]},
                                     hidden_activation,
                                     false,
                                     format("decoder_layer_{}", decoder)));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 input_shape,
                                 output_activation,
                                 false,
                                 "output_layer"));

    add_layer(make_unique<Unscaling>(input_shape));

    finalize_build(*this);
}

#ifndef OPENNN_NO_VISION

ImageClassificationNetwork::ImageClassificationNetwork(const Shape& input_shape,
                                                       const Shape& complexity_dimensions,
                                                       const Shape& output_shape)
    : NeuralNetwork(NetworkTask::ImageClassification)
{
    throw_if(input_shape.get_rank() != 3, "Input shape size is not 3.");

    auto scaling_layer = make_unique<Scaling>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    add_layer(std::move(scaling_layer));

    const Index complexity_size = complexity_dimensions.get_rank();

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

    finalize_build(*this);
}

ResNet::ResNet(const Shape& input_shape,
               const vector<Index>& blocks_per_stage,
               const Shape& initial_filters,
               const Shape& output_shape,
               bool use_bottleneck)
    : NeuralNetwork(NetworkTask::ImageClassification)
{
    throw_if(input_shape.get_rank() != 3, "ResNet: input shape must be rank 3 (H, W, C).");
    throw_if(Index(blocks_per_stage.size()) != Index(initial_filters.get_rank()),
             "ResNet: blocks_per_stage and initial_filters must have the same size.");
    throw_if(blocks_per_stage.empty(), "ResNet: at least one stage is required.");

    constexpr Index bottleneck_expansion = 4;

    auto add_conv = [&](Index input_index,
                        const Shape& kernel_shape, const char* activation,
                        const Shape& stride, const string& name) -> Index {
        add_layer(make_unique<Convolutional>(
                      get_layer(input_index)->get_output_shape(),
                      kernel_shape, activation, stride, "Same",
                       true, name),
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

    auto add_residual_conv = [&](Index input_index, Index skip_index,
                                 const Shape& kernel_shape, const string& name) -> Index {
        auto conv = make_unique<Convolutional>(
            get_layer(input_index)->get_output_shape(),
            kernel_shape, "ReLU", Shape{1, 1}, "Same",
             true, name);
        conv->set_residual(true);
        add_layer(std::move(conv), {input_index, skip_index});
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
    add_layer(std::move(scaling_layer));

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

static vector<array<float, 2>> sort_anchors_by_area(const vector<array<float, 2>>& anchors)
{
    vector<array<float, 2>> sorted = anchors;
    ranges::sort(sorted, {}, [](const array<float, 2>& a) { return a[0] * a[1]; });
    return sorted;
}

YoloNetwork::YoloNetwork(const Shape& input_shape,
                         Index classes_number,
                         const vector<array<float, 2>>& anchors,
                         Index grid_size,
                         Backbone backbone,
                         ClassActivation class_activation,
                         HeadStyle head_style,
                         BodyActivation body_activation,
                         bool use_sppf,
                         Index reg_max,
                         ModelSize model_size)
    : NeuralNetwork(NetworkTask::ObjectDetection)
{
    throw_if(input_shape.get_rank() != 3, "YoloNetwork: input shape must be rank 3 (H, W, C).");
    throw_if(classes_number <= 0 || anchors.empty(),
             "YoloNetwork: classes_number and anchors must be valid.");
    throw_if(input_shape[0] != grid_size * 32 || input_shape[1] != grid_size * 32,
             "YoloNetwork: this minimal builder expects input H/W == grid_size * 32.");
    if (head_style == HeadStyle::FPN)
    {
        throw_if(backbone != Backbone::DarknetTiny && backbone != Backbone::DarknetTinyV3
                 && backbone != Backbone::Darknet53 && backbone != Backbone::CSPDarknet53,
                 "YoloNetwork: HeadStyle::FPN requires DarknetTiny, DarknetTinyV3, Darknet53, or CSPDarknet53.");
        throw_if(ssize(anchors) != 9 && ssize(anchors) != 6,
                 "YoloNetwork: HeadStyle::FPN expects 6 anchors (2-head) or 9 anchors (3-head).");
    }
    if (head_style == HeadStyle::PANet)
    {
        throw_if(backbone != Backbone::Darknet53 && backbone != Backbone::CSPDarknet53,
                 "YoloNetwork: HeadStyle::PANet requires Backbone::Darknet53 or CSPDarknet53.");
        throw_if(ssize(anchors) != 9,
                 "YoloNetwork: HeadStyle::PANet requires exactly 9 anchors.");
    }

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

    // Prior bias for anchor-based fused detection heads ("yolo_logits*"):
    // Each fused conv bias has layout [tx,ty,tw,th,obj,c0..cN] × bpc.
    // Set objectness (pos 4) and class (pos 5..4+C) biases to -4.5951 per anchor.
    // Box coord biases (pos 0..3 per anchor) stay at 0.
    auto apply_yolo_prior_bias = [&](Index n_classes) {
        static constexpr float PRIOR_BIAS = -4.5951f;
        const Index vpb = 5 + n_classes;
        for (const auto& layer : get_layers())
        {
            auto* conv = dynamic_cast<Convolutional*>(layer.get());
            if (!conv) continue;
            if (conv->get_label().rfind("yolo_logits", 0) != 0) continue;
            auto& views = conv->get_parameter_views();
            if (views.empty() || views[0].empty()) continue;
            float* b = views[0].as<float>();
            const Index n = conv->get_kernels_number();
            for (Index k = 0; k * vpb <= n - vpb; ++k)
                std::fill(b + k * vpb + 4, b + (k + 1) * vpb, PRIOR_BIAS);
        }
    };

    // Every backbone below ends the same way, and the prior bias has to be
    // applied after the parameters are randomised or it is overwritten.
    auto finish_yolo_network = [&] {
        compile();
        set_parameters_random();
        apply_yolo_prior_bias(classes_number);
    };

    auto add_det_head = [&](Index feature_index,
                            const vector<array<float, 2>>& head_anchors,
                            const string& name) -> Index {
        const Index logits_index = add_conv(feature_index,
            Shape{1, 1, get_layer(feature_index)->get_output_shape()[2],
                  3 * (5 + classes_number)},
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

    auto add_residual_block = [&](Index input_index, Index channels, Index mid,
                                  const string& prefix, const char* c1_suffix,
                                  const char* c2_suffix, const char* act_suffix) -> Index {
        Index x = add_conv(input_index, Shape{1, 1, channels, mid},      act,        stride, true, prefix + c1_suffix);
        x       = add_conv(x,           Shape{3, 3, mid,      channels}, "Identity", stride, true, prefix + c2_suffix);
        add_layer(make_unique<Addition>(get_layer(x)->get_output_shape(), prefix + "_add"), {x, input_index});
        const Index add_index = get_layers_number() - 1;
        add_layer(make_unique<Activation>(get_layer(add_index)->get_output_shape(), act, prefix + act_suffix), {add_index});
        return get_layers_number() - 1;
    };

    auto add_yolo_neck = [&](Index idx, Index in_ch,
                             Index ch_small, Index ch_large, const string& pfx) -> Index {
        Index x = add_conv(idx, Shape{1, 1, in_ch,     ch_small}, act, stride, true, pfx+"_c1");
        x       = add_conv(x,   Shape{3, 3, ch_small, ch_large},  act, stride, true, pfx+"_c2");
        x       = add_conv(x,   Shape{1, 1, ch_large, ch_small},  act, stride, true, pfx+"_c3");
        x       = add_conv(x,   Shape{3, 3, ch_small, ch_large},  act, stride, true, pfx+"_c4");
        x       = add_conv(x,   Shape{1, 1, ch_large, ch_small},  act, stride, true, pfx+"_c5");
        return x;
    };

    auto add_top_down = [&](Index lateral_index, Index c_index,
                            const string& upper, const string& lower) -> Index {
        add_layer(make_unique<Upsample>(get_layer(lateral_index)->get_output_shape(),
                                         2, "fpn_" + upper + "_upsample"),
                  {lateral_index});
        const Index up_index = get_layers_number() - 1;

        add_layer(make_unique<Concatenation>(get_layer(c_index)->get_output_shape(),
                      vector<Index>{get_layer(up_index)->get_output_shape()[2],
                                    get_layer(c_index)->get_output_shape()[2]},
                      "fpn_" + lower + "_concatenation"),
                  {up_index, c_index});
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

        struct DarknetStage { Index channels = 0; bool pool = false; bool one_by_one = false; };

        static constexpr array stages = {
            DarknetStage{.channels =   16, .pool = true},
            DarknetStage{.channels =   32, .pool = true},
            DarknetStage{.channels =   64, .pool = true},
            DarknetStage{.channels =  128, .pool = true},
            DarknetStage{.channels =  256, .pool = true},
            DarknetStage{.channels =  512},
            DarknetStage{.channels = 1024},
            DarknetStage{.channels =  256, .one_by_one = true},
        };

        Index c3_index = -1;
        Index last_index = -1;

        for (size_t i = 0; i < stages.size(); ++i)
        {
            const DarknetStage& stage = stages[i];

            const Shape in_shape  = (i == 0) ? input_shape : get_layer(last_index)->get_output_shape();
            const Index  in_ch    = in_shape[2];
            const Index  out_ch   = stage.channels;
            const Index  ksize    = stage.one_by_one ? 1 : 3;

            add_layer(make_unique<Convolutional>(in_shape,
                                                 Shape{ksize, ksize, in_ch, out_ch},
                                                 act, stride, "Same", true,
                                                 format("dntv3_conv_{}", i + 1)));
            last_index = get_layers_number() - 1;

            if (stage.pool)
            {
                add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                               pool, pool_stride, no_padding,
                                               "MaxPooling",
                                               format("dntv3_pool_{}", i + 1)));
                last_index = get_layers_number() - 1;
            }

            if (i == 4) c3_index = get_layers_number() - 1 - (stage.pool ? 1 : 0);

        }

        if (head_style == HeadStyle::FPN)
        {
            const vector<array<float, 2>> anchors_sorted = sort_anchors_by_area(anchors);

            const vector<array<float, 2>> anchors_small(anchors_sorted.begin(),     anchors_sorted.begin() + 3);
            const vector<array<float, 2>> anchors_large(anchors_sorted.begin() + 3, anchors_sorted.end());

            const Index p5_conv = add_conv(last_index,
                Shape{3, 3, get_layer(last_index)->get_output_shape()[2], 512},
                act, stride, true, "fpn_p5_conv");
            add_det_head(p5_conv, anchors_large, "large");

            const Index p5_lateral = add_conv(last_index,
                Shape{1, 1, get_layer(last_index)->get_output_shape()[2], 128},
                act, stride, true, "fpn_p5_lateral");

            const Index p4_concat = add_top_down(p5_lateral, c3_index, "p5", "p4");

            const Index p4_conv = add_conv(p4_concat,
                Shape{3, 3, get_layer(p4_concat)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p4_conv");
            add_det_head(p4_conv, anchors_small, "small");

            finish_yolo_network();
            return;
        }
    }
    else if (backbone == Backbone::CSPDarknet53v11)
    {

        auto add_c3k2_block = [&](Index input_index, Index ch, const string& prefix) -> Index {
            const Index mid = ch / 2;
            Index x = add_conv(input_index, Shape{3, 3, ch, mid},  act,        stride, true, prefix + "_c1");
            x       = add_conv(x,           Shape{3, 3, mid, ch},  "Identity", stride, true, prefix + "_c2");
            add_layer(make_unique<Addition>(get_layer(x)->get_output_shape(), prefix + "_add"), {x, input_index});
            const Index add_index = get_layers_number() - 1;
            add_layer(make_unique<Activation>(get_layer(add_index)->get_output_shape(), act, prefix + "_act"), {add_index});
            return get_layers_number() - 1;
        };

        auto add_c3k2_stage = [&](Index input_index, Index in_ch, Index out_ch,
                                   Index n_blocks, const string& prefix, bool first_stage) -> Index {
            const Index half      = out_ch / 2;
            const Index branch_ch = first_stage ? out_ch : half;

            const Index down = add_conv(input_index, Shape{3, 3, in_ch, out_ch}, act, stride_2, true, prefix + "_down");

            Index branch2 = add_conv(down, Shape{1, 1, out_ch, branch_ch}, act, stride, true, prefix + "_s2");
            for (Index j = 0; j < n_blocks; ++j)
                branch2 = add_c3k2_block(branch2, branch_ch, prefix + format("_b{}", j + 1));
            const Index trans = add_conv(branch2, Shape{1, 1, branch_ch, branch_ch}, act, stride, true, prefix + "_trans");

            const Index branch1 = add_conv(down, Shape{1, 1, out_ch, branch_ch}, act, stride, true, prefix + "_s1");

            const Shape hw = get_layer(branch1)->get_output_shape();
            add_layer(make_unique<Concatenation>(hw, vector<Index>{branch_ch, branch_ch}, prefix + "_cat"),
                      {trans, branch1});
            const Index cat = get_layers_number() - 1;
            return add_conv(cat, Shape{1, 1, 2 * branch_ch, out_ch}, act, stride, true, prefix + "_merge");
        };

        auto scale_ch = [&](Index base) -> Index {
            float w = model_size == ModelSize::n ? 0.25f
                    : model_size == ModelSize::s ? 0.50f
                    : model_size == ModelSize::m ? 0.75f
                    : model_size == ModelSize::x ? 1.25f
                    :                              1.00f;
            return max(Index(8), Index(round(float(base) * w / 8.f) * 8));
        };
        auto scale_d = [&](Index base) -> Index {
            float d = model_size == ModelSize::n ? 0.33f
                    : model_size == ModelSize::s ? 0.33f
                    : model_size == ModelSize::m ? 0.67f
                    :                              1.00f;
            return max(Index(1), Index(round(float(base) * d)));
        };

        const vector<pair<Index, Index>> stages = {
            {scale_ch(64),   scale_d(1)},
            {scale_ch(128),  scale_d(2)},
            {scale_ch(256),  scale_d(8)},
            {scale_ch(512),  scale_d(8)},
            {scale_ch(1024), scale_d(4)},
        };

        const Index stem_ch = max(Index(8), Index(round(32.f * (
            model_size == ModelSize::n ? 0.25f :
            model_size == ModelSize::s ? 0.50f :
            model_size == ModelSize::m ? 0.75f :
            model_size == ModelSize::x ? 1.25f : 1.00f) / 8.f) * 8));

        add_layer(make_unique<Convolutional>(input_shape, Shape{3, 3, input_shape[2], stem_ch},
                                             act, stride, "Same", true, "c11_stem"));
        Index last_index = get_layers_number() - 1;
        Index c3_index = -1, c4_index = -1, c5_index = -1;
        Index in_ch = stem_ch;
        for (size_t i = 0; i < stages.size(); ++i)
        {
            const auto& [ch, nblocks] = stages[i];
            last_index = add_c3k2_stage(last_index, in_ch, ch, nblocks, format("c11_s{}", i + 1), i == 0);
            in_ch = ch;
            if (i == 2) c3_index = last_index;
            if (i == 3) c4_index = last_index;
            if (i == 4) c5_index = last_index;
        }

        if (head_style == HeadStyle::FPNv8)
        {
            constexpr Index head_ch = 64;
            const Index box_ch = 4 * max(reg_max, Index(1));

            auto add_det_head_v8_c11 = [&](Index feat_idx, const string& name) {
                const Index in_ch_h = get_layer(feat_idx)->get_output_shape()[2];
                Index box = add_conv(feat_idx, Shape{3, 3, in_ch_h, head_ch},    act,        stride, true,  name + "_box_c1");
                box       = add_conv(box,      Shape{3, 3, head_ch, head_ch},    act,        stride, true,  name + "_box_c2");
                box       = add_conv(box,      Shape{1, 1, head_ch, box_ch},     "Identity", stride, false, name + "_box_out");
                Index cls = add_conv(feat_idx, Shape{3, 3, in_ch_h, head_ch},    act,        stride, true,  name + "_cls_c1");
                cls       = add_conv(cls,      Shape{3, 3, head_ch, head_ch},    act,        stride, true,  name + "_cls_c2");
                cls       = add_conv(cls,      Shape{1, 1, head_ch, classes_number}, "Identity", stride, false, name + "_cls_out");
                const Shape hw = get_layer(box)->get_output_shape();
                add_layer(make_unique<Concatenation>(hw, vector<Index>{box_ch, classes_number}, name + "_cat"),
                          {box, cls});
                const Index cat = get_layers_number() - 1;
                add_layer(make_unique<DetectionV8>(get_layer(cat)->get_output_shape(), reg_max, name + "_det"), {cat});
            };

            const Index c5_ch = get_layer(c5_index)->get_output_shape()[2];
            const Index c4_ch = get_layer(c4_index)->get_output_shape()[2];
            const Index c3_ch = get_layer(c3_index)->get_output_shape()[2];
            const Index p5_small = c5_ch / 2;
            const Index p4_small = c4_ch / 2;
            const Index p3_small = c3_ch / 2;

            auto build_fpn_trunk_c11 = [&](Index entry, const string& pfx) -> array<Index, 3> {
                const Index p5n = add_yolo_neck(entry, c5_ch, p5_small, c5_ch, pfx + "neck_p5");
                const Index p5l = add_conv(p5n, Shape{1, 1, p5_small, p4_small}, act, stride, true, pfx + "neck_p5_lat");
                add_layer(make_unique<Upsample>(get_layer(p5l)->get_output_shape(), 2, pfx + "fpn_p5_up"), {p5l});
                const Index p5u = get_layers_number() - 1;
                add_layer(make_unique<Concatenation>(get_layer(c4_index)->get_output_shape(),
                                                     vector<Index>{p4_small, c4_ch}, pfx + "fpn_p4_cat"),
                          {p5u, c4_index});
                const Index p4n = add_yolo_neck(get_layers_number() - 1, p4_small + c4_ch, p4_small, c4_ch, pfx + "neck_p4");
                const Index p4l = add_conv(p4n, Shape{1, 1, p4_small, p3_small}, act, stride, true, pfx + "neck_p4_lat");
                add_layer(make_unique<Upsample>(get_layer(p4l)->get_output_shape(), 2, pfx + "fpn_p4_up"), {p4l});
                const Index p4u = get_layers_number() - 1;
                add_layer(make_unique<Concatenation>(get_layer(c3_index)->get_output_shape(),
                                                     vector<Index>{p3_small, c3_ch}, pfx + "fpn_p3_cat"),
                          {p4u, c3_index});
                return {p5n, p4n, add_yolo_neck(get_layers_number() - 1, p3_small + c3_ch, p3_small, c3_ch, pfx + "neck_p3")};
            };

            add_layer(make_unique<C2PSA>(get_layer(c5_index)->get_output_shape(), "c11_c2psa"), {c5_index});
            const Index c2psa_index = get_layers_number() - 1;
            const auto [p5n, p4n, p3n] = build_fpn_trunk_c11(c2psa_index, "c11_");
            const Index p5d = add_conv(p5n, Shape{3, 3, p5_small, c5_ch}, act, stride, true, "c11_neck_p5_pre");
            add_det_head_v8_c11(p5d, "c11_large");
            const Index p4d = add_conv(p4n, Shape{3, 3, p4_small, c4_ch}, act, stride, true, "c11_neck_p4_pre");
            add_det_head_v8_c11(p4d, "c11_medium");
            const Index p3d = add_conv(p3n, Shape{3, 3, p3_small, c3_ch}, act, stride, true, "c11_neck_p3_pre");
            add_det_head_v8_c11(p3d, "c11_small");

            compile();
            set_parameters_random();
            {

                static constexpr float PRIOR_BIAS = -4.5951f;
                for (const auto& layer : get_layers())
                {
                    auto* conv = dynamic_cast<Convolutional*>(layer.get());
                    if (!conv || !conv->get_label().ends_with("_cls_out")) continue;
                    auto& views = conv->get_parameter_views();
                    if (views.empty() || views[0].empty()) continue;
                    float* b = views[0].as<float>();
                    fill(b, b + conv->get_kernels_number(), PRIOR_BIAS);
                }
            }
            return;
        }

        throw runtime_error("YoloNetwork: CSPDarknet53v11 backbone only supports FPNv8 head style.");
    }
    else if (backbone == Backbone::Darknet53 || backbone == Backbone::CSPDarknet53)
    {

        const bool use_csp = (backbone == Backbone::CSPDarknet53);

        auto add_csp_stage = [&](Index input_index, Index in_ch, Index out_ch,
                                  Index n_blocks, const string& prefix, bool first_stage) -> Index {
            const Index half      = out_ch / 2;
            const Index branch_ch = first_stage ? out_ch : half;

            const Index down = add_conv(input_index, Shape{3, 3, in_ch, out_ch}, act, stride_2, true, prefix+"_down");

            Index branch2 = add_conv(down, Shape{1, 1, out_ch, branch_ch}, act, stride, true, prefix+"_s2");
            for (Index j = 0; j < n_blocks; ++j)
                branch2 = add_residual_block(branch2, branch_ch, half,
                                             prefix + format("_b{}", j+1), "_c1", "_c2", "_act");
            const Index trans = add_conv(branch2, Shape{1, 1, branch_ch, branch_ch}, act, stride, true, prefix+"_trans");

            const Index branch1 = add_conv(down, Shape{1, 1, out_ch, branch_ch}, act, stride, true, prefix+"_s1");

            const Shape hw = get_layer(branch1)->get_output_shape();
            add_layer(make_unique<Concatenation>(hw, vector<Index>{branch_ch, branch_ch}, prefix+"_cat"),
                      {trans, branch1});
            const Index cat = get_layers_number() - 1;
            return add_conv(cat, Shape{1, 1, 2 * branch_ch, out_ch}, act, stride, true, prefix+"_merge");
        };

        const vector<pair<Index,Index>> stages = {{64,1},{128,2},{256,8},{512,8},{1024,4}};

        add_layer(make_unique<Convolutional>(input_shape, Shape{3, 3, input_shape[2], 32},
                                             act, stride, "Same", true, use_csp ? "csp53_stem" : "dn53_stem"));
        Index last_index = get_layers_number() - 1;

        Index c3_index = -1, c4_index = -1, c5_index = -1;

        Index in_ch = 32;
        for (size_t i = 0; i < stages.size(); ++i)
        {
            const auto& [ch, nblocks] = stages[i];
            if (use_csp)
                last_index = add_csp_stage(last_index, in_ch, ch, nblocks, format("csp53_s{}", i+1), i == 0);
            else
            {
                last_index = add_conv(last_index, Shape{3, 3, in_ch, ch}, act, stride_2, true,
                                      format("dn53_down_{}", i+1));
                for (Index j = 0; j < nblocks; ++j)
                    last_index = add_residual_block(last_index, ch, ch / 2,
                                                    format("dn53_s{}_b{}", i+1, j+1), "_c1", "_c2", "_act");
            }
            in_ch = ch;
            if (i == 2) c3_index = last_index;
            if (i == 3) c4_index = last_index;
            if (i == 4) c5_index = last_index;
        }

        auto build_fpn_trunk = [&](Index entry, const string& pfx) -> array<Index, 3> {
            const Index p5n = add_yolo_neck(entry, 1024, 512, 1024, pfx + "neck_p5");

            const Index p5l = add_conv(p5n, Shape{1, 1, 512, 256}, act, stride, true, pfx + "neck_p5_lat");
            add_layer(make_unique<Upsample>(get_layer(p5l)->get_output_shape(), 2, pfx + "fpn_p5_up"), {p5l});
            const Index p5u = get_layers_number() - 1;

            add_layer(make_unique<Concatenation>(get_layer(c4_index)->get_output_shape(),
                                                 vector<Index>{256, 512}, pfx + "fpn_p4_cat"),
                      {p5u, c4_index});
            const Index p4n = add_yolo_neck(get_layers_number() - 1, 768, 256, 512, pfx + "neck_p4");

            const Index p4l = add_conv(p4n, Shape{1, 1, 256, 128}, act, stride, true, pfx + "neck_p4_lat");
            add_layer(make_unique<Upsample>(get_layer(p4l)->get_output_shape(), 2, pfx + "fpn_p4_up"), {p4l});
            const Index p4u = get_layers_number() - 1;

            add_layer(make_unique<Concatenation>(get_layer(c3_index)->get_output_shape(),
                                                 vector<Index>{128, 256}, pfx + "fpn_p3_cat"),
                      {p4u, c3_index});
            return {p5n, p4n, add_yolo_neck(get_layers_number() - 1, 384, 128, 256, pfx + "neck_p3")};
        };

        if (head_style == HeadStyle::FPN || head_style == HeadStyle::PANet)
        {
            throw_if(ssize(anchors) != 9, "YoloNetwork: Darknet53 FPN/PANet requires exactly 9 anchors.");

            const vector<array<float,2>> anchors_sorted = sort_anchors_by_area(anchors);
            const vector<array<float,2>> anchors_small (anchors_sorted.begin(),     anchors_sorted.begin()+3);
            const vector<array<float,2>> anchors_medium(anchors_sorted.begin()+3,   anchors_sorted.begin()+6);
            const vector<array<float,2>> anchors_large (anchors_sorted.begin()+6,   anchors_sorted.end());

            Index fpn_entry = c5_index;
            if (use_sppf)
            {
                const Shape c5_shape = get_layer(c5_index)->get_output_shape();
                const Index c5_ch   = c5_shape[2];
                const Index half_ch = c5_ch / 2;

                const Index sppf_in = add_conv(c5_index, Shape{1, 1, c5_ch, half_ch}, act, stride, true, "sppf_in");
                const Shape s_shape = get_layer(sppf_in)->get_output_shape();

                add_layer(make_unique<Pooling>(s_shape, Shape{5, 5}, Shape{1, 1}, Shape{2, 2}, "MaxPooling", "sppf_p1"), {sppf_in});
                const Index p1 = get_layers_number() - 1;
                add_layer(make_unique<Pooling>(s_shape, Shape{5, 5}, Shape{1, 1}, Shape{2, 2}, "MaxPooling", "sppf_p2"), {p1});
                const Index p2 = get_layers_number() - 1;
                add_layer(make_unique<Pooling>(s_shape, Shape{5, 5}, Shape{1, 1}, Shape{2, 2}, "MaxPooling", "sppf_p3"), {p2});
                const Index p3 = get_layers_number() - 1;

                add_layer(make_unique<Concatenation>(s_shape,
                                                     vector<Index>{half_ch, half_ch, half_ch, half_ch}, "sppf_cat"),
                          {sppf_in, p1, p2, p3});
                const Index sppf_cat = get_layers_number() - 1;

                fpn_entry = add_conv(sppf_cat, Shape{1, 1, 2 * c5_ch, c5_ch}, act, stride, true, "sppf_out");
            }

            const auto [p5n, p4n, p3n] = build_fpn_trunk(fpn_entry, "");

            if (head_style == HeadStyle::FPN)
            {

                const Index p5d = add_conv(p5n, Shape{3, 3, 512, 1024}, act, stride, true, "neck_p5_pre");
                add_det_head(p5d, anchors_large, "large");
                const Index p4d = add_conv(p4n, Shape{3, 3, 256, 512}, act, stride, true, "neck_p4_pre");
                add_det_head(p4d, anchors_medium, "medium");
                const Index p3d = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "neck_p3_pre");
                add_det_head(p3d, anchors_small, "small");
            }
            else
            {

                const Index p3d = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "neck_p3_pre");
                add_det_head(p3d, anchors_small, "small");

                auto add_pan_block = [&](Index idx, Index in_ch, Index ch_s, Index ch_l, const string& pfx) -> Index {
                    Index x = add_conv(idx, Shape{1, 1, in_ch, ch_s}, act, stride, true, pfx+"_c1");
                    x       = add_conv(x,   Shape{3, 3, ch_s,  ch_l}, act, stride, true, pfx+"_c2");
                    x       = add_conv(x,   Shape{1, 1, ch_l,  ch_s}, act, stride, true, pfx+"_c3");
                    return x;
                };

                const Index n3_down = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride_2, true, "pan_n3_down");
                add_layer(make_unique<Concatenation>(get_layer(p4n)->get_output_shape(),
                                                     vector<Index>{256, 256}, "pan_n4_cat"),
                          {n3_down, p4n});
                const Index n4c = get_layers_number() - 1;
                const Index n4n = add_pan_block(n4c, 512, 256, 512, "pan_n4");
                const Index n4d = add_conv(n4n, Shape{3, 3, 256, 512}, act, stride, true, "pan_n4_pre");
                add_det_head(n4d, anchors_medium, "medium");

                const Index n4_down = add_conv(n4n, Shape{3, 3, 256, 512}, act, stride_2, true, "pan_n4_down");
                add_layer(make_unique<Concatenation>(get_layer(p5n)->get_output_shape(),
                                                     vector<Index>{512, 512}, "pan_n5_cat"),
                          {n4_down, p5n});
                const Index n5c = get_layers_number() - 1;
                const Index n5n = add_pan_block(n5c, 1024, 512, 1024, "pan_n5");
                const Index n5d = add_conv(n5n, Shape{3, 3, 512, 1024}, act, stride, true, "pan_n5_pre");
                add_det_head(n5d, anchors_large, "large");
            }

            finish_yolo_network();
            return;
        }

        if (head_style == HeadStyle::FPNv8)
        {

            constexpr Index head_ch = 64;

            const Index box_ch = 4 * max(reg_max, Index(1));

            auto add_det_head_v8 = [&](Index feat_idx, const string& name) {
                const Index in_ch = get_layer(feat_idx)->get_output_shape()[2];

                Index box = add_conv(feat_idx, Shape{3,3,in_ch,head_ch},        act,        stride, true,  name+"_box_c1");
                box       = add_conv(box,      Shape{3,3,head_ch,head_ch},      act,        stride, true,  name+"_box_c2");
                box       = add_conv(box,      Shape{1,1,head_ch,box_ch},       "Identity", stride, false, name+"_box_out");

                Index cls = add_conv(feat_idx, Shape{3,3,in_ch,head_ch},               act,        stride, true,  name+"_cls_c1");
                cls       = add_conv(cls,      Shape{3,3,head_ch,head_ch},             act,        stride, true,  name+"_cls_c2");
                cls       = add_conv(cls,      Shape{1,1,head_ch,classes_number},      "Identity", stride, false, name+"_cls_out");

                const Shape hw = get_layer(box)->get_output_shape();
                add_layer(make_unique<Concatenation>(hw, vector<Index>{box_ch, classes_number}, name+"_cat"),
                          {box, cls});
                const Index cat = get_layers_number() - 1;
                add_layer(make_unique<DetectionV8>(get_layer(cat)->get_output_shape(), reg_max, name+"_det"), {cat});
            };

            const auto [p5n, p4n, p3n] = build_fpn_trunk(c5_index, "v8_");

            const Index p5d = add_conv(p5n, Shape{3, 3, 512, 1024}, act, stride, true, "v8_neck_p5_pre");
            add_det_head_v8(p5d, "v8_large");
            const Index p4d = add_conv(p4n, Shape{3, 3, 256, 512}, act, stride, true, "v8_neck_p4_pre");
            add_det_head_v8(p4d, "v8_medium");
            const Index p3d = add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "v8_neck_p3_pre");
            add_det_head_v8(p3d, "v8_small");

            compile();
            set_parameters_random();
            {
                static constexpr float PRIOR_BIAS = -4.5951f;
                for (const auto& layer : get_layers())
                {
                    auto* conv = dynamic_cast<Convolutional*>(layer.get());
                    if (!conv || !conv->get_label().ends_with("_cls_out")) continue;
                    auto& views = conv->get_parameter_views();
                    if (views.empty() || views[0].empty()) continue;
                    float* b = views[0].as<float>();
                    fill(b, b + conv->get_kernels_number(), PRIOR_BIAS);
                }
            }
            return;
        }
    }
    else
    {
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
            const auto& [channels, blocks_number] = stages[i];
            const Index input_channels = get_layer(last_index)->get_output_shape()[2];

            last_index = add_conv(last_index,
                Shape{3, 3, input_channels, channels}, act,
                stride_2, true, format("darknet_down_{}", i + 1));

            for (Index j = 0; j < blocks_number; ++j)
                last_index = add_residual_block(last_index, channels,
                    max<Index>(channels / 2, 1),
                    format("darknet_s{}_b{}", i + 1, j), "_conv1", "_conv2", "_relu");

            if (i == 1) c3_index = last_index;
            if (i == 2) c4_index = last_index;
            if (i == 3) c5_index = last_index;
        }

        if (head_style == HeadStyle::FPN)
        {
            throw_if(ssize(anchors) != 9,
                     "YoloNetwork: DarknetTiny FPN (3-head) requires exactly 9 anchors.");
            const vector<array<float, 2>> anchors_sorted = sort_anchors_by_area(anchors);

            const vector<array<float, 2>> anchors_small (anchors_sorted.begin(),     anchors_sorted.begin() + 3);
            const vector<array<float, 2>> anchors_medium(anchors_sorted.begin() + 3, anchors_sorted.begin() + 6);
            const vector<array<float, 2>> anchors_large (anchors_sorted.begin() + 6, anchors_sorted.end());

            const Index p5_lateral = add_conv(c5_index,
                Shape{1, 1, get_layer(c5_index)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p5_lateral");
            add_det_head(p5_lateral, anchors_large, "large");

            const Index p4_concatenation = add_top_down(p5_lateral, c4_index, "p5", "p4");

            const Index p4_lateral = add_conv(p4_concatenation,
                Shape{1, 1, get_layer(p4_concatenation)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p4_lateral");
            add_det_head(p4_lateral, anchors_medium, "medium");

            const Index p3_concatenation = add_top_down(p4_lateral, c3_index, "p4", "p3");

            const Index p3_lateral = add_conv(p3_concatenation,
                Shape{1, 1, get_layer(p3_concatenation)->get_output_shape()[2], 128},
                act, stride, true, "fpn_p3_lateral");
            add_det_head(p3_lateral, anchors_small, "small");

            finish_yolo_network();
            return;
        }
    }

    const Index detection_channels = ssize(anchors) * (5 + classes_number);

    add_layer(make_unique<Convolutional>(get_output_shape(),
                                         Shape{1, 1, get_output_shape()[2], detection_channels},
                                         "Identity", stride, "Same", false,
                                         "yolo_logits"));
    // Note: prior bias for this single-head path is applied below after set_parameters_random().

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

    finish_yolo_network();
}

TextClassificationNetwork::TextClassificationNetwork(const Shape& input_shape,
                                                     const Shape& complexity_dimensions,
                                                     const Shape& output_shape,
                                                     PoolingMethod pooling_method)
    : NeuralNetwork(NetworkTask::TextClassification)
{
    const Index vocabulary_size = input_shape[0];
    const Index sequence_length = input_shape[1];
    const Index embedding_dimension = input_shape[2];
    const Index heads_number = complexity_dimensions[0];
    const Index hidden_neurons = complexity_dimensions.get_rank() > 1 ? complexity_dimensions[1] : 64;

    add_layer(make_unique<Tokenizer>(Shape{sequence_length}, "tokenizer"), {-1});

    auto embedding_layer = make_unique<Embedding>(Shape({vocabulary_size, sequence_length}),
                                                  embedding_dimension,
                                                  "embedding_layer");
    embedding_layer->set_scale_embedding(true);
    embedding_layer->set_add_positional_encoding(true);
    // The pooling below has to know where each sequence ends, and the positional
    // encoding this Embedding adds already means a padded row is not the zero
    // row anyone downstream could recognise it by.
    embedding_layer->set_export_valid_lengths(true);
    add_layer(std::move(embedding_layer));

    auto attention_layer = make_unique<MultiHeadAttention>(
        Shape({sequence_length, embedding_dimension}),
        heads_number,
        "multihead_attention_layer");

    // No set_zero_padded_queries here. It asked attention to write exactly-zero
    // rows at padded query positions, and paid for them by vetoing cuDNN's
    // fused attention for the whole network. The only reader of those zeros was
    // the pooling below, recovering the sequence length by looking for them;
    // it reads the Embedding's exported lengths now, so the demand is gone and
    // this network's attention can be fused.
    add_layer(std::move(attention_layer));

    add_layer(make_unique<Pooling3d>(get_output_shape(), pooling_method));

    add_layer(make_unique<Dense>(get_output_shape(), Shape({hidden_neurons}), "ReLU", false, "dense_layer_1"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                 false,
                                 "classification_layer"));

    finalize_build(*this);
}

static Index add_residual_and_norm(NeuralNetwork& network,
                                   const Shape& shape,
                                   const string& norm_label,
                                   Index left_index, Index right_index)
{
    auto norm = make_unique<Normalization3d>(shape, norm_label);
    norm->set_fuse_add(true);
    network.add_layer(std::move(norm), {left_index, right_index});
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

Transformer::Transformer()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

Transformer::Transformer(Index input_sequence_length,
                         Index decoder_sequence_length,
                         Index input_vocabulary_size,
                         Index output_vocabulary_size,
                         Index embedding_dimension,
                         Index heads_number,
                         Index feed_forward_dimension,
                         Index layers_number)
    : NeuralNetwork(NetworkTask::LanguageModeling)
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
    // Both embeddings export, and the two records stay apart because they are
    // held per layer. The normalizations between the blocks shift a padded row
    // off zero as soon as training moves their bias, and from there no layer
    // downstream can recover where a sequence ended by looking at it.
    decoder_embedding->set_export_valid_lengths(true);
    add_layer(std::move(decoder_embedding), {decoder_tokenizer_index});
    Index current_decoder_index = get_layers_number() - 1;

    add_layer(make_unique<Tokenizer>(Shape{input_sequence_length}, "encoder_tokenizer"), {-2});
    const Index encoder_tokenizer_index = get_layers_number() - 1;

    auto encoder_embedding = make_unique<Embedding>(
        Shape{input_vocabulary_size, input_sequence_length},
        embedding_dimension, "encoder_embedding");
    encoder_embedding->set_scale_embedding(true);
    encoder_embedding->set_add_positional_encoding(true);
    encoder_embedding->set_export_valid_lengths(true);
    add_layer(std::move(encoder_embedding), {encoder_tokenizer_index});
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
        add_layer(std::move(decoder_self_attention), {current_decoder_index});
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

    finalize_build(*this);
}

template<typename Apply>
static void apply_and_recompile(NeuralNetwork& network, Apply apply)
{
    const auto forward_before = network.get_forward_specs(1);
    const auto backward_before = network.get_backward_specs(1);

    for (const auto& layer : network.get_layers())
        if (layer) apply(*layer);

    recompile_if_specs_changed(network, forward_before, backward_before);
}

static void set_attention_and_dense_dropout(NeuralNetwork& network, float new_dropout_rate,
                                            initializer_list<string_view> dense_prefixes)
{
    apply_and_recompile(network, [&](Layer& layer)
    {
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(&layer))
            mha->set_dropout_rate(new_dropout_rate);
        else if (starts_with_any(layer.get_label(), dense_prefixes))
            if (auto* dense = dynamic_cast<Dense*>(&layer))
                dense->set_dropout_rate(new_dropout_rate);
    });
}

void Transformer::set_dropout_rate(const float new_dropout_rate)
{
    set_attention_and_dense_dropout(*this, new_dropout_rate,
                                    {"encoder_internal_dense", "encoder_external_dense",
                                     "decoder_internal_dense", "decoder_external_dense"});
}

void Transformer::set_attention_sdpa_min_sequence_length(Index new_threshold)
{
    apply_and_recompile(*this, [&](Layer& layer)
    {
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(&layer))
            mha->set_sdpa_min_sequence_length(new_threshold);
    });
}

Index Transformer::get_input_sequence_length() const
{
    return get_layer("encoder_embedding")->get_input_shape()[0];
}

Index Transformer::get_decoder_sequence_length() const
{
    return get_layer("decoder_embedding")->get_input_shape()[0];
}

TextGenerationNetwork::TextGenerationNetwork()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
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
    : NeuralNetwork(NetworkTask::LanguageModeling)
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
    add_layer(std::move(embedding), {tokenizer_index});
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
        add_layer(std::move(self_attention), {attention_input_index});
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

    finalize_build(*this);
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

    auto word_embeddings = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length}, hidden_size, "word_embeddings");
    word_embeddings->set_learned_positional(true);
    word_embeddings->set_export_valid_lengths(true);
    net.add_layer(std::move(word_embeddings), {-1});
    const Index word_index = net.get_layers_number() - 1;

    net.add_layer(make_unique<Embedding>(
                      Shape{type_vocabulary_size + 1, sequence_length}, hidden_size, "token_type_embeddings"),
                  {-2});
    const Index type_index = net.get_layers_number() - 1;

    Index current = add_residual_and_norm(net, seq_hidden, "embeddings_layer_norm", word_index, type_index);

    for (Index i = 0; i < layers_number; ++i)
    {
        const string sfx = format("_{}", i + 1);

        net.add_layer(make_unique<MultiHeadAttention>(seq_hidden, heads_number, "attention" + sfx),
                      {current});
        const Index attention_index = net.get_layers_number() - 1;

        const Index attention_norm_index =
            add_residual_and_norm(net, seq_hidden, "attention_layer_norm" + sfx, current, attention_index);

        const Index feed_forward_index = add_feed_forward(net, seq_hidden, intermediate_size,
            "intermediate" + sfx, "feed_forward_output" + sfx, "GELU");

        current = add_residual_and_norm(net, seq_hidden, "output_layer_norm" + sfx,
                                        attention_norm_index, feed_forward_index);
    }

    return current;
}

Bert::Bert()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

Bert::Bert(Index sequence_length,
           Index vocabulary_size,
           Index hidden_size,
           Index heads_number,
           Index intermediate_size,
           Index layers_number,
           Index type_vocabulary_size)
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
    add_bert_encoder(*this, sequence_length, vocabulary_size, hidden_size, heads_number,
                     intermediate_size, layers_number, type_vocabulary_size);
    finalize_build(*this);
}

Qwen3::Qwen3()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

Qwen3::Qwen3(Index sequence_length,
             Index vocabulary_size,
             Index hidden_size,
             Index layers_number,
             Index query_heads,
             Index key_value_heads,
             Index head_dimension,
             Index intermediate_size,
             float rope_theta,
             float rms_epsilon)
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
    throw_if(sequence_length == 0 || vocabulary_size == 0 || hidden_size == 0 ||
             layers_number == 0 || query_heads == 0 || key_value_heads == 0 ||
             head_dimension == 0 || intermediate_size == 0,
             "Qwen3: all dimensions must be > 0.");
    throw_if(query_heads % key_value_heads != 0,
             "Qwen3: query_heads must be divisible by key_value_heads.");

    auto embedding = make_unique<Embedding>(Shape{vocabulary_size + 1, sequence_length}, hidden_size, "embed_tokens");
    embedding->set_scale_embedding(false);
    embedding->set_weights_follow_compute_dtype(true);
    add_layer(std::move(embedding), {-1});
    Index current = get_layers_number() - 1;

    const Shape block{sequence_length, hidden_size};

    auto add_norm = [&](const string& name, Index source)
    {
        auto norm = make_unique<Normalization3d>(block, name);
        norm->set_method(NormalizationMethod::RMS);
        norm->set_epsilon(rms_epsilon);
        add_layer(std::move(norm), {source});
        return get_layers_number() - 1;
    };

    auto add_linear = [&](const Shape& in_shape, Index out_features, const string& name, Index source)
    {
        auto dense = make_unique<Dense>(in_shape, Shape{out_features}, "Identity", false, name);
        dense->set_use_bias(false);
        add_layer(std::move(dense), {source});
        return get_layers_number() - 1;
    };

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = "_" + to_string(i);

        const Index input_norm = add_norm("input_norm" + suffix, current);
        add_layer(make_unique<GroupedQueryAttention>(block, query_heads, key_value_heads, head_dimension,
                                                     rope_theta, rms_epsilon,   true,
                                                     "attn" + suffix), {input_norm});
        const Index attention = get_layers_number() - 1;
        add_layer(make_unique<Addition>(block, "attn_add" + suffix), {current, attention});
        const Index residual = get_layers_number() - 1;

        const Index post_norm = add_norm("post_norm" + suffix, residual);

        auto gate_up = make_unique<Dense>(block, Shape{intermediate_size}, "Identity", false, "gate_up" + suffix);
        gate_up->set_use_bias(false);
        gate_up->set_gated(true);
        add_layer(std::move(gate_up), {post_norm});
        const Index ffn = get_layers_number() - 1;
        const Index down = add_linear(Shape{sequence_length, intermediate_size}, hidden_size, "down" + suffix, ffn);
        static_cast<Dense*>(layers[size_t(down)].get())->set_transposed_inference(true);
        add_layer(make_unique<Addition>(block, "ffn_add" + suffix), {residual, down});
        current = get_layers_number() - 1;
    }

    add_norm("final_norm", current);
    current = get_layers_number() - 1;

    const Index lm_head = add_linear(block, vocabulary_size + 1, "lm_head", current);
    static_cast<Dense*>(layers[size_t(lm_head)].get())->set_tied_weight_source(layers.front().get());

    compile();
    set_parameters_random();
}

void TextGenerationNetwork::set_dropout_rate(const float new_dropout_rate)
{
    set_attention_and_dense_dropout(*this, new_dropout_rate,
                                    {"internal_dense", "external_dense"});
}

void TextGenerationNetwork::set_attention_sdpa_auto(bool new_sdpa_auto)
{
    apply_and_recompile(*this, [&](Layer& layer)
    {
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(&layer))
            mha->set_sdpa_auto(new_sdpa_auto);
    });
}

Index TextGenerationNetwork::get_sequence_length() const
{
    return get_layer("embedding")->get_input_shape()[0];
}

namespace
{

template <typename Network>
auto& get_tokenizer_layer(Network& network, const string& label, const char* method)
{
    using TokenizerType = conditional_t<is_const_v<Network>, const Tokenizer, Tokenizer>;
    TokenizerType* tokenizer_layer = nullptr;

    try
    {
        tokenizer_layer =
            dynamic_cast<TokenizerType*>(network.get_layer(label).get());
    }
    catch (const exception&)
    {
    }

    throw_if(!tokenizer_layer,
             format("{}: network has no '{}' layer. Rebuild the network or "
                    "re-save the model with a tokenizer.",
                    method, label));

    return *tokenizer_layer;
}

}

Transformer::Transformer(const filesystem::path& path)
    : NeuralNetwork(path, NetworkTask::LanguageModeling)
{
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

TextGenerationNetwork::TextGenerationNetwork(const filesystem::path& path)
    : NeuralNetwork(path, NetworkTask::LanguageModeling)
{
}

void TextGenerationNetwork::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::set_tokenizer")
        .set_tokenizer(std::move(new_tokenizer));
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

void TextClassificationNetwork::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "tokenizer", "TextClassificationNetwork::set_tokenizer")
        .set_tokenizer(std::move(new_tokenizer));
}

const TokenizerOperator* TextClassificationNetwork::get_tokenizer() const
{
    return get_tokenizer_layer(*this, "tokenizer", "TextClassificationNetwork::get_tokenizer").get_tokenizer();
}

MatrixR TextClassificationNetwork::calculate_text_outputs(
    const Tensor<string, 1>& input_documents)
{
    const Tokenizer& tokenizer_layer = get_tokenizer_layer(
        *this, "tokenizer", "TextClassificationNetwork::calculate_text_outputs");
    const TokenizerOperator* tokenizer = tokenizer_layer.get_tokenizer();

    throw_if(!tokenizer || tokenizer->get_vocabulary_size() == 0,
             "TextClassificationNetwork::calculate_text_outputs: the tokenizer "
             "has no vocabulary; call set_tokenizer() first.");

    const Index sequence_length = tokenizer_layer.get_output_shape()[0];
    const Index batch_size = input_documents.size();
    MatrixR inputs = MatrixR::Zero(batch_size, sequence_length);

    for (Index i = 0; i < batch_size; ++i)
    {
        const vector<Index> ids =
            tokenizer->encode_sequence(input_documents.data()[i], sequence_length);

        for (Index j = 0; j < min(ssize(ids), sequence_length); ++j)
            inputs(i, j) = float(ids[size_t(j)]);
    }

    return calculate_outputs(inputs);
}

BertForSequenceClassification::BertForSequenceClassification()
    : NeuralNetwork(NetworkTask::TextClassification)
{
}

BertForSequenceClassification::BertForSequenceClassification(Index sequence_length,
                                                             Index vocabulary_size,
                                                             Index hidden_size,
                                                             Index heads_number,
                                                             Index intermediate_size,
                                                             Index layers_number,
                                                             Index labels_number,
                                                             Index type_vocabulary_size)
    : NeuralNetwork(NetworkTask::TextClassification)
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

    finalize_build(*this);
}

void BertForSequenceClassification::set_dropout_rate(const float new_dropout_rate)
{
    set_attention_and_dense_dropout(*this, new_dropout_rate,
                                    {"feed_forward_output", "pooler"});
}

#endif

Index load_darknet_backbone(NeuralNetwork& network,
                            const filesystem::path& weights_path,
                            Index n_backbone_convs)
{
    FILE* f = fopen(weights_path.string().c_str(), "rb");
    throw_if(!f, "load_darknet_backbone: cannot open file: " + weights_path.string());

    int32_t header[3];
    int64_t seen;
    throw_if(fread(header, sizeof(int32_t), 3, f) != 3,
             "load_darknet_backbone: failed to read header int32s.");
    throw_if(fread(&seen, sizeof(int64_t), 1, f) != 1,
             "load_darknet_backbone: failed to read header seen.");

    cout << "Darknet weights header: major=" << header[0]
         << " minor=" << header[1]
         << " revision=" << header[2]
         << " seen=" << seen << "\n";

    Index loaded = 0;
    const auto& layers = network.get_layers();
    for (size_t li = 0; li < layers.size() && loaded < n_backbone_convs; ++li)
    {
        auto* conv = dynamic_cast<Convolutional*>(layers[li].get());
        if (!conv) continue;

        conv->load_darknet_weights(f);
        ++loaded;
        cout << format("Loaded backbone conv {}/{} from {}\n", loaded, n_backbone_convs, weights_path.string());
    }

    fclose(f);
    return loaded;
}

Index load_darknet_backbone_v11(NeuralNetwork& network,
                                const filesystem::path& weights_path)
{
    FILE* f = fopen(weights_path.string().c_str(), "rb");
    throw_if(!f, "load_darknet_backbone_v11: cannot open file: " + weights_path.string());

    int32_t header[3];
    int64_t seen;
    throw_if(fread(header, sizeof(int32_t), 3, f) != 3,
             "load_darknet_backbone_v11: failed to read header.");
    throw_if(fread(&seen, sizeof(int64_t), 1, f) != 1,
             "load_darknet_backbone_v11: failed to read header seen.");

    cout << "Darknet weights header: major=" << header[0]
         << " minor=" << header[1]
         << " revision=" << header[2]
         << " seen=" << seen << "\n";

    static const pair<const char*, size_t> targets[] = {
        {"c11_stem",    0},
        {"c11_s1_down", 0},
        {"c11_s2_down", 42368},
        {"c11_s3_down", 79872},
        {"c11_s4_down", 811520},
        {"c11_s5_down", 3228672},
    };

    map<string, Convolutional*> label_to_conv;
    for (const auto& layer : network.get_layers())
    {
        auto* conv = dynamic_cast<Convolutional*>(layer.get());
        if (conv) label_to_conv[conv->get_label()] = conv;
    }

    Index loaded = 0;
    for (const auto& [label, skip_floats] : targets)
    {
        if (skip_floats > 0)
            fseek(f, long(skip_floats) * long(sizeof(float)), SEEK_CUR);

        auto it = label_to_conv.find(label);
        if (it == label_to_conv.end())
        {
            cout << "load_darknet_backbone_v11: layer \"" << label << "\" not found — skipping.\n";
            continue;
        }

        it->second->load_darknet_weights(f);
        ++loaded;
        cout << "Loaded pretrained downsampling conv \"" << label << "\" from yolov4.conv.137\n";
    }

    fclose(f);
    return loaded;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
