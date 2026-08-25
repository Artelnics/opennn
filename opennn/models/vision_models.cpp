//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I S I O N   M O D E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/models/models.h"

#include <utility>

#include "opennn/core/string_utilities.h"
#include "opennn/neural_network/layers/activation_layer.h"
#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/layers/clamping_layer.h"
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
#include "opennn/neural_network/layers/upsampling_layer.h"

namespace opennn
{

static void finalize_build(NeuralNetwork& network)
{
    network.compile();
    network.set_parameters_glorot();
}

static void bias_v8_class_logits(NeuralNetwork& network)
{
    constexpr float PRIOR_BIAS = -4.5951f;

    for (const unique_ptr<Layer>& layer : network.get_layers())
    {
        auto* const convolutional = dynamic_cast<Convolutional*>(layer.get());

        if (!convolutional || !convolutional->get_label().ends_with("_cls_out"))
            continue;

        vector<TensorView>& views = convolutional->get_parameter_views();

        if (views.empty() || views[0].empty())
            continue;

        float* const biases = views[0].as<float>();

        fill(biases, biases + convolutional->get_kernels_number(), PRIOR_BIAS);
    }
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
        return add_layer(make_unique<Convolutional>(
                             get_layer(input_index)->get_output_shape(),
                             kernel_shape, activation, stride, "Same",
                              true, name),
                         {input_index});
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
        return add_layer(std::move(conv), {input_index, skip_index});
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

    last_index = add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                                Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                                                "MaxPooling", "stem_pool"),
                           {last_index});

    for (size_t i = 0; i < blocks_per_stage.size(); ++i)
        for (Index j = 0; j < blocks_per_stage[i]; ++j)
            last_index = use_bottleneck
                ? add_bottleneck_block(last_index, i, j, initial_filters[i])
                : add_basic_block(last_index, i, j, initial_filters[i]);

    const Shape pre_pool = get_layer(last_index)->get_output_shape();
    last_index = add_layer(make_unique<Pooling>(pre_pool,
                                                Shape{pre_pool[0], pre_pool[1]},
                                                Shape{1, 1}, Shape{0, 0},
                                                "AveragePooling", "global_avg_pool"),
                           {last_index});

    last_index = add_layer(make_unique<Flatten>(get_layer(last_index)->get_output_shape()),
                           {last_index});

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

struct BackboneFeatures
{
    Index c3 = -1;
    Index c4 = -1;
    Index c5 = -1;
};

struct YoloBuilder
{
    NeuralNetwork& network;
    const char* act;
    Shape stride;
    Index classes_number;
    YoloNetwork::ClassActivation class_activation;
    Index reg_max;

    Index add_layer(unique_ptr<Layer> layer, const vector<Index>& sources) const
    {
        return network.add_layer(std::move(layer), sources);
    }

    const unique_ptr<Layer>& get_layer(Index index) const { return network.get_layer(index); }
    const vector<unique_ptr<Layer>>& get_layers() const { return network.get_layers(); }
    Index get_layers_number() const { return network.get_layers_number(); }

    BackboneFeatures add_vgg_backbone(const Shape& input_shape) const
    {
        const vector<Index> filters = {32, 64, 128, 256, 512};
        const Shape pool{2, 2};
        const Shape pool_stride{2, 2};
        const Shape no_padding{0, 0};
        Index last_index = -1;

        for (Index i = 0; i < ssize(filters); ++i)
        {
            const Shape conv_input_shape = (i == 0)
                ? input_shape
                : get_layer(last_index)->get_output_shape();

            last_index = add_layer(make_unique<Convolutional>(
                conv_input_shape,
                Shape{3, 3, conv_input_shape[2], filters[size_t(i)]},
                act, stride, "Same", true,
                format("yolo_conv_{}", i + 1)), {});

            last_index = add_layer(make_unique<Pooling>(
                get_layer(last_index)->get_output_shape(), pool, pool_stride,
                no_padding, "MaxPooling", format("yolo_pool_{}", i + 1)), {});
        }

        last_index = add_layer(make_unique<Convolutional>(
            get_layer(last_index)->get_output_shape(),
            Shape{3, 3, get_layer(last_index)->get_output_shape()[2], 1024},
            act, stride, "Same", true, "yolo_conv_6"), {});

        return {.c5 = last_index};
    }

    BackboneFeatures add_darknet_tiny_v3_backbone(const Shape& input_shape) const
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

        const Shape pool{2, 2};
        const Shape pool_stride{2, 2};
        const Shape no_padding{0, 0};
        Index c3_index = -1;
        Index last_index = -1;

        for (size_t i = 0; i < stages.size(); ++i)
        {
            const DarknetStage& stage = stages[i];
            const Shape in_shape = (i == 0)
                ? input_shape
                : get_layer(last_index)->get_output_shape();
            const Index kernel_size = stage.one_by_one ? 1 : 3;

            last_index = add_layer(make_unique<Convolutional>(
                in_shape,
                Shape{kernel_size, kernel_size, in_shape[2], stage.channels},
                act, stride, "Same", true,
                format("dntv3_conv_{}", i + 1)), {});

            if (stage.pool)
                last_index = add_layer(make_unique<Pooling>(
                    get_layer(last_index)->get_output_shape(),
                    pool, pool_stride, no_padding, "MaxPooling",
                    format("dntv3_pool_{}", i + 1)), {});

            if (i == 4)
                c3_index = get_layers_number() - 1 - (stage.pool ? 1 : 0);
        }

        return {.c3 = c3_index, .c5 = last_index};
    }

    Index add_sppf(Index input_index,
                   Index channels,
                   const string& prefix,
                   const function<Index(Index, const Shape&, const string&)>& add_block) const
    {
        const Index half = channels / 2;

        const Index projected = add_block(input_index, Shape{1, 1, channels, half}, prefix + "_in");
        const Shape shape = get_layer(projected)->get_output_shape();

        const Index pool_1 = add_layer(make_unique<Pooling>(
            shape, Shape{5, 5}, Shape{1, 1}, Shape{2, 2}, "MaxPooling", prefix + "_p1"), {projected});
        const Index pool_2 = add_layer(make_unique<Pooling>(
            shape, Shape{5, 5}, Shape{1, 1}, Shape{2, 2}, "MaxPooling", prefix + "_p2"), {pool_1});
        const Index pool_3 = add_layer(make_unique<Pooling>(
            shape, Shape{5, 5}, Shape{1, 1}, Shape{2, 2}, "MaxPooling", prefix + "_p3"), {pool_2});

        const Index concatenated = add_layer(make_unique<Concatenation>(
            shape, vector<Index>{half, half, half, half}, prefix + "_cat"),
            {projected, pool_1, pool_2, pool_3});

        return add_block(concatenated, Shape{1, 1, 4 * half, channels}, prefix + "_out");
    }

    Index add_conv(Index input_index, const Shape& kernel_shape,
                   const char* activation, const Shape& kernel_stride,
                   bool batch_norm, const string& name) const
    {
        const bool needs_own_layer =
            activation_needs_input(ActivationOperator::from_string(activation));

        const Index convolution_index = add_layer(make_unique<Convolutional>(
                                                      get_layer(input_index)->get_output_shape(),
                                                      kernel_shape, needs_own_layer ? "Identity" : activation,
                                                      kernel_stride, "Same",
                                                      batch_norm, name),
                                                  {input_index});

        if (!needs_own_layer) return convolution_index;

        return add_layer(make_unique<Activation>(get_layer(convolution_index)->get_output_shape(),
                                                 activation, name + "_act"),
                         {convolution_index});
    }

    void apply_yolo_prior_bias(Index n_classes) const
    {
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
    }

    void finish_yolo_network() const
    {
        network.compile();
        network.set_parameters_random();
        apply_yolo_prior_bias(classes_number);
    }

    Index add_det_head(Index feature_index,
                       const vector<array<float, 2>>& head_anchors,
                       const string& name) const
    {
        const Index logits_index = add_conv(feature_index,
            Shape{1, 1, get_layer(feature_index)->get_output_shape()[2],
                  3 * (5 + classes_number)},
            "Identity", stride, false, "yolo_logits_" + name);

        const Index detection_index = add_layer(make_unique<Detection>(
                                                    get_layer(logits_index)->get_output_shape(),
                                                    head_anchors, "detection_" + name),
                                                {logits_index});
        static_cast<Detection&>(*get_layers().back())
            .set_class_activation(class_activation);
        return detection_index;
    }

    Index add_residual_block(Index input_index, Index channels, Index mid,
                             const string& prefix, const char* c1_suffix,
                             const char* c2_suffix, const char* act_suffix) const
    {
        Index x = add_conv(input_index, Shape{1, 1, channels, mid},      act,        stride, true, prefix + c1_suffix);
        x       = add_conv(x,           Shape{3, 3, mid,      channels}, "Identity", stride, true, prefix + c2_suffix);
        const Index add_index = add_layer(make_unique<Addition>(get_layer(x)->get_output_shape(), prefix + "_add"), {x, input_index});
        return add_layer(make_unique<Activation>(get_layer(add_index)->get_output_shape(), act, prefix + act_suffix), {add_index});
    }

    BackboneFeatures add_darknet_tiny_backbone(const Shape& input_shape) const
    {
        static const vector<pair<Index, Index>> stages = {
            { 64, 1},
            {128, 1},
            {256, 1},
            {512, 1},
        };

        const Shape stride_2{2, 2};
        Index last_index = add_layer(make_unique<Convolutional>(
            input_shape, Shape{3, 3, input_shape[2], 32},
            act, stride_2, "Same", true, "darknet_stem"), {});

        BackboneFeatures features;
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
                    format("darknet_s{}_b{}", i + 1, j),
                    "_conv1", "_conv2", "_relu");

            if (i == 1) features.c3 = last_index;
            if (i == 2) features.c4 = last_index;
            if (i == 3) features.c5 = last_index;
        }

        return features;
    }

    BackboneFeatures add_darknet53_backbone(const Shape& input_shape, bool use_csp) const
    {
        const Shape stride_2{2, 2};

        const auto add_csp_stage = [&](Index input_index, Index in_channels,
                                       Index out_channels, Index blocks_number,
                                       const string& prefix, bool first_stage) -> Index
        {
            const Index half = out_channels / 2;
            const Index branch_channels = first_stage ? out_channels : half;

            const Index down = add_conv(input_index,
                Shape{3, 3, in_channels, out_channels}, act, stride_2, true,
                prefix + "_down");

            Index branch_2 = add_conv(down,
                Shape{1, 1, out_channels, branch_channels}, act, stride, true,
                prefix + "_s2");
            for (Index j = 0; j < blocks_number; ++j)
                branch_2 = add_residual_block(branch_2, branch_channels, half,
                    prefix + format("_b{}", j + 1), "_c1", "_c2", "_act");
            const Index transformed = add_conv(branch_2,
                Shape{1, 1, branch_channels, branch_channels}, act, stride, true,
                prefix + "_trans");

            const Index branch_1 = add_conv(down,
                Shape{1, 1, out_channels, branch_channels}, act, stride, true,
                prefix + "_s1");

            const Shape spatial = get_layer(branch_1)->get_output_shape();
            const Index concatenated = add_layer(make_unique<Concatenation>(
                spatial, vector<Index>{branch_channels, branch_channels},
                prefix + "_cat"), {transformed, branch_1});
            return add_conv(concatenated,
                Shape{1, 1, 2 * branch_channels, out_channels}, act, stride, true,
                prefix + "_merge");
        };

        static const vector<pair<Index, Index>> stages = {
            {64, 1}, {128, 2}, {256, 8}, {512, 8}, {1024, 4}
        };

        Index last_index = add_layer(make_unique<Convolutional>(
            input_shape, Shape{3, 3, input_shape[2], 32},
            act, stride, "Same", true,
            use_csp ? "csp53_stem" : "dn53_stem"), {});

        BackboneFeatures features;
        Index input_channels = 32;
        for (size_t i = 0; i < stages.size(); ++i)
        {
            const auto& [channels, blocks_number] = stages[i];
            if (use_csp)
                last_index = add_csp_stage(last_index, input_channels, channels,
                    blocks_number, format("csp53_s{}", i + 1), i == 0);
            else
            {
                last_index = add_conv(last_index,
                    Shape{3, 3, input_channels, channels}, act, stride_2, true,
                    format("dn53_down_{}", i + 1));
                for (Index j = 0; j < blocks_number; ++j)
                    last_index = add_residual_block(last_index, channels, channels / 2,
                        format("dn53_s{}_b{}", i + 1, j + 1),
                        "_c1", "_c2", "_act");
            }

            input_channels = channels;
            if (i == 2) features.c3 = last_index;
            if (i == 3) features.c4 = last_index;
            if (i == 4) features.c5 = last_index;
        }

        return features;
    }

    Index scale_csp_v11_channels(Index base, YoloNetwork::ModelSize model_size) const
    {
        const float width = model_size == YoloNetwork::ModelSize::n ? 0.25f
                          : model_size == YoloNetwork::ModelSize::s ? 0.50f
                          : model_size == YoloNetwork::ModelSize::m ? 0.75f
                          : model_size == YoloNetwork::ModelSize::x ? 1.25f
                          :                                              1.00f;

        return max(Index(8), Index(round(float(base) * width / 8.f) * 8));
    }

    Index scale_csp_v11_depth(Index base, YoloNetwork::ModelSize model_size) const
    {
        const float depth = model_size == YoloNetwork::ModelSize::n ? 0.33f
                          : model_size == YoloNetwork::ModelSize::s ? 0.33f
                          : model_size == YoloNetwork::ModelSize::m ? 0.67f
                          :                                              1.00f;

        return max(Index(1), Index(round(float(base) * depth)));
    }

    Index add_csp_v11_block(Index input_index,
                            const Shape& kernel,
                            const Shape& kernel_stride,
                            const string& name) const
    {
        const Index convolution = add_conv(input_index, kernel, "Identity",
                                           kernel_stride, true, name);

        return add_layer(make_unique<Activation>(get_layer(convolution)->get_output_shape(),
                                                 act, name + "_act"),
                         {convolution});
    }

    Index add_c2f(Index input_index,
                  Index input_channels,
                  Index output_channels,
                  Index blocks_number,
                  bool shortcut,
                  const string& prefix) const
    {
        const Index half = output_channels / 2;
        const Index branch_1 = add_csp_v11_block(input_index,
            Shape{1, 1, input_channels, half}, stride, prefix + "_cv1a");
        const Index branch_2 = add_csp_v11_block(input_index,
            Shape{1, 1, input_channels, half}, stride, prefix + "_cv1b");

        vector<Index> concatenation_inputs = {branch_1, branch_2};
        Index block_input = branch_2;

        for (Index i = 0; i < blocks_number; ++i)
        {
            const string block_prefix = prefix + format("_b{}", i + 1);
            Index block = add_csp_v11_block(block_input,
                Shape{3, 3, half, half}, stride, block_prefix + "_cv1");
            block = add_csp_v11_block(block,
                Shape{3, 3, half, half}, stride, block_prefix + "_cv2");

            if (shortcut)
                block = add_layer(make_unique<Addition>(get_layer(block)->get_output_shape(),
                                                        block_prefix + "_add"),
                                  {block, block_input});

            concatenation_inputs.push_back(block);
            block_input = block;
        }

        const Shape spatial_shape = get_layer(branch_1)->get_output_shape();
        const Index concatenation = add_layer(make_unique<Concatenation>(
            spatial_shape, vector<Index>(2 + blocks_number, half), prefix + "_cat"),
            concatenation_inputs);

        return add_csp_v11_block(concatenation,
            Shape{1, 1, (2 + blocks_number) * half, output_channels},
            stride, prefix + "_cv2");
    }

    BackboneFeatures add_csp_darknet53_v11_backbone(
        const Shape& input_shape,
        YoloNetwork::ModelSize model_size) const
    {
        const Index c1 = scale_csp_v11_channels(64, model_size);
        const Index c2 = scale_csp_v11_channels(128, model_size);
        const Index c3 = scale_csp_v11_channels(256, model_size);
        const Index c4 = scale_csp_v11_channels(512, model_size);
        const Index c5 = scale_csp_v11_channels(1024, model_size);

        const Index d1 = scale_csp_v11_depth(3, model_size);
        const Index d2 = scale_csp_v11_depth(6, model_size);
        const Index d3 = scale_csp_v11_depth(6, model_size);
        const Index d4 = scale_csp_v11_depth(3, model_size);
        const Shape stride_2{2, 2};

        Index last_index = add_layer(make_unique<Convolutional>(
            input_shape, Shape{3, 3, input_shape[2], c1},
            "Identity", stride_2, "Same", true, "c8_stem"), {});
        last_index = add_layer(make_unique<Activation>(
            get_layer(last_index)->get_output_shape(), act, "c8_stem_act"),
            {last_index});

        last_index = add_csp_v11_block(last_index,
            Shape{3, 3, c1, c2}, stride_2, "c8_s1_down");
        last_index = add_c2f(last_index, c2, c2, d1, true, "c8_s1");

        last_index = add_csp_v11_block(last_index,
            Shape{3, 3, c2, c3}, stride_2, "c8_s2_down");
        last_index = add_c2f(last_index, c3, c3, d2, true, "c8_s2");
        const Index c3_index = last_index;

        last_index = add_csp_v11_block(last_index,
            Shape{3, 3, c3, c4}, stride_2, "c8_s3_down");
        last_index = add_c2f(last_index, c4, c4, d3, true, "c8_s3");
        const Index c4_index = last_index;

        last_index = add_csp_v11_block(last_index,
            Shape{3, 3, c4, c5}, stride_2, "c8_s4_down");
        last_index = add_c2f(last_index, c5, c5, d4, true, "c8_s4");
        last_index = add_sppf(last_index, c5, "c8_sppf",
            [&](Index next_input, const Shape& kernel, const string& name)
            {
                return add_csp_v11_block(next_input, kernel, stride, name);
            });

        return {.c3 = c3_index, .c4 = c4_index, .c5 = last_index};
    }

    Index add_yolo_neck(Index idx, Index in_ch,
                        Index ch_small, Index ch_large, const string& pfx) const
    {
        Index x = add_conv(idx, Shape{1, 1, in_ch,     ch_small}, act, stride, true, pfx+"_c1");
        x       = add_conv(x,   Shape{3, 3, ch_small, ch_large},  act, stride, true, pfx+"_c2");
        x       = add_conv(x,   Shape{1, 1, ch_large, ch_small},  act, stride, true, pfx+"_c3");
        x       = add_conv(x,   Shape{3, 3, ch_small, ch_large},  act, stride, true, pfx+"_c4");
        x       = add_conv(x,   Shape{1, 1, ch_large, ch_small},  act, stride, true, pfx+"_c5");
        return x;
    }

    Index add_top_down(Index lateral_index, Index c_index,
                       const string& upper, const string& lower) const
    {
        const Index up_index = add_layer(make_unique<Upsampling>(get_layer(lateral_index)->get_output_shape(),
                                                                 2, "fpn_" + upper + "_upsampling"),
                                         {lateral_index});

        return add_layer(make_unique<Concatenation>(get_layer(c_index)->get_output_shape(),
                             vector<Index>{get_layer(up_index)->get_output_shape()[2],
                                           get_layer(c_index)->get_output_shape()[2]},
                             "fpn_" + lower + "_concatenation"),
                         {up_index, c_index});
    }

    void add_v8_detection_head(Index feature_index,
                               const string& name,
                               Index head_channels,
                               Index box_channels,
                               const function<Index(Index, const Shape&, const string&)>& add_block) const
    {
        const Index input_channels = get_layer(feature_index)->get_output_shape()[2];

        Index box = add_block(feature_index, Shape{3, 3, input_channels, head_channels}, name + "_box_c1");
        box       = add_block(box, Shape{3, 3, head_channels, head_channels}, name + "_box_c2");
        box       = add_conv(box, Shape{1, 1, head_channels, box_channels},
                             "Identity", stride, false, name + "_box_out");

        Index classes = add_block(feature_index, Shape{3, 3, input_channels, head_channels}, name + "_cls_c1");
        classes       = add_block(classes, Shape{3, 3, head_channels, head_channels}, name + "_cls_c2");
        classes       = add_conv(classes, Shape{1, 1, head_channels, classes_number},
                                 "Identity", stride, false, name + "_cls_out");

        const Shape spatial = get_layer(box)->get_output_shape();

        const Index concatenated = add_layer(make_unique<Concatenation>(
            spatial, vector<Index>{box_channels, classes_number}, name + "_cat"), {box, classes});

        add_layer(make_unique<DetectionV8>(get_layer(concatenated)->get_output_shape(),
                                           reg_max, name + "_det"), {concatenated});
    }
};

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
        throw_if(backbone == Backbone::DarknetTinyV3 && ssize(anchors) != 6,
                 "YoloNetwork: DarknetTinyV3 with HeadStyle::FPN is 2-head and requires exactly 6 anchors.");
    }
    if (head_style == HeadStyle::PANet)
    {
        throw_if(backbone != Backbone::Darknet53 && backbone != Backbone::CSPDarknet53,
                 "YoloNetwork: HeadStyle::PANet requires Backbone::Darknet53 or CSPDarknet53.");
        throw_if(ssize(anchors) != 9,
                 "YoloNetwork: HeadStyle::PANet requires exactly 9 anchors.");
    }

    const bool is_darknet53_family =
        backbone == Backbone::Darknet53 || backbone == Backbone::CSPDarknet53;

    throw_if(head_style == HeadStyle::FPNv8
             && !is_darknet53_family && backbone != Backbone::CSPDarknet53v11,
             "YoloNetwork: HeadStyle::FPNv8 requires Darknet53, CSPDarknet53 or CSPDarknet53v11.");

    throw_if(backbone == Backbone::CSPDarknet53v11 && head_style != HeadStyle::FPNv8,
             "YoloNetwork: CSPDarknet53v11 backbone only supports FPNv8 head style.");

    throw_if(reg_max > 1 && head_style != HeadStyle::FPNv8,
             "YoloNetwork: reg_max applies to HeadStyle::FPNv8 only; the anchor heads ignore it.");

    throw_if(use_sppf && !(is_darknet53_family
                           && (head_style == HeadStyle::FPN || head_style == HeadStyle::PANet)),
             "YoloNetwork: use_sppf applies to Darknet53/CSPDarknet53 with FPN or PANet only.");

    throw_if(model_size != ModelSize::l && backbone != Backbone::CSPDarknet53v11,
             "YoloNetwork: model_size applies to the CSPDarknet53v11 backbone only.");

    const char* act = (body_activation == BodyActivation::LeakyReLU) ? "LeakyReLU"
                    : (body_activation == BodyActivation::SiLU)      ? "SiLU"
                    :                                                   "ReLU";

    const Shape stride{1, 1};

    const YoloBuilder builder{*this, act, stride, classes_number, class_activation, reg_max};
    const Shape stride_2{2, 2};

    if (backbone == Backbone::Vgg)
        builder.add_vgg_backbone(input_shape);
    else if (backbone == Backbone::DarknetTinyV3)
    {
        const BackboneFeatures features = builder.add_darknet_tiny_v3_backbone(input_shape);
        const Index c3_index = features.c3;
        const Index last_index = features.c5;

        if (head_style == HeadStyle::FPN)
        {
            const vector<array<float, 2>> anchors_sorted = sort_anchors_by_area(anchors);

            const vector<array<float, 2>> anchors_small(anchors_sorted.begin(),     anchors_sorted.begin() + 3);
            const vector<array<float, 2>> anchors_large(anchors_sorted.begin() + 3, anchors_sorted.end());

            const Index p5_conv = builder.add_conv(last_index,
                Shape{3, 3, get_layer(last_index)->get_output_shape()[2], 512},
                act, stride, true, "fpn_p5_conv");
            builder.add_det_head(p5_conv, anchors_large, "large");

            const Index p5_lateral = builder.add_conv(last_index,
                Shape{1, 1, get_layer(last_index)->get_output_shape()[2], 128},
                act, stride, true, "fpn_p5_lateral");

            const Index p4_concat = builder.add_top_down(p5_lateral, c3_index, "p5", "p4");

            const Index p4_conv = builder.add_conv(p4_concat,
                Shape{3, 3, get_layer(p4_concat)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p4_conv");
            builder.add_det_head(p4_conv, anchors_small, "small");

            builder.finish_yolo_network();
            return;
        }
    }
    else if (backbone == Backbone::CSPDarknet53v11)
    {
        const BackboneFeatures features =
            builder.add_csp_darknet53_v11_backbone(input_shape, model_size);
        const Index p3_idx = features.c3;
        const Index p4_idx = features.c4;
        const Index p5_idx = features.c5;

        const Index c3 = builder.scale_csp_v11_channels(256, model_size);
        const Index c4 = builder.scale_csp_v11_channels(512, model_size);
        const Index c5 = builder.scale_csp_v11_channels(1024, model_size);

        if (head_style == HeadStyle::FPNv8)
        {
            const Index n12_ch = builder.scale_csp_v11_channels(512, model_size);
            const Index n15_ch = builder.scale_csp_v11_channels(256, model_size);
            const Index n18_ch = builder.scale_csp_v11_channels(512, model_size);
            const Index n21_ch = builder.scale_csp_v11_channels(1024, model_size);
            const Index nd_n = builder.scale_csp_v11_depth(3, model_size);

            add_layer(make_unique<Upsampling>(get_layer(p5_idx)->get_output_shape(), 2, "c8_fpn_p5_upsampling"), {p5_idx});
            add_layer(make_unique<Concatenation>(get_layer(p4_idx)->get_output_shape(),
                                                 vector<Index>{c5,c4}, "c8_fpn_p4_cat"),
                      {get_layers_number()-1, p4_idx});
            const Index c8_n12 = builder.add_c2f(get_layers_number()-1, c5+c4, n12_ch, nd_n, false, "c8_n12");

            add_layer(make_unique<Upsampling>(get_layer(c8_n12)->get_output_shape(), 2, "c8_fpn_p4_upsampling"), {c8_n12});
            add_layer(make_unique<Concatenation>(get_layer(p3_idx)->get_output_shape(),
                                                 vector<Index>{n12_ch,c3}, "c8_fpn_p3_cat"),
                      {get_layers_number()-1, p3_idx});
            const Index c8_n15 = builder.add_c2f(get_layers_number()-1, n12_ch+c3, n15_ch, nd_n, false, "c8_n15");

            const Index n15_down = builder.add_csp_v11_block(c8_n15, Shape{3,3,n15_ch,n15_ch}, stride_2, "c8_pan_n4_down");
            add_layer(make_unique<Concatenation>(get_layer(c8_n12)->get_output_shape(),
                                                 vector<Index>{n15_ch,n12_ch}, "c8_pan_n4_cat"),
                      {n15_down, c8_n12});
            const Index c8_n18 = builder.add_c2f(get_layers_number()-1, n15_ch+n12_ch, n18_ch, nd_n, false, "c8_n18");

            const Index n18_down = builder.add_csp_v11_block(c8_n18, Shape{3,3,n18_ch,n18_ch}, stride_2, "c8_pan_n5_down");
            add_layer(make_unique<Concatenation>(get_layer(p5_idx)->get_output_shape(),
                                                 vector<Index>{n18_ch,c5}, "c8_pan_n5_cat"),
                      {n18_down, p5_idx});
            const Index c8_n21 = builder.add_c2f(get_layers_number()-1, n18_ch+c5, n21_ch, nd_n, false, "c8_n21");

            constexpr Index head_ch = 64;
            const Index box_ch = 4 * max(reg_max, Index(1));

            const auto cba_block = [&](Index in, const Shape& kernel, const string& label)
                                   { return builder.add_csp_v11_block(in, kernel, stride, label); };

            builder.add_v8_detection_head(c8_n15, "c8_small",  head_ch, box_ch, cba_block);
            builder.add_v8_detection_head(c8_n18, "c8_medium", head_ch, box_ch, cba_block);
            builder.add_v8_detection_head(c8_n21, "c8_large",  head_ch, box_ch, cba_block);

            compile();
            set_parameters_random();
            bias_v8_class_logits(*this);
            return;
        }

    }
    else if (backbone == Backbone::Darknet53 || backbone == Backbone::CSPDarknet53)
    {
        const bool use_csp = (backbone == Backbone::CSPDarknet53);
        const BackboneFeatures features = builder.add_darknet53_backbone(input_shape, use_csp);
        const Index c3_index = features.c3;
        const Index c4_index = features.c4;
        const Index c5_index = features.c5;

        auto build_fpn_trunk = [&](Index entry, const string& pfx) -> array<Index, 3> {
            const Index p5n = builder.add_yolo_neck(entry, 1024, 512, 1024, pfx + "neck_p5");

            const Index p5l = builder.add_conv(p5n, Shape{1, 1, 512, 256}, act, stride, true, pfx + "neck_p5_lat");
            const Index p5u = add_layer(make_unique<Upsampling>(get_layer(p5l)->get_output_shape(), 2, pfx + "fpn_p5_upsampling"), {p5l});

            const Index p4c = add_layer(make_unique<Concatenation>(get_layer(c4_index)->get_output_shape(),
                                                                   vector<Index>{256, 512}, pfx + "fpn_p4_cat"),
                                        {p5u, c4_index});
            const Index p4n = builder.add_yolo_neck(p4c, 768, 256, 512, pfx + "neck_p4");

            const Index p4l = builder.add_conv(p4n, Shape{1, 1, 256, 128}, act, stride, true, pfx + "neck_p4_lat");
            const Index p4u = add_layer(make_unique<Upsampling>(get_layer(p4l)->get_output_shape(), 2, pfx + "fpn_p4_upsampling"), {p4l});

            const Index p3c = add_layer(make_unique<Concatenation>(get_layer(c3_index)->get_output_shape(),
                                                                   vector<Index>{128, 256}, pfx + "fpn_p3_cat"),
                                        {p4u, c3_index});
            return {p5n, p4n, builder.add_yolo_neck(p3c, 384, 128, 256, pfx + "neck_p3")};
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
                fpn_entry = builder.add_sppf(c5_index, get_layer(c5_index)->get_output_shape()[2], "sppf",
                                     [&](Index in, const Shape& kernel, const string& name)
                                     { return builder.add_conv(in, kernel, act, stride, true, name); });

            const auto [p5n, p4n, p3n] = build_fpn_trunk(fpn_entry, "");

            if (head_style == HeadStyle::FPN)
            {

                const Index p5d = builder.add_conv(p5n, Shape{3, 3, 512, 1024}, act, stride, true, "neck_p5_pre");
                builder.add_det_head(p5d, anchors_large, "large");
                const Index p4d = builder.add_conv(p4n, Shape{3, 3, 256, 512}, act, stride, true, "neck_p4_pre");
                builder.add_det_head(p4d, anchors_medium, "medium");
                const Index p3d = builder.add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "neck_p3_pre");
                builder.add_det_head(p3d, anchors_small, "small");
            }
            else
            {

                const Index p3d = builder.add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "neck_p3_pre");
                builder.add_det_head(p3d, anchors_small, "small");

                auto add_pan_block = [&](Index idx, Index in_ch, Index ch_s, Index ch_l, const string& pfx) -> Index {
                    Index x = builder.add_conv(idx, Shape{1, 1, in_ch, ch_s}, act, stride, true, pfx+"_c1");
                    x       = builder.add_conv(x,   Shape{3, 3, ch_s,  ch_l}, act, stride, true, pfx+"_c2");
                    x       = builder.add_conv(x,   Shape{1, 1, ch_l,  ch_s}, act, stride, true, pfx+"_c3");
                    return x;
                };

                const Index n3_down = builder.add_conv(p3n, Shape{3, 3, 128, 256}, act, stride_2, true, "pan_n3_down");
                const Index n4c = add_layer(make_unique<Concatenation>(get_layer(p4n)->get_output_shape(),
                                                                       vector<Index>{256, 256}, "pan_n4_cat"),
                                            {n3_down, p4n});
                const Index n4n = add_pan_block(n4c, 512, 256, 512, "pan_n4");
                const Index n4d = builder.add_conv(n4n, Shape{3, 3, 256, 512}, act, stride, true, "pan_n4_pre");
                builder.add_det_head(n4d, anchors_medium, "medium");

                const Index n4_down = builder.add_conv(n4n, Shape{3, 3, 256, 512}, act, stride_2, true, "pan_n4_down");
                const Index n5c = add_layer(make_unique<Concatenation>(get_layer(p5n)->get_output_shape(),
                                                                       vector<Index>{512, 512}, "pan_n5_cat"),
                                            {n4_down, p5n});
                const Index n5n = add_pan_block(n5c, 1024, 512, 1024, "pan_n5");
                const Index n5d = builder.add_conv(n5n, Shape{3, 3, 512, 1024}, act, stride, true, "pan_n5_pre");
                builder.add_det_head(n5d, anchors_large, "large");
            }

            builder.finish_yolo_network();
            return;
        }

        if (head_style == HeadStyle::FPNv8)
        {

            constexpr Index head_ch = 64;

            const Index box_ch = 4 * max(reg_max, Index(1));

            const auto conv_block = [&](Index in, const Shape& kernel, const string& label)
                                    { return builder.add_conv(in, kernel, act, stride, true, label); };

            const auto [p5n, p4n, p3n] = build_fpn_trunk(c5_index, "v8_");

            const Index p5d = builder.add_conv(p5n, Shape{3, 3, 512, 1024}, act, stride, true, "v8_neck_p5_pre");
            builder.add_v8_detection_head(p5d, "v8_large", head_ch, box_ch, conv_block);
            const Index p4d = builder.add_conv(p4n, Shape{3, 3, 256, 512}, act, stride, true, "v8_neck_p4_pre");
            builder.add_v8_detection_head(p4d, "v8_medium", head_ch, box_ch, conv_block);
            const Index p3d = builder.add_conv(p3n, Shape{3, 3, 128, 256}, act, stride, true, "v8_neck_p3_pre");
            builder.add_v8_detection_head(p3d, "v8_small", head_ch, box_ch, conv_block);

            compile();
            set_parameters_random();
            bias_v8_class_logits(*this);
            return;
        }
    }
    else
    {
        const BackboneFeatures features = builder.add_darknet_tiny_backbone(input_shape);
        const Index c3_index = features.c3;
        const Index c4_index = features.c4;
        const Index c5_index = features.c5;

        if (head_style == HeadStyle::FPN)
        {
            throw_if(ssize(anchors) != 9,
                     "YoloNetwork: DarknetTiny FPN (3-head) requires exactly 9 anchors.");
            const vector<array<float, 2>> anchors_sorted = sort_anchors_by_area(anchors);

            const vector<array<float, 2>> anchors_small (anchors_sorted.begin(),     anchors_sorted.begin() + 3);
            const vector<array<float, 2>> anchors_medium(anchors_sorted.begin() + 3, anchors_sorted.begin() + 6);
            const vector<array<float, 2>> anchors_large (anchors_sorted.begin() + 6, anchors_sorted.end());

            const Index p5_lateral = builder.add_conv(c5_index,
                Shape{1, 1, get_layer(c5_index)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p5_lateral");
            builder.add_det_head(p5_lateral, anchors_large, "large");

            const Index p4_concatenation = builder.add_top_down(p5_lateral, c4_index, "p5", "p4");

            const Index p4_lateral = builder.add_conv(p4_concatenation,
                Shape{1, 1, get_layer(p4_concatenation)->get_output_shape()[2], 256},
                act, stride, true, "fpn_p4_lateral");
            builder.add_det_head(p4_lateral, anchors_medium, "medium");

            const Index p3_concatenation = builder.add_top_down(p4_lateral, c3_index, "p4", "p3");

            const Index p3_lateral = builder.add_conv(p3_concatenation,
                Shape{1, 1, get_layer(p3_concatenation)->get_output_shape()[2], 128},
                act, stride, true, "fpn_p3_lateral");
            builder.add_det_head(p3_lateral, anchors_small, "small");

            builder.finish_yolo_network();
            return;
        }
    }

    const Index detection_channels = ssize(anchors) * (5 + classes_number);

    add_layer(make_unique<Convolutional>(get_output_shape(),
                                         Shape{1, 1, get_output_shape()[2], detection_channels},
                                         "Identity", stride, "Same", false,
                                         "yolo_logits"));

    add_layer(make_unique<Detection>(get_output_shape(), anchors, "detection_layer"));
    static_cast<Detection&>(*get_layers().back())
        .set_class_activation(class_activation);

    add_layer(make_unique<NonMaxSuppression>(get_output_shape(),
                                             ssize(anchors),
                                             0.5f,
                                             0.4f,
                                             "non_max_suppression_layer"));

    builder.finish_yolo_network();
}

#endif

namespace
{

using DarknetFile = unique_ptr<FILE, int (*)(FILE*)>;

DarknetFile open_darknet_weights(const filesystem::path& weights_path, const char* who)
{
    DarknetFile file(fopen(weights_path.string().c_str(), "rb"), &fclose);
    throw_if(!file, "{}: cannot open file: {}", who, weights_path.string());

    int32_t header[3];
    int64_t seen;
    throw_if(fread(header, sizeof(int32_t), 3, file.get()) != 3,
             "{}: failed to read header.", who);
    throw_if(fread(&seen, sizeof(int64_t), 1, file.get()) != 1,
             "{}: failed to read header seen.", who);

    cout << "Darknet weights header: major=" << header[0]
         << " minor=" << header[1]
         << " revision=" << header[2]
         << " seen=" << seen << "\n";

    return file;
}

}

Index load_darknet_backbone(NeuralNetwork& network,
                            const filesystem::path& weights_path,
                            Index n_backbone_convs)
{
    const DarknetFile file = open_darknet_weights(weights_path, "load_darknet_backbone");
    FILE* const f = file.get();

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

    return loaded;
}

Index load_darknet_backbone_v11(NeuralNetwork& network,
                                const filesystem::path& weights_path)
{
    const DarknetFile file = open_darknet_weights(weights_path, "load_darknet_backbone_v11");
    FILE* const f = file.get();

    static const pair<const char*, size_t> targets[] = {
        {"c8_stem",    0},
        {"c8_s1_down", 0},
        {"c8_s2_down", 42368},
        {"c8_s3_down", 79872},
        {"c8_s4_down", 811520},
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
        throw_if(it == label_to_conv.end(),
                 "load_darknet_backbone_v11: layer {} is not in the network; "
                 "the backbone does not match this loader.", label);

        it->second->load_darknet_weights(f);
        ++loaded;
        cout << "Loaded pretrained downsampling conv \"" << label << "\" from yolov4.conv.137\n";
    }

    return loaded;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
