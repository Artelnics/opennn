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


// Every v8 detection head starts with its classification logits biased low, so
// the first epochs are not spent unlearning a prior that says every cell holds
// an object. Both FPNv8 branches -- CSPDarknet53v11 and Darknet53/CSPDarknet53
// -- built this loop for themselves, character for character.
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
                                  const char* clamping_method)
{
    network.add_layer(make_unique<Dense>(network.get_output_shape(),
                                         output_shape,
                                         "Identity",
                                         false,
                                         output_label));

    network.add_layer(make_unique<Unscaling>(output_shape));

    auto clamping = make_unique<Clamping>(output_shape);
    if (clamping_method) clamping->set_clamping_method(clamping_method);
    network.add_layer(std::move(clamping));
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
                                             const Shape& output_shape,
                                             const string& hidden_activation)
    : NeuralNetwork(NetworkTask::Classification)
{
    add_layer(make_unique<Scaling>(input_shape));

    add_dense_stack(*this, complexity_dimensions, hidden_activation);

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

    add_regression_output(*this, output_shape, "forecasting_layer", "NoClamping");

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

    add_regression_output(*this, output_shape, "forecasting_layer", "NoClamping");

    finalize_build(*this);
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape)
    : NeuralNetwork(NetworkTask::AutoAssociation)
{
    // Shape::operator[] is unchecked; its four-argument sibling below already
    // validates the same two inputs.
    throw_if(input_shape.empty(),
             "AutoAssociationNetwork: input shape cannot be empty.");
    throw_if(complexity_dimensions.empty(),
             "AutoAssociationNetwork: complexity dimensions cannot be empty.");

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

// The YoloNetwork constructor's shared builders. They were eight lambdas
// declared before the backbone branches, capturing the same six things: the
// network being built and the five configuration values below. As a struct they
// live outside the constructor, which is what lets the constructor shrink and
// the per-backbone branches eventually move out too.
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
        // SiLU and the GELUs need their pre-activation input, which a fused
        // convolution does not keep, so they travel as a standalone Activation
        // layer - what the v8 path spells out as add_cba. The convolution used
        // to accept them and quietly substitute Identity, which built a
        // linear neck and said nothing.
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

    // Prior bias for anchor-based fused detection heads ("yolo_logits*"):
    // Each fused conv bias has layout [tx,ty,tw,th,obj,c0..cN] × bpc.
    // Set objectness (pos 4) and class (pos 5..4+C) biases to -4.5951 per anchor.
    // Box coord biases (pos 0..3 per anchor) stay at 0.
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

    // Every backbone below ends the same way, and the prior bias has to be
    // applied after the parameters are randomised or it is overwritten.
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
        static_cast<Detection&>(*get_layers().back()).set_class_activation(
            class_activation == YoloNetwork::ClassActivation::Sigmoid
            ? Detection::ClassActivation::Sigmoid
            : Detection::ClassActivation::Softmax);
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

    // The v8 detection head: two 3x3 blocks into a 1x1 projection, once for the
    // box branch and once for the class branch, concatenated and handed to
    // DetectionV8. Like SPPF, the two FPNv8 branches build the same graph and
    // differ only in how they wrap a 3x3 block, so that is the parameter. The
    // two 1x1 output projections are identical in both and stay here.
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
        // DarknetTinyV3 is the only 2-head FPN backbone: it slices
        // anchors[3..end) for its large head, so nine anchors put six of them
        // on a conv sized for three and the failure surfaced much later as an
        // unrelated divisibility message from DetectionOperator::set.
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

    // The remaining parameters are read by some branches and not others, and a
    // branch that does not read one used to build a network that silently
    // ignored it -- reg_max on an anchor head, model_size on anything but v11,
    // use_sppf on a backbone with no SPPF to insert. Rejected here instead, so
    // the caller learns at construction rather than from a model that trains
    // to the wrong shape.
    const bool is_darknet53_family =
        backbone == Backbone::Darknet53 || backbone == Backbone::CSPDarknet53;

    throw_if(head_style == HeadStyle::FPNv8
             && !is_darknet53_family && backbone != Backbone::CSPDarknet53v11,
             "YoloNetwork: HeadStyle::FPNv8 requires Darknet53, CSPDarknet53 or CSPDarknet53v11.");

    // Moved up from the end of the CSPDarknet53v11 branch, which threw only
    // after building the whole backbone.
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

    // Everything the shared builders need; they live outside the constructor.
    const YoloBuilder builder{*this, act, stride, classes_number, class_activation, reg_max};
    const Shape stride_2{2, 2};
    const Shape pool{2, 2};
    const Shape pool_stride{2, 2};
    const Shape no_padding{0, 0};

    // Spatial pyramid pooling - fast: a 1x1 down-projection, three chained
    // 5x5 max-pools at stride 1, the four streams concatenated, and a 1x1 back
    // up. Both users build the same graph but wrap their convolutions
    // differently -- the v11 branch splits the activation into its own layer --
    // so the block builder is a parameter rather than baked in.
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

            last_index = add_layer(make_unique<Convolutional>(in_shape,
                                                              Shape{ksize, ksize, in_ch, out_ch},
                                                              act, stride, "Same", true,
                                                              format("dntv3_conv_{}", i + 1)));

            if (stage.pool)
            {
                last_index = add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                                            pool, pool_stride, no_padding,
                                                            "MaxPooling",
                                                            format("dntv3_pool_{}", i + 1)));
            }

            if (i == 4) c3_index = get_layers_number() - 1 - (stage.pool ? 1 : 0);

        }

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
        // Official YOLOv8-compatible backbone: C2f blocks, SiLU, SPPF, PANet C2f neck.
        // Layer prefix "c8_" distinguishes from the old C3k2 ("c11_") implementation.
        //
        // SiLU is not supported as an inline activation in Convolutional (it needs the
        // pre-activation input for its derivative). Every Conv block is therefore built
        // as Conv(Identity, BN) + Activation(act) — matching the official YOLOv8 Conv
        // module: Conv2d → BatchNorm2d → SiLU.

        auto scale_ch = [&](Index base) -> Index {
            const float w = model_size == ModelSize::n ? 0.25f
                          : model_size == ModelSize::s ? 0.50f
                          : model_size == ModelSize::m ? 0.75f
                          : model_size == ModelSize::x ? 1.25f
                          :                              1.00f;
            return max(Index(8), Index(round(float(base) * w / 8.f) * 8));
        };
        auto scale_d = [&](Index base) -> Index {
            const float d = model_size == ModelSize::n ? 0.33f
                          : model_size == ModelSize::s ? 0.33f
                          : model_size == ModelSize::m ? 0.67f
                          :                              1.00f;
            return max(Index(1), Index(round(float(base) * d)));
        };

        // Conv+BN+act block (activation in a separate layer so SiLU works)
        auto add_cba = [&](Index in, const Shape& kernel, const Shape& kstride,
                            const string& name) -> Index {
            Index c = builder.add_conv(in, kernel, "Identity", kstride, true, name);
            return add_layer(make_unique<Activation>(get_layer(c)->get_output_shape(), act, name+"_act"), {c});
        };

        // C2f: two independent 1×1 convs (cv1a, cv1b) equivalent to official cv1→chunk(2),
        // followed by n bottleneck blocks, all outputs concatenated, merged by cv2.
        auto add_c2f = [&](Index input_idx, Index in_ch, Index out_ch, Index n,
                            bool shortcut, const string& prefix) -> Index {
            const Index half = out_ch / 2;
            const Index cv1a = add_cba(input_idx, Shape{1,1,in_ch,half}, stride, prefix+"_cv1a");
            const Index cv1b = add_cba(input_idx, Shape{1,1,in_ch,half}, stride, prefix+"_cv1b");
            vector<Index> cat_inputs = {cv1a, cv1b};
            Index bn_in = cv1b;
            for (Index j = 0; j < n; ++j) {
                const string bpfx = prefix + format("_b{}", j + 1);
                Index bx = add_cba(bn_in, Shape{3,3,half,half}, stride, bpfx+"_cv1");
                bx       = add_cba(bx,   Shape{3,3,half,half}, stride, bpfx+"_cv2");
                if (shortcut) {
                    bx = add_layer(make_unique<Addition>(get_layer(bx)->get_output_shape(), bpfx+"_add"), {bx, bn_in});
                }
                cat_inputs.push_back(bx);
                bn_in = bx;
            }
            const Shape hw = get_layer(cv1a)->get_output_shape();
            const Index cat_idx = add_layer(make_unique<Concatenation>(hw, vector<Index>(2 + n, half), prefix+"_cat"), cat_inputs);
            return add_cba(cat_idx, Shape{1,1,(2+n)*half,out_ch}, stride, prefix+"_cv2");
        };

        // Backbone channel widths (YOLOv8s at width=0.5: 32/64/128/256/512)
        const Index c1 = scale_ch(64);    // stem out
        const Index c2 = scale_ch(128);   // stage 1 out
        const Index c3 = scale_ch(256);   // stage 2 out  (P3, stride=8)
        const Index c4 = scale_ch(512);   // stage 3 out  (P4, stride=16)
        const Index c5 = scale_ch(1024);  // stage 4 out + SPPF (P5, stride=32)

        // Backbone depths (YOLOv8s at depth=0.33: 1/2/2/1)
        const Index d1 = scale_d(3);
        const Index d2 = scale_d(6);
        const Index d3 = scale_d(6);
        const Index d4 = scale_d(3);

        // Stem: Conv(k=3, s=2) + act
        Index x = add_layer(make_unique<Convolutional>(input_shape, Shape{3,3,input_shape[2],c1},
                                                       "Identity", stride_2, "Same", true, "c8_stem"));
        x = add_layer(make_unique<Activation>(get_layer(x)->get_output_shape(), act, "c8_stem_act"), {x});

        // Stage 1: Conv(s=2) + C2f
        x = add_cba(x, Shape{3,3,c1,c2}, stride_2, "c8_s1_down");
        x = add_c2f(x, c2, c2, d1, true, "c8_s1");

        // Stage 2: Conv(s=2) + C2f  — P3 feature (stride=8)
        x = add_cba(x, Shape{3,3,c2,c3}, stride_2, "c8_s2_down");
        x = add_c2f(x, c3, c3, d2, true, "c8_s2");
        const Index p3_idx = x;

        // Stage 3: Conv(s=2) + C2f  — P4 feature (stride=16)
        x = add_cba(x, Shape{3,3,c3,c4}, stride_2, "c8_s3_down");
        x = add_c2f(x, c4, c4, d3, true, "c8_s3");
        const Index p4_idx = x;

        // Stage 4: Conv(s=2) + C2f
        x = add_cba(x, Shape{3,3,c4,c5}, stride_2, "c8_s4_down");
        x = add_c2f(x, c5, c5, d4, true, "c8_s4");

        x = builder.add_sppf(x, c5, "c8_sppf",
                     [&](Index in, const Shape& kernel, const string& name)
                     { return add_cba(in, kernel, stride, name); });
        const Index p5_idx = x;  // SPPF output (stride=32)

        if (head_style == HeadStyle::FPNv8)
        {
            // Head channel widths
            const Index n12_ch = scale_ch(512);   // FPN P4 C2f output (256 for 's')
            const Index n15_ch = scale_ch(256);   // FPN P3 C2f output (128 for 's')
            const Index n18_ch = scale_ch(512);   // PAN P4 C2f output (256 for 's')
            const Index n21_ch = scale_ch(1024);  // PAN P5 C2f output (512 for 's')
            const Index nd_n   = scale_d(3);      // neck C2f repeat count (1 for 's')

            // FPN top-down path
            add_layer(make_unique<Upsampling>(get_layer(p5_idx)->get_output_shape(), 2, "c8_fpn_p5_upsampling"), {p5_idx});
            add_layer(make_unique<Concatenation>(get_layer(p4_idx)->get_output_shape(),
                                                 vector<Index>{c5,c4}, "c8_fpn_p4_cat"),
                      {get_layers_number()-1, p4_idx});
            const Index c8_n12 = add_c2f(get_layers_number()-1, c5+c4, n12_ch, nd_n, false, "c8_n12");

            add_layer(make_unique<Upsampling>(get_layer(c8_n12)->get_output_shape(), 2, "c8_fpn_p4_upsampling"), {c8_n12});
            add_layer(make_unique<Concatenation>(get_layer(p3_idx)->get_output_shape(),
                                                 vector<Index>{n12_ch,c3}, "c8_fpn_p3_cat"),
                      {get_layers_number()-1, p3_idx});
            const Index c8_n15 = add_c2f(get_layers_number()-1, n12_ch+c3, n15_ch, nd_n, false, "c8_n15");

            // PAN bottom-up path
            const Index n15_down = add_cba(c8_n15, Shape{3,3,n15_ch,n15_ch}, stride_2, "c8_pan_n4_down");
            add_layer(make_unique<Concatenation>(get_layer(c8_n12)->get_output_shape(),
                                                 vector<Index>{n15_ch,n12_ch}, "c8_pan_n4_cat"),
                      {n15_down, c8_n12});
            const Index c8_n18 = add_c2f(get_layers_number()-1, n15_ch+n12_ch, n18_ch, nd_n, false, "c8_n18");

            const Index n18_down = add_cba(c8_n18, Shape{3,3,n18_ch,n18_ch}, stride_2, "c8_pan_n5_down");
            add_layer(make_unique<Concatenation>(get_layer(p5_idx)->get_output_shape(),
                                                 vector<Index>{n18_ch,c5}, "c8_pan_n5_cat"),
                      {n18_down, p5_idx});
            const Index c8_n21 = add_c2f(get_layers_number()-1, n18_ch+c5, n21_ch, nd_n, false, "c8_n21");

            // Decoupled box+cls detection heads with DFL (box_c1/c2 and cls_c1/c2 use SiLU)
            constexpr Index head_ch = 64;
            const Index box_ch = 4 * max(reg_max, Index(1));

            const auto cba_block = [&](Index in, const Shape& kernel, const string& label)
                                   { return add_cba(in, kernel, stride, label); };

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

        auto add_csp_stage = [&](Index input_index, Index in_ch, Index out_ch,
                                  Index n_blocks, const string& prefix, bool first_stage) -> Index {
            const Index half      = out_ch / 2;
            const Index branch_ch = first_stage ? out_ch : half;

            const Index down = builder.add_conv(input_index, Shape{3, 3, in_ch, out_ch}, act, stride_2, true, prefix+"_down");

            Index branch2 = builder.add_conv(down, Shape{1, 1, out_ch, branch_ch}, act, stride, true, prefix+"_s2");
            for (Index j = 0; j < n_blocks; ++j)
                branch2 = builder.add_residual_block(branch2, branch_ch, half,
                                             prefix + format("_b{}", j+1), "_c1", "_c2", "_act");
            const Index trans = builder.add_conv(branch2, Shape{1, 1, branch_ch, branch_ch}, act, stride, true, prefix+"_trans");

            const Index branch1 = builder.add_conv(down, Shape{1, 1, out_ch, branch_ch}, act, stride, true, prefix+"_s1");

            const Shape hw = get_layer(branch1)->get_output_shape();
            const Index cat = add_layer(make_unique<Concatenation>(hw, vector<Index>{branch_ch, branch_ch}, prefix+"_cat"),
                                        {trans, branch1});
            return builder.add_conv(cat, Shape{1, 1, 2 * branch_ch, out_ch}, act, stride, true, prefix+"_merge");
        };

        const vector<pair<Index,Index>> stages = {{64,1},{128,2},{256,8},{512,8},{1024,4}};

        Index last_index = add_layer(make_unique<Convolutional>(input_shape, Shape{3, 3, input_shape[2], 32},
                                                                act, stride, "Same", true, use_csp ? "csp53_stem" : "dn53_stem"));

        Index c3_index = -1, c4_index = -1, c5_index = -1;

        Index in_ch = 32;
        for (size_t i = 0; i < stages.size(); ++i)
        {
            const auto& [ch, nblocks] = stages[i];
            if (use_csp)
                last_index = add_csp_stage(last_index, in_ch, ch, nblocks, format("csp53_s{}", i+1), i == 0);
            else
            {
                last_index = builder.add_conv(last_index, Shape{3, 3, in_ch, ch}, act, stride_2, true,
                                      format("dn53_down_{}", i+1));
                for (Index j = 0; j < nblocks; ++j)
                    last_index = builder.add_residual_block(last_index, ch, ch / 2,
                                                    format("dn53_s{}_b{}", i+1, j+1), "_c1", "_c2", "_act");
            }
            in_ch = ch;
            if (i == 2) c3_index = last_index;
            if (i == 3) c4_index = last_index;
            if (i == 4) c5_index = last_index;
        }

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
        const vector<pair<Index, Index>> stages = {
            { 64, 1},
            {128, 1},
            {256, 1},
            {512, 1},
        };

        Index last_index = add_layer(make_unique<Convolutional>(input_shape,
                                                                Shape{3, 3, input_shape[2], 32},
                                                                act, stride_2, "Same", true,
                                                                "darknet_stem"));

        Index c3_index = -1;
        Index c4_index = -1;
        Index c5_index = -1;

        for (size_t i = 0; i < stages.size(); ++i)
        {
            const auto& [channels, blocks_number] = stages[i];
            const Index input_channels = get_layer(last_index)->get_output_shape()[2];

            last_index = builder.add_conv(last_index,
                Shape{3, 3, input_channels, channels}, act,
                stride_2, true, format("darknet_down_{}", i + 1));

            for (Index j = 0; j < blocks_number; ++j)
                last_index = builder.add_residual_block(last_index, channels,
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

    builder.finish_yolo_network();
}

TextClassificationNetwork::TextClassificationNetwork(const Shape& input_shape,
                                                     const Shape& complexity_dimensions,
                                                     const Shape& output_shape,
                                                     PoolingMethod pooling_method)
    : NeuralNetwork(NetworkTask::TextClassification)
{
    // Shape::operator[] is unchecked, so a short shape read past the end of the
    // fixed dims array and built the network from whatever was there.
    throw_if(input_shape.get_rank() < 3,
             "TextClassificationNetwork: the input shape must be "
             "{{vocabulary_size, sequence_length, embedding_dimension}}, got rank {}.",
             input_shape.get_rank());
    throw_if(complexity_dimensions.get_rank() < 1,
             "TextClassificationNetwork: the complexity dimensions must name at least the "
             "number of heads.");

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
    return network.add_layer(std::move(norm), {left_index, right_index});
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
    return network.add_layer(make_unique<Dense>(Shape{seq_len, ff_dim}, Shape{emb_dim},
                                                "Identity", false, external_label));
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

    const Index decoder_tokenizer_index = add_layer(make_unique<Tokenizer>(Shape{decoder_sequence_length}, "decoder_tokenizer"), {-1});

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
    Index current_decoder_index = add_layer(std::move(decoder_embedding), {decoder_tokenizer_index});

    const Index encoder_tokenizer_index = add_layer(make_unique<Tokenizer>(Shape{input_sequence_length}, "encoder_tokenizer"), {-2});

    auto encoder_embedding = make_unique<Embedding>(
        Shape{input_vocabulary_size, input_sequence_length},
        embedding_dimension, "encoder_embedding");
    encoder_embedding->set_scale_embedding(true);
    encoder_embedding->set_add_positional_encoding(true);
    encoder_embedding->set_export_valid_lengths(true);
    Index current_encoder_index = add_layer(std::move(encoder_embedding), {encoder_tokenizer_index});

    const Shape encoder_shape{input_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        const Index attn_index = add_layer(make_unique<MultiHeadAttention>(encoder_shape, heads_number,
                                                                           "encoder_self_attention" + suffix),
                                           {current_encoder_index});

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
        const Index self_attn_index = add_layer(std::move(decoder_self_attention), {current_decoder_index});

        const Index norm1_index = add_residual_and_norm(*this, decoder_shape,
            "decoder_self_attention_normalization" + suffix,
            current_decoder_index, self_attn_index);

        const Index cross_attn_index = add_layer(make_unique<MultiHeadAttention>(decoder_shape, encoder_shape,
                                                                                 heads_number,
                                                                                 "cross_attention" + suffix),
                                                 {norm1_index, encoder_final_output_index});

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

    const Index tokenizer_index = add_layer(make_unique<Tokenizer>(Shape{sequence_length}, "tokenizer"), {-1});

    auto embedding = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length},
        embedding_dimension, "embedding");
    embedding->set_scale_embedding(scale_embedding);
    if (learned_positional)
        embedding->set_learned_positional(true);
    else
        embedding->set_add_positional_encoding(true);
    Index current_index = add_layer(std::move(embedding), {tokenizer_index});

    const Shape block_shape{sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        Index attention_input_index = current_index;

        if (pre_normalization)
        {
            attention_input_index = add_layer(make_unique<Normalization3d>(block_shape,
                                                                           "attention_normalization" + suffix),
                                              {current_index});
        }

        auto self_attention = make_unique<MultiHeadAttention>(
            block_shape, heads_number, "self_attention" + suffix);
        self_attention->set(sequence_length, sequence_length,
                            embedding_dimension, heads_number,
                            true,
                            "self_attention" + suffix);
        const Index attn_index = add_layer(std::move(self_attention), {attention_input_index});

        if (pre_normalization)
        {
            const Index residual_index = add_layer(make_unique<Addition>(block_shape, "attention_addition" + suffix),
                                                   {current_index, attn_index});

            add_layer(make_unique<Normalization3d>(block_shape,
                                                   "dense_normalization" + suffix),
                      {residual_index});

            const Index ff_index = add_feed_forward(*this, block_shape, feed_forward_dimension,
                "internal_dense" + suffix,
                "external_dense" + suffix,
                feed_forward_activation);

            current_index = add_layer(make_unique<Addition>(block_shape, "dense_addition" + suffix),
                                      {residual_index, ff_index});
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
    const Index word_index = net.add_layer(std::move(word_embeddings), {-1});

    const Index type_index = net.add_layer(make_unique<Embedding>(
                                               Shape{type_vocabulary_size + 1, sequence_length}, hidden_size, "token_type_embeddings"),
                                           {-2});

    Index current = add_residual_and_norm(net, seq_hidden, "embeddings_layer_norm", word_index, type_index);

    for (Index i = 0; i < layers_number; ++i)
    {
        const string sfx = format("_{}", i + 1);

        const Index attention_index = net.add_layer(make_unique<MultiHeadAttention>(seq_hidden, heads_number, "attention" + sfx),
                                                    {current});

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
    Index current = add_layer(std::move(embedding), {-1});

    const Shape block{sequence_length, hidden_size};

    auto add_norm = [&](const string& name, Index source)
    {
        auto norm = make_unique<Normalization3d>(block, name);
        norm->set_method(NormalizationMethod::RMS);
        norm->set_epsilon(rms_epsilon);
        return add_layer(std::move(norm), {source});
    };

    auto add_linear = [&](const Shape& in_shape, Index out_features, const string& name, Index source)
    {
        auto dense = make_unique<Dense>(in_shape, Shape{out_features}, "Identity", false, name);
        dense->set_use_bias(false);
        return add_layer(std::move(dense), {source});
    };

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = "_" + to_string(i);

        const Index input_norm = add_norm("input_norm" + suffix, current);
        const Index attention = add_layer(make_unique<GroupedQueryAttention>(block, query_heads, key_value_heads, head_dimension,
                                                                             rope_theta, rms_epsilon,   true,
                                                                             "attn" + suffix), {input_norm});
        const Index residual = add_layer(make_unique<Addition>(block, "attn_add" + suffix), {current, attention});

        const Index post_norm = add_norm("post_norm" + suffix, residual);

        auto gate_up = make_unique<Dense>(block, Shape{intermediate_size}, "Identity", false, "gate_up" + suffix);
        gate_up->set_use_bias(false);
        gate_up->set_gated(true);
        const Index ffn = add_layer(std::move(gate_up), {post_norm});
        const Index down = add_linear(Shape{sequence_length, intermediate_size}, hidden_size, "down" + suffix, ffn);
        static_cast<Dense*>(layers[size_t(down)].get())->set_transposed_inference(true);
        current = add_layer(make_unique<Addition>(block, "ffn_add" + suffix), {residual, down});
    }

    current = add_norm("final_norm", current);

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

namespace
{

template <typename Network>
// label is a const char*, not a const string&: every call site passes a literal,
// and a reference parameter bound to the resulting temporary makes GCC's
// -Wdangling-reference fire on a reference that actually points into the
// network's layer storage.
auto& get_tokenizer_layer(Network& network, const char* label, const char* method)
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

// Both loaders open the same file format, and both used to leak the handle:
// Convolutional::load_darknet_weights throws on any short read, and a bare
// fopen/fclose pair around the loop never runs its fclose on that path - on
// Windows the file then stays locked for the life of the process.
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

    // The CSPDarknet53v11 builder labels its layers c8_*; these were the labels
    // of the older C3k2 implementation it replaced, so every lookup missed, the
    // function printed six warnings and returned 0 - and the caller reported
    // that pretrained weights had been loaded.
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
        // Throwing, not skipping: loading nothing while reporting success is
        // exactly how this went unnoticed once the backbone was relabelled.
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
