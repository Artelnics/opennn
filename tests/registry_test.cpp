#include "tests/pch.h"

#include "opennn/core/json.h"
#include "opennn/registry.h"
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
#include "opennn/neural_network/layers/layer.h"
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
#include "opennn/training_strategy/optimizer.h"
#include "opennn/model_selection/inputs_selection.h"

using namespace opennn;

namespace
{

unique_ptr<Layer> make_serializable_layer(LayerType type)
{
    using enum LayerType;

    switch (type)
    {
    case Activation:
        return make_unique<opennn::Activation>(Shape{3}, "SiLU", "activation_roundtrip");

    case Addition:
        return make_unique<opennn::Addition>(Shape{2, 3}, "addition_roundtrip", 3);

    case Bounding:
    {
        auto layer = make_unique<opennn::Bounding>(Shape{3}, "bounding_roundtrip");
        layer->set_lower_bound(0, -2.0f);
        layer->set_upper_bound(1, 3.0f);
        return layer;
    }

    case Concatenation:
        return make_unique<opennn::Concatenation>(
            Shape{4, 4, 5}, vector<Index>{2, 3}, "concatenation_roundtrip");

    case Convolutional:
    {
        auto layer = make_unique<opennn::Convolutional>(
            Shape{6, 6, 2}, Shape{3, 3, 2, 2}, "LeakyReLU", Shape{2, 2},
            "Same", true, "convolutional_roundtrip");
        layer->set_residual(true);
        return layer;
    }

    case Dense:
    {
        auto layer = make_unique<opennn::Dense>(
            Shape{3}, Shape{2}, "ReLU", true, "dense_roundtrip");
        layer->set_use_bias(false);
        layer->set_transposed_inference(true);
        layer->set_dropout_rate(0.125f);
        return layer;
    }

    case Detection:
    {
        vector<opennn::array<float, 2>> anchors = {{0.5f, 0.75f}, {1.0f, 1.5f}};
        auto layer = make_unique<opennn::Detection>(
            Shape{4, 4, 14}, anchors, "detection_roundtrip");
        layer->set_class_activation(Detection::ClassActivation::Sigmoid);
        return layer;
    }

    case DetectionV8:
        return make_unique<opennn::DetectionV8>(Shape{4, 5, 14}, 3, "detection_v8_roundtrip");

    case Embedding:
    {
        auto layer = make_unique<opennn::Embedding>(Shape{11, 4}, 6, "embedding_roundtrip");
        layer->set_scale_embedding(true);
        layer->set_learned_positional(true);
        layer->set_export_valid_lengths(true);
        layer->set_weights_follow_compute_dtype(true);
        layer->set_dropout_rate(0.125f);
        return layer;
    }

    case Flatten:
    {
        auto layer = make_unique<opennn::Flatten>(Shape{2, 3, 4});
        layer->set_label("flatten_roundtrip");
        return layer;
    }

    case LongShortTermMemory:
    {
        auto layer = make_unique<opennn::LongShortTermMemory>(
            Shape{3, 2}, Shape{4}, "ReLU", "Sigmoid", "lstm_roundtrip");
        layer->set_return_sequences(true);
        return layer;
    }

    case MultiHeadAttention:
    {
        auto layer = make_unique<opennn::MultiHeadAttention>(Shape{4, 8}, 2, "mha_roundtrip");
        layer->set_zero_padded_queries(true);
        layer->set_dropout_rate(0.125f);
        layer->set_sdpa_auto(false);
        layer->set_sdpa_min_sequence_length(37);
        return layer;
    }

    case Normalization3d:
    {
        auto layer = make_unique<opennn::Normalization3d>(Shape{4, 8}, "normalization_roundtrip");
        layer->set_fuse_add(true);
        layer->set_epsilon(0.002f);
        return layer;
    }

    case GroupedQueryAttention:
        return make_unique<opennn::GroupedQueryAttention>(
            Shape{5, 8}, 4, 2, 2, 500000.0f, 0.0002f, false, "gqa_roundtrip");

    case NonMaxSuppression:
        return make_unique<opennn::NonMaxSuppression>(
            Shape{4, 4, 14}, 2, 0.35f, 0.55f, "nms_roundtrip");

    case Pooling:
        return make_unique<opennn::Pooling>(
            Shape{6, 6, 2}, Shape{2, 3}, Shape{2, 1}, Shape{1, 0},
            "AveragePooling", "pooling_roundtrip");

    case Pooling3d:
        return make_unique<opennn::Pooling3d>(
            Shape{5, 8}, PoolingMethod::FirstToken, "pooling_3d_roundtrip");

    case Recurrent:
    {
        auto layer = make_unique<opennn::Recurrent>(
            Shape{3, 2}, Shape{4}, "ReLU", "recurrent_roundtrip");
        layer->set_return_sequences(true);
        return layer;
    }

    case Scaling:
    {
        auto layer = make_unique<opennn::Scaling>(Shape{3});
        layer->set_label("scaling_roundtrip");
        layer->set_descriptives({Descriptives(-2.0f, 4.0f, 1.0f, 1.5f),
                                 Descriptives(-3.0f, 5.0f, 0.5f, 2.0f),
                                 Descriptives(0.0f, 8.0f, 3.0f, 2.5f)});
        layer->set_scalers(vector<string>{"MinimumMaximum", "MeanStandardDeviation", "None"});
        return layer;
    }

    case Tokenizer:
    {
        auto layer = make_unique<opennn::Tokenizer>(Shape{4}, "tokenizer_roundtrip");
        layer->set_vocabulary({"<unk>", "alpha", "beta", "gamma"});
        return layer;
    }

    case Unscaling:
    {
        auto layer = make_unique<opennn::Unscaling>(Shape{3}, "unscaling_roundtrip");
        layer->set_descriptives({Descriptives(-2.0f, 4.0f, 1.0f, 1.5f),
                                 Descriptives(-3.0f, 5.0f, 0.5f, 2.0f),
                                 Descriptives(0.0f, 8.0f, 3.0f, 2.5f)});
        layer->set_scalers(vector<string>{"MinimumMaximum", "MeanStandardDeviation", "None"});
        return layer;
    }

    case Upsample:
        return make_unique<opennn::Upsample>(Shape{3, 4, 2}, 3, "upsample_roundtrip");

    case C2PSA:
        return make_unique<opennn::C2PSA>(Shape{4, 4, 8}, "c2psa_roundtrip");

    case Count:
        break;
    }

    throw runtime_error("Cannot create a serialization fixture for an unknown layer type.");
}

string serialize_layer(const Layer& layer)
{
    JsonWriter writer;
    layer.to_JSON(writer);
    return writer.c_str();
}

void expect_nondefault_fields(const LayerType type, const JsonDocument& document)
{
    const Json* root = document.first_child(layer_type_to_string(type));
    ASSERT_NE(root, nullptr);
    ASSERT_TRUE(root->has("Trainable"));
    EXPECT_FALSE(read_json_bool(root, "Trainable"));

    using enum LayerType;
    switch (type)
    {
    case Dense:
        EXPECT_FALSE(read_json_bool(root, "UseBias"));
        EXPECT_TRUE(read_json_bool(root, "TransposedInference"));
        EXPECT_FLOAT_EQ(read_json_float(root, "DropoutRate"), 0.125f);
        break;

    case Embedding:
        EXPECT_TRUE(read_json_bool(root, "WeightsFollowComputeDtype"));
        EXPECT_FLOAT_EQ(read_json_float(root, "DropoutRate"), 0.125f);
        break;

    case MultiHeadAttention:
        EXPECT_TRUE(read_json_bool(root, "ZeroPaddedQueries"));
        EXPECT_FLOAT_EQ(read_json_float(root, "DropoutRate"), 0.125f);
        EXPECT_FALSE(read_json_bool(root, "SdpaAuto"));
        EXPECT_EQ(read_json_index(root, "SdpaMinSequenceLength"), 37);
        break;

    default:
        break;
    }
}

}

TEST(RegistryTest, AllComponentNamesConstruct)
{
    const vector<string> optimizer_names = {
        "AdaptiveMomentEstimation",
        "LevenbergMarquardt",
        "QuasiNewtonMethod",
        "StochasticGradientDescent"
    };

    const vector<string> inputs_selection_names = {
        "GeneticAlgorithm",
        "GrowingInputs"
    };

    const auto& layer_entries = layer_type_map().get_entries();
    ASSERT_EQ(layer_entries.size(), static_cast<size_t>(LayerType::Count));

    for (size_t i = 0; i < layer_entries.size(); ++i)
    {
        const auto& [type, name] = layer_entries[i];
        EXPECT_EQ(type, static_cast<LayerType>(i)) << name;

        const unique_ptr<Layer> layer = create_layer(name);
        ASSERT_NE(layer, nullptr) << name;
        EXPECT_EQ(string_to_layer_type(name), type) << name;
        EXPECT_EQ(layer->get_type(), type) << name;
        EXPECT_EQ(layer->get_name(), name) << name;
        EXPECT_EQ(layer_type_to_string(type), name) << name;
    }

    for (const string& name : optimizer_names)
        EXPECT_NE(create_optimizer(name), nullptr) << name;

    for (const string& name : inputs_selection_names)
        EXPECT_NE(create_inputs_selection(name), nullptr) << name;
}

TEST(RegistryTest, AliasesConstructConfiguredComponents)
{
    EXPECT_TRUE(ranges::none_of(layer_type_map().get_entries(),
                               [](const auto& entry) { return entry.second == "Concatenate"; }));

    const unique_ptr<Layer> concatenation = create_layer("Concatenate");
    EXPECT_EQ(string_to_layer_type("Concatenate"), LayerType::Concatenation);
    EXPECT_EQ(concatenation->get_type(), LayerType::Concatenation);
    EXPECT_EQ(concatenation->get_name(), "Concatenation");

    const unique_ptr<Layer> rms_normalization = create_layer("RMSNormalization3d");
    EXPECT_EQ(string_to_layer_type("RMSNormalization3d"), LayerType::Normalization3d);
    EXPECT_EQ(rms_normalization->get_type(), LayerType::Normalization3d);
    EXPECT_EQ(rms_normalization->get_name(), "Normalization3d");

    const auto* configured_normalization =
        dynamic_cast<const Normalization3d*>(rms_normalization.get());
    ASSERT_NE(configured_normalization, nullptr);
    EXPECT_EQ(configured_normalization->get_method(), NormalizationMethod::RMS);

    const string rms_json = serialize_layer(*rms_normalization);
    JsonDocument rms_document;
    rms_document.set_root(Json::parse(rms_json));

    EXPECT_EQ(rms_document.first_child("RMSNormalization3d"), nullptr);
    const Json* rms_root = rms_document.first_child("Normalization3d");
    ASSERT_NE(rms_root, nullptr);
    EXPECT_EQ(read_json_string(rms_root, "Method"), "RMS");

    const unique_ptr<Layer> restored = create_layer("Normalization3d");
    restored->from_JSON(rms_document);
    const auto* restored_normalization =
        dynamic_cast<const Normalization3d*>(restored.get());
    ASSERT_NE(restored_normalization, nullptr);
    EXPECT_EQ(restored_normalization->get_method(), NormalizationMethod::RMS);

    Json legacy_body = *rms_root;
    erase_if(legacy_body.as_object(),
             [](const auto& field) { return field.first == "Method"; });
    const JsonDocument legacy_document =
        JsonDocument::wrap("RMSNormalization3d", std::move(legacy_body));

    const unique_ptr<Layer> legacy_restored = create_layer("RMSNormalization3d");
    legacy_restored->from_JSON(legacy_document);
    const auto* legacy_normalization =
        dynamic_cast<const Normalization3d*>(legacy_restored.get());
    ASSERT_NE(legacy_normalization, nullptr);
    EXPECT_EQ(legacy_normalization->get_method(), NormalizationMethod::RMS);
}

TEST(RegistryTest, EveryLayerStateRoundTripsThroughJSONAndTheFactory)
{
    for (const auto& [type, name] : layer_type_map().get_entries())
    {
        SCOPED_TRACE(name);

        const unique_ptr<Layer> original = make_serializable_layer(type);
        ASSERT_EQ(original->get_type(), type);
        ASSERT_EQ(original->get_name(), name);
        if (original->get_is_trainable()) original->set_is_trainable(false);

        const string original_json = serialize_layer(*original);
        JsonDocument document;
        document.set_root(Json::parse(original_json));
        expect_nondefault_fields(type, document);

        const unique_ptr<Layer> restored = create_layer(name);
        restored->from_JSON(document);

        EXPECT_EQ(restored->get_type(), type);
        EXPECT_EQ(restored->get_name(), name);
        EXPECT_EQ(restored->get_label(), original->get_label());
        EXPECT_EQ(restored->get_is_trainable(), original->get_is_trainable());
        EXPECT_EQ(restored->get_input_shape(), original->get_input_shape());
        EXPECT_EQ(restored->get_output_shape(), original->get_output_shape());
        EXPECT_EQ(serialize_layer(*restored), original_json);
    }
}

TEST(RegistryTest, UnknownComponentThrows)
{
    EXPECT_THROW(layer_type_to_string(LayerType::Count), runtime_error);
    EXPECT_THROW(string_to_layer_type("Unknown"), runtime_error);
    EXPECT_THROW(create_layer("Unknown"), runtime_error);
    EXPECT_THROW(create_optimizer("Unknown"), runtime_error);
    EXPECT_THROW(create_inputs_selection("Unknown"), runtime_error);
}
