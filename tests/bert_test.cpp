#include "pch.h"
#include "numerical_derivatives.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "opennn/tensor_types.h"
#include "opennn/configuration.h"
#include "opennn/standard_networks.h"
#include "opennn/neural_network.h"
#include "opennn/forward_propagation.h"
#include "opennn/tabular_dataset.h"
#include "opennn/dense_layer.h"
#include "opennn/pooling_layer_3d.h"
#include "opennn/embedding_layer.h"
#include "opennn/multihead_attention_layer.h"
#include "opennn/loss.h"

using namespace opennn;

namespace
{
    Index embedding_roundtrip_mismatches(bool learned_positional, bool add_positional)
    {
        const Index batch = 2, seq = 6, vocab = 10, hidden = 8;

        NeuralNetwork net;
        auto embedding = make_unique<Embedding>(Shape{vocab, seq}, hidden, "emb");
        embedding->set_add_positional_encoding(add_positional);
        embedding->set_learned_positional(learned_positional);
        net.add_layer(move(embedding), {-1});
        net.compile();
        net.set_parameters_random();

        vector<float> ids(size_t(batch * seq));
        for (Index i = 0; i < batch * seq; ++i)
            ids[size_t(i)] = float(1 + i % (vocab - 1));

        auto make_inputs = [&]() { return vector<TensorView>{ TensorView(ids.data(), {batch, seq}) }; };

        ForwardPropagation forward_before(batch, &net);
        vector<TensorView> inputs_before = make_inputs();
        net.forward_propagate(inputs_before, forward_before, false);
        const TensorView out_before = forward_before.get_outputs();
        const vector<float> expected(out_before.as<float>(), out_before.as<float>() + out_before.size());

        const string path = (filesystem::temp_directory_path() / "opennn_emb_roundtrip.json").string();
        net.save(path);

        NeuralNetwork loaded;
        loaded.load(path);

        ForwardPropagation forward_after(batch, &loaded);
        vector<TensorView> inputs_after = make_inputs();
        loaded.forward_propagate(inputs_after, forward_after, false);
        const TensorView out_after = forward_after.get_outputs();

        Index mismatches = 0;
        const float* values = out_after.as<float>();
        for (Index i = 0; i < out_after.size(); ++i)
            if (fabs(values[i] - expected[size_t(i)]) > 1e-5f) ++mismatches;

        error_code error;
        filesystem::remove(path, error);
        filesystem::remove(filesystem::path(path).replace_extension(".bin"), error);
        return mismatches;
    }
}

TEST(BertTest, EmbeddingPlainSaveLoad)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    EXPECT_EQ(embedding_roundtrip_mismatches(                       false,                    false), 0);
    Configuration::instance().set();
}

TEST(BertTest, EmbeddingLearnedPositionalSaveLoad)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    EXPECT_EQ(embedding_roundtrip_mismatches(                       true,                    true), 0);
    Configuration::instance().set();
}

TEST(BertTest, SaveLoadRoundTripSegmentZero)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch = 2, seq = 6, vocab = 30, hidden = 8, heads = 2;
    const Index intermediate = 16, layers = 2;

    Bert bert(seq, vocab, hidden, heads, intermediate, layers);
    bert.set_parameters_random();

    vector<float> input_ids(size_t(batch * seq));
    vector<float> token_type_ids(size_t(batch * seq), 0.0f);
    for (Index i = 0; i < batch * seq; ++i)
        input_ids[size_t(i)] = float(1 + i % (vocab - 1));

    auto make_inputs = [&]() {
        return vector<TensorView>{
            TensorView(input_ids.data(),      {batch, seq}),
            TensorView(token_type_ids.data(), {batch, seq})
        };
    };

    ForwardPropagation forward_before(batch, &bert);
    vector<TensorView> inputs_before = make_inputs();
    bert.forward_propagate(inputs_before, forward_before, false);
    const TensorView out_before = forward_before.get_outputs();
    const vector<float> expected(out_before.as<float>(), out_before.as<float>() + out_before.size());

    const string path = (filesystem::temp_directory_path() / "opennn_bert_seg0.json").string();
    bert.save(path);

    NeuralNetwork loaded;
    loaded.load(path);

    ForwardPropagation forward_after(batch, &loaded);
    vector<TensorView> inputs_after = make_inputs();
    loaded.forward_propagate(inputs_after, forward_after, false);
    const TensorView out_after = forward_after.get_outputs();

    ASSERT_EQ(out_after.size(), out_before.size());
    Index mismatches = 0;
    const float* values = out_after.as<float>();
    for (Index i = 0; i < out_after.size(); ++i)
        if (fabs(values[i] - expected[size_t(i)]) > 1e-5f) ++mismatches;
    EXPECT_EQ(mismatches, 0);

    error_code error;
    filesystem::remove(path, error);
    filesystem::remove(filesystem::path(path).replace_extension(".bin"), error);
    Configuration::instance().set();
}

TEST(BertTest, AttentionSaveLoad)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch = 2, seq = 6, vocab = 10, hidden = 8, heads = 2;

    NeuralNetwork net;
    net.add_layer(make_unique<Embedding>(Shape{vocab, seq}, hidden, "emb"), {-1});
    net.add_layer(make_unique<MultiHeadAttention>(Shape{seq, hidden}, heads, "attn"));
    net.compile();
    net.set_parameters_random();

    vector<float> ids(size_t(batch * seq));
    for (Index i = 0; i < batch * seq; ++i)
        ids[size_t(i)] = float(1 + i % (vocab - 1));

    auto make_inputs = [&]() { return vector<TensorView>{ TensorView(ids.data(), {batch, seq}) }; };

    ForwardPropagation forward_before(batch, &net);
    vector<TensorView> inputs_before = make_inputs();
    net.forward_propagate(inputs_before, forward_before, false);
    const TensorView out_before = forward_before.get_outputs();
    const vector<float> expected(out_before.as<float>(), out_before.as<float>() + out_before.size());

    const string path = (filesystem::temp_directory_path() / "opennn_attn_roundtrip.json").string();
    net.save(path);

    NeuralNetwork loaded;
    loaded.load(path);

    ForwardPropagation forward_after(batch, &loaded);
    vector<TensorView> inputs_after = make_inputs();
    loaded.forward_propagate(inputs_after, forward_after, false);
    const TensorView out_after = forward_after.get_outputs();

    ASSERT_EQ(out_after.size(), out_before.size());
    Index mismatches = 0;
    const float* values = out_after.as<float>();
    for (Index i = 0; i < out_after.size(); ++i)
        if (fabs(values[i] - expected[size_t(i)]) > 1e-5f) ++mismatches;
    EXPECT_EQ(mismatches, 0);

    error_code error;
    filesystem::remove(path, error);
    filesystem::remove(filesystem::path(path).replace_extension(".bin"), error);
    Configuration::instance().set();
}

TEST(BertTest, ForwardShapeAndFinite)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch        = 2;
    const Index seq          = 6;
    const Index vocab        = 30;
    const Index hidden       = 8;
    const Index heads        = 2;
    const Index intermediate = 16;
    const Index layers       = 2;

    Bert bert(seq, vocab, hidden, heads, intermediate, layers);

    EXPECT_EQ(bert.get_sequence_length(), seq);
    EXPECT_EQ(bert.get_hidden_size(), hidden);
    EXPECT_EQ(bert.get_heads_number(), heads);

    vector<float> input_ids(size_t(batch * seq));
    vector<float> token_type_ids(size_t(batch * seq));
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < seq; ++s)
        {
            input_ids[size_t(b * seq + s)]      = float(1 + (b * seq + s) % (vocab - 1));
            token_type_ids[size_t(b * seq + s)] = float(s >= seq / 2 ? 1 : 0);
        }

    ForwardPropagation forward_propagation(batch, &bert);
    vector<TensorView> inputs = {
        TensorView(input_ids.data(),      {batch, seq}),
        TensorView(token_type_ids.data(), {batch, seq})
    };
    bert.forward_propagate(inputs, forward_propagation, false);

    const TensorView output = forward_propagation.get_outputs();
    ASSERT_EQ(output.shape.rank, 3);
    EXPECT_EQ(output.shape[0], batch);
    EXPECT_EQ(output.shape[1], seq);
    EXPECT_EQ(output.shape[2], hidden);

    const float* values = output.as<float>();
    for (Index i = 0; i < output.size(); ++i)
        EXPECT_TRUE(isfinite(values[i])) << "non-finite output at " << i;

    Configuration::instance().set();
}

TEST(BertTest, ForSequenceClassificationForward)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch = 2, seq = 6, vocab = 30, hidden = 8, heads = 2;
    const Index intermediate = 16, layers = 2, labels = 3;

    BertForSequenceClassification model(seq, vocab, hidden, heads, intermediate, layers, labels);

    vector<float> input_ids(size_t(batch * seq));
    vector<float> token_type_ids(size_t(batch * seq));
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < seq; ++s)
        {
            input_ids[size_t(b * seq + s)]      = float(1 + (b * seq + s) % (vocab - 1));
            token_type_ids[size_t(b * seq + s)] = 1.0f;
        }

    ForwardPropagation forward_propagation(batch, &model);
    vector<TensorView> inputs = {
        TensorView(input_ids.data(),      {batch, seq}),
        TensorView(token_type_ids.data(), {batch, seq})
    };
    model.forward_propagate(inputs, forward_propagation, false);

    const TensorView output = forward_propagation.get_outputs();
    ASSERT_EQ(output.shape.rank, 2);
    EXPECT_EQ(output.shape[0], batch);
    EXPECT_EQ(output.shape[1], labels);

    const float* p = output.as<float>();
    for (Index b = 0; b < batch; ++b)
    {
        float row_sum = 0.0f;
        for (Index c = 0; c < labels; ++c)
        {
            const float v = p[b * labels + c];
            EXPECT_TRUE(isfinite(v));
            EXPECT_GE(v, -1e-6f);
            EXPECT_LE(v, 1.0f + 1e-6f);
            row_sum += v;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }

    Configuration::instance().set();
}

TEST(BertTest, FirstTokenPoolGradientCheck)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = 6;
    const Index seq            = 4;
    const Index features_in    = 5;
    const Index features       = 6;
    const Index targets_number = 3;

    TabularDataset dataset(samples_number, {seq, features_in}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{seq, features_in}, Shape{features}, "Identity"));
    neural_network.add_layer(make_unique<Pooling3d>(Shape{seq, features}, PoolingMethod::FirstToken));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{features}, Shape{targets_number}, "Identity"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));

    Configuration::instance().set();
}

TEST(BertTest, SaveLoadRoundTrip)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch = 2, seq = 6, vocab = 30, hidden = 8, heads = 2;
    const Index intermediate = 16, layers = 2;

    Bert bert(seq, vocab, hidden, heads, intermediate, layers);
    bert.set_parameters_random();

    vector<float> input_ids(size_t(batch * seq));
    vector<float> token_type_ids(size_t(batch * seq));
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < seq; ++s)
        {
            input_ids[size_t(b * seq + s)]      = float(1 + (b * seq + s) % (vocab - 1));
            token_type_ids[size_t(b * seq + s)] = float(s >= seq / 2 ? 1 : 0);
        }

    auto make_inputs = [&]() {
        return vector<TensorView>{
            TensorView(input_ids.data(),      {batch, seq}),
            TensorView(token_type_ids.data(), {batch, seq})
        };
    };

    ForwardPropagation forward_before(batch, &bert);
    vector<TensorView> inputs_before = make_inputs();
    bert.forward_propagate(inputs_before, forward_before, false);
    const TensorView output_before = forward_before.get_outputs();
    const vector<float> expected(output_before.as<float>(),
                                      output_before.as<float>() + output_before.size());

    {
        ForwardPropagation forward_again(batch, &bert);
        vector<TensorView> inputs_again = make_inputs();
        bert.forward_propagate(inputs_again, forward_again, false);
        const TensorView output_again = forward_again.get_outputs();
        ASSERT_EQ(output_again.size(), output_before.size());
        const float* again = output_again.as<float>();
        for (Index i = 0; i < output_again.size(); ++i)
            EXPECT_NEAR(again[i], expected[size_t(i)], 1e-5f) << "NON-DETERMINISTIC forward at " << i;
    }

    const Index parameters_size = bert.get_parameters_size();
    const vector<float> parameters_before(bert.get_parameters_data(),
                                               bert.get_parameters_data() + parameters_size);

    const string path = (filesystem::temp_directory_path() / "opennn_bert_roundtrip.json").string();
    bert.save(path);

    NeuralNetwork loaded;
    loaded.load(path);

    ASSERT_EQ(loaded.get_parameters_size(), parameters_size);
    EXPECT_EQ(0, memcmp(loaded.get_parameters_data(), parameters_before.data(),
                             size_t(parameters_size) * sizeof(float)));

    ForwardPropagation forward_after(batch, &loaded);
    vector<TensorView> inputs_after = make_inputs();
    loaded.forward_propagate(inputs_after, forward_after, false);
    const TensorView output_after = forward_after.get_outputs();

    ASSERT_EQ(output_after.size(), output_before.size());
    const float* values = output_after.as<float>();
    for (Index i = 0; i < output_after.size(); ++i)
        EXPECT_NEAR(values[i], expected[size_t(i)], 1e-5f) << "output mismatch at " << i;

    error_code error;
    filesystem::remove(path, error);
    filesystem::remove(filesystem::path(path).replace_extension(".bin"), error);

    Configuration::instance().set();
}
