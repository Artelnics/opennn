//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   A U D I T   ( T E M P O R A R Y   D R I V E R )
//
//   Measurement driver, not a regression test: builds representative
//   transformer / LSTM training setups exactly the way Optimizer::train does
//   (joint forward/delta arena) and dumps the memory_debug breakdown.
//   Run with OPENNN_MEMORY_DEBUG=1 and --gtest_filter=MemoryAudit.*

#include "tests/pch.h"

#include "opennn/core/configuration.h"
#include "opennn/core/memory_debug.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/training_context.h"

using namespace opennn;

namespace
{

// Mirrors the mini-batch path in Optimizer::train: co-planned lifetimes,
// joint arena, deltas bound into the forward arena.
void audit(const string& label, NeuralNetwork& network, Loss& loss, const Index batch_size)
{
    memory_debug::reset();

    TrainingContext context(batch_size, loss, false, nullptr,
                            /*joint_gradient_arena*/ true);
    ForwardPropagation& forward_propagation = context.forward;
    BackPropagation& back_propagation = context.backward;

    const double mib = 1024.0 * 1024.0;
    cout << "\n[AUDIT] ===== " << label << " =====\n"
         << "[AUDIT] batch=" << batch_size
         << " fp_arena_mib=" << fixed << setprecision(2) << double(forward_propagation.arena.byte_size()) / mib
         << " bp_arena_mib=" << double(back_propagation.arena.byte_size()) / mib
         << " gradient_allocation_mib=" << double(back_propagation.gradient.byte_size()) / mib
         << " gradient_logical_mib=" << double(back_propagation.gradient_logical_bytes()) / mib
         << " parameters=" << network.get_parameters_number() << "\n";

    memory_debug::print(cout);
}

void audit_transformer(const string& label, const float dropout_rate)
{
    const Index sequence_length = 256;
    const Index vocabulary_size = 4000;
    const Index embedding_dimension = 256;
    const Index heads_number = 8;
    const Index feed_forward_dimension = 1024;
    const Index layers_number = 4;
    const Index batch_size = 32;

    Transformer transformer(sequence_length, sequence_length,
                            vocabulary_size, vocabulary_size,
                            embedding_dimension, heads_number,
                            feed_forward_dimension, layers_number);
    transformer.set_dropout_rate(dropout_rate);

    Loss loss(&transformer, nullptr);
    loss.set_error(Loss::Error::CrossEntropy3d);

    audit(label + format(" seq={} embed={} heads={} ffn={} layers={} dropout={}",
                         sequence_length, embedding_dimension, heads_number,
                         feed_forward_dimension, layers_number, dropout_rate),
          transformer, loss, batch_size);
}

void audit_transformer_benchmark(const string& label)
{
    constexpr Index sequence_length = 130;
    constexpr Index vocabulary_size = 20000;
    constexpr Index embedding_dimension = 1024;
    constexpr Index heads_number = 16;
    constexpr Index feed_forward_dimension = 4096;
    constexpr Index layers_number = 2;
    constexpr Index batch_size = 32;

    Transformer transformer(sequence_length, sequence_length,
                            vocabulary_size, vocabulary_size,
                            embedding_dimension, heads_number,
                            feed_forward_dimension, layers_number);
    transformer.set_attention_sdpa_min_sequence_length(128);

    Loss loss(&transformer, nullptr);
    loss.set_error(Loss::Error::CrossEntropy3d);

    audit(label + format(" seq={} embed={} heads={} ffn={} layers={} sdpa_min={}",
                         sequence_length, embedding_dimension, heads_number,
                         feed_forward_dimension, layers_number, 128),
          transformer, loss, batch_size);
}

void audit_lstm(const string& label)
{
    const Index time_steps = 200;
    const Index input_features = 64;
    const Index hidden_features = 256;
    const Index batch_size = 64;

    NeuralNetwork network;
    network.add_layer(make_unique<LongShortTermMemory>(
                          Shape{time_steps, input_features}, Shape{hidden_features}),
                      {-1});
    network.add_layer(make_unique<opennn::Dense>(Shape{hidden_features}, Shape{1}, "Identity"),
                      {network.get_layers_number() - 1});
    network.compile();

    Loss loss(&network, nullptr);
    loss.set_error(Loss::Error::MeanSquaredError);

    audit(label + format(" T={} in={} hidden={}", time_steps, input_features, hidden_features),
          network, loss, batch_size);
}

}

TEST(MemoryAudit, TransformerCpuFp32DropoutOn)
{
    audit_transformer("transformer cpu fp32", 0.1f);
}

TEST(MemoryAudit, TransformerCpuFp32DropoutOff)
{
    audit_transformer("transformer cpu fp32", 0.0f);
}

TEST(MemoryAudit, LstmCpuFp32)
{
    audit_lstm("lstm cpu fp32");
}

#ifdef OPENNN_HAS_CUDA

TEST(MemoryAudit, TransformerCudaFp32DropoutOn)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    audit_transformer("transformer cuda fp32", 0.1f);
}

TEST(MemoryAudit, TransformerCudaFp32DropoutOff)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    audit_transformer("transformer cuda fp32", 0.0f);
}

TEST(MemoryAudit, TransformerCudaBf16DropoutOn)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    audit_transformer("transformer cuda bf16", 0.1f);
}

TEST(MemoryAudit, TransformerBenchmarkCudaBf16)
{
    Configuration::instance().set(Device::CUDA, Type::BF16);
    audit_transformer_benchmark("transformer benchmark cuda bf16");
}

TEST(MemoryAudit, LstmCudaFp32)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    audit_lstm("lstm cuda fp32");
}

#endif
