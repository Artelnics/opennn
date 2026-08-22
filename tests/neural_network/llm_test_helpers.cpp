//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L L M   T E S T   H E L P E R S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tests/pch.h"

#include "tests/neural_network/llm_test_helpers.h"

#include <cstring>
#include <random>

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/device_backend.h"
#endif

using namespace opennn;

namespace opennn_test
{

unique_ptr<Qwen3> make_qwen(const Dims& dims)
{
    return make_unique<Qwen3>(dims.seq, dims.vocab, dims.hidden, dims.layers,
                              dims.q_heads, dims.kv_heads, dims.head_dim,
                              dims.intermediate, 1000000.0f, 1.0e-6f);
}


void fill_parameters(NeuralNetwork& network)
{
    mt19937 rng(21);
    normal_distribution<float> distribution(0.0f, 0.05f);

    for (auto& layer : network.get_layers())
        for (auto& view : layer->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] = distribution(rng);
}


unique_ptr<Qwen3> make_filled_qwen(const Dims& dims)
{
    unique_ptr<Qwen3> network = make_qwen(dims);
    fill_parameters(*network);
    return network;
}


void run(NeuralNetwork& network,
         ForwardPropagation& forward_propagation,
         vector<float>& window,
         const vector<Index>& ids,
         Index past)
{
    const Index count = Index(ids.size());

    for (Index i = 0; i < count; ++i)
        window[size_t(i)] = float(ids[size_t(i)]);

    forward_propagation.past_length = past;
    forward_propagation.set_active_sequence_length(count);

    vector<TensorView> inputs = { TensorView(window.data(), {1, count}) };

    network.forward_propagate(inputs, forward_propagation, false);
}


vector<float> logits_row(const ForwardPropagation& forward_propagation, Index position)
{
    const TensorView output = forward_propagation.get_outputs();
    const Index vocabulary = output.get_shape().back();

    vector<float> row(size_t(vocabulary), 0.0f);

    const Index element_bytes = Index(type_bytes(output.get_type()));

    vector<char> host(size_t(vocabulary) * size_t(element_bytes));

    const char* source = static_cast<const char*>(output.get_data())
                       + size_t(position) * size_t(vocabulary) * size_t(element_bytes);

#ifdef OPENNN_HAS_CUDA
    if (output.is_cuda())
    {
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(host.data(), source, Index(host.size()), Device::CUDA, Device::CPU, stream);
        device::synchronize(stream);
    }
    else
#endif
        memcpy(host.data(), source, host.size());

    if (output.is_fp32())
    {
        memcpy(row.data(), host.data(), size_t(vocabulary) * sizeof(float));
        return row;
    }

    const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(host.data());

    for (Index i = 0; i < vocabulary; ++i)
    {
        const uint32_t bits = uint32_t(bf16[size_t(i)]) << 16;
        memcpy(&row[size_t(i)], &bits, sizeof(float));
    }

    return row;
}


float max_difference(const vector<float>& a, const vector<float>& b)
{
    EXPECT_EQ(a.size(), b.size());

    float result = 0.0f;

    for (size_t i = 0; i < min(a.size(), b.size()); ++i)
        result = max(result, abs(a[i] - b[i]));

    return result;
}


void round_parameters_to_bf16(NeuralNetwork& network)
{
    for (auto& layer : network.get_layers())
        for (TensorView& view : layer->get_parameter_views())
            for (Index i = 0; i < view.size(); ++i)
                view.as<float>()[i] =
                    bfloat16_to_float_host(float_to_bfloat16_host(view.as<float>()[i]));
}

}
