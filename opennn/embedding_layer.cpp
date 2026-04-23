//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "embedding_layer.h"
#include "neural_network.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "random_utilities.h"

namespace opennn
{

Embedding::Embedding(const Shape& new_input_shape,
                     Index new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);

    name = "Embedding";
    layer_type = LayerType::Embedding;
}

// Getters

Shape Embedding::get_output_shape() const
{
    return {sequence_length, embedding_dimension};
}

vector<Shape> Embedding::get_parameter_shapes() const
{
    return {{vocabulary_size, embedding_dimension}}; // weights
}

// Setters

void Embedding::set(const Index new_vocabulary_size,
                    Index new_sequence_length,
                    Index new_embedding_dimension,
                    const string& new_label)
{
    sequence_length = new_sequence_length;
    vocabulary_size = new_vocabulary_size;
    embedding_dimension = new_embedding_dimension;
    embedding_scale = sqrt(static_cast<type>(new_embedding_dimension));
    label = new_label;

    positional_encoding = MatrixR::Zero(new_sequence_length, new_embedding_dimension);

    const type half_depth = type(new_embedding_dimension) / 2;

    VectorR divisors(new_embedding_dimension);
    for(Index j = 0; j < new_embedding_dimension; ++j)
        divisors(j) = pow(type(10000), (j < Index(half_depth) ? j : j - Index(half_depth)) / half_depth);

    #pragma omp parallel for collapse(2)
    for(Index i = 0; i < new_sequence_length; ++i)
        for(Index j = 0; j < new_embedding_dimension; ++j)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / divisors(j))
                : cos(i / divisors(j));
#ifdef OPENNN_WITH_CUDA

    const Index pe_size = new_sequence_length * new_embedding_dimension;
    positional_encoding_device.resize(pe_size);
    positional_encoding_device.resize_device(pe_size);
    CHECK_CUDA(cudaMemcpy(positional_encoding_device.device(),
                          positional_encoding.data(),
                          pe_size * sizeof(float),
                          cudaMemcpyHostToDevice));

#endif
}

// Parameter initialization

void Embedding::set_parameters_random()
{
    if(parameters[Weight].empty()) return;

    MatrixMap weights = matrix_map(parameters[Weight]);
    set_random_normal(weights, type(0), type(1));

    weights.row(0).setZero();
}

void Embedding::set_parameters_glorot()
{
    if(parameters[Weight].empty()) return;

    const type limit = sqrt(type(6.0) / (vocabulary_size + embedding_dimension));

    MatrixMap weights = matrix_map(parameters[Weight]);

    weights.setRandom();
    weights *= limit;

    weights.row(0).setZero();
}

#ifdef OPENNN_WITH_CUDA

void Embedding::init_cuda(Index batch_size)
{
    if(dropout_rate <= type(0)) return;
    if(sequence_length == 0 || embedding_dimension == 0) return;

    if(dropout_arguments.descriptor)    { cudnnDestroyDropoutDescriptor(dropout_arguments.descriptor); dropout_arguments.descriptor = nullptr; }
    if(dropout_arguments.states)        { cudaFree(dropout_arguments.states);        dropout_arguments.states = nullptr; }
    if(dropout_arguments.reserve_space) { cudaFree(dropout_arguments.reserve_space); dropout_arguments.reserve_space = nullptr; }

    cudnnTensorDescriptor_t temp_desc = nullptr;
    cudnnCreateTensorDescriptor(&temp_desc);
    cudnnSetTensor4dDescriptor(temp_desc, CUDNN_TENSOR_NHWC, CUDNN_ACTIVATION_DTYPE,
                               static_cast<int>(batch_size),
                               static_cast<int>(embedding_dimension),
                               static_cast<int>(sequence_length),
                               1);

    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_arguments.descriptor));
    CHECK_CUDNN(cudnnDropoutGetStatesSize(Device::get_cudnn_handle(), &dropout_arguments.states_size));
    CHECK_CUDA(cudaMalloc(&dropout_arguments.states, dropout_arguments.states_size));
    CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_arguments.descriptor, Device::get_cudnn_handle(),
                                          static_cast<float>(dropout_rate),
                                          dropout_arguments.states, dropout_arguments.states_size,
                                          static_cast<unsigned long long>(random_integer(0, 1 << 30))));
    CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(temp_desc, &dropout_arguments.reserve_size));
    CHECK_CUDA(cudaMalloc(&dropout_arguments.reserve_space, dropout_arguments.reserve_size));

    dropout_arguments.rate = dropout_rate;

    cudnnDestroyTensorDescriptor(temp_desc);
}

#endif

// Forward / back propagation

void Embedding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    const Index batch_size = forward_propagation.batch_size;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const Index total_elements = batch_size * sequence_length * embedding_dimension;

        const float* positional_encoding_data = add_positional_encoding
            ? positional_encoding_device.device()
            : nullptr;

        TensorView& output_view = forward_views[Output][0];
        output_view.dispatch([&](auto tag) {
            using T = decltype(tag);
            embedding_forward_cuda<T>(
                total_elements,
                forward_views[Input][0].as<float>(),
                parameters[Weight].as<float>(),
                positional_encoding_data,
                output_view.as<T>(),
                sequence_length, embedding_dimension, vocabulary_size,
                scale_embedding, add_positional_encoding);
        });
    }
    else
#endif
    {
        const Index total_tokens = batch_size * sequence_length;

        const TensorView& output_view = forward_views[Output][0];
        MatrixMap outputs(output_view.data, total_tokens, embedding_dimension);

        const MatrixMap weights(parameters[Weight].data, vocabulary_size, embedding_dimension);

        const type* input_indices = forward_views[Input][0].data;

        static std::atomic<bool> out_of_range_warned{false};

        #pragma omp parallel for
        for(Index i = 0; i < total_tokens; ++i)
        {
            const Index token_id = static_cast<Index>(input_indices[i]);

            if(token_id < 0 || token_id >= weights.rows())
            {
                if(!out_of_range_warned.exchange(true))
                    std::cerr << "Embedding warning: token id " << token_id
                              << " out of range [0, " << weights.rows()
                              << "); zeroing row. Further warnings suppressed.\n";
                outputs.row(i).setZero();
                continue;
            }

            outputs.row(i).noalias() = weights.row(token_id);

            if(scale_embedding)
                outputs.row(i) *= embedding_scale;

            if(add_positional_encoding && token_id > 0)
                outputs.row(i) += positional_encoding.row(i % sequence_length);
        }
    }

    if (is_training && dropout_rate > type(0))
        dropout(forward_views[Output][0], dropout_arguments);
}

void Embedding::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    TensorView& output_gradient = backward_views[OutputGradient][0];

    if (dropout_rate > type(0))
        dropout_gradient(output_gradient, output_gradient, dropout_arguments);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const Index total_elements = forward_propagation.batch_size * sequence_length * embedding_dimension;

        output_gradient.dispatch([&](auto tag) {
            using T = decltype(tag);
            embedding_backward_cuda<T>(
                total_elements,
                forward_views[Input][0].as<float>(),
                output_gradient.as<T>(),
                gradient_views[Weight].as<float>(),
                embedding_dimension, vocabulary_size, scale_embedding);
        });

        return;
    }
#endif

    embedding_backward(forward_views[Input][0],
                       output_gradient,
                       gradient_views[Weight],
                       embedding_dimension,
                       scale_embedding);
}

// Serialization

void Embedding::from_XML(const XmlDocument& document)
{
    const XmlElement* embedding_layer_element = get_xml_root(document, "Embedding");

    const string new_label = read_xml_string(embedding_layer_element, "Label");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_label);

    set_scale_embedding(read_xml_bool(embedding_layer_element, "ScaleEmbedding"));
    set_add_positional_encoding(read_xml_bool(embedding_layer_element, "AddPositionalEncoding"));
}

void Embedding::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Embedding");

    write_xml_properties(printer, {
        {"Label", label},
        {"VocabularySize", to_string(get_vocabulary_size())},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingSize", to_string(get_embedding_dimension())},
        {"ScaleEmbedding", to_string(scale_embedding)},
        {"AddPositionalEncoding", to_string(add_positional_encoding)}
    });

    printer.close_element();
}

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
