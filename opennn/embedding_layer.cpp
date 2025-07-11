//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "embedding_layer.h"

namespace opennn
{

Embedding::Embedding(const dimensions& new_input_dimensions,
                     const Index& new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_dimensions[0], new_input_dimensions[1], new_embedding_dimension, new_label);

    name = "Embedding";

    label = new_label;
}


Index Embedding::get_vocabulary_size() const
{
    return weights.dimension(0);
}


Index Embedding::get_sequence_length() const
{
    return sequence_length;
}


Index Embedding::get_embedding_dimension() const
{
    return weights.dimension(1);
}


dimensions Embedding::get_input_dimensions() const
{
    return { sequence_length };
}


dimensions Embedding::get_output_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


vector<pair<type *, Index> > Embedding::get_parameter_pairs() const
{
    return {{(type*)(weights.data()), weights.size()}};
}


void Embedding::set(const Index& new_vocabulary_size,
                    const Index& new_sequence_length,
                    const Index& new_embedding_dimension,
                    const string& new_label)
{
    sequence_length = new_sequence_length;

    weights.resize(new_vocabulary_size, new_embedding_dimension);

    set_parameters_random();

    positional_encoding.resize(sequence_length, new_embedding_dimension);

    positional_encoding.setZero();

    const type half_depth = type(new_embedding_dimension)/2;

    #pragma omp parallel for collapse(2)

    for (Index i = 0; i < sequence_length; i++)
        for (Index j = 0; j < new_embedding_dimension; j++)
            positional_encoding(i, j) = (j < Index(half_depth))
            ? sin(i / pow(10000, j / half_depth))
            : cos(i / pow(10000, (j - Index(half_depth)) / half_depth));

    label = new_label;
}


void Embedding::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Embedding::set_parameters_random()
{
    if(weights.size() == 0) return;

    // First row must be 0s because input value 0 is padding

    weights.setRandom();
    weights.chip(0, 0).setZero();
}


void Embedding::embedding_lookup(const Tensor<type, 2>& inputs, Tensor<type, 3>& outputs)
{
    const Index batch_size = inputs.dimension(0);
    const type coefficient = sqrt(type(get_embedding_dimension()));

    // #pragma omp parallel for

    for (Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        auto sample_output = outputs.chip(sample_index, 0);

        for (Index word_index = 0; word_index < sequence_length; word_index++)
        {
            const Index token_id = inputs(sample_index, word_index);

            if (token_id < 0 || token_id >= weights.dimension(0))
                throw runtime_error("Invalid token_id \n");

            const auto embedding = weights.chip(token_id, 0);

            if(scale_embedding)
                sample_output.chip(word_index, 0) = embedding*coefficient;
            else
                sample_output.chip(word_index, 0) = embedding;
        }
    }
}


void Embedding::add_positional_encodings(Tensor<type, 3>& embeddings) const
{ 
    const Index batch_size = embeddings.dimension(0);
    const Index embedding_dimension = embeddings.dimension(2);

    embeddings.device(*thread_pool_device) += positional_encoding
        .reshape(array<Index, 3>({1, sequence_length, embedding_dimension}))
        .broadcast(array<Index, 3>({batch_size, 1, 1}));
}


void Embedding::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                  const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map<2>(input_pairs[1]);

    EmbeddingForwardPropagation* embedding_forward_propagation =
        static_cast<EmbeddingForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = embedding_forward_propagation->outputs;

    embedding_lookup(inputs, outputs);

    if(positional_encoding_xxx)
        add_positional_encodings(outputs);

    if(is_training && dropout_rate > 0)
        dropout(outputs, dropout_rate);
}


void Embedding::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>&,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index embedding_dimension = get_embedding_dimension();

    const Index batch_size = input_pairs[0].second[0];
    const Index sequence_length = input_pairs[0].second[1];

    const TensorMap<Tensor<type, 2>> inputs = tensor_map<2>(input_pairs[0]);

    if (delta_pairs.size() > 1)
        add_deltas(delta_pairs);

    TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_pairs[0]);

    // Back propagation

    EmbeddingBackPropagation* embedding_back_propagation =
        static_cast<EmbeddingBackPropagation*>(back_propagation.get());

    Tensor<type, 2>& weight_deltas = embedding_back_propagation->weight_deltas;

    weight_deltas.setZero();

    if(scale_embedding)
        deltas.device(*thread_pool_device) = deltas*sqrt(type(embedding_dimension));

    #pragma omp parallel for

    for(Index sample_index = 0; sample_index < batch_size; sample_index++)
    {
        auto sample_deltas = deltas.chip(sample_index, 0);

        for(Index word_index = 0; word_index < sequence_length; word_index++)
        {
            weight_deltas.chip(Index(inputs(sample_index, word_index)), 0)
            += sample_deltas.chip(word_index, 0);
        }
    }
}


void Embedding::print() const
{
    cout << "Embedding Layer" << endl;
    cout << "Label: " << label << endl;
    cout << "Type: Embedding" << endl;

    cout << "Input dimensions: ";
    print_vector(get_input_dimensions());

    cout << "Output dimensions: ";
    print_vector(get_output_dimensions());

    cout << "Vocabulary size: " << get_vocabulary_size() << endl;
    cout << "Sequence length: " << get_sequence_length() << endl;
    cout << "Embedding dimension: " << get_embedding_dimension() << endl;

    cout << "Dropout rate: " << dropout_rate << endl;

    cout << "Weights dimensions: " << weights.dimensions() << endl;

    //cout << "Weights:\n " << weights << endl;
    //cout << "Positional encoding:\n" << positional_encoding << endl;
}


void Embedding::from_XML(const XMLDocument& document)
{
    const XMLElement* embedding_layer_element = document.FirstChildElement("Embedding");

    if(!embedding_layer_element)
        throw runtime_error("Embedding element is nullptr.\n");

    const string new_name = read_xml_string(embedding_layer_element, "Name");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_name);

    Index index = 0;

    //set_parameters(to_type_vector(read_xml_string(embedding_layer_element, "Parameters"), " "), index);
}


void Embedding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Embedding");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "VocabularySize", to_string(get_vocabulary_size()));
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingSize", to_string(get_embedding_dimension()));

    //Tensor<type, 1> parameters;
    //get_parameters(parameters);

    //add_xml_element(printer, "Parameters", tensor_to_string(parameters));

    printer.CloseElement();  
}


EmbeddingForwardPropagation::EmbeddingForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> EmbeddingForwardPropagation::get_output_pair() const
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index sequence_length = embedding_layer->get_sequence_length();

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();

    return {(type*)outputs.data(), {batch_size, sequence_length, embedding_dimension}};
}


void EmbeddingForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    const Embedding* embedding_layer = static_cast<Embedding*>(new_layer);

    batch_size = new_batch_size;

    const Index sequence_length = embedding_layer->get_sequence_length();

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();

    // Outputs

    outputs.resize(batch_size, sequence_length, embedding_dimension);
}


void EmbeddingForwardPropagation::print() const
{
    cout << "Outputs dimensions:" << endl;
    //       cout << output_dimensions << endl;
    cout << "Outputs:" << endl;
    //       cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
}


EmbeddingBackPropagation::EmbeddingBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> EmbeddingBackPropagation::get_input_derivative_pairs() const
{
    return vector<pair<type*, dimensions>>();
}

vector<pair<type*, Index>> EmbeddingBackPropagation::get_parameter_delta_pairs() const
{
    return {
        {(type*)weight_deltas.data(), weight_deltas.size()}
    };
}



void EmbeddingBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    const Embedding* embedding_layer = static_cast<Embedding*>(new_layer);

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();
    const Index vocabulary_size = embedding_layer->get_vocabulary_size();

    weight_deltas.resize(vocabulary_size, embedding_dimension);
}


void EmbeddingBackPropagation::print() const
{
}

#ifdef OPENNN_CUDA

void Embedding::forward_propagate_cuda(const vector<float*>& inputs_device,
                                       unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                       const bool& is_training)
{

}


void Embedding::back_propagate_cuda(const vector<float*>&,
                                    const vector<float*>&,
                                    unique_ptr<LayerForwardPropagationCuda>&,
                                    unique_ptr<LayerBackPropagationCuda>&) const
{

}


vector<pair<float*, Index>> Embedding::get_parameter_pair_device() const
{
    return vector<pair<float*, Index>>();
}


void Embedding::allocate_parameters_device()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    CHECK_CUDA(cudaMalloc(&weights_device, inputs_number * outputs_number * sizeof(float)));
}


void Embedding::free_parameters_device()
{
    cudaFree(weights_device);

    weights_device = nullptr;
}


void Embedding::copy_parameters_device()
{
    if (!weights_device)
        cout << "Weights device is null" << endl;

    CHECK_CUDA(cudaMemcpy(weights_device, weights.data(), weights.size() * sizeof(type), cudaMemcpyHostToDevice));
}


void Embedding::copy_parameters_host()
{
    if (!weights_device)
        cout << "Synaptic weights is null" << endl;

    CHECK_CUDA(cudaMemcpy(weights.data(), weights_device, weights.size() * sizeof(type), cudaMemcpyDeviceToHost));
}


EmbeddingForwardPropagationCuda::EmbeddingForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void EmbeddingForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;
}


void EmbeddingForwardPropagationCuda::print() const
{

}


EmbeddingBackPropagationCuda::EmbeddingBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


vector<pair<float*, Index>> EmbeddingBackPropagationCuda::get_parameter_delta_pair_device() const
{
    return vector<pair<float*, Index>>();
}


void EmbeddingBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;
}


void EmbeddingBackPropagationCuda::print() const
{

}

REGISTER(LayerForwardPropagationCuda, EmbeddingForwardPropagationCuda, "Embedding")
REGISTER(LayerBackPropagationCuda, EmbeddingBackPropagationCuda, "Embedding")

//REGISTER_FORWARD_CUDA("Embedding", EmbeddingForwardPropagationCuda);
//REGISTER_BACK_CUDA("Embedding", EmbeddingBackPropagationCuda);

#endif

REGISTER(Layer, Embedding, "Embedding")
REGISTER(LayerForwardPropagation, EmbeddingForwardPropagation, "Embedding")
REGISTER(LayerBackPropagation, EmbeddingBackPropagation, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
