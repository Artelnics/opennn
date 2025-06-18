//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include "tensors.h"

namespace opennn
{

class NeuralNetwork;

//struct ForwardPropagation;
struct NeuralNetworkBackPropagation;
struct NeuralNetworkBackPropagationLM;

#ifdef OPENNN_CUDA
struct ForwardPropagationCuda;
struct NeuralNetworkBackPropagationCuda;
#endif



struct ForwardPropagation
{
    ForwardPropagation(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    pair<type*, dimensions> get_last_trainable_layer_outputs_pair() const;

    vector<vector<pair<type*, dimensions>>> get_layer_input_pairs(const vector<pair<type*, dimensions>>&, const bool&) const;

    void print() const;

    Index samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerForwardPropagation>> layers;
};


class NeuralNetwork
{

public:

   enum class ModelType{Default,
                        AutoAssociation,
                        Approximation,
                        Classification,
                        Forecasting,
                        ImageClassification,
                        TextClassification};

   NeuralNetwork(const NeuralNetwork::ModelType& = NeuralNetwork::ModelType::Default,
                 const dimensions& = {},
                 const dimensions& = {},
                 const dimensions& = {});

   NeuralNetwork(const filesystem::path&);

   void add_layer(unique_ptr<Layer>, 
                  const vector<Index>& = vector<Index>());

   bool validate_layer_type(const Layer::Type&) const;

   // Get

   bool has(const Layer::Type&) const;

   bool is_empty() const;

   const vector<string>& get_input_names() const;
   Index get_input_index(const string&) const;

   ModelType get_model_type() const;
   string get_model_type_string() const;

   const vector<string>& get_output_names() const;
   Index get_output_index(const string&) const;

   const vector<unique_ptr<Layer>>& get_layers() const;
   const unique_ptr<Layer>& get_layer(const Index&) const;
   const unique_ptr<Layer>& get_layer(const string&) const;

   Index get_layer_index(const string&) const;

   const vector<vector<Index>>& get_layer_input_indices() const;
   vector<vector<Index>> get_layer_output_indices() const;

   Index find_input_index(const vector<Index>&, const Index&) const;

   Layer* get_first(const Layer::Type&) const;

   const bool& get_display() const;

   // Set

   void set(const NeuralNetwork::ModelType& = NeuralNetwork::ModelType::Default,
            const dimensions& = {}, 
            const dimensions& = {},
            const dimensions& = {});

   void set_approximation(const dimensions&, const dimensions&, const dimensions&);
   void set_classification(const dimensions&, const dimensions&, const dimensions&);
   void set_forecasting(const dimensions&, const dimensions&, const dimensions&);
   void set_auto_association(const dimensions&, const dimensions&, const dimensions&);
   void set_image_classification(const dimensions&, const dimensions&, const dimensions&);
   void set_text_classification(const dimensions&, const dimensions&, const dimensions&);

   void set(const filesystem::path&);

   void set_layers_number(const Index&);

   void set_layer_input_indices(const vector<vector<Index>>&);
   void set_layer_inputs_indices(const Index&, const vector<Index>&);

   void set_layer_inputs_indices(const string&, const vector<string>&);
   void set_layer_inputs_indices(const string&, const initializer_list<string>&);
   void set_layer_inputs_indices(const string&, const string&);

   void set_model_type(const ModelType&);
   void set_model_type_string(const string&);
   void set_input_names(const vector<string>&);
   void set_output_names(const vector<string>&);

   void set_input_dimensions(const dimensions&);

   void set_default();

   void set_threads_number(const int&);

   void set_display(const bool&);

   // Layers

   static bool is_trainable(const Layer::Type&);

   Index get_layers_number() const;
   Index get_layers_number(const Layer::Type&) const;

   Index get_first_trainable_layer_index() const;
   Index get_last_trainable_layer_index() const;

   // Architecture

   Index get_inputs_number() const;
   Index get_outputs_number() const;

   dimensions get_input_dimensions() const;
   dimensions get_output_dimensions() const;

   // Parameters

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   vector<Index> get_layer_parameter_numbers() const;

   void set_parameters(const Tensor<type, 1>&) const;

   // Parameters initialization

   void set_parameters_constant(const type&) const;

   void set_parameters_random() const;

   // Output

   template <Index input_rank, Index output_rank>
   Tensor<type, output_rank> calculate_outputs(const Tensor<type, input_rank>& inputs)
   {
       const Index layers_number = get_layers_number();

       if (layers_number == 0)
           return Tensor<type, output_rank>();

       const Index batch_size = inputs.dimension(0);
       const Index inputs_number = inputs.dimension(1);

       ForwardPropagation forward_propagation(batch_size, this);

       dimensions input_dimensions;
       input_dimensions.reserve(input_rank);
       input_dimensions.push_back(batch_size);

       if constexpr (input_rank >= 2) input_dimensions.push_back(inputs.dimension(1));
       if constexpr (input_rank >= 3) input_dimensions.push_back(inputs.dimension(2));
       if constexpr (input_rank >= 4) input_dimensions.push_back(inputs.dimension(3));
       static_assert(input_rank >= 2 && input_rank <= 4, "Unsupported input rank");

       const pair<type*, dimensions> input_pair((type*)inputs.data(), {{batch_size, inputs_number}});

       forward_propagate({input_pair}, forward_propagation, false);

       const pair<type*, dimensions> outputs_pair
           = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

       if constexpr (output_rank == 2)
           return tensor_map<2>(outputs_pair);
       else if constexpr (output_rank == 3)
           return tensor_map<3>(outputs_pair);
       else if constexpr (output_rank == 4)
           return tensor_map<4>(outputs_pair);
       else
           static_assert(output_rank >= 2 && output_rank <= 4, "Unsupported output rank");
   }

   Tensor<type, 2> calculate_scaled_outputs(type*, Tensor<Index, 1>& );

   Tensor<type, 2> calculate_directional_inputs(const Index&, const Tensor<type, 1>&, const type&, const type&, const Index& = 101) const;

   Index calculate_image_output(const filesystem::path&);

   // Serialization

   Tensor<string, 2> get_dense2d_layers_information() const;
   Tensor<string, 2> get_probabilistic_layer_information() const;

   void from_XML(const XMLDocument&);
   void inputs_from_XML(const XMLElement*);
   void layers_from_XML(const XMLElement*);
   void outputs_from_XML(const XMLElement*);

   void to_XML(XMLPrinter&) const;

   virtual void print() const;
   void save(const filesystem::path&) const;
   void save_parameters(const filesystem::path&) const;

   void load(const filesystem::path&);
   void load_parameters_binary(const filesystem::path&);

   vector<string> get_layer_names() const;
   vector<string> get_layer_types_string() const;

   void save_outputs(Tensor<type, 2>&, const filesystem::path&);

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          ForwardPropagation&,
                          const bool& = false) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          const Tensor<type, 1>&, 
                          ForwardPropagation&) const;

   string get_expression() const;


#ifdef OPENNN_CUDA

public:

    void create_cuda() const;
    void destroy_cuda() const;

    void allocate_parameters_device();
    void free_parameters_device();
    void copy_parameters_device();
    void copy_parameters_host();

    void forward_propagate_cuda(const vector<float*>&,
                                ForwardPropagationCuda&,
                                const bool& = false) const;

    void set_parameters_cuda(const float*);

    float* calculate_outputs_cuda(float*, const Index&);

protected:

    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;

#endif

protected:

   string name = "neural_network";

   NeuralNetwork::ModelType model_type = NeuralNetwork::ModelType::Default;

   vector<string> input_names;

   vector<string> output_names;

   vector<unique_ptr<Layer>> layers;

   vector<vector<Index>> layer_input_indices;

   bool display = true;

};

struct NeuralNetworkBackPropagation
{
    NeuralNetworkBackPropagation(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagation>>& get_layers() const;

    void print() const;

    NeuralNetwork* get_neural_network() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagation>> layers;
};


#ifdef OPENNN_CUDA

struct NeuralNetworkBackPropagationCuda
{
    NeuralNetworkBackPropagationCuda(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationCuda>>& get_layers() const;

    void print();

    void free();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationCuda>> layers;
};

#endif


struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network = nullptr)
    {
        neural_network = new_neural_network;
    }

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationLM>>& get_layers() const
    {
        return layers;
    }

    NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }


    void print()
    {
        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers[i]->print();
        }
    }

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationLM>> layers;
};


#ifdef OPENNN_CUDA

struct ForwardPropagationCuda
{
    ForwardPropagationCuda(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    float* get_last_trainable_layer_outputs_device() const;

    vector<vector<float*>> get_layer_inputs_device(const vector<float*>&, const bool&) const;

    void print();

    void free();

    Index samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerForwardPropagationCuda>> layers;
};

#endif


}

#endif

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
